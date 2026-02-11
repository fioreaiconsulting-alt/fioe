import re, json, requests
from chat_utils import dedupe, detect_seniority, extract_sectors_regex, extract_country_hint, country_to_cc
from chat_state import history_for_prompt
import os
import sys
import threading
import time

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY","")
GEMINI_MODEL   = os.getenv("GEMINI_SOURCING_MODEL","gemini-2.5-flash-lite")
HF_API_KEY     = os.getenv("HUGGINGFACE_API_KEY","")
# allow alternate env var forms for HF model names; keep common default
HF_MODEL       = os.getenv("HUGGINGFACE_MODEL","meta-llama/Meta-Llama-3-8B-Instruct")
HF_TIMEOUT     = float(os.getenv("HUGGINGFACE_TIMEOUT","45"))

_gemini=False
try:
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        _gemini=True
    else:
        genai=None
except Exception:
    genai=None
    _gemini=False

def hf_headers():
    return {"Authorization": f"Bearer {HF_API_KEY}", "Content-Type":"application/json"}

def hf_invoke(prompt, max_new_tokens=512):
    if not HF_API_KEY: return None
    try:
        r=requests.post(f"https://api-inference.huggingface.co/models/{HF_MODEL}",
                        headers=hf_headers(),
                        json={"inputs":prompt,"parameters":{"max_new_tokens":max_new_tokens,"return_full_text":False}},
                        timeout=HF_TIMEOUT)
        if r.status_code!=200: return None
        data=r.json()
        if isinstance(data,list) and data:
            return (data[0].get("generated_text") or "").strip()
        if isinstance(data,dict):
            return (data.get("generated_text") or "").strip()
    except Exception:
        return None
    return None

# Robust JSON fragment extractor used for model outputs
def _extract_json_fragment(text: str):
    """
    Attempts to find and parse a JSON object inside an arbitrary text block.
    Repairs common formatting issues (trailing commas) when possible.
    Returns parsed dict or None.
    """
    if not isinstance(text, str):
        return None
    s = text.strip()
    # Find first { and the matching } by scanning (to avoid capturing too much)
    start = s.find("{")
    if start == -1:
        return None
    # Try to find a matching closing brace by simple heuristic: find last '}' after start
    end = s.rfind("}")
    if end == -1 or end <= start:
        return None
    frag = s[start:end+1]
    try:
        return json.loads(frag)
    except Exception:
        # Attempt simple repairs: remove trailing commas before } or ]
        repaired = re.sub(r",\s*}", "}", frag)
        repaired = re.sub(r",\s*\]", "]", repaired)
        # Remove non-JSON prefixes/suffixes (e.g., "JSON:" or similar)
        repaired = re.sub(r'^[^{]*', '', repaired)
        try:
            return json.loads(repaired)
        except Exception:
            # As a last resort, try to find any {...} balanced pairs using a stack scan to extract the smallest valid JSON
            stack = []
            for i, ch in enumerate(s[start:]):
                if ch == "{":
                    stack.append(i+start)
                elif ch == "}":
                    if stack:
                        start_idx = stack[0]
                        end_idx = i+start
                        candidate = s[start_idx:end_idx+1]
                        try:
                            return json.loads(candidate)
                        except Exception:
                            continue
            return None

def gemini_json_extract(text):
    return _extract_json_fragment(text)

# --- Affected section: expanded normalization + preamble stripping for robust title extraction ---
_REQUEST_PREAMBLES = re.compile(
    r"""^\s*(
        i\s+(?:want\s+to\s+find|want\s+to|need(?:\s+a|\s+an)?|am\s+looking\s+for|would\s+like\s+to\s+find|wish\s+to\s+find)\s+|
        let'?s\s+(?:try|do|find|search)\s+|
        look(?:ing)?\s+for\s+|
        search(?:\s+for)?\s+|
        find\s+|
        hire(?:\s+a|\s+an)?\s+|hiring\s+|recruit(?:ing)?\s+for\s+
    )""",
    re.I | re.X
)

def _strip_request_preambles(text: str) -> str:
    if not isinstance(text, str): return text
    s = text.strip()
    s = re.sub(_REQUEST_PREAMBLES, "", s).strip()
    s = re.sub(r"\s+(?:please|thanks|thank\s+you|instead)\s*$", "", s, flags=re.I).strip()
    return s

# --- Affected section: improved job_title validation for suggestion inputs ---
_META_PATTERNS = re.compile(
    r'\b('
    r'tell me( the)?( search)?( parameter| parameters)?'
    r'|what( is| are)?( the)? (parameter|parameters|options|capabilities)'
    r'|how to'
    r'|show me'
    r'|help'
    r'|i need (help|assistance)'
    r'|explain'
    r'|guide me'
    r')\b', re.I
)

def _looks_like_meta_query(text: str) -> bool:
    return bool(_META_PATTERNS.search(text or ""))

def _singleton_country(text: str) -> str:
    t=(text or "").strip()
    if not t: return ""
    return t if country_to_cc(t) else ""

def _looks_like_user_feedback_or_command(text: str) -> bool:
    s = text.lower().strip()
    return (
        s.startswith("yes") or
        "need" in s or
        "more" in s or
        "suggest" in s or
        "but" in s or
        "provide" in s or
        len(s.split()) > 60
    )

def _split_slash_input(text: str):
    parts = [x.strip() for x in text.split("/") if x.strip()]
    result = {"job_title": "", "companies": [], "sectors": []}
    if len(parts) == 3:
        result["job_title"] = parts[0]
        result["companies"] = [c.strip() for c in parts[1].split(",") if c.strip()]
        result["sectors"] = [parts[2]]
    elif len(parts) == 2:
        result["job_title"] = parts[0]
        if "," in parts[1]:
            result["companies"] = [c.strip() for c in parts[1].split(",") if c.strip()]
        else:
            result["sectors"] = [parts[1]]
    return result if result["job_title"] else None

def _strip_find_prefix(text: str) -> str:
    if not isinstance(text, str): return text
    return re.sub(r'^\s*find\s+', '', text, flags=re.I).strip()

def _split_job_titles_list(title_block: str):
    if not isinstance(title_block, str): return []
    parts = [p.strip() for p in re.split(r'[;,]+', title_block) if p and p.strip()]
    cleaned = []
    for p in parts:
        if len(re.sub(r'[^A-Za-z]', '', p)) < 2:
            continue
        cleaned.append(p)
    return dedupe(cleaned)

def regex_parse(msg: str):
    if not isinstance(msg, str):
        msg = ""
    s=_strip_request_preambles(msg.strip())

    m=re.match(r"^(?P<title>.+?)\s+in\s+(?P<country>.+?)\s+at\s+(?P<company>.+)$", s, re.I)
    if not m: m=re.match(r"^(?P<title>.+?)\s+at\s+(?P<company>.+)$", s, re.I)
    if not m: m=re.match(r"^(?P<title>.+?)\s+in\s+(?P<country>.+)$", s, re.I)
    if not m: m=re.match(r"^(?:look(?:ing)?\s+for\s+)?(?P<title>.+?)\s+in\s+(?P<country>[A-Za-z \.-]{2,})$", s, re.I)
    if not m: m=re.match(r"^(?:let'?s\s+(?:try|do|find|search)\s+)?(?P<title>.+?)\s+(?:search\s+)?instead$", s, re.I)
    if not m: m=re.match(r"^(?:look(?:ing)?\s+for\s+)?(?P<title>.+?)$", s, re.I)

    if m:
        gd=m.groupdict()
        comp_raw=(gd.get("company") or "").strip()
        companies=[c.strip() for c in comp_raw.split(",") if c.strip()] if comp_raw else []
        title_raw=(gd.get("title") or "").strip()
        titles=_split_job_titles_list(title_raw) if title_raw else []
        primary = titles[0] if titles else title_raw
        return {
            "job_title": primary,
            "job_titles": titles if titles else ([primary] if primary else []),
            "country": (gd.get("country") or "").strip(),
            "companies": companies,
            "sectors": extract_sectors_regex(s)
        }

    maybe_country=_singleton_country(s)
    if maybe_country:
        return {"job_title":"", "job_titles": [], "country":maybe_country, "companies":[], "sectors":extract_sectors_regex(s)}

    if _looks_like_meta_query(s):
        return {"job_title":"", "job_titles": [], "country":"", "companies":[], "sectors":extract_sectors_regex(s)}

    slash_result = _split_slash_input(s)
    if slash_result:
        return {
            "job_title": slash_result.get("job_title",''),
            "job_titles": [slash_result.get("job_title",'')] if slash_result.get("job_title") else [],
            "country":"",
            "companies": slash_result.get("companies", []),
            "sectors": slash_result.get("sectors", []),
        }

    titles = _split_job_titles_list(s)
    if titles:
        return {"job_title": titles[0], "job_titles": titles, "country":"", "companies":[], "sectors":extract_sectors_regex(s)}

    if (
        not re.search(r'[?]', s)
        and len(s) <= 180
        and not re.search(r'\b(and|or|the|a|an|to|from|with|for|how|what|which|where)\b', s.lower())
        and not _looks_like_user_feedback_or_command(s)
    ):
        return {"job_title": s.strip(), "job_titles":[s.strip()], "country":"", "companies":[], "sectors":extract_sectors_regex(s)}

    return {"job_title":"", "job_titles": [], "country":"", "companies":[], "sectors":extract_sectors_regex(s)}

# --- Affected section: strengthen LLM-first extraction with strict JSON + HF fallback + salvage + reload ---
def _trigger_reload():
    """
    4th-level fallback: schedule process exit so external supervisor restarts service.
    Adjust exit code or mechanism per deployment (systemd, Docker, PM2, etc.)
    """
    def exit_in_delay():
        time.sleep(1)
        try:
            sys.exit(100)
        except SystemExit:
            pass
    threading.Thread(target=exit_in_delay, daemon=True).start()

def llm_parse(msg: str, hist: str):
    """
    Extraction order:
      1) Gemini (strict JSON) → validate
      2) HuggingFace Inference (strict JSON) → validate
      3) Python regex/hardcode salvage
      4) Backend reload + user-facing error marker

    Sector differentiation enhancement:
      - Instruct LLM to separate sectors (industry/domain) from companies & country.
      - Post-process to remove country if it duplicates a company name and is not a valid country.
      - Reclassify obvious industry words (e.g. pharma, gaming) from country/company collisions into sectors.
    Returns dict (never None).
    """
    sector_aliases = {
        "pharma","pharmaceutical","pharmaceuticals","biotech","gaming","game","media",
        "medical","medical device","healthcare","life sciences","fintech","finance",
        "semiconductor","chip","automotive","ecommerce","retail","clinical","cardiovascular"
    }

    base_prompt = (
        "Extract sourcing parameters as STRICT JSON with keys exactly: "
        "{job_title, country, companies, sectors}.\n"
        "Definitions:\n"
        "- job_title: the role being searched (string; may be empty).\n"
        "- country: geographic country ONLY (full name). If no real country given, leave empty.\n"
        "- companies: list of organization/employer names ONLY.\n"
        "- sectors: list of industries/domains (e.g. pharma, biotech, gaming, media, healthcare, semiconductor, automotive, retail, ecommerce, fintech). "
        "Map any such words even if prefixed by 'in' or trailing after the role.\n"
        "Rules:\n"
        "- Do NOT put a company into country.\n"
        "- If a token could be both company and sector, prefer company if clearly an organization; otherwise sector.\n"
        "- If the user says 'in Pharma' treat 'Pharma' as a sector, NOT a country and NOT a company.\n"
        "- If the user only gives company (no country), leave country empty.\n"
        "- Output JSON ONLY. No commentary.\n"
        f"History:\n{hist}\nInput:\n{msg}\nJSON:"
    )

    def _normalize_obj(obj: dict):
        if not isinstance(obj, dict): return None
        jt = (obj.get("job_title") or "").strip()
        ct = (obj.get("country") or "").strip()
        companies=[c.strip() for c in obj.get("companies") or [] if isinstance(c,str) and c.strip()]
        sectors=[s.strip() for s in obj.get("sectors") or [] if isinstance(s,str) and s.strip()]
        # Dedup
        companies = dedupe(companies)
        sectors = dedupe(sectors)

        # If country duplicates a company name and not a valid country → clear country.
        if ct and any(ct.lower() == c.lower() for c in companies) and not country_to_cc(ct):
            ct = ""

        # If country is actually a sector word (not a valid country, matches alias) → move to sectors.
        if ct and ct.lower() in sector_aliases and not country_to_cc(ct):
            if ct not in sectors:
                sectors.append(ct)
            ct = ""

        # If any company value is a pure sector alias (and not a multi-word typical company), move to sectors.
        refined_companies=[]
        for c in companies:
            lower_c = c.lower()
            if lower_c in sector_aliases and not country_to_cc(c):
                if c not in sectors:
                    sectors.append(c)
            else:
                refined_companies.append(c)
        companies = refined_companies

        # Fallback: if sectors empty, attempt regex extraction
        if not sectors:
            sectors = extract_sectors_regex(msg)

        return {
            "job_title": jt,
            "job_titles": [jt] if jt else [],
            "country": ct,
            "companies": companies,
            "sectors": dedupe(sectors)
        }

    # 1) Gemini first
    if _gemini and genai:
        try:
            model=genai.GenerativeModel(GEMINI_MODEL)
            resp=model.generate_content(base_prompt)
            obj=gemini_json_extract((resp.text or ""))
            norm=_normalize_obj(obj) if obj else None
            if norm and (norm["job_title"] or norm["country"] or norm["companies"] or norm["sectors"]):
                return norm
        except Exception:
            pass

    # 2) HF fallback
    raw=None
    if HF_API_KEY:
        raw=hf_invoke(base_prompt, max_new_tokens=300)
    if raw:
        obj=gemini_json_extract(raw)
        norm=_normalize_obj(obj) if obj else None
        if norm and (norm["job_title"] or norm["country"] or norm["companies"] or norm["sectors"]):
            return norm

    # 3) Python regex salvage
    reg = regex_parse(msg)
    if isinstance(reg, dict) and (
        reg.get("job_title") or reg.get("country") or reg.get("companies") or reg.get("sectors")
    ):
        # Apply same collision rules on salvage
        ct = (reg.get("country") or "").strip()
        companies = reg.get("companies",[]) or []
        sectors = reg.get("sectors",[]) or []
        if ct and any(ct.lower()==c.lower() for c in companies) and not country_to_cc(ct):
            ct=""
        if ct and ct.lower() in sector_aliases and not country_to_cc(ct):
            if ct not in sectors: sectors.append(ct)
            ct=""
        # Sector fallback again if missing
        if not sectors:
            sectors = extract_sectors_regex(msg)
        return {
            "job_title": (reg.get("job_title") or "").strip(),
            "job_titles": reg.get("job_titles") if reg.get("job_titles") else ([reg.get("job_title")] if reg.get("job_title") else []),
            "country": ct,
            "companies": companies,
            "sectors": dedupe(sectors)
        }

    # 4) All failed -> trigger reload + return error marker
    _trigger_reload()
    return {
        "job_title": "",
        "job_titles": [],
        "country": "",
        "companies": [],
        "sectors": [],
        "error": "Internal refresh triggered. Please resend your request in a few seconds."
    }

# --- Affected section: New JD analysis function ---
def analyze_job_description(text: str):
    """
    Analyzes job description text using Gemini (or HF/regex fallback) to extract key parameters.
    Returns tuple: (summary_string, missing_list)
    summary_string follows:
      "Are you seeking a (seniority level) (job title) in the (sector) based in (country)?"
    missing_list contains any of: "seniority", "job_title", "sector", "country" that were not found.
    """
    if not text or not isinstance(text, str):
        return ("Could not analyze empty job description.", ["job_title", "country", "sector", "seniority"])

    # Helper to assemble summary and missing tags
    def _build_summary(jt, seniority, sector, country):
        missing = []
        jt_s = (jt or "").strip()
        senior_s = (seniority or "").strip()
        sector_s = (sector or "").strip()
        country_s = (country or "").strip()

        if not senior_s:
            missing.append("seniority")
        if not jt_s:
            missing.append("job_title")
        if not sector_s:
            missing.append("sector")
        if not country_s:
            missing.append("country")

        # Friendly defaults for wording
        senior_word = senior_s or "the"
        # If seniority missing, we will omit the level in readable form
        if senior_s:
            lead = f"{senior_s} {jt_s}" if jt_s else f"{senior_s} role"
        else:
            lead = jt_s or "this role"

        sector_phrase = f"in the {sector_s}" if sector_s else ""
        country_phrase = f" based in {country_s}" if country_s else ""

        summary = f"Are you seeking a {lead} {sector_phrase}{country_phrase}?".replace("  ", " ").strip()
        # normalize spacing when sector_phrase empty
        summary = re.sub(r'\s+', ' ', summary)
        return summary, missing

    # 1) Try Gemini
    parsed = None
    if _gemini and genai:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            prompt = (
                "Analyze the following Job Description and extract key sourcing parameters.\n"
                "Return STRICT JSON with keys: {job_title, seniority, sector, country}.\n"
                "Rules:\n"
                "- job_title: Specific role title (e.g. 'Product Manager', 'Software Engineer').\n"
                "- seniority: Level inferred (e.g. 'Associate', 'Senior', 'Lead', 'Manager', 'Director'). If unsure, set to empty string.\n"
                "- sector: The industry/domain (e.g. 'Healthcare > Medical Devices', 'Gaming', 'Fintech').\n"
                "- country: The geographic location for the role (country name) or empty.\n\n"
                f"Input Text:\n{text[:10000]}\n\nJSON:"
            )
            resp = model.generate_content(prompt)
            obj = gemini_json_extract((resp.text or ""))
            if isinstance(obj, dict):
                parsed = {
                    "job_title": (obj.get("job_title") or "").strip(),
                    "seniority": (obj.get("seniority") or "").strip(),
                    "sector": (obj.get("sector") or "").strip(),
                    "country": (obj.get("country") or "").strip()
                }
        except Exception:
            parsed = None

    # 2) HF fallback (if Gemini not available or failed)
    if parsed is None and HF_API_KEY:
        try:
            hf_resp = hf_invoke(
                "Extract JSON {job_title, seniority, sector, country} from the following text:\n\n" + text[:10000],
                max_new_tokens=300
            )
            if hf_resp:
                obj = gemini_json_extract(hf_resp)
                if isinstance(obj, dict):
                    parsed = {
                        "job_title": (obj.get("job_title") or "").strip(),
                        "seniority": (obj.get("seniority") or "").strip(),
                        "sector": (obj.get("sector") or "").strip(),
                        "country": (obj.get("country") or "").strip()
                    }
        except Exception:
            parsed = None

    # 3) Regex salvage fallback
    if parsed is None:
        try:
            # Try to salvage basic fields via regex helpers
            jt = ""
            country = extract_country_hint(text)
            sectors = extract_sectors_regex(text)
            sector = sectors[0] if sectors else ""
            # Try to find likely title: look for "Title: ..." or first heading-like line
            m = re.search(r'(?mi)^(?:title|role|position)[:\s\-]+(.+)$', text)
            if m:
                jt = m.group(1).strip()
            else:
                # fallback to first non-empty line up to 6 words
                for line in text.splitlines():
                    sline = line.strip()
                    if not sline:
                        continue
                    if len(sline.split()) <= 8:
                        jt = sline
                        break
            seniority = detect_seniority(text) or ""
            parsed = {
                "job_title": jt or "",
                "seniority": seniority or "",
                "sector": sector or "",
                "country": country or ""
            }
        except Exception:
            parsed = {"job_title": "", "seniority": "", "sector": "", "country": ""}

    # Ensure parsed is dict
    if not isinstance(parsed, dict):
        parsed = {"job_title": "", "seniority": "", "sector": "", "country": ""}

    summary, missing = _build_summary(parsed.get("job_title"), parsed.get("seniority"), parsed.get("sector"), parsed.get("country"))

    return (summary, missing)


def prepare_source(msg: str, hist_path: str):
    hist=history_for_prompt(hist_path)
    parsed=llm_parse(msg, hist)
    # If LLM produced nothing meaningful and no error marker, fallback again to regex_parse (defensive)
    if not parsed.get("job_title") and not parsed.get("country") and not parsed.get("companies") and not parsed.get("sectors") and not parsed.get("error"):
        parsed = regex_parse(msg)

    jt=(parsed.get("job_title") or "").strip()
    if jt:
        if country_to_cc(jt) and not parsed.get("country"):
            parsed["country"]=jt
            parsed["job_title"]=""
            parsed["job_titles"]=[]
        elif _looks_like_meta_query(jt):
            parsed["job_title"]=""
            parsed["job_titles"]=[]

    if not parsed.get("sectors"):
        parsed["sectors"]=extract_sectors_regex(msg)
    parsed["seniority"]=detect_seniority(parsed.get("job_title","")) or None
    if "job_titles" not in parsed:
        parsed["job_titles"] = [parsed["job_title"]] if parsed.get("job_title") else []
    return parsed

def middleware_normalize(msg: str, hist_path: str):
    base=prepare_source(msg, hist_path)
    missing=[]
    if not base.get("job_title"): missing.append("job_title")
    if not base.get("country"): missing.append("country")
    return {
        "original": msg,
        "job_title": base.get("job_title",""),
        "job_titles": base.get("job_titles", []),
        "country": base.get("country",""),
        "companies": base.get("companies",[]),
        "sectors": base.get("sectors",[]),
        "seniority": base.get("seniority"),
        "missing": missing,
        "error": base.get("error","")
    }