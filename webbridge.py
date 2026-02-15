import logging
import os
import threading
import time
import uuid
from csv import DictWriter
from datetime import datetime
import re
import json
import requests
import io
import hashlib
import difflib
from flask import Flask, request, send_from_directory, jsonify, abort, Response, stream_with_context

# Import DispatcherMiddleware to mount the second app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__, static_url_path='', static_folder='.')

# Set a secret key for session security (shared with data_sorter if integrated)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "change-me-in-production-webbridge")

# Affected section: lightweight CORS support for local development
def _apply_cors_headers(response):
    try:
        origin = request.headers.get('Origin')
        # Logic: Echo origin if present to support credentials, else wildcard (no credentials)
        if origin:
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Vary'] = 'Origin'
        else:
            response.headers['Access-Control-Allow-Origin'] = '*'
            # Credentials cannot be true if Origin is *
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = request.headers.get('Access-Control-Request-Headers', 'Content-Type, Authorization')
    except Exception:
        pass
    return response

@app.after_request
def _apply_cors(response):
    return _apply_cors_headers(response)

@app.route('/', methods=['OPTIONS'])
def _options_root():
    resp = app.make_response(('', 204))
    return _apply_cors_headers(resp)

@app.route('/<path:path>', methods=['OPTIONS'])
def _options(path):
    resp = app.make_response(('', 204))
    return _apply_cors_headers(resp)
# End affected section (CORS)

logging.basicConfig(level=logging.INFO, format="(%(asctime)s) | %(levelname)s | %(message)s")
logger = logging.getLogger("AutoSourcingServer")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define searchxls path for final Excel output
SEARCH_XLS_DIR = r"F:\Recruiting Tools\Autosourcing\searchxls"
os.makedirs(SEARCH_XLS_DIR, exist_ok=True)

GOOGLE_CSE_API_KEY = os.getenv("GOOGLE_CSE_API_KEY") or os.getenv("GOOGLE_CUSTOM_SEARCH_API_KEY")
GOOGLE_CSE_CX = os.getenv("GOOGLE_CSE_CX") or os.getenv("GOOGLE_CUSTOM_SEARCH_CX")

# diagnostic: masked preview of key the process sees (safe)
if GOOGLE_CSE_API_KEY:
    try:
        k = GOOGLE_CSE_API_KEY
        logger.info("GOOGLE_CSE_API_KEY=%s", (k[:4] + "..." + k[-4:]) if len(k)>8 else "SET")
    except Exception:
        logger.info("GOOGLE_CSE_API_KEY=SET")
else:
    logger.info("GOOGLE_CSE_API_KEY=NOT_SET")
logger.info("GOOGLE_CSE_CX=%s", GOOGLE_CSE_CX or "NOT_SET")

SEARCH_RESULTS_TARGET = int(os.getenv("SEARCH_RESULTS_TARGET", "50"))
CSE_PAGE_SIZE = min(int(os.getenv("CSE_PAGE_SIZE", "10")), 10)
CSE_PAGE_DELAY = float(os.getenv("CSE_PAGE_DELAY", "0.5"))

MIN_PLATFORM_RESULTS = int(os.getenv("MIN_PLATFORM_RESULTS", "8"))
MAX_PLATFORM_PAGES = int(os.getenv("MAX_PLATFORM_PAGES", "3"))

# New: maximum companies to return from suggestions (user requested)
MAX_COMPANY_SUGGESTIONS = int(os.getenv("MAX_COMPANY_SUGGESTIONS", "25"))

# CV Translation and Assessment Constants
CV_TRANSLATION_MAX_CHARS = 10000  # Max chars to translate (balances API limits and CV comprehensiveness)
LANG_DETECTION_SAMPLE_LENGTH = 1000  # Sample size for language detection (sufficient for accurate detection)
CV_ANALYSIS_MAX_CHARS = 15000  # Max CV text for Gemini analysis (ensures complete parsing within API limits)
MAX_COMMENT_LENGTH = 500  # Maximum length for overall assessment comment (UI/UX limit, DB supports unlimited)
COMMENT_TRUNCATE_LENGTH = MAX_COMMENT_LENGTH - 3  # Account for "..." ellipsis when truncating
ASSESSMENT_EXCELLENT_THRESHOLD = 80  # Score threshold for "Excellent" rating
ASSESSMENT_GOOD_THRESHOLD = 60  # Score threshold for "Good" rating
ASSESSMENT_MODERATE_THRESHOLD = 40  # Score threshold for "Moderate" rating

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_SUGGEST_MODEL = os.getenv("GEMINI_SUGGEST_MODEL", "gemini-2.5-flash-lite")
try:
    if GEMINI_API_KEY:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
    else:
        genai = None
        logger.warning("GEMINI_API_KEY not set. Gemini features disabled.")
except Exception as e:
    genai = None
    logger.warning(f"Gemini init failed: {e}")

TRANSLATION_ENABLED = os.getenv("TRANSLATION_ENABLED", "1") != "0"
TRANSLATION_PROVIDER = (os.getenv("TRANSLATION_PROVIDER", "auto") or "auto").lower()
TRANSLATOR_BASE = (os.getenv("TRANSLATOR_BASE", "") or "").rstrip("/")
NLLB_TIMEOUT = float(os.getenv("NLLB_TIMEOUT", "15.0"))
BRAND_TRANSLATE_WITH_NLLB = os.getenv("BRAND_TRANSLATE_WITH_NLLB", "0") == "1"

SINGAPORE_CONTEXT = os.getenv("SG_CONTEXT", "1") == "1"

SEARCH_RULES_PATH = os.path.join(BASE_DIR, "search_target_rules.json")
def _load_search_rules():
    try:
        if os.path.isfile(SEARCH_RULES_PATH):
            with open(SEARCH_RULES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data
    except Exception as e:
        logger.warning(f"[SearchRules] Failed to load {SEARCH_RULES_PATH}: {e}")
    return None

SEARCH_RULES = _load_search_rules()

DATA_SORTER_RULES_PATH = os.path.join(BASE_DIR, "static", "data_sorter.json")
if not os.path.isfile(DATA_SORTER_RULES_PATH):
    # Fallback check
    DATA_SORTER_RULES_PATH = os.path.join(BASE_DIR, "data_sorter.json")

def _load_data_sorter_rules():
    try:
        if os.path.isfile(DATA_SORTER_RULES_PATH):
            with open(DATA_SORTER_RULES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"[DataSorter] Loaded rules from {DATA_SORTER_RULES_PATH}")
                return data
    except Exception as e:
        logger.warning(f"[DataSorter] Failed to load {DATA_SORTER_RULES_PATH}: {e}")
    return None

DATA_SORTER_RULES = _load_data_sorter_rules()

def get_reference_mapping(job_title: str):
    """
    Look up job_title in DATA_SORTER_RULES to find augmented fields
    (seniority, family, geographic, sector).
    """
    if not DATA_SORTER_RULES or not job_title:
        return None
    
    t_lower = str(job_title).strip().lower()
    if not t_lower:
        return None

    # 1. Direct mapping
    mappings = DATA_SORTER_RULES.get("mappings", {})
    if t_lower in mappings:
        return mappings[t_lower]

    # 2. Regex patterns
    patterns = DATA_SORTER_RULES.get("patterns", [])
    if patterns:
        for p in patterns:
            try:
                regex = p.get("regex")
                if regex and re.search(regex, t_lower, re.IGNORECASE):
                    return p # Returns dict with family, seniority etc
            except Exception:
                continue
    
    return None

def _infer_region_from_country(country: str) -> str:
    """
    Infer the geographic region from a country name using DATA_SORTER_RULES (GeoCountries).
    """
    if not country:
        return ""
    
    # 1. Try to use loaded rules
    if DATA_SORTER_RULES:
        geo_map = DATA_SORTER_RULES.get("GeoCountries", {})
        if geo_map:
            c_clean = country.strip()
            # Handle "United States (USA)" -> "united states"
            if "(" in c_clean:
                c_clean = c_clean.split("(")[0].strip()
            
            c_lower = c_clean.lower()
            
            for region, countries in geo_map.items():
                for c_item in countries:
                    # Check exact match
                    if c_item.lower() == c_lower:
                        return region
                    # Check if rule item has suffix e.g. "United States (USA)"
                    if "(" in c_item:
                        base = c_item.split("(")[0].strip().lower()
                        if base == c_lower:
                            return region
    
    # 2. Fallback internal map (if JSON missing or lookup failed)
    # Basic mapping for key regions often encountered
    _FALLBACK_GEO = {
        "Asia": ["singapore","japan","china","india","korea","south korea","malaysia","thailand","vietnam","indonesia","philippines","taiwan","hong kong"],
        "Western Europe": ["uk","united kingdom","germany","france","spain","italy","netherlands","ireland","switzerland","sweden","norway","denmark","finland","belgium","austria","portugal"],
        "North America": ["usa","united states","us","canada","mexico"],
        "Australia/Oceania": ["australia","new zealand"]
    }
    
    c_lower = country.strip().lower()
    for region, countries in _FALLBACK_GEO.items():
        if c_lower in countries:
            return region
            
    return ""

def _map_gemini_seniority_to_dropdown(seniority_text: str, total_experience_years=None) -> str:
    """
    Normalize freeform seniority to one of: "Associate", "Manager", "Director".
    Rules (priority):
      1) Exact match to dropdown values -> use it.
      2) Director tokens (includes principal/staff/expert per requirement) -> "Director"
      3) Experience thresholds if numeric years available: >=10 -> Director; 5-9 -> Manager; <5 -> Associate
      4) Manager tokens -> "Manager"
      5) Associate/junior tokens -> "Associate"
      6) Fallback: "" (empty = no selection)
    """
    if not seniority_text and total_experience_years is None:
        return ""
    s = (seniority_text or "").strip().lower()

    # Exact canonical
    if s in {"associate", "manager", "director"}:
        return s.capitalize()

    # Director tokens (including Principal/Staff/Expert -> Director)
    director_tokens = [
        "director", "vice president", "vp", "vice-president", "head of", "head ",
        "chief ", "cxo", "executive director", "group director",
        "principal", "staff", "expert"
    ]
    for tok in director_tokens:
        if tok in s:
            return "Director"

    # Experience numeric override (if provided)
    try:
        if total_experience_years is not None:
            years = float(total_experience_years)
            if years >= 10:
                return "Director"
            if years >= 5:
                return "Manager"
            if years >= 0:
                return "Associate"
    except Exception:
        pass

    # Manager tokens (exclude principal/staff/expert already handled)
    manager_tokens = ["manager", "mgr", "team lead", "lead", "supervisor", "senior", "team-lead", "teamlead"]
    for tok in manager_tokens:
        if tok in s:
            return "Manager"

    # Associate / junior tokens
    associate_tokens = ["associate", "junior", "intern", "entry-level", "trainee", "graduate"]
    for tok in associate_tokens:
        if tok in s:
            return "Associate"

    # Conservative fallback for 'senior' alone
    if "senior" in s:
        return "Manager"

    return ""

# Helper for deduplication (Needed for heuristics)
def dedupe(seq):
    out=[]; seen=set()
    for x in seq:
        k=str(x).lower()
        if k in seen: continue
        seen.add(k); out.append(x)
    return out

# ---- NEW: Load sectors.json index once for server-side sector matching ----
SECTORS_JSON_PATH = os.path.join(BASE_DIR, "sectors.json")
SECTORS_INDEX = []  # list of labels (strings) in human-friendly form, e.g. "Financial Services > Banking"

def _load_sectors_index():
    global SECTORS_INDEX
    try:
        if os.path.isfile(SECTORS_JSON_PATH):
            with open(SECTORS_JSON_PATH, "r", encoding="utf-8") as fh:
                sdata = json.load(fh) or []
            labels = []
            for s in sdata:
                # s is expected to be a dict with keys sector, subsectors, domains etc.
                try:
                    sect = s.get("sector") if isinstance(s, dict) else None
                    if sect:
                        # subsectors -> industries
                        if isinstance(s.get("subsectors"), list) and s.get("subsectors"):
                            for ss in s.get("subsectors", []):
                                subname = ss.get("name") if isinstance(ss, dict) else None
                                if subname and isinstance(ss.get("industries"), list) and ss.get("industries"):
                                    for ind in ss.get("industries", []):
                                        labels.append(" > ".join([sect, subname, ind]))
                                else:
                                    if subname:
                                        labels.append(" > ".join([sect, subname]))
                        # domains
                        if isinstance(s.get("domains"), list) and s.get("domains"):
                            for d in s.get("domains", []):
                                labels.append(" > ".join([sect, d]))
                        # fallback to sector only
                        if not s.get("subsectors") and not s.get("domains"):
                            labels.append(sect)
                except Exception:
                    continue
            # dedupe while preserving order
            seen = set()
            out = []
            for l in labels:
                if not isinstance(l, str): continue
                clean = l.strip()
                if not clean: continue
                key = clean.lower()
                if key in seen: continue
                seen.add(key)
                out.append(clean)
            SECTORS_INDEX = out
        else:
            SECTORS_INDEX = []
    except Exception as e:
        logger.warning(f"[SectorsIndex] failed to load {SECTORS_JSON_PATH}: {e}")
        SECTORS_INDEX = []

# immediately load sectors index (best-effort)
_load_sectors_index()

# Helper functions for sector matching (new)
def _token_set(s):
    if not s: return set()
    return set(re.findall(r'\w+', s.lower()))

def _find_best_sector_match_for_text(candidate):
    """
    Given an arbitrary candidate string (e.g., "Air Conditioning / HVAC"),
    find the best-matching label from SECTORS_INDEX by token overlap.
    Returns the matched label (exact wording from sectors.json) or None.
    """
    try:
        if not candidate or not SECTORS_INDEX:
            return None
        cand_tokens = _token_set(candidate)
        if not cand_tokens:
            return None
        best = None
        best_score = 0
        # Prefer longer labels (more specific) when tie
        for label in SECTORS_INDEX:
            label_tokens = _token_set(label)
            if not label_tokens:
                continue
            score = len(cand_tokens & label_tokens)
            if score > best_score or (score == best_score and best and len(label) > len(best)):
                best_score = score
                best = label
        if best_score > 0:
            return best
        return None
    except Exception:
        return None

# Small explicit keyword -> sectors.json label mapping to handle cases like HVAC -> Machinery
# Keys are lowercase keywords; values are exact labels expected to exist (or closely match) in SECTORS_INDEX
# NOTE: pharma/clinical mapping removed per user request (do not auto-apply pharma heuristics)
_KEYWORD_TO_SECTOR_LABEL = {
    "hvac": "Industrial & Manufacturing > Machinery",
    "air conditioning": "Industrial & Manufacturing > Machinery",
    "air solutions": "Industrial & Manufacturing > Machinery",
    "software": "Technology > Software",
    "cloud": "Technology > Cloud & Infrastructure",
    "infrastructure": "Technology > Cloud & Infrastructure",
    "ai": "Technology > AI & Data",
    "artificial intelligence": "Technology > AI & Data",
    "machine learning": "Technology > AI & Data",
    # Financial keywords mapping added: map to Financial Services domains present in sectors.json
    "bank": "Financial Services > Banking",
    "banking": "Financial Services > Banking",
    "insurance": "Financial Services > Insurance",
    "investment": "Financial Services > Investment & Asset Management",
    "asset management": "Financial Services > Investment & Asset Management",
    "asset-management": "Financial Services > Investment & Asset Management",
    "wealth": "Financial Services > Investment & Asset Management",
    "fintech": "Financial Services > Fintech",
    # Removed 'clinical', 'pharma', 'biotech' mappings to avoid automatic pharma sector assignment
    "gaming": "Media, Gaming & Entertainment > Gaming",
    "ecommerce": "Consumer & Retail > E-commerce",
    "renewable": "Energy & Environment > Renewable Energy",
    "aerospace": "Industrial & Manufacturing > Aerospace & Defense"
}

def _map_keyword_to_sector_label(text):
    """
    Search for keywords in text and return a sectors.json label if found and present in SECTORS_INDEX.
    """
    try:
        txt = (text or "").lower()
        for kw, label in _KEYWORD_TO_SECTOR_LABEL.items():
            if kw in txt:
                # Ensure the label exists in SECTORS_INDEX (case-insensitive)
                for l in SECTORS_INDEX:
                    if l.lower() == label.lower():
                        return l
                # As fallback, try partial containment
                for l in SECTORS_INDEX:
                    if label.lower() in l.lower():
                        return l
        return None
    except Exception:
        return None

# --------------------------------------------------------------------------
# Helpers and modifications to avoid injecting pharma by default
# --------------------------------------------------------------------------

# set of tokens to identify pharma/biotech companies (lowercase substrings)
_PHARMA_KEYWORDS = {
    "pharma", "pharmaceutical", "pharmaceuticals", "pfizer", "roche", "novartis", "gsk", "glaxosmith", "sanofi",
    "astrazeneca", "bayer", "takeda", "cs l", "cs l", "sino", "biopharm", "sun pharma", "daiichi", "daiichi", "daiichi sankyo",
    "medtronic", "abbott", "baxter", "stryker", "bd", "csll", "cs l", "novotech", "iqvia", "labcorp", "icon", "parexel", "ppd",
    "syneos", "tigermed", "ppd"
}

def _is_pharma_company(name: str) -> bool:
    if not name or not isinstance(name, str): return False
    n = name.lower()
    for kw in _PHARMA_KEYWORDS:
        if kw in n:
            return True
    return False

def _sectors_allow_pharma(sectors):
    """
    Decide whether pharma companies should be allowed given selected/derived sectors.
    Returns True only when sectors clearly indicate healthcare/pharma/biotech/clinical contexts.
    """
    if not sectors:
        return False
    for s in sectors:
        if not isinstance(s, str):
            continue
        txt = s.lower()
        if any(k in txt for k in ("health", "healthcare", "pharma", "pharmaceutical", "biotech", "clinical", "medical", "pharmaceuticals", "biotechnology", "biopharma", "clinical research")):
            return True
    return False

# --------------------------------------------------------------------------

def _compute_search_target(job_titles, country, companies, auto_suggest_companies, sectors, languages, current_role, seniority=None,
                           channel_count=0, platform_count=0):
    if not SEARCH_RULES:
        return None
    if not (job_titles and isinstance(job_titles, list) and len(job_titles) > 0):
        return None
    if not (country and isinstance(country, str) and country.strip()):
        return None
    base_cfg = SEARCH_RULES.get("base", {})
    weights = SEARCH_RULES.get("weights", {})
    per_additional = SEARCH_RULES.get("per_additional", {})
    bounds = SEARCH_RULES.get("bounds", {})
    min_v = int(bounds.get("min", 10))
    max_v = int(bounds.get("max", 100))
    target = int(base_cfg.get("withJobAndLocation", SEARCH_RESULTS_TARGET))
    uniq_companies = set()
    for c in (companies or []):
        if isinstance(c, str) and c.strip():
            uniq_companies.add(c.strip().lower())
    for c in (auto_suggest_companies or []):
        if isinstance(c, str) and c.strip():
            uniq_companies.add(c.strip().lower())
    company_count = len(uniq_companies)
    sector_count = len({s.strip().lower() for s in (sectors or []) if isinstance(s, str) and s.strip()})
    language_count = len({l.strip().lower() for l in (languages or []) if isinstance(l, str) and l.strip()})
    current_role_flag = bool(current_role)
    if company_count > 0:
        target -= int(weights.get("company", 0))
        if company_count > 1:
            target -= int(per_additional.get("company", 0)) * (company_count - 1)
    if sector_count > 0:
        target -= int(weights.get("sector", 0))
        if sector_count > 1:
            target -= int(per_additional.get("sector", 0)) * (sector_count - 1)
    if language_count > 0:
        target -= int(weights.get("language", 0))
    if language_count > 1:
        target -= int(per_additional.get("language", 0)) * (language_count - 1)
    if current_role_flag:
        target -= int(weights.get("currentRole", 0))
    if channel_count > 0:
        target += int(weights.get("channel", 0))
        if channel_count > 1:
            target += int(per_additional.get("channel", 0)) * (channel_count - 1)
    if platform_count > 0:
        target += int(weights.get("platform", 0))
        if platform_count > 1:
            target += int(per_additional.get("platform", 0)) * (platform_count - 1)
    try:
        if seniority:
            srules = SEARCH_RULES.get("seniority_rules") or {}
            s_key = None
            s_lower = str(seniority).strip().lower()
            for k in srules.keys():
                if str(k).strip().lower() == s_lower:
                    s_key = k
                    break
            if s_key is not None:
                sval = srules.get(s_key)
                if isinstance(sval, dict):
                    s_weight = int(sval.get("weight", 0))
                else:
                    s_weight = int(sval or 0)
                target -= s_weight
    except Exception as e:
        logger.warning(f"[SearchRules] seniority adjustment skipped: {e}")
    if target < min_v:
        target = min_v
    if target > max_v:
        target = max_v
    return int(target)

@app.post("/preview_target")
def preview_target():
    data = request.get_json(force=True, silent=True) or {}
    job_titles = data.get('jobTitles') or []
    country = (data.get('country') or '').strip()
    companies = data.get('companyNames') or []
    auto_suggest_companies = data.get('autoSuggestedCompanyNames') or []
    sectors = data.get('selectedSectors') or data.get('sectors') or []
    languages = data.get('languages') or []
    current_role = bool(data.get('currentRole'))
    seniority = (data.get('seniority') or data.get('Seniority') or '').strip() or None
    channel_count = int(bool(data.get("channelGaming"))) + int(bool(data.get("channelMedia"))) + int(bool(data.get("channelTechnology")))
    platform_count = 0
    pq = data.get("xrayPlatformQueries")
    if isinstance(pq, list):
        platform_count = len(pq)
    target = _compute_search_target(job_titles, country, companies, auto_suggest_companies, sectors, languages,
                                    current_role, seniority, channel_count, platform_count)
    return jsonify({"target": target}), 200

def _extract_json_object(text: str):
    if not text: return None
    s=text.strip(); start=s.find('{'); end=s.rfind('}')
    if start!=-1 and end!=-1 and end>start:
        try: return json.loads(s[start:end+1])
        except Exception: return None
    return None

# ... [Translation functions kept as is] ...
NLLB_LANG = {
    "en": "eng_Latn","fr":"fra_Latn","de":"deu_Latn","es":"spa_Latn","it":"ita_Latn","pt":"por_Latn","ja":"jpn_Jpan",
    "zh":"zho_Hans","zh-hans":"zho_Hans","zh-hant":"zho_Hant","nl":"nld_Latn","pl":"pol_Latn","cs":"ces_Latn",
    "ru":"rus_Cyrl","ko":"kor_Hang","vi":"vie_Latn","th":"tha_Thai","sv":"swe_Latn","no":"nob_Latn","da":"dan_Latn",
    "fi":"fin_Latn","tr":"tur_Latn"
}
def _map_lang(code: str, default: str):
    c=(code or "").strip().lower()
    return NLLB_LANG.get(c) or NLLB_LANG.get(default.lower()) or "eng_Latn"
def _nllb_available() -> bool:
    return bool(TRANSLATION_ENABLED and TRANSLATOR_BASE)
def nllb_translate(text: str, src_lang: str, tgt_lang: str):
    if not _nllb_available():
        return None
    url=f"{TRANSLATOR_BASE}/translate"
    payload={"text":text,"src":_map_lang(src_lang or "en","en"),"tgt":_map_lang(tgt_lang or "en","en"),"max_length":200}
    try:
        r=requests.post(url, json=payload, timeout=NLLB_TIMEOUT)
        if r.status_code!=200:
            logger.warning(f"[NLLB] HTTP {r.status_code}: {r.text}")
            return None
        data=r.json()
        return (data.get("translation") or "").strip() or None
    except Exception as e:
        logger.warning(f"[NLLB] error: {e}")
        return None
def gemini_translate_plain(text: str, target_lang: str, source_lang: str="en"):
    if not (genai and GEMINI_API_KEY): return None
    prompt=(f"Translate from {source_lang} to {target_lang}. Keep proper nouns if commonly untranslated. Output only the final text.\n\n{text}")
    try:
        model=genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
        resp=model.generate_content(prompt)
        out=(resp.text or "").strip()
        out=re.sub(r'^\s*["“”\'`]+|["“”\'`]+\s*$', '', out)
        return out or None
    except Exception as e:
        logger.warning(f"[Gemini Translate] {e}")
    return None
def gemini_translate_company(text: str, target_lang: str, source_lang: str="en"):
    if not (genai and GEMINI_API_KEY): return None
    prompt=(f"Translate the company or organization name from {source_lang} to {target_lang}. "
            f"If the brand is commonly kept in {target_lang}, keep it unchanged. Output only the final name.\n\n{text}")
    try:
        model=genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
        resp=model.generate_content(prompt)
        out=(resp.text or "").strip()
        out=re.sub(r'^\s*["“”\'`]+|["“”\'`]+\s*$', '', out)
        return out or None
    except Exception as e:
        logger.warning(f"[Gemini Company Translate] {e}")
    return None
def translate_text_pipeline(text: str, target_lang: str, source_lang: str="en"):
    if not TRANSLATION_ENABLED or not text or not target_lang:
        return {"translated": text, "engine": "disabled", "status": "unchanged"}
    provider=TRANSLATION_PROVIDER
    if provider in ("nllb","auto") and _nllb_available():
        out=nllb_translate(text, source_lang, target_lang)
        if out:
            return {"translated": out, "engine":"nllb", "status":"translated" if out.lower()!=text.lower() else "unchanged"}
        if provider=="nllb":
            return {"translated": text, "engine":"nllb", "status":"fallback_original"}
    if provider in ("gemini","auto") and genai and GEMINI_API_KEY:
        out=gemini_translate_plain(text, target_lang, source_lang)
        if out:
            return {"translated": out, "engine":"gemini", "status":"translated" if out.lower()!=text.lower() else "unchanged"}
        if provider=="gemini":
            return {"translated": text, "engine":"gemini", "status":"fallback_original"}
    return {"translated": text, "engine":"fallback", "status":"unchanged"}
@app.post("/translate")
def translate_endpoint():
    data=request.get_json(force=True, silent=True) or {}
    text=(data.get("text") or "").strip()
    target_lang=(data.get("target_lang") or "").strip().lower()
    source_lang=(data.get("source_lang") or "en").strip().lower()
    if not text or not target_lang:
        return jsonify({"error":"text and target_lang required"}), 400
    return jsonify(translate_text_pipeline(text, target_lang, source_lang)), 200
@app.post("/translate_company")
def translate_company_endpoint():
    data=request.get_json(force=True, silent=True) or {}
    text=(data.get("text") or "").strip()
    target_lang=(data.get("target_lang") or "").strip().lower()
    source_lang=(data.get("source_lang") or "en").strip().lower()
    if not text or not target_lang:
        return jsonify({"translated": text, "engine":"fallback", "status":"unchanged"})
    out=gemini_translate_company(text, target_lang, source_lang) if (genai and GEMINI_API_KEY) else None
    if not out and BRAND_TRANSLATE_WITH_NLLB and _nllb_available():
        out=nllb_translate(text, source_lang, target_lang)
    if not out:
        return jsonify({"translated": text, "engine":"fallback", "status":"unchanged"})
    return jsonify({"translated": out, "engine":"gemini" if (genai and GEMINI_API_KEY) else "nllb",
                    "status":"translated" if out.lower()!=text.lower() else "unchanged"})

@app.post("/gemini/analyze_jd")
def gemini_analyze_jd():
    """
    Accepts JSON body: { "username": "...", "text": "...", "country": "..." }
    If username provided but text empty, server may fetch stored JD from DB (optional).
    
    Implements workflow:
    1. Identify companies mentioned in JD
    2. Determine sectors from identified companies (using sectors.json)
    3. If no companies found, infer sector from JD content
    4. Filter companies by legal entity in specified country
    5. Always identify at least one sector (using sectors.json)
    6. Derive second sector from skillset (if applicable)
    7. Generate at least 2 job titles (original + suggested variant)
    
    Returns JSON:
    {
      "job_title": "...",  # Single title for backward compatibility
      "job_titles": [...],  # Array of at least 2 job titles (original + suggestions)
      "seniority": "...",
      "sectors": [...],  # Always mapped to sectors.json, may include skillset-based sector
      "companies": [...],  # Filtered by country legal entity
      "country": "...",
      "summary": "...",
      "missing": [...],
      "skills": [...],
      "raw": "raw model output"
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    text_input = (data.get("text") or "").strip()
    sectors_data = data.get("sectors") or []
    country = (data.get("country") or "").strip()

    # If no text but username provided, attempt to read JD from login.jd column (best-effort)
    if not text_input and username:
        try:
            import psycopg2
            conn = psycopg2.connect(host=os.getenv("PGHOST","localhost"), port=int(os.getenv("PGPORT","5432")), user=os.getenv("PGUSER","postgres"), password=os.getenv("PGPASSWORD","") or "orlha", dbname=os.getenv("PGDATABASE","candidate_db"))
            cur = conn.cursor()
            cur.execute("SELECT jd FROM login WHERE username = %s", (username,))
            row = cur.fetchone()
            cur.close()
            conn.close()
            text_input = (row[0] or "").strip() if row and row[0] else text_input
        except Exception:
            text_input = text_input

    if not text_input:
        return jsonify({"error":"No JD text provided or found for user"}), 400

    if not genai or not GEMINI_API_KEY:
        # Fallback: simple heuristics if gemini not configured
        try:
            from chat_extract import analyze_job_description as heuristic_analyze
            s, missing = heuristic_analyze(text_input)
            return jsonify({"summary": s, "missing": missing, "parsed": {}, "skills": [], "companies": [], "raw": "", "observation": ""}), 200
        except Exception:
            return jsonify({"error":"Gemini not available and no heuristic fallback"}), 503

    try:
        # -------------------------
        # STEP 1: Identify companies mentioned in JD
        # -------------------------
        identified_companies = []
        company_identification_note = ""
        
        company_prompt = (
            "You are a recruiting assistant. Analyze the following job description and identify ALL company names mentioned.\n"
            "Return STRICT JSON with this structure:\n"
            "{ \"companies\": [\"Company Name 1\", \"Company Name 2\", ...] }\n"
            "Rules:\n"
            "- Include the hiring company if explicitly mentioned\n"
            "- Include client companies, partner companies, or competitor companies mentioned\n"
            "- Use official company names (e.g., 'Microsoft' not 'MS', 'Johnson & Johnson' not 'J&J')\n"
            "- Do NOT include generic industry terms (e.g., 'tech companies', 'pharma firms')\n"
            "- Return empty array if no specific companies are mentioned\n"
            "\nJOB DESCRIPTION:\n" + (text_input[:15000]) + "\n\nJSON:"
        )
        
        model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
        company_resp = model.generate_content(company_prompt)
        company_raw = (company_resp.text or "").strip()
        company_obj = _extract_json_object(company_raw) or {}
        
        raw_companies = company_obj.get("companies") or []
        if isinstance(raw_companies, list):
            for c in raw_companies:
                if isinstance(c, str) and c.strip():
                    identified_companies.append(c.strip())
        
        company_identification_note = f"Identified {len(identified_companies)} companies in JD." if identified_companies else "No companies identified in JD."
        
        # -------------------------
        # STEP 2: Main JD Analysis with enhanced prompt
        # -------------------------
        
        # Build sectors reference for prompt
        sectors_list = ""
        if sectors_data:
            sectors_list = "\n\nAVAILABLE SECTORS:\n" + json.dumps(sectors_data, indent=2)
        
        # Include identified companies in the analysis prompt
        companies_context = ""
        if identified_companies:
            companies_context = f"\n\nIDENTIFIED COMPANIES: {', '.join(identified_companies)}\n"
            companies_context += "Use these companies to help determine the appropriate sector(s) from the available sectors list."

        # Build strict JSON request to Gemini
        prompt = (
            "You are a recruiting assistant. Analyze the job description and return STRICT JSON with keys:\n"
            "{ parsed: { job_title, seniority, sector, country, skills }, missing: [...], summary: string, suggestions: [...], justification: string, observation: string, raw: string }\n"
            "IMPORTANT:\n"
            "- You MUST identify at least one sector. Use your best judgment if unclear.\n"
            "- Multiple sectors may be assigned if the role spans multiple domains.\n"
            "- Match sectors to the AVAILABLE SECTORS list provided below.\n"
            + companies_context
            + sectors_list
            + "\nJOB DESCRIPTION:\n" + (text_input[:15000]) + "\n\nJSON:"
        )

        resp = model.generate_content(prompt)
        raw_out = (resp.text or "").strip()
        parsed_obj = _extract_json_object(raw_out) or {}
        parsed = parsed_obj.get("parsed", {})
        
        # Normalize output
        job_title = (parsed.get("job_title") or parsed.get("role") or "").strip()
        seniority = (parsed.get("seniority") or "").strip()
        sector = parsed.get("sector") or ""
        sectors = parsed.get("sectors") or ([sector] if sector else [])
        if not country:  # Use country from analysis if not provided in request
            country = (parsed.get("country") or parsed.get("location") or "").strip()
        skills = parsed.get("skills") or parsed_obj.get("skills") or []
        if isinstance(skills, str) and skills.strip():
            skills = [s.strip() for s in skills.split(",") if s.strip()]
        suggestions = parsed_obj.get("suggestions") or []
        summary = parsed_obj.get("summary") or ""
        missing = parsed_obj.get("missing") if isinstance(parsed_obj.get("missing"), list) else []
        justification = parsed_obj.get("justification") or parsed_obj.get("reason") or ""
        observation = parsed_obj.get("observation") or parsed_obj.get("justification") or ""

        # Ensure missing computed if not present
        if not isinstance(missing, list):
            missing = []
            if not job_title: missing.append("job_title")
            if not seniority: missing.append("seniority")
            if not sectors: missing.append("sector")
            if not country: missing.append("country")

        # If skills still empty, optionally run a local heuristic (if helper available)
        if not skills:
            try:
                from chat_gemini_review import extract_skills_heuristic
                skills = extract_skills_heuristic(text_input, job_title, sectors[0] if sectors else "", "")
            except Exception:
                skills = []

        # -------------------------
        # New: Heuristic derivation for missing/ambiguous seniority and sector
        # If Gemini left seniority or sector blank or ambiguous, apply conservative heuristics.
        # This supplements Gemini output (and appends rationale into justification/observation).
        # -------------------------
        def derive_seniority_from_text(jd_text: str, jt: str):
            """
            Simple rules for seniority inference from job title/text:
              - If title contains senior/lead/principal/manager/director -> map accordingly
              - Else if description explicitly mentions 'senior' or '5+ years' -> Senior
              - Else default to empty (unknown)
            """
            try:
                text_lower = (jd_text or "").lower()
                jt_lower = (jt or "").lower()
                # title-based
                if re.search(r'\b(senior|sr\.?\b|principal|lead|head|staff)\b', jt_lower) or re.search(r'\b(senior|sr\.?)\b', text_lower):
                    return "Senior", "Detected 'senior/lead/principal' token in title/text"
                if re.search(r'\b(manager|director|vp|vice president)\b', jt_lower) or re.search(r'\b(manager|director|vp|vice president)\b', text_lower):
                    # manager/director are higher-level seniorities
                    if re.search(r'\bdirector|vp|vice president\b', jt_lower + " " + text_lower):
                        return "Director", "Detected director/vp in title/text"
                    return "Manager", "Detected 'manager' in title/text"
                # experience hint
                if re.search(r'\b(\d+\+?\s+years? of experience|5\+ years|7\+ years|10\+ years)\b', text_lower):
                    return "Senior", "Years-of-experience hint in JD"
                # otherwise unknown
                return "", ""
            except Exception:
                return "", ""

        def derive_sector_from_text(jd_text: str, jt: str):
            """
            Determine sector heuristically:
              - First, try to match exact or long-form labels from sectors.json (loaded into SECTORS_INDEX)
                by searching for label phrases in the JD text (longest match wins).
              - If no sectors.json label matches, check keyword->label mapping (strict).
              - If neither yields a match, return empty (we do NOT return freeform sector strings).
            """
            try:
                text_lower = (jd_text or "").lower()
                jt_lower = (jt or "").lower()

                # 1) Try sectors.json labels (longest-match strategy by substring)
                best_match = ""
                best_orig = ""
                for label in SECTORS_INDEX:
                    lbl_low = label.lower()
                    if lbl_low and lbl_low in text_lower:
                        # prefer the longest matched label (more specific)
                        if len(lbl_low) > len(best_match):
                            best_match = lbl_low
                            best_orig = label
                if best_orig:
                    return best_orig, "Matched sectors.json label"

                # 1b) Try direct mapping from the whole job title or text to SECTORS_INDEX using token overlap
                # (helps when the model returns a sector phrase that doesn't match as substring)
                mapped_from_text = _find_best_sector_match_for_text(jd_text)
                if mapped_from_text:
                    return mapped_from_text, "Matched sectors.json label via token overlap"

                mapped_from_title = _find_best_sector_match_for_text(jt)
                if mapped_from_title:
                    return mapped_from_title, "Matched sectors.json label via title token overlap"

                # 2) Keyword -> sector label mapping (STRICT mapping to sectors.json labels)
                kw_map = _map_keyword_to_sector_label(jd_text) or _map_keyword_to_sector_label(jt)
                if kw_map:
                    return kw_map, "Mapped via keyword to sectors.json label"

                # 3) Do NOT return freeform labels; instead return empty to indicate no strict sectors.json match
                return "", ""
            except Exception:
                return "", ""

        # Try to map any sectors returned by Gemini to sectors.json labels (strict mapping)
        heuristic_notes = []
        mapped_sectors = []
        try:
            if sectors and isinstance(sectors, (list, tuple)):
                for cand in sectors:
                    if not cand or not isinstance(cand, str): continue
                    # sometimes the model returns slashed lists; break them up
                    parts = re.split(r'[\/,;|]+', cand)
                    for p in parts:
                        p = p.strip()
                        if not p: continue
                        mapped = _find_best_sector_match_for_text(p) or _map_keyword_to_sector_label(p)
                        if mapped and mapped not in mapped_sectors:
                            mapped_sectors.append(mapped)
            elif sector and isinstance(sector, str) and sector.strip():
                parts = re.split(r'[\/,;|]+', sector)
                for p in parts:
                    p = p.strip()
                    if not p: continue
                    mapped = _find_best_sector_match_for_text(p) or _map_keyword_to_sector_label(p)
                    if mapped and mapped not in mapped_sectors:
                        mapped_sectors.append(mapped)
            # ALWAYS use mapped sectors (even if empty) - do NOT keep unmapped sectors
            # This ensures ONLY sectors.json validated sectors are used
            sectors = mapped_sectors  # Replace with mapped sectors (empty if no valid mapping)
            if mapped_sectors:
                sector = mapped_sectors[0]  # Only set if we have valid mappings
                heuristic_notes.append("sector mapped from model output to sectors.json label(s)")
            else:
                sector = ""  # Clear single sector if no valid mapping
        except (KeyError, ValueError, AttributeError, TypeError) as e:
            # Only catch expected mapping errors, log for debugging
            logger.warning(f"Sector mapping error: {e}")
            sectors = []  # Clear sectors on error to ensure no unmapped sectors slip through
            sector = ""

        # Apply derivation if needed (only when no mapping from model exists)
        if not seniority:
            derived_sen, note = derive_seniority_from_text(text_input, job_title)
            if derived_sen:
                seniority = derived_sen
                heuristic_notes.append(f"seniority derived: {note}")
        if not sectors or (isinstance(sectors, list) and len(sectors)==0):
            derived_sector, note = derive_sector_from_text(text_input, job_title)
            if derived_sector:
                # derived_sector should already be a sectors.json label (per new logic)
                sectors = [derived_sector]
                sector = derived_sector
                heuristic_notes.append(f"sector derived: {note}")

        # If we made heuristic derivations or mappings, append explanation to justification/observation
        if heuristic_notes:
            note_text = " Heuristic derivation applied: " + "; ".join(heuristic_notes) + "."
            if justification:
                justification = justification.strip()
                # avoid duplicating punctuation
                if not justification.endswith("."):
                    justification += "."
                justification += note_text
            else:
                justification = note_text.strip()
            if observation:
                if not observation.endswith("."):
                    observation += "."
                observation += " " + " ".join(heuristic_notes) + "."
            else:
                observation = " ".join(heuristic_notes) + "."

        # -------------------------
        # STEP 3: Filter identified companies by legal entity in specified country
        # Only suggest companies that have a legal entity in the specified country
        # -------------------------
        valid_companies = []
        if identified_companies and country:
            for company in identified_companies:
                if _has_local_presence(company, country):
                    valid_companies.append(company)
            
            if len(valid_companies) < len(identified_companies):
                filtered_count = len(identified_companies) - len(valid_companies)
                company_identification_note += f" Filtered {filtered_count} companies without legal entity in {country}."
        elif identified_companies:
            # If no country specified, include all identified companies
            valid_companies = identified_companies
        
        # -------------------------
        # STEP 4: Determine sectors based on identified companies (if available)
        # If companies were identified and mapped to sectors, those should take precedence
        # -------------------------
        company_based_sectors = []
        if valid_companies:
            # Try to determine sectors from the identified companies
            for company in valid_companies:
                # Check if company matches any bucket in BUCKET_COMPANIES
                company_lower = company.lower().strip()
                for bucket_name, bucket_data in BUCKET_COMPANIES.items():
                    for region in ["global", "apac"]:
                        region_companies = bucket_data.get(region, [])
                        if any(company_lower == c.lower().strip() for c in region_companies):
                            # Map bucket to sector
                            sector_from_bucket = _bucket_to_sector_label(bucket_name)
                            if sector_from_bucket and sector_from_bucket not in company_based_sectors:
                                company_based_sectors.append(sector_from_bucket)
                            break
            
            # If we found sectors from companies, use them (but keep any additional sectors from JD analysis)
            if company_based_sectors:
                # Merge company-based sectors with JD-derived sectors (deduplicate)
                # IMPORTANT: Only merge sectors that were successfully mapped to sectors.json
                # At this point, 'sectors' contains ONLY sectors.json validated sectors
                for s in sectors:
                    if s and s not in company_based_sectors:
                        company_based_sectors.append(s)
                sectors = company_based_sectors
                heuristic_notes.append(f"sectors determined from identified companies")
        
        # -------------------------
        # STEP 4.5: Derive second sector based on skillset
        # When companies are identified (first sector), derive additional sector from skills
        # This ensures multi-sector coverage: company-based + skillset-based
        # -------------------------
        def derive_sector_from_skills(skills_list, existing_sectors):
            """
            Derive a sector from the skillset that is different from existing sectors.
            Maps common skill patterns to sectors.json labels.
            
            Example: ["AWS", "Cloud", "Kubernetes"] -> "Technology > Cloud & Infrastructure"
            """
            if not skills_list:
                return None, ""
            
            skills_text = " ".join([str(s).lower() for s in skills_list if s])
            
            # Skill-to-sector mapping patterns (all map to sectors.json)
            skill_patterns = [
                # Cloud & Infrastructure
                (["cloud", "aws", "azure", "gcp", "kubernetes", "docker", "devops", "terraform", "infrastructure"], 
                 "Technology > Cloud & Infrastructure"),
                # AI & Data
                (["machine learning", "ml", "ai", "artificial intelligence", "data science", "python", "tensorflow", "pytorch", "nlp"],
                 "Technology > AI & Data"),
                # Cybersecurity
                (["security", "cybersecurity", "penetration testing", "siem", "firewall", "encryption"],
                 "Technology > Cybersecurity"),
                # Software Development
                (["java", "javascript", "react", "node", "sql", "api", "backend", "frontend", "full stack"],
                 "Technology > Software"),
                # Gaming
                (["unity", "unreal", "game engine", "game development", "3d", "animation"],
                 "Media, Gaming & Entertainment > Gaming"),
                # Healthcare/Clinical
                (["clinical", "medical", "patient", "healthcare", "hospital", "diagnosis"],
                 "Healthcare > HealthTech"),
                # Finance
                (["trading", "financial analysis", "investment", "portfolio", "risk management", "bloomberg"],
                 "Financial Services > Investment & Asset Management"),
                # Manufacturing/Engineering
                (["manufacturing", "plc", "scada", "automation", "robotics", "lean", "six sigma"],
                 "Industrial & Manufacturing > Machinery"),
            ]
            
            # Find matching sectors based on skills
            for keywords, sector_label in skill_patterns:
                if any(keyword in skills_text for keyword in keywords):
                    # Verify the sector exists in sectors.json and isn't already in existing sectors
                    if sector_label in SECTORS_INDEX and sector_label not in existing_sectors:
                        return sector_label, f"Derived from skillset: {', '.join(keywords[:3])}"
            
            return None, ""
        
        # Apply skillset-based sector derivation if we have skills and at least one existing sector
        if skills and len(skills) > 0 and len(sectors) > 0:
            skillset_sector, skillset_note = derive_sector_from_skills(skills, sectors)
            if skillset_sector:
                sectors.append(skillset_sector)
                heuristic_notes.append(f"second sector from skillset: {skillset_note}")
        
        # -------------------------
        # STEP 5: Ensure at least one sector is always identified
        # -------------------------
        if not sectors or (isinstance(sectors, list) and len(sectors) == 0):
            # Apply sector derivation as fallback
            derived_sector, note = derive_sector_from_text(text_input, job_title)
            if derived_sector:
                sectors = [derived_sector]
                sector = derived_sector
                heuristic_notes.append(f"sector derived (fallback): {note}")
            else:
                # Last resort: assign a generic sector based on job title keywords
                sectors = ["Other"]
                heuristic_notes.append("sector set to 'Other' as fallback")

        # Update justification/observation with company identification and filtering notes
        if company_identification_note:
            note_text = f" {company_identification_note}"
            if justification:
                justification = justification.strip()
                if not justification.endswith("."):
                    justification += "."
                justification += note_text
            else:
                justification = note_text.strip()

        # If we made heuristic derivations or mappings, append explanation to justification/observation
        if heuristic_notes:
            note_text = " Heuristic derivation applied: " + "; ".join(heuristic_notes) + "."
            if justification:
                justification = justification.strip()
                # avoid duplicating punctuation
                if not justification.endswith("."):
                    justification += "."
                justification += note_text
            else:
                justification = note_text.strip()
            if observation:
                if not observation.endswith("."):
                    observation += "."
                observation += " " + " ".join(heuristic_notes) + "."
            else:
                observation = " ".join(heuristic_notes) + "."

        # Recompute missing after all derivations
        missing = []
        if not job_title: missing.append("job_title")
        if not seniority: missing.append("seniority")
        if not sectors: missing.append("sector")
        if not country: missing.append("country")
        
        # -------------------------
        # STEP 6: Job Title Inference - Generate at least 2 job titles
        # As per requirement: must return at least two job titles:
        # 1. Original job title from JD
        # 2. Closest matched job title from Job Title Suggestion process
        # -------------------------
        job_titles = []
        
        # Add original job title if present
        if job_title:
            job_titles.append(job_title)
        
        # Get suggested job titles using the suggestion system
        try:
            # Call the suggestion system to get related job titles
            suggested_titles = []
            if job_title or sectors:  # Need at least one of these for suggestions
                # Use first sector to infer industry if available
                industry = "Non-Gaming"  # Default industry for suggestion system
                if sectors and sectors[0]:
                    # Map sector to industry context for better suggestions
                    sector_lower = sectors[0].lower()
                    if "gaming" in sector_lower or "entertainment" in sector_lower:
                        industry = "Gaming"
                    # Non-Gaming is appropriate default for most professional roles
                
                gem_suggestions = _gemini_suggestions(
                    job_titles=[job_title] if job_title else [],
                    companies=valid_companies,  # Use identified companies for context
                    industry=industry,
                    languages=None,
                    sectors=sectors,
                    country=country
                )
                
                if gem_suggestions and gem_suggestions.get("job", {}).get("related"):
                    suggested_titles = gem_suggestions.get("job", {}).get("related", [])
                else:
                    # Fallback to heuristic suggestions with company context
                    suggested_titles = _heuristic_job_suggestions(
                        job_titles=[job_title] if job_title else [],
                        companies=valid_companies,  # Pass companies for better suggestions
                        industry=industry,
                        languages=None,
                        sectors=sectors
                    ) or []
            
            # Add the closest matched job title (first suggestion)
            if suggested_titles:
                # Filter out the original job title if it appears in suggestions
                for suggested in suggested_titles:
                    if suggested and isinstance(suggested, str):
                        suggested_clean = suggested.strip()
                        # Avoid duplicates (case-insensitive comparison)
                        if not any(jt.lower() == suggested_clean.lower() for jt in job_titles):
                            job_titles.append(suggested_clean)
                            break  # Only add the first (closest) match
        except Exception as e:
            logger.warning(f"Failed to get job title suggestions: {e}")
        
        # Ensure we have at least 2 job titles as required
        # If we only have 1 (or 0), add a generic variant
        if len(job_titles) < 2:
            if job_title:
                # Create a variant by adding "Senior" if not already present
                if "senior" not in job_title.lower():
                    job_titles.append(f"Senior {job_title}")
                else:
                    # Remove "Senior" to create a variant
                    variant = re.sub(r'\bSenior\s+', '', job_title, flags=re.IGNORECASE).strip()
                    if variant and variant != job_title:
                        job_titles.append(variant)
                    else:
                        # Add "Lead" variant
                        job_titles.append(f"Lead {job_title}")
            else:
                # No job title provided at all - use sector-specific defaults
                # These are placeholder titles when no better inference is possible
                if sectors and sectors[0] != "Other":
                    # Extract sector name for more specific title generation
                    sector_name = sectors[0].split(">")[-1].strip() if ">" in sectors[0] else sectors[0]
                    # Generate sector-appropriate titles (these are fallback placeholders)
                    job_titles = [f"{sector_name} Professional", f"Senior {sector_name} Professional"]
                else:
                    # Ultimate fallback for unknown sectors
                    job_titles = ["Professional", "Senior Professional"]
        
        # Update justification to note job title inference
        if len(job_titles) >= 2:
            title_note = f" Generated {len(job_titles)} job title variants (original + suggested)."
            if justification:
                justification = justification.strip()
                if not justification.endswith("."):
                    justification += "."
                justification += title_note
            else:
                justification = title_note.strip()
        
        # -------------------------
        # End enhanced workflow
        # -------------------------

        out = {
            "job_title": job_title,  # Keep single job_title for backward compatibility
            "job_titles": job_titles,  # NEW: Array of at least 2 job titles
            "seniority": seniority,
            "sectors": sectors if isinstance(sectors, list) else ([sectors] if sectors else []),
            "companies": valid_companies,  # Include filtered companies with legal entity in country
            "country": country,
            "summary": summary,
            "missing": missing,
            "suggestions": suggestions,
            "justification": justification,
            "observation": observation,
            "skills": skills,
            "raw": raw_out
        }
        return jsonify(out), 200
    except Exception as e:
        logger.exception("Gemini analyze_jd failed")
        return jsonify({"error": str(e)}), 500

# Helper: persist jskillset (and fallback columns) for a username
def _persist_jskillset(username: str, skills):
    """
    Persist the provided skills (list or CSV/string) into the login table.
    Prefers column order: jskillset, jskills, skills, skillset
    Attempts to write JSON array first; falls back to comma-separated text.
    Returns (ok:bool, message:str)
    """
    if not username:
        return False, "username required"

    # Normalize skills to deduped list preserving order
    skills_list = []
    if isinstance(skills, str):
        parts = [p.strip() for p in re.split(r'[,\n;]+', skills) if p.strip()]
        skills_list = parts
    elif isinstance(skills, list):
        skills_list = [str(s).strip() for s in skills if str(s).strip()]
    else:
        # unknown format
        try:
            skills_list = list(skills) if skills else []
        except Exception:
            skills_list = []

    deduped = []
    seen = set()
    for s in skills_list:
        k = s.lower()
        if k not in seen:
            seen.add(k)
            deduped.append(s)

    # Ensure a JSON-serializable list
    final_skills = [str(s) for s in deduped]

    try:
        import psycopg2
        from psycopg2 import sql
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()
        
        # Explicit check for jskillset existence to be safe
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='login' AND column_name='jskillset'
        """)
        has_jskillset = bool(cur.fetchone())
        
        preferred = "jskillset" if has_jskillset else "skills"

        # Format skillset as comma-separated string (no brackets/quotes)
        # Per requirement: Remove enclosing brackets and quotes
        # Example: ["algorithms", "data structures"] -> algorithms, data structures
        formatted_skills = ", ".join(final_skills)

        # Try updating as plain text (comma-separated)
        try:
            cur.execute(sql.SQL("UPDATE login SET {} = %s WHERE username = %s").format(sql.Identifier(preferred)),
                        (formatted_skills, username))
            if cur.rowcount == 0:
                conn.commit()
                cur.close(); conn.close()
                return False, "username not found"
            conn.commit()
            cur.close(); conn.close()
            logger.info(f"[PersistSkills] Updated {preferred} for {username} (comma-separated).")
            return True, f"updated {preferred} as comma-separated"
        except Exception as e_update:
            conn.rollback()
            cur.close(); conn.close()
            logger.warning(f"[PersistSkills] Failed to persist into {preferred} for {username}: {e_update}")
            return False, f"DB write failed: {e_update}"
    except Exception as e:
        logger.warning(f"[PersistSkills] DB connection or discovery failed: {e}")
        return False, f"DB error: {e}"

# Helper: fetch jskillset for a username
def _fetch_jskillset(username: str):
    """
    Attempt to retrieve a user's skillset from login table.
    Explicitly checks 'jskillset' column first, then 'skills'.
    Returns a list of skill strings (possibly empty).
    """
    if not username:
        return []
    try:
        import psycopg2
        from psycopg2 import sql
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur = conn.cursor()
        
        # Check if jskillset column exists to prevent query errors
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_schema='public' AND table_name='login' AND column_name='jskillset'
        """)
        has_jskillset = bool(cur.fetchone())
        
        if has_jskillset:
            cur.execute("SELECT jskillset FROM login WHERE username=%s", (username,))
            row = cur.fetchone()
            if row and row[0]:
                val = row[0]
                if isinstance(val, list): return val
                if isinstance(val, str):
                    try: return json.loads(val)
                    except: return [s.strip() for s in val.split(',') if s.strip()]
        
        # Fallback to skills column
        cur.execute("SELECT skills FROM login WHERE username=%s", (username,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        
        if row and row[0]:
            val = row[0]
            if isinstance(val, list): return val
            if isinstance(val, str):
                try: 
                    parsed = json.loads(val)
                    if isinstance(parsed, list): return parsed
                except: 
                    pass
                return [s.strip() for s in val.split(',') if s.strip()]
                
        return []
    except Exception as e:
        logger.error(f"[_fetch_jskillset] Error: {e}")
        return []

def _fetch_jskillset_from_process(linkedinurl: str):
    """
    Retrieve jskillset from process table for a specific candidate.
    This is used for cross-checking extracted skillsets against stored jskillset.
    Returns a list of skill strings (possibly empty).
    """
    if not linkedinurl:
        return []
    try:
        import psycopg2
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur = conn.cursor()
        
        # Normalize LinkedIn URL
        normalized_url = linkedinurl.strip().rstrip('/').lower()
        if not normalized_url.startswith('http'):
            normalized_url = 'https://' + normalized_url
        
        # Check if jskillset column exists in process table
        cur.execute("""
            SELECT column_name FROM information_schema.columns 
            WHERE table_schema='public' AND table_name='process' AND column_name='jskillset'
        """)
        has_jskillset = bool(cur.fetchone())
        
        if has_jskillset:
            cur.execute("SELECT jskillset FROM process WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl))=%s", (normalized_url,))
            row = cur.fetchone()
            if row and row[0]:
                val = row[0]
                if isinstance(val, list):
                    cur.close()
                    conn.close()
                    return val
                if isinstance(val, str):
                    cur.close()
                    conn.close()
                    try:
                        return json.loads(val)
                    except:
                        return [s.strip() for s in val.split(',') if s.strip()]
        
        # Fallback to jskill or skillset column
        for col in ['jskill', 'skillset', 'skills']:
            try:
                cur.execute(f"SELECT {col} FROM process WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl))=%s", (normalized_url,))
                row = cur.fetchone()
                if row and row[0]:
                    val = row[0]
                    if isinstance(val, list):
                        cur.close()
                        conn.close()
                        return val
                    if isinstance(val, str):
                        cur.close()
                        conn.close()
                        try:
                            parsed = json.loads(val)
                            if isinstance(parsed, list):
                                return parsed
                        except:
                            pass
                        return [s.strip() for s in val.split(',') if s.strip()]
            except:
                continue
        
        cur.close()
        conn.close()
        return []
    except Exception as e:
        logger.error(f"[_fetch_jskillset_from_process] Error: {e}")
        return []

def _sync_login_jskillset_to_process(username: str, linkedinurl: str, normalized_linkedin: str):
    """
    Copy user's skillset from login table (jskillset/skills) to process table (jskillset/jskill/skills)
    for the candidate. Since 'linkedinurl' may not exist in login table, 
    we use 'username' to find source data in login, and 'linkedinurl' to identify target row in process.
    
    IMPORTANT: This function should write to 'jskillset' or 'jskill' column in process table 
    if available, to avoid overwriting candidate's own 'skillset' or 'skills' column.
    """
    try:
        import psycopg2
        from psycopg2 import sql
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()

        # 1. Find source column in login
        # Priority: jskillset > jskills > skills > skillset
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema='public' AND table_name='login'
            AND column_name IN ('jskillset','jskills','skills','skillset')
        """)
        login_cols = {r[0].lower() for r in cur.fetchall()}
        login_skill_col = None
        for cand in ['jskillset', 'jskills', 'skills', 'skillset']:
            if cand in login_cols:
                login_skill_col = cand
                break
        
        if not login_skill_col:
            cur.close(); conn.close(); return

        # 2. Read skill value from login using username
        skill_val = None
        if username:
            cur.execute(sql.SQL("SELECT {} FROM login WHERE username=%s LIMIT 1").format(sql.Identifier(login_skill_col)), (username,))
            r = cur.fetchone()
            if r and r[0]:
                v = r[0]
                if isinstance(v, (list, tuple)):
                    # If stored as JSON/array in DB, convert to comma-string for compatibility or re-serialize
                    # Let's normalize to comma-string for broader compatibility unless destination is jsonb
                    skill_val = ",".join(str(x).strip() for x in v if str(x).strip())
                else:
                    skill_val = str(v).strip()

        if not skill_val:
            cur.close(); conn.close(); return

        # 3. Find dest column in process
        # Correct logic: Prefer 'jskillset' or 'jskill' to store JOB/TARGET skills. 
        # Only fallback to 'skills'/'skillset' if explicit target cols are missing, 
        # but be careful not to overwrite extracted candidate skills.
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema='public' AND table_name='process'
            AND column_name IN ('jskillset','jskills','jskill','skills','skillset', 'normalized_linkedin', 'linkedinurl')
        """)
        process_cols = {r[0].lower() for r in cur.fetchall()}
        
        # Priority for destination: jskillset > jskills > jskill
        process_skill_col = None
        for cand in ['jskillset', 'jskills', 'jskill']:
            if cand in process_cols:
                process_skill_col = cand
                break
        
        # If no specific 'jskill*' column exists, skip to avoid overwriting candidate skills
        if not process_skill_col:
            cur.close(); conn.close(); return

        # 4. Update process using linkedinurl/normalized as key to identify the record
        updated = 0
        if normalized_linkedin and 'normalized_linkedin' in process_cols:
            cur.execute(sql.SQL("UPDATE process SET {} = %s WHERE normalized_linkedin = %s").format(sql.Identifier(process_skill_col)), (skill_val, normalized_linkedin))
            updated = cur.rowcount
        if updated == 0 and linkedinurl and 'linkedinurl' in process_cols:
            cur.execute(sql.SQL("UPDATE process SET {} = %s WHERE linkedinurl = %s").format(sql.Identifier(process_skill_col)), (skill_val, linkedinurl))
        
        conn.commit()
        cur.close(); conn.close()
    except Exception as e:
        logger.warning(f"[_sync_login_jskillset_to_process] failed: {e}")

def _gemini_talent_pool_suggestion(skills_list):
    """
    Ask Gemini to propose:
      - job_titles: array of job titles that encompass all skills
      - companies: array of company names (should prefer cross-sector diversity)
    Returns tuple (job_titles, companies, raw_response) where lists may be empty.
    Falls back to heuristic generation if Gemini not available or fails.
    """
    job_titles = []
    companies = []
    raw = ""
    if not skills_list:
        return job_titles, companies, raw

    # Try to use Gemini if configured and client library available
    if genai and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
            prompt = (
                "You are an assistant that maps skill lists to candidate-facing job titles and representative target companies.\n"
                "INPUT: A JSON array of skill tokens. Example: [\"Python\",\"Django\",\"PostgreSQL\"]\n"
                "OUTPUT: Return STRICT JSON ONLY with keys: {job_titles, companies}.\n"
                "- job_titles: an array (max 8) of concise job titles that together cover the given skillset. Prefer real-world titles (e.g., 'Backend Engineer', 'Data Engineer', 'ML Engineer').\n"
                "- companies: an array (max 20) of company names that commonly hire for such skills. Ensure companies are drawn from different sectors where possible; avoid listing multiple companies from same industry cluster when alternatives exist.\n"
                "Rules:\n"
                "- Do not include commentary. Only the JSON object.\n"
                "- Deduplicate outputs. Order job_titles by relevance. Order companies by sector diversity.\n\n"
                f"SKILLS:\n{json.dumps(skills_list, ensure_ascii=False)}\n\nJSON:"
            )
            resp = model.generate_content(prompt)
            raw = (resp.text or "").strip()
            
            parsed = _extract_json_object(raw)
            if isinstance(parsed, dict):
                jt = parsed.get("job_titles") or parsed.get("jobs") or []
                comp = parsed.get("companies") or parsed.get("company") or []
                if isinstance(jt, str):
                    jt = [s.strip() for s in re.split(r'[,\n;]+', jt) if s.strip()]
                if isinstance(comp, str):
                    comp = [s.strip() for s in re.split(r'[,\n;]+', comp) if s.strip()]
                job_titles = [str(x).strip() for x in jt if str(x).strip()]
                companies = [str(x).strip() for x in comp if str(x).strip()]
                # Ensure cross-sector uniqueness: naive dedupe by normalized sector keyword (best-effort)
                # We cannot reliably detect sector server-side here; leave as returned but dedupe exact companies
                job_titles = dedupe(job_titles)[:8]
                # Apply server-side cap to companies returned by Gemini as well
                companies = dedupe(companies)[:MAX_COMPANY_SUGGESTIONS]
                return job_titles, companies, raw
        except Exception:
            # fallthrough to heuristic
            pass

    # Heuristic fallback: map skills keywords to typical titles and companies
    try:
        s_lower = " ".join(skills_list).lower()
        # simple title heuristics
        title_candidates = []
        if any(k in s_lower for k in ["machine learning","ml","pytorch","tensorflow","scikit"]):
            title_candidates += ["Machine Learning Engineer", "Data Scientist", "ML Research Engineer"]
        if any(k in s_lower for k in ["sql","postgres","mysql","mongodb","nosql","spark","hadoop","etl","data pipeline"]):
            title_candidates += ["Data Engineer", "ETL Engineer", "Analytics Engineer"]
        if any(k in s_lower for k in ["aws","azure","gcp","kubernetes","docker","terraform","devops","sre"]):
            title_candidates += ["DevOps Engineer", "Site Reliability Engineer", "Cloud Infrastructure Engineer"]
        if any(k in s_lower for k in ["react","angular","vue","frontend","javascript","typescript","css","html"]):
            title_candidates += ["Frontend Engineer", "UI Engineer"]
        if any(k in s_lower for k in ["java","spring","c#","dotnet","c++","golang","go","backend","api","rest","grpc"]):
            title_candidates += ["Backend Engineer", "Software Engineer"]
        if any(k in s_lower for k in ["product","roadmap","stakeholder","prerogative","product manager"]):
            title_candidates += ["Product Manager", "Technical Product Manager"]
        # generic fallbacks
        if not title_candidates:
            title_candidates = ["Software Engineer", "Product Manager", "Data Scientist", "DevOps Engineer"]
        job_titles = dedupe(title_candidates)[:8]

        # companies heuristics by sector keywords
        # companies heuristics by sector keywords (stricter matching)
        company_pool = []

        # Helper local regex check for whole-word presence
        def has_kw(patterns):
            for p in patterns:
                if re.search(r'\b' + re.escape(p) + r'\b', s_lower):
                    return True
            return False

        if has_kw(["gaming","game","graphics","render"]):
            company_pool += ["Ubisoft", "Electronic Arts", "Unity Technologies"]

        if has_kw(["bank","payments","fintech"]):
            company_pool += ["DBS Bank", "OCBC Bank", "Standard Chartered", "Stripe", "Visa", "Mastercard"]

        # NOTE: Pharma heuristic removed entirely per user request.
        # Earlier versions added pharma companies when "pharma"/"clinical" tokens matched.
        # We purposely DO NOT inject pharmaceutical company names here.

        if has_kw(["cloud","aws","azure","gcp","kubernetes"]):
            company_pool += ["Amazon", "Google", "Microsoft", "IBM", "Oracle"]

        if has_kw(["retail","ecommerce","shop"]):
            company_pool += ["Amazon", "Shopify", "Sea Limited", "Shopee", "Lazada"]

        # add some generic tech companies for broad matches
        company_pool += ["Google", "Microsoft", "Amazon", "Facebook (Meta)", "Apple", "Nvidia", "Intel", "Accenture", "Capgemini"]

        # Deduplicate and cap the companies list as requested
        companies = dedupe(company_pool)[:MAX_COMPANY_SUGGESTIONS]
        raw = json.dumps({"job_titles": job_titles, "companies": companies}, ensure_ascii=False)
        return job_titles, companies, raw
    except Exception:
        return [], [], ""

@app.post("/highlight_talent_pools")
def highlight_talent_pools():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    
    if not username:
        return jsonify({"error": "username required"}), 400

    try:
        # 1. Fetch user's persisted skillset
        skills = _fetch_jskillset(username) or []
        if not skills:
            # Try once more with a slight delay in case of replication lag (unlikely in local dev but safe)
            time.sleep(0.5)
            skills = _fetch_jskillset(username) or []
            
        if not skills:
            return jsonify({"error": "No skillset found in profile", "code": "no_skills"}), 200

        # 2. Ask Gemini (or fallback) for job titles + cross-sector companies
        job_titles, companies, raw = _gemini_talent_pool_suggestion(skills)

        # 3. If both lists empty, respond with error
        if not job_titles and not companies:
            return jsonify({
                "error": "Could not generate suggestions based on skills", 
                "skills_count": len(skills)
            }), 200

        # 4. Return structured response
        return jsonify({
            "job": {"related": job_titles},
            "company": {"related": companies},
            "skills_count": len(skills),
            "engine": "gemini" if (genai and GEMINI_API_KEY) else "heuristic"
        }), 200

    except Exception as e:
        logger.error(f"[Highlight Talent Pools] {e}")
        return jsonify({"error": str(e)}), 500

# --- START PATCH: restore gemini_company_job_extract and rebate assessment endpoints ---

# Ensure _normalize_linkedin_to_path exists (define only if not already defined)
try:
    _normalize_linkedin_to_path  # type: ignore
except NameError:
    def _normalize_linkedin_to_path(linkedin_url: str) -> str:
        if not linkedin_url:
            return ""
        s = linkedin_url.split('?', 1)[0].strip()
        path = re.sub(r'^https?://[^/]+', '', s, flags=re.I)
        path = path.lower().rstrip('/')
        return path

@app.post("/gemini/company_job_extract")
def gemini_company_job_extract():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text required"}), 400
    if not (genai and GEMINI_API_KEY):
        return jsonify({"error": "Gemini not configured"}), 503
    try:
        model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
        prompt = (
            "SYSTEM:\n"
            "You are given raw OCR text from a professional profile or CV timeline. "
            "Identify the CURRENT (most recent) employment. Return STRICT JSON only:\n"
            "{\"company\":\"<company>\",\"job_title\":\"<job title>\"}\n"
            "If unsure, still make the best inference. Do not add commentary.\n\n"
            f"TEXT:\n{text}\n\nJSON:"
        )
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
        obj = _extract_json_object(raw)
        if not isinstance(obj, dict):
            return jsonify({"error": "Gemini did not return valid JSON"}), 422
        company = (obj.get("company") or "").strip()
        job_title = (obj.get("job_title") or obj.get("jobTitle") or "").strip()
        company = re.sub(r'^\s*["“”`]+|["“”`]+\s*$', '', company)
        job_title = re.sub(r'^\s*["“”`]+|["“”`]+\s*$', '', job_title)
        return jsonify({"company": company, "job_title": job_title}), 200
    except Exception as e:
        logger.warning(f"[Gemini Company/Job Extract] {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/gemini/rebate_validate")
def gemini_rebate_validate():
    data = request.get_json(force=True, silent=True) or {}
    job_title = (data.get("job_title") or data.get("jobTitle") or "").strip()
    role_tag = (data.get("role_tag") or data.get("roleTag") or "").strip()
    justification = (data.get("justification") or "").strip()
    
    # NEW: Capture extra context for persistent updates
    username = (data.get("username") or "").strip()
    linkedinurl = (data.get("linkedinurl") or "").strip()
    normalized_linkedin = _normalize_linkedin_to_path(linkedinurl)

    full_experience_list = data.get("experience_list") or data.get("experience") or []
    full_experience_text = (data.get("experience_text") or "").strip()

    if not job_title or not role_tag:
        return jsonify({"error": "job_title and role_tag required"}), 400
    has_exp = (isinstance(full_experience_list, list) and len(full_experience_list) > 0) or bool(full_experience_text)
    if not has_exp:
        return jsonify({"error": "experience_text or experience_list required for rebate assessment", "code": 412}), 412
        
    # --- TRIGGER jskill UPDATE ON REBATE VALIDATION ---
    # When rebate assessment is triggered, ensure role_tag is persisted into 'process' table as 'jskill'.
    if role_tag and (linkedinurl or normalized_linkedin):
        try:
            import psycopg2
            from psycopg2 import sql
            pg_host=os.getenv("PGHOST","localhost")
            pg_port=int(os.getenv("PGPORT","5432"))
            pg_user=os.getenv("PGUSER","postgres")
            pg_password=os.getenv("PGPASSWORD","") or "orlha"
            pg_db=os.getenv("PGDATABASE","candidate_db")
            conn_rt=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
            cur_rt=conn_rt.cursor()
            
            # Check for jskill column existence
            cur_rt.execute("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema='public' AND table_name='process' AND column_name='jskill'
            """)
            has_jskill = bool(cur_rt.fetchone())
            
            if has_jskill:
                updated_count = 0
                # Use full role_tag without truncation (DB columns now TEXT type)
                # Try by normalized first
                if normalized_linkedin:
                    cur_rt.execute("UPDATE process SET jskill=%s WHERE normalized_linkedin=%s", (role_tag, normalized_linkedin))
                    updated_count = cur_rt.rowcount
                # Fallback to exact URL
                if updated_count == 0 and linkedinurl:
                    cur_rt.execute("UPDATE process SET jskill=%s WHERE linkedinurl=%s", (role_tag, linkedinurl))
                
                conn_rt.commit()
            
            cur_rt.close(); conn_rt.close()
            
            # Now trigger jskillset sync from login to process
            _sync_login_jskillset_to_process(username, linkedinurl, normalized_linkedin)

        except Exception as e_rt:
            logger.warning(f"[Rebate Validate] Failed to sync role_tag to jskill: {e_rt}")
    # --------------------------------------------------

    if not (genai and GEMINI_API_KEY):
        return jsonify({"error": "Gemini not configured"}), 503
    try:
        model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)

        exp_lines = []
        if isinstance(full_experience_list, list):
            for x in full_experience_list:
                if isinstance(x, str):
                    t = x.strip()
                    if t:
                        exp_lines.append(t)
                elif isinstance(x, dict):
                    title = (x.get("title") or x.get("job_title") or x.get("jobTitle") or "").strip()
                    company = (x.get("company") or "").strip()
                    start = (x.get("start") or x.get("start_year") or "").strip()
                    end = (x.get("end") or x.get("end_year") or "").strip()
                    if title or company:
                        segs = [seg for seg in [title, company, f"{start} to {end}".strip()] if seg and seg.strip()]
                        exp_lines.append(", ".join(segs))
        if not exp_lines and full_experience_text:
            exp_lines.append(full_experience_text)

        prompt = (
            "SYSTEM: You perform rebate eligibility assessment based on experience history.\n"
            "Return ONLY JSON: {\"relevant\":true|false, \"reasoning\":\"...\"}\n\n"
            "Decision Rules:\n"
            "- PRIORITY: Use the LATEST (most recent) experience entry as the primary signal.\n"
            "- Invalid rebate (relevant=true): latest role directly matches the searched role title/seniority (e.g., manager) AND experience is relevant to the role_tag.\n"
            "- Valid rebate (relevant=false): latest role is irrelevant, mismatched domain/field, or different seniority level (too junior/senior), "
            "  EVEN IF earlier history contains relevant experience.\n"
            "- Appeals: If a justification is provided, consider the entire history for context, but latest role still takes precedence.\n\n"
            "Inputs:\n"
            f"- searched_role_tag: {role_tag}\n"
            f"- extracted_job_title (from latest experience): {job_title}\n"
            f"- justification: {justification or '(none)'}\n"
            "- experience_history_latest_first:\n"
            + ("\n".join([f"  - {line}" for line in exp_lines]) if exp_lines else "  - (none) ") +
            "\n\n"
            "Output JSON only (no comments):"
        )

        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
        obj = _extract_json_object(raw)
        if not isinstance(obj, dict):
            jt_lower = job_title.lower()
            rt_lower = role_tag.lower()
            heuristic_rel = (jt_lower in rt_lower) or (rt_lower in jt_lower)
            searched_title = role_tag
            human_reason = ("We compared the candidate’s current job title to the searched "
                            f"\"{searched_title}\" and found a {'strong' if heuristic_rel else 'insufficient'} match based on title tokens.")
            return jsonify({"relevant": bool(heuristic_rel), "reasoning": "Heuristic fallback based on title-token match.", "human_reason": human_reason})

        relevant = bool(obj.get("relevant"))
        reasoning = (obj.get("reasoning") or "").strip() or "No reasoning provided."

        searched_title = role_tag
        if relevant:
            human_reason = (
                "The latest role and responsibilities align with the searched "
                f"\"{searched_title}\", so the profile is considered relevant and not eligible for rebate."
            )
        else:
            human_reason = (
                "The candidate’s most recent role and responsibilities do not match the searched "
                f"\"{searched_title}\". Based on the latest-role priority rule, this profile qualifies for rebate."
            )

        return jsonify({"relevant": relevant, "reasoning": reasoning, "human_reason": human_reason}), 200
    except Exception as e:
        logger.warning(f"[Gemini Rebate Validate] {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/gemini/experience_format")
def gemini_experience_format():
    data = request.get_json(force=True, silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "text required"}), 400
    if not (genai and GEMINI_API_KEY):
        return jsonify({"error": "Gemini not configured"}), 503
    try:
        model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
        prompt = (
            "SYSTEM:\n"
            "You are given unstructured experience/education text from a profile or CV.\n"
            "Return STRICT JSON only with these keys:\n"
            "{\n"
            "  \"experience\": [\"Job Title, Company, StartYear to EndYear|present\", ...],\n"
            "  \"education\": [\"University Name, Degree Type, Discipline\", ...],\n"
            "  \"language\": [\"Language Name [optional proficiency]\", ...]\n"
            "}\n"
            "Rules:\n"
            "- Each experience line must be: Job Title, Company, YYYY to YYYY OR YYYY to present.\n"
            "- Only include Education if a university is detected.\n"
            "- Include Language if any languages are explicitly mentioned; include proficiency if given, otherwise just the language name.\n"
            "- No commentary, no extra keys. Output only valid JSON.\n\n"
            f"TEXT:\n{text}\n\nJSON:"
        )
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
        obj = _extract_json_object(raw)
        if not isinstance(obj, dict):
            logger.warning(f"[Gemini Experience Format] Unparsable response: {raw[:200]}")
            return jsonify({"error": "Gemini did not return valid JSON"}), 422

        experience = obj.get("experience") or []
        education = obj.get("education") or []
        language = obj.get("language") or obj.get("language") or []

        if isinstance(experience, str):
            experience = [experience]
        if isinstance(education, str):
            education = [education]
        if isinstance(language, str):
            language = [language]

        exp_out = [str(x).strip() for x in experience if str(x).strip()]
        edu_out = [str(x).strip() for x in education if str(x).strip()]
        lang_out = [str(x).strip() for x in language if str(x).strip()]

        return jsonify({"experience": exp_out, "education": edu_out, "language": lang_out}), 200
    except Exception as e:
        logger.warning(f"[Gemini Experience Format] {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/gemini/assess_profile")
def gemini_assess_profile():
    data = request.get_json(force=True, silent=True) or {}
    linkedinurl = (data.get("linkedinurl") or "").strip()
    job_title = (data.get("job_title") or data.get("jobtitle") or "").strip()
    role_tag = (data.get("role_tag") or data.get("roleTag") or "").strip()
    company = (data.get("company") or "").strip()
    country = (data.get("country") or "").strip()
    seniority = (data.get("seniority") or "").strip()
    sector = (data.get("sector") or "").strip()
    experience_text = (data.get("experience_text") or "").strip()
    username = (data.get("username") or "").strip()
    userid = (data.get("userid") or "").strip()
    custom_weights = data.get("custom_weights") or {}
    assessment_level = (data.get("assessment_level") or "L1").strip().upper()  # L1 or L2
    tenure = data.get("tenure")  # Average tenure value

    # 1. Fetch Target Skillset (jskillset) from process table (cross-check source)
    # Per requirement: Cross-check against jskillset column in process table, not login table
    target_skills = []
    if linkedinurl:
        target_skills = _fetch_jskillset_from_process(linkedinurl)
    # Fallback to login table if process table doesn't have jskillset
    if not target_skills and username:
        target_skills = _fetch_jskillset(username)
    
    # 2. Fetch Candidate Skillset from process table if available
    # NEW: Use vskillset instead of skillset for Gemini assessment
    candidate_skills = []
    try:
        candidate_skills = data.get("skillset") or []
        if not candidate_skills and linkedinurl:
            normalized = _normalize_linkedin_to_path(linkedinurl)
            import psycopg2
            pg_host=os.getenv("PGHOST","localhost")
            pg_port=int(os.getenv("PGPORT","5432"))
            pg_user=os.getenv("PGUSER","postgres")
            pg_password=os.getenv("PGPASSWORD","") or "orlha"
            pg_db=os.getenv("PGDATABASE","candidate_db")
            conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
            cur=conn.cursor()
            
            # Check if vskillset column exists (prioritize vskillset over skillset)
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='process' AND column_name IN ('vskillset', 'skillset')")
            available_cols = {r[0] for r in cur.fetchall()}
            
            # Try to fetch vskillset first (High probability skills only)
            if 'vskillset' in available_cols:
                row = None
                if normalized:
                    cur.execute("SELECT vskillset FROM process WHERE normalized_linkedin = %s", (normalized,))
                    row = cur.fetchone()
                if not row and linkedinurl:
                    cur.execute("SELECT vskillset FROM process WHERE linkedinurl = %s", (linkedinurl,))
                    row = cur.fetchone()
                
                if row and row[0]:
                    vskillset_val = row[0]
                    # Parse vskillset JSON and extract High skills
                    try:
                        if isinstance(vskillset_val, str):
                            vskillset_data = json.loads(vskillset_val)
                        else:
                            vskillset_data = vskillset_val
                        
                        if isinstance(vskillset_data, list):
                            # Extract skills with High category
                            # Validate that both 'skill' and 'category' keys exist
                            # Only High probability skills are considered valid
                            candidate_skills = [
                                item.get("skill") for item in vskillset_data 
                                if isinstance(item, dict) 
                                and item.get("skill")  # Ensure skill exists
                                and item.get("category") == "High"
                            ]
                            logger.info(f"[Assess] Using vskillset: {len(candidate_skills)} High skills extracted")
                    except Exception as e_vs:
                        logger.warning(f"[Assess] Failed to parse vskillset, falling back to skillset: {e_vs}")
                        candidate_skills = []
            
            # Fallback to skillset if vskillset not available or empty
            if not candidate_skills and 'skillset' in available_cols:
                row = None
                if normalized:
                    cur.execute("SELECT skillset FROM process WHERE normalized_linkedin = %s", (normalized,))
                    row = cur.fetchone()
                if not row and linkedinurl:
                    cur.execute("SELECT skillset FROM process WHERE linkedinurl = %s", (linkedinurl,))
                    row = cur.fetchone()
                
                if row and row[0]:
                    val = row[0]
                    if isinstance(val, str):
                        candidate_skills = [s.strip() for s in val.split(',') if s.strip()]
                    elif isinstance(val, list):
                        candidate_skills = val
                
                # Log fallback usage after val extraction
                if candidate_skills:
                    logger.info(f"[Assess] Using skillset (fallback): {len(candidate_skills)} skills")
            
            cur.close(); conn.close()
    except Exception as e:
        logger.warning(f"[Assess] Failed to fetch candidate skills: {e}")

    # 3. NEW: Fetch Process Hints (jskillset/jskills/jskill) from process table
    process_skills = []
    try:
        # Prefer jskillset stored in login (source-of-truth for target/jobs-related skills)
        if username:
            try:
                login_skills = _fetch_jskillset(username) or []
                if isinstance(login_skills, list) and login_skills:
                    process_skills = [str(x).strip() for x in login_skills if str(x).strip()]
            except Exception as e_fetch_login:
                logger.warning(f"[Assess] _fetch_jskillset failed for user '{username}': {e_fetch_login}")

        # Fallback: if login has none, try the process table's jskill* hints (legacy)
        if not process_skills and linkedinurl:
            import psycopg2
            pg_host=os.getenv("PGHOST","localhost")
            pg_port=int(os.getenv("PGPORT","5432"))
            pg_user=os.getenv("PGUSER","postgres")
            pg_password=os.getenv("PGPASSWORD","") or "orlha"
            pg_db=os.getenv("PGDATABASE","candidate_db")
            conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
            cur=conn.cursor()
            
            # Determine available columns
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_schema='public' AND table_name='process' AND column_name IN ('jskillset','jskills','jskill')
            """)
            avail = {r[0] for r in cur.fetchall()}
            
            # Priority
            col_to_use = None
            if 'jskillset' in avail: col_to_use = 'jskillset'
            elif 'jskills' in avail: col_to_use = 'jskills'
            elif 'jskill' in avail: col_to_use = 'jskill'
            
            if col_to_use:
                normalized = _normalize_linkedin_to_path(linkedinurl)
                row = None
                if normalized:
                    cur.execute(f"SELECT {col_to_use} FROM process WHERE normalized_linkedin = %s", (normalized,))
                    row = cur.fetchone()
                if not row:
                    cur.execute(f"SELECT {col_to_use} FROM process WHERE linkedinurl = %s", (linkedinurl,))
                    row = cur.fetchone()
                    
                if row and row[0]:
                    val = row[0]
                    if isinstance(val, list): process_skills = [str(x).strip() for x in val]
                    elif isinstance(val, str):
                        try:
                            # json is already imported globally, no need to import again
                            parsed = json.loads(val)
                            if isinstance(parsed, list): process_skills = [str(x).strip() for x in parsed]
                            else: process_skills = [s.strip() for s in val.split(',') if s.strip()]
                        except:
                            process_skills = [s.strip() for s in val.split(',') if s.strip()]
                            
            cur.close(); conn.close()
    except Exception as e:
        logger.warning(f"[Assess] Failed to fetch process_skills: {e}")

    # NEW: Trigger vskillset inference BEFORE assessment
    # This populates the vskillset column and MERGES confirmed skills with existing skillset
    vskillset_results = None  # Initialize to avoid NameError later
    try:
        # Check prerequisites for vskillset inference
        if not linkedinurl:
            logger.info(f"[Gemini Assess -> vskillset] Skipped: No linkedinurl provided")
        elif not target_skills or len(target_skills) == 0:
            logger.info(f"[Gemini Assess -> vskillset] Skipped: No target_skills for linkedin='{linkedinurl}'")
        else:
            import psycopg2
            pg_host = os.getenv("PGHOST", "localhost")
            pg_port = int(os.getenv("PGPORT", "5432"))
            pg_user = os.getenv("PGUSER", "postgres")
            pg_password = os.getenv("PGPASSWORD", "") or "orlha"
            pg_db = os.getenv("PGDATABASE", "candidate_db")
            
            conn = psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
            cur = conn.cursor()
            
            # Normalize linkedin URL
            normalized = linkedinurl.lower().strip().rstrip('/')
            if not normalized.startswith('http'):
                normalized = 'https://' + normalized
            
            # Fetch experience and existing skillset from process table
            experience_text = ""
            existing_skillset = []
            
            cur.execute("""
                SELECT experience, skillset 
                FROM process 
                WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl)) = %s
                LIMIT 1
            """, (normalized,))
            row = cur.fetchone()
            
            if row:
                experience_text = (row[0] or "").strip()
                # Parse existing skillset
                if row[1]:
                    skillset_val = row[1]
                    if isinstance(skillset_val, str):
                        try:
                            # Try JSON parse first
                            existing_skillset = json.loads(skillset_val)
                            if not isinstance(existing_skillset, list):
                                existing_skillset = []
                        except (json.JSONDecodeError, ValueError):
                            # Fallback to comma-separated
                            existing_skillset = [s.strip() for s in skillset_val.split(',') if s.strip()]
                    elif isinstance(skillset_val, list):
                        existing_skillset = skillset_val
            
            # Use experience as profile context
            profile_context = experience_text
            
            if not profile_context:
                logger.info(f"[Gemini Assess -> vskillset] Skipped: No experience data for linkedin='{linkedinurl}'")
            elif not genai or not GEMINI_API_KEY:
                logger.warning(f"[Gemini Assess -> vskillset] Skipped: Gemini not configured")
            else:
                # Call Gemini to evaluate each skill
                model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
                
                prompt = f"""SYSTEM:
You are an expert technical recruiter evaluating candidate skillsets based on their work experience.

TASK:
For each skill in the list below, evaluate the candidate's likely proficiency based on their experience.
Assign a probability score (0-100) and categorize as Low (<40), Medium (40-74), or High (75-100).
Provide clear reasoning based on job titles, companies, and experience patterns.

CANDIDATE PROFILE:
{profile_context[:3000]}

SKILLS TO EVALUATE:
{json.dumps(target_skills, ensure_ascii=False)}

OUTPUT FORMAT (JSON):
{{
  "evaluations": [
    {{
      "skill": "skill_name",
      "probability": 0-100,
      "category": "Low|Medium|High",
      "reason": "Brief explanation based on companies and roles"
    }}
  ]
}}

Return ONLY the JSON object, no other text."""
                
                resp = model.generate_content(prompt)
                raw_text = (resp.text or "").strip()
                
                # Extract JSON from response
                parsed = _extract_json_object(raw_text)
                
                if parsed and "evaluations" in parsed:
                    results = parsed["evaluations"]
                    
                    # Ensure all required fields are present
                    for item in results:
                        if "probability" not in item:
                            item["probability"] = 50
                        if "category" not in item:
                            prob = item.get("probability", 50)
                            if prob >= 75:
                                item["category"] = "High"
                            elif prob >= 40:
                                item["category"] = "Medium"
                            else:
                                item["category"] = "Low"
                        if "reason" not in item:
                            item["reason"] = "No reasoning provided"
                    
                    # Persist vskillset to database
                    vskillset_json = json.dumps(results, ensure_ascii=False)
                    
                    # Get confirmed skills from vskillset (High only)
                    confirmed_skills = [item["skill"] for item in results if item["category"] == "High"]
                    
                    # MERGE with existing skillset (not replace)
                    # Preserve order: keep existing skills first, then add new ones (avoiding duplicates)
                    existing_set = set(existing_skillset)
                    merged_skillset = existing_skillset + [skill for skill in confirmed_skills if skill not in existing_set]
                    # Ensure all skills are strings before joining
                    skillset_str = ", ".join([str(s) for s in merged_skillset if s])
                    
                    # Check if vskillset column exists
                    cur.execute("""
                        SELECT column_name 
                        FROM information_schema.columns
                        WHERE table_schema='public' AND table_name='process' 
                          AND column_name IN ('vskillset', 'skillset')
                    """)
                    available_cols = {r[0] for r in cur.fetchall()}
                    
                    # Update process table
                    updates = []
                    if 'vskillset' in available_cols:
                        updates.append("vskillset = %s")
                    if 'skillset' in available_cols:
                        updates.append("skillset = %s")
                    
                    if updates:
                        update_sql = f"UPDATE process SET {', '.join(updates)} WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl)) = %s"
                        
                        update_values = []
                        if 'vskillset' in available_cols:
                            update_values.append(vskillset_json)
                        if 'skillset' in available_cols:
                            update_values.append(skillset_str)
                        update_values.append(normalized)
                        
                        cur.execute(update_sql, tuple(update_values))
                        conn.commit()
                        
                        logger.info(f"[Gemini Assess -> vskillset] Populated vskillset and merged {len(confirmed_skills)} skills into skillset for linkedin='{linkedinurl}'")
                        logger.info(f"[Gemini Assess -> vskillset] Merged skillset has {len(merged_skillset)} total skills: {merged_skillset[:10]}")  # Log first 10 skills
                        
                        # Update candidate_skills so assessment uses the merged skillset
                        candidate_skills = merged_skillset
                        
                        # Store vskillset results for later inclusion in response
                        vskillset_results = results
            
            cur.close()
            conn.close()
    except Exception as e_vskillset:
        logger.warning(f"[Gemini Assess -> vskillset] Failed to populate vskillset: {e_vskillset}")

    # Pack data for core logic
    profile_data = {
        "job_title": job_title,
        "role_tag": role_tag,
        "company": company,
        "country": country,
        "seniority": seniority,
        "sector": sector,
        "experience_text": experience_text,
        "target_skills": target_skills,
        "candidate_skills": candidate_skills,
        "process_skills": process_skills,
        "custom_weights": custom_weights,
        "linkedinurl": linkedinurl,
        "assessment_level": assessment_level,  # L1 = extractive only, L2 = contextual inference
        "tenure": tenure,  # Average tenure per employer
        "vskillset_results": vskillset_results  # vskillset inference results for scoring
    }
    
    # Log data completeness before assessment
    missing_fields = []
    if not job_title and not role_tag:
        missing_fields.append("job_title/role_tag")
    if not company:
        missing_fields.append("company")
    if not country:
        missing_fields.append("country")
    if not sector:
        missing_fields.append("sector")
    if not seniority:
        missing_fields.append("seniority")
    if tenure is None:
        missing_fields.append("tenure")
    if not candidate_skills or len(candidate_skills) == 0:
        missing_fields.append("skillset")
    
    if missing_fields:
        logger.warning(f"[Gemini Assess] Proceeding with incomplete data for linkedin='{linkedinurl}'. Missing fields: {', '.join(missing_fields)}")
    else:
        logger.info(f"[Gemini Assess] All required fields present for linkedin='{linkedinurl}'")

    # Reference Mapping Augmentation
    try:
        ref_map = get_reference_mapping(job_title)
        if ref_map:
            # Apply mapped fields if available and existing field is empty or override preferred
            if not profile_data.get("seniority") and ref_map.get("seniority"):
                profile_data["seniority"] = ref_map["seniority"]
            
            if not profile_data.get("sector") and (ref_map.get("family") or ref_map.get("job_family")):
                profile_data["sector"] = ref_map.get("family") or ref_map.get("job_family")
            
            if not profile_data.get("country") and (ref_map.get("geographic") or ref_map.get("country")):
                profile_data["country"] = ref_map.get("geographic") or ref_map.get("country")
    except Exception as e:
        logger.warning(f"[Gemini Assess] Reference mapping application failed: {e}")

    # Perform Assessment
    try:
        out_obj = _core_assess_profile(profile_data)
        if not out_obj:
            logger.error("[Gemini Assess] _core_assess_profile returned None")
            return jsonify({"error": "Assessment failed - no result returned"}), 500
        
        # Add vskillset to output if it was generated
        if vskillset_results:
            out_obj["vskillset"] = vskillset_results
    except Exception as e:
        logger.error(f"[Gemini Assess] _core_assess_profile failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Assessment failed: {str(e)}"}), 500

    # NEW: Persist Level 1 assessment into the 'rating' column of the process table (if present).
    try:
        import psycopg2
        from psycopg2 import sql
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()

        # Check if 'rating' column exists
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'process' AND column_name = 'rating'
        """)
        if cur.fetchone():
            # Create a copy of out_obj without vskillset for the rating column
            rating_obj = {k: v for k, v in out_obj.items() if k != "vskillset"}
            rating_payload = json.dumps(rating_obj, ensure_ascii=False)
            normalized = None
            try:
                # Use helper if available
                normalized = _normalize_linkedin_to_path(linkedinurl)
            except Exception:
                normalized = None

            updated = 0
            if normalized:
                try:
                    cur.execute("UPDATE process SET rating = %s WHERE normalized_linkedin = %s", (rating_payload, normalized))
                    updated = cur.rowcount
                    conn.commit()
                except Exception:
                    conn.rollback()
                    updated = 0
            if updated == 0:
                try:
                    cur.execute("UPDATE process SET rating = %s WHERE linkedinurl = %s", (rating_payload, linkedinurl))
                    updated = cur.rowcount
                    conn.commit()
                except Exception:
                    conn.rollback()
                    updated = 0
            logger.info(f"[Gemini Assess -> DB rating] Updated rating for linkedin='{linkedinurl}' normalized='{normalized}' updated_rows={updated}")
        
        # --- NEW: Trigger role_tag -> jskill sync during assessment ---
        # If we successfully assessed, ensure process.jskill is updated with role_tag
        if role_tag:
            try:
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema='public' AND table_name='process' AND column_name='jskill'
                """)
                has_jskill = bool(cur.fetchone())
                if has_jskill:
                    js_updated = 0
                    if normalized:
                        cur.execute("UPDATE process SET jskill=%s WHERE normalized_linkedin=%s", (role_tag, normalized))
                        js_updated = cur.rowcount
                    if js_updated == 0:
                        cur.execute("UPDATE process SET jskill=%s WHERE linkedinurl=%s", (role_tag, linkedinurl))
                    conn.commit()
                
                # Now trigger jskillset sync from login to process
                _sync_login_jskillset_to_process(username, linkedinurl, normalized)

            except Exception as e_js:
                logger.warning(f"[Assess -> jskill] Sync failed: {e_js}")
        # -----------------------------------------------------------------
        
        # Patch: safer owner update (replace the existing owner-setting block in gemini_assess_profile)
        try:
            # Only attempt when we have values to set and a linkedinurl
            if (username or userid) and linkedinurl:
                # Discover which columns exist in process table (we will check for username, userid and normalized_linkedin)
                cur.execute("""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = 'process'
                      AND column_name IN ('username','userid','normalized_linkedin','linkedinurl')
                """)
                process_cols = {r[0] for r in cur.fetchall()}

                # Build update parts and parameters
                update_parts = []
                params = []

                if 'username' in process_cols and username:
                    update_parts.append("username = COALESCE(username, %s)")
                    params.append(username)
                if 'userid' in process_cols and userid:
                    update_parts.append("userid = COALESCE(userid, %s)")
                    params.append(userid)

                if update_parts:
                    # Determine WHERE clause depending on whether normalized_linkedin exists
                    try:
                        norm = _normalize_linkedin_to_path(linkedinurl)
                    except Exception:
                        norm = None

                    if 'normalized_linkedin' in process_cols and norm:
                        sql_update = "UPDATE process SET " + ", ".join(update_parts) + " WHERE normalized_linkedin = %s OR linkedinurl = %s"
                        params.extend([norm, linkedinurl])
                    else:
                        # fallback: update by linkedinurl only
                        sql_update = "UPDATE process SET " + ", ".join(update_parts) + " WHERE linkedinurl = %s"
                        params.append(linkedinurl)

                    try:
                        cur.execute(sql_update, tuple(params))
                        conn.commit()
                        logger.info(f"[Assess -> owner] Set username/userid for linkedin={linkedinurl} (rows={cur.rowcount})")
                    except Exception as e_up:
                        conn.rollback()
                        logger.warning(f"[Assess -> set owner] failed to set userid/username: {e_up}")
        except Exception as e:
            logger.warning(f"[Assess -> set owner] unexpected error: {e}")

        cur.close(); conn.close()
    except Exception as e:
        logger.warning(f"[Gemini Assess -> DB rating] {e}")

    return jsonify(out_obj), 200

# ... [Login/Register/Auth functions kept as is] ...
@app.post("/login")
def login_account():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    if not (username and password):
        return jsonify({"error":"username and password required"}), 400

    try:
        import common_auth
        hash_password_fn = getattr(common_auth, "hash_password", None)
        verify_password_fn = getattr(common_auth, "verify_password", None)
    except Exception:
        hash_password_fn = None
        verify_password_fn = None

    try:
        import psycopg2
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()
        cur.execute("SELECT password, userid, cemail, fullname, role_tag, COALESCE(token,0) FROM login WHERE username=%s", (username,))
        row=cur.fetchone()
        cur.close(); conn.close()
        if not row:
            return jsonify({"error":"Invalid credentials"}), 401
        stored_pw, userid, cemail, fullname, role_tag, token_val = row
        stored_pw = stored_pw or ""

        if verify_password_fn:
            ok = False
            try:
                ok = bool(verify_password_fn(stored_pw, password))
            except Exception:
                ok = False
            if not ok:
                return jsonify({"error":"Invalid credentials"}), 401
        else:
            def _local_hash_password(p: str) -> str:
                import hashlib
                salt = os.getenv("PASSWORD_SALT", "")
                return hashlib.sha256((salt + p).encode("utf-8")).hexdigest()
            hashed = hash_password_fn(password) if hash_password_fn else _local_hash_password(password)
            if stored_pw != hashed and stored_pw != password:
                return jsonify({"error":"Invalid credentials"}), 401

        return jsonify({"ok": True, "userid": userid or "", "username": username, "cemail": cemail or "", "fullname": fullname or "", "role_tag": role_tag or "", "token": int(token_val or 0)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/register")
def register_account():
    data = request.get_json(force=True, silent=True) or {}

    fullname   = (data.get("fullname") or "").strip()
    corporation = (data.get("corporation") or "").strip()
    cemail     = (data.get("cemail") or "").strip()
    username   = (data.get("username") or "").strip()
    password   = data.get("password") or ""
    userid     = (data.get("userid") or "").strip()
    created_at = (data.get("created_at") or "").strip()

    if not (fullname and cemail and username and password):
        return jsonify({"error": "fullname, cemail, username, password are required"}), 400

    if not userid:
        userid = str(uuid.uuid4().int % 9000000 + 1000000)

    try:
        import common_auth
        hash_password_fn = getattr(common_auth, "hash_password", None)
    except Exception:
        hash_password_fn = None

    if hash_password_fn:
        try:
            hashed_pw = hash_password_fn(password)
        except Exception:
            hashed_pw = None
    else:
        hashed_pw = None

    if not hashed_pw:
        def _local_hash_password(p: str) -> str:
            import hashlib
            salt = os.getenv("PASSWORD_SALT", "")
            return hashlib.sha256((salt + p).encode("utf-8")).hexdigest()
        hashed_pw = _local_hash_password(password)

    try:
        import psycopg2
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")

        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()

        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='login'
        """)
        login_cols = {r[0].lower() for r in cur.fetchall()}

        if "username" in login_cols and "cemail" in login_cols:
            cur.execute("SELECT 1 FROM login WHERE username=%s OR cemail=%s LIMIT 1", (username, cemail))
            if cur.fetchone():
                cur.close(); conn.close()
                return jsonify({"error": "Username or email already registered"}), 409
        elif "username" in login_cols:
            cur.execute("SELECT 1 FROM login WHERE username=%s LIMIT 1", (username,))
            if cur.fetchone():
                cur.close(); conn.close()
                return jsonify({"error": "Username already registered"}), 409
        elif "cemail" in login_cols:
            cur.execute("SELECT 1 FROM login WHERE cemail=%s LIMIT 1", (cemail,))
            if cur.fetchone():
                cur.close(); conn.close()
                return jsonify({"error": "Email already registered"}), 409

        insert_cols = []
        insert_vals = []

        for col, val in [
            ("userid", userid),
            ("username", username),
            ("password", hashed_pw),
            ("fullname", fullname),
            ("cemail", cemail)
        ]:
            if col in login_cols:
                insert_cols.append(col)
                insert_vals.append(val)

        if "corporation" in login_cols and corporation:
            insert_cols.append("corporation"); insert_vals.append(corporation)
        if "created_at" in login_cols and created_at:
            insert_cols.append("created_at"); insert_vals.append(created_at)
        if "role_tag" in login_cols:
            insert_cols.append("role_tag"); insert_vals.append("")
        elif "roletag" in login_cols:
            insert_cols.append("roletag"); insert_vals.append("")
        if "token" in login_cols:
            insert_cols.append("token"); insert_vals.append(0)

        if not insert_cols:
            cur.close(); conn.close()
            return jsonify({"error": "No compatible columns found for registration"}), 500

        col_sql = ", ".join(insert_cols)
        placeholders = ", ".join(["%s"] * len(insert_cols))
        cur.execute(f"INSERT INTO login ({col_sql}) VALUES ({placeholders})", insert_vals)
        conn.commit()
        cur.close(); conn.close()

        return jsonify({"ok": True, "message": "Registration successful", "username": username, "userid": userid}), 200
    except Exception as e:
        logger.error(f"[Register] {e}")
        return jsonify({"error": str(e)}), 500

@app.get("/user/resolve")
def user_resolve():
    username = (request.args.get("username") or "").strip()
    if not username:
        return jsonify({"error": "username required"}), 400
    try:
        import psycopg2
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()
        cur.execute("SELECT userid, fullname, role_tag, COALESCE(token,0) FROM login WHERE username=%s", (username,))
        row = cur.fetchone()
        cur.close(); conn.close()
        if not row:
            return jsonify({"error":"not found"}), 404
        return jsonify({"userid": row[0] or "", "fullname": row[1] or "", "role_tag": (row[2] or ""), "token": int(row[3] or 0)}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==================== VSkillset Integration Endpoints ====================

@app.get("/user/jskillset")
def get_user_jskillset():
    """
    GET /user/jskillset?username=<username>
    Returns the user's jskillset from the login table.
    Response: { "jskillset": ["Python", "C++", ...] }
    """
    username = (request.args.get("username") or "").strip()
    if not username:
        return jsonify({"error": "username required"}), 400
    
    try:
        skills = _fetch_jskillset(username)
        return jsonify({"jskillset": skills}), 200
    except Exception as e:
        logger.error(f"[get_user_jskillset] Error: {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/vskillset/infer")
def vskillset_infer():
    """
    POST /vskillset/infer
    Body: { 
        linkedinurl: "<url>", 
        skills: ["Python", "C++", ...], 
        assessment_level: "L1"|"L2", 
        username: "<optional>" 
    }
    
    Uses Gemini to evaluate each skill based on experience/cv.
    Returns: { 
        results: [ 
            { skill: "Python", probability: 85, category: "High", reason: "..." },
            ...
        ], 
        persisted: true 
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    linkedinurl = (data.get("linkedinurl") or "").strip()
    skills = data.get("skills", [])
    assessment_level = (data.get("assessment_level") or "L1").upper()
    username = (data.get("username") or "").strip()
    
    if not linkedinurl or not skills:
        return jsonify({"error": "linkedinurl and skills required"}), 400
    
    if not isinstance(skills, list) or len(skills) == 0:
        return jsonify({"error": "skills must be a non-empty array"}), 400
    
    if not (genai and GEMINI_API_KEY):
        return jsonify({"error": "Gemini not configured"}), 503
    
    try:
        import psycopg2
        from psycopg2 import sql
        pg_host = os.getenv("PGHOST", "localhost")
        pg_port = int(os.getenv("PGPORT", "5432"))
        pg_user = os.getenv("PGUSER", "postgres")
        pg_password = os.getenv("PGPASSWORD", "") or "orlha"
        pg_db = os.getenv("PGDATABASE", "candidate_db")
        
        conn = psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur = conn.cursor()
        
        # Normalize linkedin URL
        normalized = linkedinurl.lower().strip().rstrip('/')
        if not normalized.startswith('http'):
            normalized = 'https://' + normalized
        
        # Fetch experience and cv from process table
        experience_text = ""
        cv_text = ""
        
        # Try by normalized_linkedin first, then linkedinurl
        cur.execute("""
            SELECT experience, cv 
            FROM process 
            WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl)) = %s 
               OR normalized_linkedin = %s
            LIMIT 1
        """, (normalized, normalized))
        row = cur.fetchone()
        
        if row:
            experience_text = (row[0] or "").strip()
            cv_text = (row[1] or "").strip()
        
        # Use experience as primary, cv as fallback
        profile_context = experience_text if experience_text else cv_text
        
        if not profile_context:
            cur.close()
            conn.close()
            return jsonify({
                "error": "No experience or CV data found for this profile",
                "results": [],
                "persisted": False
            }), 404
        
        # Call Gemini to evaluate each skill
        model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
        
        prompt = f"""SYSTEM:
You are an expert technical recruiter evaluating candidate skillsets based on their work experience.

TASK:
For each skill in the list below, evaluate the candidate's likely proficiency based on their experience.
Assign a probability score (0-100) and categorize as Low (<40), Medium (40-74), or High (75-100).
Provide clear reasoning based on job titles, companies, and experience patterns.

CANDIDATE PROFILE:
{profile_context[:3000]}

SKILLS TO EVALUATE:
{json.dumps(skills, ensure_ascii=False)}

OUTPUT FORMAT (JSON):
{{
  "evaluations": [
    {{
      "skill": "skill_name",
      "probability": 0-100,
      "category": "Low|Medium|High",
      "reason": "Brief explanation based on companies and roles"
    }}
  ]
}}

Return ONLY the JSON object, no other text."""
        
        resp = model.generate_content(prompt)
        raw_text = (resp.text or "").strip()
        
        # Extract JSON from response
        parsed = _extract_json_object(raw_text)
        
        if not parsed or "evaluations" not in parsed:
            logger.warning(f"[vskillset_infer] Gemini returned invalid JSON: {raw_text[:200]}")
            # Fallback: create basic results
            results = []
            for skill in skills:
                results.append({
                    "skill": skill,
                    "probability": 50,
                    "category": "Medium",
                    "reason": "Unable to parse Gemini response"
                })
        else:
            results = parsed["evaluations"]
        
        # Ensure all required fields are present
        for item in results:
            if "probability" not in item:
                item["probability"] = 50
            if "category" not in item:
                prob = item.get("probability", 50)
                if prob >= 75:
                    item["category"] = "High"
                elif prob >= 40:
                    item["category"] = "Medium"
                else:
                    item["category"] = "Low"
            if "reason" not in item:
                item["reason"] = "No reasoning provided"
        
        # Persist to database
        # 1. Store full annotated results in vskillset column (JSON)
        # 2. Store only High skills in skillset column as comma-separated string
        
        vskillset_json = json.dumps(results, ensure_ascii=False)
        confirmed_skills = [item["skill"] for item in results if item["category"] == "High"]
        # Ensure all skills are strings before joining
        skillset_str = ", ".join([str(s) for s in confirmed_skills if s])
        
        # Check if vskillset column exists
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='process' 
              AND column_name IN ('vskillset', 'skillset')
        """)
        available_cols = {r[0] for r in cur.fetchall()}
        
        # Update process table
        updates = []
        if 'vskillset' in available_cols:
            updates.append("vskillset = %s")
        if 'skillset' in available_cols:
            updates.append("skillset = %s")
        
        if updates:
            update_sql = f"UPDATE process SET {', '.join(updates)} WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl)) = %s OR normalized_linkedin = %s"
            
            update_values = []
            if 'vskillset' in available_cols:
                update_values.append(vskillset_json)
            if 'skillset' in available_cols:
                update_values.append(skillset_str)
            update_values.extend([normalized, normalized])
            
            cur.execute(update_sql, tuple(update_values))
            conn.commit()
        
        cur.close()
        conn.close()
        
        return jsonify({
            "results": results,
            "persisted": True,
            "confirmed_skills": confirmed_skills
        }), 200
        
    except Exception as e:
        logger.error(f"[vskillset_infer] Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "persisted": False}), 500

@app.get("/process/skillsets")
def get_process_skillsets():
    """
    GET /process/skillsets?linkedin=<linkedinurl>
    Returns the persisted skillset and vskillset for a candidate.
    Response: { 
        "skillset": ["Python", "C++", ...], 
        "vskillset": [ 
            { "skill": "Python", "probability": 85, "category": "High", "reason": "..." },
            ...
        ] 
    }
    """
    linkedinurl = (request.args.get("linkedin") or "").strip()
    if not linkedinurl:
        return jsonify({"error": "linkedin parameter required"}), 400
    
    try:
        import psycopg2
        pg_host = os.getenv("PGHOST", "localhost")
        pg_port = int(os.getenv("PGPORT", "5432"))
        pg_user = os.getenv("PGUSER", "postgres")
        pg_password = os.getenv("PGPASSWORD", "") or "orlha"
        pg_db = os.getenv("PGDATABASE", "candidate_db")
        
        conn = psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur = conn.cursor()
        
        # Normalize linkedin URL
        normalized = linkedinurl.lower().strip().rstrip('/')
        if not normalized.startswith('http'):
            normalized = 'https://' + normalized
        
        # Check which columns exist
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns
            WHERE table_schema='public' AND table_name='process' 
              AND column_name IN ('vskillset', 'skillset')
        """)
        available_cols = {r[0] for r in cur.fetchall()}
        
        # Build SELECT query based on available columns
        select_cols = []
        if 'vskillset' in available_cols:
            select_cols.append('vskillset')
        if 'skillset' in available_cols:
            select_cols.append('skillset')
        
        if not select_cols:
            cur.close()
            conn.close()
            return jsonify({"skillset": [], "vskillset": []}), 200
        
        query = f"SELECT {', '.join(select_cols)} FROM process WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl)) = %s OR normalized_linkedin = %s LIMIT 1"
        cur.execute(query, (normalized, normalized))
        row = cur.fetchone()
        
        cur.close()
        conn.close()
        
        if not row:
            return jsonify({"skillset": [], "vskillset": []}), 200
        
        result = {}
        col_idx = 0
        
        if 'vskillset' in available_cols:
            vskillset_raw = row[col_idx]
            col_idx += 1
            if vskillset_raw:
                if isinstance(vskillset_raw, str):
                    try:
                        result["vskillset"] = json.loads(vskillset_raw)
                    except (json.JSONDecodeError, ValueError):
                        result["vskillset"] = []
                elif isinstance(vskillset_raw, list):
                    result["vskillset"] = vskillset_raw
                else:
                    result["vskillset"] = []
            else:
                result["vskillset"] = []
        else:
            result["vskillset"] = []
        
        if 'skillset' in available_cols:
            skillset_raw = row[col_idx]
            if skillset_raw:
                if isinstance(skillset_raw, str):
                    try:
                        parsed = json.loads(skillset_raw)
                        if isinstance(parsed, list):
                            result["skillset"] = parsed
                        else:
                            result["skillset"] = [s.strip() for s in skillset_raw.split(',') if s.strip()]
                    except (json.JSONDecodeError, ValueError):
                        result["skillset"] = [s.strip() for s in skillset_raw.split(',') if s.strip()]
                elif isinstance(skillset_raw, list):
                    result["skillset"] = skillset_raw
                else:
                    result["skillset"] = []
            else:
                result["skillset"] = []
        else:
            result["skillset"] = []
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"[get_process_skillsets] Error: {e}")
        return jsonify({"error": str(e)}), 500

# ==================== End VSkillset Integration ====================

# ... rest of file unchanged beyond this point ...

# Suggestion code: caching, supplemental lists, enforcement with sector-aware filtering
SUGGEST_CACHE = {}
SUGGEST_CACHE_LOCK = threading.Lock()
MAX_SUGGESTIONS_PER_TAG = int(os.getenv("MAX_SUGGESTIONS_PER_TAG", "6"))
COMPANY_SUGGESTIONS_LIMIT = int(os.getenv("COMPANY_SUGGESTIONS_LIMIT", "30"))

def _clean_list(items, limit=20):
    out=[]; seen=set()
    for x in items or []:
        if not isinstance(x,str): continue
        t=re.sub(r'\s+',' ',x).strip()
        if not t: continue
        k=t.lower()
        if k in seen: continue
        t=re.sub(r'[;,/]+$','',t)
        seen.add(k); out.append(t)
        if len(out)>=limit: break
    return out

def _country_to_region(country: str):
    c=(country or "").strip().lower()
    if not c: return None
    apac={"singapore","japan","taiwan","hong kong","china","south korea","korea","vietnam","thailand","malaysia","indonesia","philippines","australia","new zealand","india"}
    emea={"united kingdom","uk","england","ireland","germany","france","spain","italy","portugal","belgium","netherlands","switzerland","austria","poland","czech republic","czechia","sweden","norway","denmark","finland"}
    amer={"united states","usa","us","canada","mexico","brazil","argentina","chile","colombia"}
    if c in apac: return "apac"
    if c in emea: return "emea"
    if c in amer: return "americas"
    return None

COMPANY_REGION_PRESENCE = {
    "iqvia": {"apac","emea","americas"},
    "labcorp drug development": {"apac","emea","americas"},
    "labcorp": {"apac","emea","americas"},
    "ppd": {"apac","emea","americas"},
    "parexel": {"apac","emea","americas"},
    "icon": {"apac","emea","americas"},
    "syneos health": {"apac","emea","americas"},
    "novotech": {"apac"},
    "tigermed": {"apac"},
    "pfizer": {"apac","emea","americas"},
    "roche": {"apac","emea","americas"},
    "novartis": {"apac","emea","americas"},
    "johnson & johnson": {"apac","emea","americas"},
    "merck": {"apac","emea","americas"},
    "gsk": {"apac","emea","americas"},
    "sanofi": {"apac","emea","americas"},
    "astrazeneca": {"apac","emea","americas"},
    "bayer": {"apac","emea","americas"},
}

def _has_local_presence(company: str, country: str) -> bool:
    if not country: return True
    region=_country_to_region(country)
    k=(company or "").strip().lower()
    pres=COMPANY_REGION_PRESENCE.get(k)
    if pres:
        if region and region in pres: return True
        if country.strip().lower() in pres: return True
        return False
    return True

CRO_COMPETITORS = ["IQVIA","Labcorp Drug Development","Labcorp","ICON","Parexel","PPD","Syneos Health","Novotech","Tigermed"]
CRA_ADJACENT_ROLES = ["Clinical Trial Associate","Site Manager","Clinical Research Coordinator","Clinical Operations Lead","Study Start-Up Specialist","Clinical Project Manager"]

_BANNED_GENERIC_COMPANY_PHRASES = {
    "gaming studio","game studio","tech company","technology company","software company","pharma company","pharmaceutical company",
    "biotech company","marketing agency","consulting firm","it services provider","design agency","media company","advertising agency",
    "creative studio","blockchain company","web3 company","healthcare company","medical company","diagnostics company","clinical research company",
    "research organization","manufacturing company","energy company","data company"
}

def _is_real_company(name: str) -> bool:
    if not name or not isinstance(name, str):
        return False
    n = name.strip()
    if len(n) < 3:
        return False
    lower = n.lower()
    if lower in _BANNED_GENERIC_COMPANY_PHRASES:
        return False
    if re.search(r'[A-Z]', n):
        return True
    if '&' in n:
        return True
    return False

def _supplement_companies(existing, country: str, limit: int, sectors=None):
    """
    Add companies from BUCKET_COMPANIES until we reach the desired limit,
    but do NOT include pharma companies unless sectors explicitly allow pharma.
    """
    pool=[]
    seen=set(x.lower() for x in existing)
    allow_pharma = _sectors_allow_pharma(sectors)
    for bucket, data in BUCKET_COMPANIES.items():
        for group in ("global","apac"):
            for c in data.get(group,[]) or []:
                cl=c.strip()
                if not cl: continue
                if cl.lower() in seen: continue
                # Skip pharma unless allowed by sectors
                if not allow_pharma and _is_pharma_company(cl):
                    continue
                if not _has_local_presence(cl, country):
                    continue
                pool.append(cl)
                seen.add(cl.lower())
                if len(existing)+len(pool) >= limit:
                    break
            if len(existing)+len(pool) >= limit:
                break
        if len(existing)+len(pool) >= limit:
            break
    return existing + pool[:max(0, limit-len(existing))]

def _enforce_company_limit(raw_list, country: str, limit: int, sectors=None):
    """
    Clean raw_list of strings into a limited list of companies.
    If result is shorter than limit, supplement from bucket list but avoid pharma unless sectors allow it.
    """
    cleaned=[]
    seen=set()
    allow_pharma = _sectors_allow_pharma(sectors)
    for c in raw_list or []:
        if not isinstance(c,str): continue
        t=c.strip()
        if not t: continue
        k=t.lower()
        if k in seen: continue
        if not _is_real_company(t):
            continue
        # Skip pharma unless allowed
        if not allow_pharma and _is_pharma_company(t):
            continue
        if not _has_local_presence(t, country):
            continue
        seen.add(k); cleaned.append(t)
        if len(cleaned) >= limit:
            break
    if len(cleaned) < limit:
        cleaned = _supplement_companies(cleaned, country, limit, sectors)
    return cleaned[:limit]

def _gemini_suggestions(job_titles, companies, industry, languages=None, sectors=None, country: str = None):
    if not (GEMINI_API_KEY and genai): return None
    languages = languages or []
    sectors = sectors or []
    locality_hint = "Prioritize Singapore/APAC relevance where naturally applicable." if SINGAPORE_CONTEXT else ""
    
    # Add country-specific filtering instruction
    country_filter_hint = ""
    if country:
        country_filter_hint = f"\n- When suggesting companies, ONLY recommend companies with a legal entity or registered presence in {country}.\n- Exclude companies that do not operate in {country}.\n"
    
    input_obj = {
        "sectors": sectors,
        "jobTitles": job_titles,
        "companies": companies,
        "languages": languages,
        "location": (country or "").strip()
    }
    company_limit = COMPANY_SUGGESTIONS_LIMIT
    job_limit = MAX_SUGGESTIONS_PER_TAG
    prompt = (
        "SYSTEM:\nYou are a sourcing assistant. Produce concise, boolean-friendly suggestions.\n"
        "Return STRICT JSON ONLY in the form:\n"
        "{\"job\":{\"related\":[...]}, \"company\":{\"related\":[]}}\n"
        f"Hard requirements:\n"
        f"- Provide EXACTLY {job_limit} distinct, real, professional job title variants in job.related (if context allows; otherwise fill remaining with closest relevant titles).\n"
        f"- Provide EXACTLY {company_limit} distinct, real, company or organization names in company.related.\n"
        "- Company names MUST be real, brand-level entities (e.g., 'Ubisoft', 'Electronic Arts', 'Pfizer').\n"
        "- DO NOT output generic placeholders (e.g., 'Gaming Studio', 'Tech Company', 'Pharma Company', 'Consulting Firm', 'Marketing Agency').\n"
        + country_filter_hint +
        "- No duplicates, no commentary, no extra keys.\n"
        "- If insufficient context, fill remaining slots with well-known global or APAC companies relevant to the sectors/location.\n"
        "- Maintain JSON key order as shown.\n"
        f"{locality_hint}\n\nINPUT(JSON): {json.dumps(input_obj, ensure_ascii=False)}\n\nJSON:"
    )
    try:
        model=genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
        resp=model.generate_content(prompt)
        text=(resp.text or "").strip()
        start=text.find('{'); end=text.rfind('}')
        if start!=-1 and end!=-1 and end>start:
            parsed=json.loads(text[start:end+1])
            out={"job":{"related":[]}, "company":{"related":[]}}
            if isinstance(parsed,dict):
                jr=parsed.get("job",{}).get("related",[])
                cr=parsed.get("company",{}).get("related",[])
                jr_clean=_clean_list([s for s in jr if isinstance(s,str)], job_limit)
                if len(jr_clean) < job_limit:
                    heuristic_extra=_heuristic_job_suggestions(job_titles or jr_clean, industry, languages, sectors) or []
                    for h in heuristic_extra:
                        if h not in jr_clean and len(jr_clean) < job_limit:
                            jr_clean.append(h)
                # Pass sectors to enforce function so it can avoid adding pharma unless allowed
                cr_enforced=_enforce_company_limit(cr, country, company_limit, sectors)
                out["job"]["related"]=jr_clean[:job_limit]
                out["company"]["related"]=cr_enforced[:company_limit]
            return out
    except Exception as e:
        logger.warning(f"[Gemini Suggest] Failure: {e}")
    return None

def _heuristic_job_suggestions(job_titles, companies, industry, languages=None, sectors=None):
    out=set()
    languages = languages or []
    sectors = sectors or []
    for jt in job_titles:
        base=jt.strip()
        if not base: continue
        if "Senior" not in base and len(out)<MAX_SUGGESTIONS_PER_TAG: out.add(f"Senior {base}")
        if "Lead" not in base and len(out)<MAX_SUGGESTIONS_PER_TAG: out.add(f"Lead {base}")
        if industry=="Gaming" and "Game" not in base and len(out)<MAX_SUGGESTIONS_PER_TAG: out.add(f"Game {base}")
        if "Manager" not in base and not base.endswith("Manager") and len(out)<MAX_SUGGESTIONS_PER_TAG: out.add(f"{base} Manager")
        if len(out)>=MAX_SUGGESTIONS_PER_TAG: break
    if languages and len(out)<MAX_SUGGESTIONS_PER_TAG:
        for lang in languages:
            for role in [f"{lang} Translator", f"{lang} Interpreter", f"{lang} Localization", f"{lang} Linguist"]:
                if len(out)>=MAX_SUGGESTIONS_PER_TAG: break
                out.add(role)
            if len(out)>=MAX_SUGGESTIONS_PER_TAG: break
    if len(out)<MAX_SUGGESTIONS_PER_TAG:
        sect_join=" ".join(sectors).lower()
        jt_join=" ".join(job_titles).lower()
        if ("clinical research" in sect_join) or ("cra" in jt_join) or ("clinical research associate" in jt_join):
            for jt in CRA_ADJACENT_ROLES:
                if len(out)>=COMPANY_SUGGESTIONS_LIMIT: break
                out.add(jt)
    return dedupe(list(out))[:MAX_SUGGESTIONS_PER_TAG]

def _heuristic_company_suggestions(companies, languages=None, sectors=None, country: str = None):
    out=set()
    sectors = sectors or []
    for c in companies:
        base=c.strip()
        if not base: continue
        if base.endswith("Inc") or base.endswith("Inc."):
            cleaned=base.replace("Inc.","").replace("Inc","").strip()
            if cleaned: out.add(cleaned)
        if "Labs" not in base and len(out)<COMPANY_SUGGESTIONS_LIMIT: out.add(f"{base} Labs")
        if "Studio" not in base and len(out)<COMPANY_SUGGESTIONS_LIMIT: out.add(f"{base} Studio")
        if len(out)>=COMPANY_SUGGESTIONS_LIMIT: break
    if len(out)<COMPANY_SUGGESTIONS_LIMIT:
        sect_join=" ".join(sectors).lower()
        comp_join=" ".join(companies).lower()
        cro_context=("clinical research" in sect_join) or any(k in comp_join for k in ["iqvia","ppd","labcorp","parexel","icon","syneos","novotech","tigermed"])
        if cro_context:
            for cro in CRO_COMPETITORS:
                if len(out)>=COMPANY_SUGGESTIONS_LIMIT: break
                if _has_local_presence(cro, country):
                    out.add(cro)
    filtered=[c for c in out if _has_local_presence(c, country)]
    final=_enforce_company_limit(filtered, country, COMPANY_SUGGESTIONS_LIMIT)
    return final[:COMPANY_SUGGESTIONS_LIMIT]

def _prioritize_cross_sector(sets):
    freq={}
    for s in sets:
        for c in s: freq[c]=freq.get(c,0)+1
    cross=[c for c,f in freq.items() if f>1]; single=[c for c,f in freq.items() if f==1]
    ordered=[]; seen=set()
    for s in sets:
        for c in s:
            if c in cross and c not in seen: ordered.append(c); seen.add(c)
    for s in sets:
        for c in s:
            if c in single and c not in seen: ordered.append(c); seen.add(c)
    return ordered

# BUCKET_COMPANIES extended with financial_services bucket and other existing buckets
BUCKET_COMPANIES = {
    "pharma_biotech":{"global":["Pfizer","Roche","Novartis","Johnson & Johnson","Merck","GSK","Sanofi","AstraZeneca","Bayer"],"apac":["Takeda","CSL","Sino Biopharm","Sun Pharma","Daiichi Sankyo"]},
    "medical_devices":{"global":["Johnson & Johnson","Medtronic","Abbott","Baxter","Stryker","BD","Philips Healthcare","Siemens Healthineers"],"apac":["Terumo","Nipro","Wuxi AppTec (Devices)"]},
    "diagnostics":{"global":["Roche Diagnostics","Siemens Healthineers","Abbott Diagnostics","BD","Qiagen","Bio-Rad"],"apac":["Sysmex","Mindray"]},
    "clinical_research":{"global":["IQVIA","Labcorp","ICON","Parexel","PPD","Syneos Health"],"apac":["Novotech","Tigermed"]},
    "healthtech":{"global":["Philips","Siemens Healthineers","GE HealthCare","Cerner (Oracle Health)","Epic Systems"],"apac":["HealthHub","IHiS","Ramsay Sime Darby Health Care"]},
    "technology":{"global":["Microsoft","Amazon Web Services","Google Cloud","Snowflake","Databricks"],"apac":["Tencent Cloud","Alibaba Cloud"]},
    "manufacturing":{"global":["Siemens","ABB","Rockwell Automation","Schneider Electric","Bosch"],"apac":["Mitsubishi Electric","FANUC","Yaskawa"]},
    "energy":{"global":["Shell","BP","TotalEnergies","Schneider Electric","Siemens Energy"],"apac":["PETRONAS","Sembcorp","Keppel"]},
    "gaming":{"global":["Sony Interactive Entertainment","Ubisoft","Electronic Arts","Nintendo","Activision Blizzard"],"apac":["Tencent","NetEase","Bandai Namco"]},
    "web3":{"global":["Coinbase","Consensys","Binance","Circle"],"apac":["OKX","Bybit"]},
    # New: Financial Services bucket to align with sectors.json "Financial Services > ..."
    "financial_services":{
        "global": [
            "J.P. Morgan", "Goldman Sachs", "Morgan Stanley", "BlackRock", "UBS", "Credit Suisse", "HSBC", "Citi", "BNP Paribas", "Deutsche Bank",
            "Standard Chartered", "State Street", "Northern Trust", "Schroders", "Fidelity"
        ],
        "apac": [
            "Samsung Life Insurance", "Hana Financial Investment", "Mirae Asset", "KB Asset Management", "NH Investment & Securities",
            "Korea Investment & Securities", "Shinhan Investment Corp", "Samsung Securities","Samsung Fire & Marine Insurance","Hyundai Marine & Fire Insurance",
            "DB Insurance","Meritz Fire & Marine Insurance","Tong Yang Securities","Woori Investment Bank","Daishin Securities","Hana Securities",
            "Kiwoom Securities","KTB Investment & Securities","Eugene Investment & Securities","Korea Life Insurance","LSM Investment",
            "Shinhan BNP Paribas Asset Management","Samsung SDS","LG CNS","SK C&C","POSCO ICT","Hyundai Information Technology","Hanmi Financial",
            "Nonghyup Bank","Lotte Card"
        ]
    }
}

BUCKET_JOB_TITLES = {
    "pharma_biotech":["Regulatory Affairs Manager","Clinical Research Associate","Pharmacovigilance Specialist","Medical Affairs Manager","Quality Assurance Specialist","CMC Scientist","Biostatistician","Clinical Project Manager"],
    "medical_devices":["Regulatory Affairs Manager","Quality Engineer","Clinical Affairs Specialist","Design Control Engineer","Risk Management Engineer","Product Manager (Medical Device)","Manufacturing Engineer"],
    "diagnostics":["IVD Regulatory Specialist","Quality Systems Engineer","Clinical Application Specialist","Assay Development Scientist","Validation Engineer"],
    "clinical_research":["CRA","Senior CRA","Clinical Project Manager","Clinical Trial Manager","Study Start-Up Specialist"],
    "healthtech":["Product Manager","Clinical Informatics Lead","Healthcare Data Scientist","Interoperability Engineer","Implementation Consultant"],
    "technology":["Software Engineer","ML Engineer","Data Scientist","Solutions Architect","Security Engineer","MLOps Engineer"],
    "manufacturing":["Manufacturing Engineer","Quality Engineer","Process Engineer","Supply Chain Analyst","Automation Engineer"],
    "energy":["Energy Analyst","Grid Integration Engineer","Sustainability Manager","HSE Engineer"],
    "gaming":["Game Producer","Gameplay Engineer","Level Designer","Technical Artist"],
    "web3":["Blockchain Engineer","Smart Contract Developer","Web3 Product Manager"],
    "other":["Project Manager","Operations Manager","Business Analyst","Data Analyst"],
    "financial_services":["Investment Analyst","Product Manager (Wealth/Investment)","Portfolio Manager","Risk Analyst","Payments Product Manager","Fintech Product Manager","Relationship Manager","Asset Manager"]
}

def _heuristic_multi_sector(selected, user_job_title, user_company, languages=None):
    languages = languages or []
    # Use canonical bucket mapping to map selected sector labels to BUCKET_COMPANIES keys
    buckets=[_canon_sector_bucket(x) for x in selected] or ["other"]
    per_sets=[]
    for b in buckets:
        entries=BUCKET_COMPANIES.get(b, {})
        vals=entries.get("global", [])
        if SINGAPORE_CONTEXT:
            vals=list(dict.fromkeys(entries.get("apac", []) + vals))
        per_sets.append(set(vals))
    companies=_prioritize_cross_sector(per_sets)
    jobs=[]; seen=set()
    for b in buckets:
        for t in BUCKET_JOB_TITLES.get(b, []):
            k=t.lower()
            if k not in seen:
                seen.add(k); jobs.append(t)
    if not jobs:
        jobs=BUCKET_JOB_TITLES["other"][:]
    if languages:
        for lang in languages:
            for role in [f"{lang} Translator", f"{lang} Interpreter", f"{lang} Localization", f"{lang} Linguist"]:
                if role.lower() not in seen:
                    jobs.insert(0, role); seen.add(role.lower())
    companies=_enforce_company_limit(companies, None, 20)
    return {"job":{"related":jobs[:15]}, "company":{"related":companies[:20]}}

# Ensure canon mapping includes financial keywords
def _normalize_sector_name(s: str):
    s=(s or "").strip().lower()
    rep={"pharmaceutical":"pharmaceuticals","pharma":"pharmaceuticals","biotech":"biotechnology","med device":"medical devices",
         "medical device":"medical devices","devices":"medical devices","medtech":"medical devices","diagnostic":"diagnostics",
         "health tech":"healthtech","health tech.":"healthtech","healthcare tech":"healthtech","web3":"web3 & blockchain",
         "blockchain":"web3 & blockchain","ai":"ai & data","data":"ai & data","cyber security":"cybersecurity"}
    return rep.get(s, s).replace("&amp;","&").strip()

def _canon_sector_bucket(name: str):
    s=_normalize_sector_name(name)
    if not s:
        return "other"
    # Financial mappings
    if any(k in s for k in ["financial", "finance", "bank", "banking", "insurance", "investment", "asset", "asset management", "asset-management", "fintech", "wealth"]):
        return "financial_services"
    if any(k in s for k in ["pharmaceutical","pharmaceuticals","biotech","biotechnology"]): return "pharma_biotech"
    if "medical device" in s or "medtech" in s or "devices" in s: return "medical_devices"
    if "diagnostic" in s: return "diagnostics"
    if "healthtech" in s or "health tech" in s: return "healthtech"
    if "clinical_research" in s or "clinical research" in s: return "clinical_research"
    if "software" in s or "saas" in s or "technology" in s or "ai & data" in s or "ai" in s: return "technology"
    if "cybersecurity" in s: return "cybersecurity"
    if "automotive" in s or "manufactur" in s or "industrial" in s: return "manufacturing"
    if "energy" in s or "renewable" in s: return "energy"
    if "gaming" in s: return "gaming"
    if "web3" in s or "blockchain" in s: return "web3"
    return "other"

def _bucket_to_sector_label(bucket_name: str):
    """
    Map bucket names (from BUCKET_COMPANIES) to sectors.json labels.
    Returns a sector label that should exist in SECTORS_INDEX, or None.
    """
    bucket_to_label = {
        "pharma_biotech": "Healthcare > Pharmaceuticals",
        "medical_devices": "Healthcare > Medical Devices",
        "diagnostics": "Healthcare > Diagnostics",
        "clinical_research": "Healthcare > Clinical Research",
        "healthtech": "Healthcare > HealthTech",
        "technology": "Technology",
        "manufacturing": "Industrial & Manufacturing",
        "energy": "Energy & Environment",
        "gaming": "Media, Gaming & Entertainment > Gaming",
        "web3": "Emerging & Cross-Sector > Web3 & Blockchain",
        "financial_services": "Financial Services",
        "cybersecurity": "Technology > Cybersecurity",
        "other": None
    }
    
    label = bucket_to_label.get(bucket_name)
    # Verify the label exists in SECTORS_INDEX before returning
    if label and label in SECTORS_INDEX:
        return label
    
    # If exact match not found, try to find a partial match in SECTORS_INDEX
    if label:
        label_lower = label.lower()
        for idx_label in SECTORS_INDEX:
            idx_label_lower = idx_label.lower()
            if label_lower in idx_label_lower or idx_label_lower in label_lower:
                return idx_label
    
    return None

@app.post("/suggest")
def suggest():
    data = request.get_json(force=True, silent=True) or {}
    job_titles = data.get("jobTitles") or []
    companies = data.get("companies") or []
    industry = data.get("industry") or "Non-Gaming"
    languages = data.get("languages") or []
    sectors = data.get("sectors") or data.get("selectedSectors") or []
    country = (data.get("country") or "").strip()
    key = (tuple(sorted([jt.strip().lower() for jt in job_titles])),
           tuple(sorted([c.strip().lower() for c in companies])),
           industry.lower(),
           tuple(sorted([str(x).lower() for x in languages])),
           tuple(sorted([str(x).lower() for x in sectors])),
           country.lower())
    with SUGGEST_CACHE_LOCK:
        cached=SUGGEST_CACHE.get(key)
    if cached:
        return jsonify(cached)

    user_jobs_clean = [jt.strip() for jt in job_titles if isinstance(jt, str) and jt.strip()]

    gem=_gemini_suggestions(job_titles, companies, industry, languages, sectors, country)
    if gem:
        existing_companies = {c.lower() for c in companies if isinstance(c, str) and c.strip()}
        gem_job_raw = [s for s in gem.get("job", {}).get("related", []) if isinstance(s, str) and s.strip()]
        gem_comp_raw = [s for s in gem.get("company", {}).get("related", []) if isinstance(s, str) and s.strip()]
        gem_job_filtered = [s for s in gem_job_raw if not any(s.strip().lower() == uj.lower() for uj in user_jobs_clean)]
        gem_comp_filtered = [s for s in gem_comp_raw if s.strip().lower() not in existing_companies]
        combined_jobs = list(gem_job_filtered)
        for uj in reversed(user_jobs_clean):
            if not any(uj.lower() == existing.lower() for existing in combined_jobs):
                combined_jobs.insert(0, uj)
        final_job_list = _clean_list(combined_jobs, MAX_SUGGESTIONS_PER_TAG)[:MAX_SUGGESTIONS_PER_TAG]
        final_company_list = gem_comp_filtered[:COMPANY_SUGGESTIONS_LIMIT]
        payload = {
            "job": {"related": final_job_list},
            "company": {"related": final_company_list},
            "engine": "gemini"
        }
    else:
        heuristic_jobs = _heuristic_job_suggestions(job_titles, industry, languages, sectors) or []
        heuristic_companies = _heuristic_company_suggestions(companies, languages, sectors, country) or []
        combined_jobs = list(heuristic_jobs)
        for uj in reversed(user_jobs_clean):
            if not any(uj.lower() == existing.lower() for existing in combined_jobs):
                combined_jobs.insert(0, uj)
        final_job_list = _clean_list(combined_jobs, MAX_SUGGESTIONS_PER_TAG)[:MAX_SUGGESTIONS_PER_TAG]
        final_company_list = heuristic_companies[:COMPANY_SUGGESTIONS_LIMIT]
        payload = {
            "job": {"related": final_job_list},
            "company": {"related": final_company_list},
            "engine": "heuristic"
        }

    with SUGGEST_CACHE_LOCK:
        SUGGEST_CACHE[key]=payload
    return jsonify(payload)

@app.post("/sector_suggest")
def sector_suggest():
    data = request.get_json(force=True, silent=True) or {}
    sectors_list = data.get("selectedSectors") or ([data.get("selectedSector")] if data.get("selectedSector") else [])
    sectors_list=[s for s in sectors_list if isinstance(s,str) and s.strip()]
    user_company=(data.get("userCompany") or "").strip()
    user_job_title=(data.get("userJobTitle") or "").strip()
    languages = data.get("languages") or []
    if not sectors_list and not user_company and not user_job_title and not languages:
        return jsonify({"job":{"related":[]}, "company":{"related":[]}}), 200
    normalized=[]
    for s in sectors_list:
        parts=[p.strip() for p in re.split(r'>', s) if p.strip()]
        normalized.append(parts[-1] if parts else s)
    normalized=[n for n in normalized if n]
    gem=_gemini_multi_sector(normalized, user_job_title, user_company, languages)
    if gem and (gem.get("job",{}).get("related") or gem.get("company",{}).get("related")):
        return jsonify(gem), 200
    return jsonify(_heuristic_multi_sector(normalized, user_job_title, user_company, languages)), 200

JOBS = {}
JOBS_LOCK = threading.Lock()
PERSIST_JOBS_TO_FILES = os.getenv("PERSIST_JOBS_TO_FILES", "1") == "1"
JOB_FILE_PREFIX="job_"; JOB_FILE_SUFFIX=".json"
def _job_file(job_id: str): return os.path.join(OUTPUT_DIR, f"{JOB_FILE_PREFIX}{job_id}{JOB_FILE_SUFFIX}")
def persist_job(job_id: str):
    if not PERSIST_JOBS_TO_FILES: return
    try:
        with JOBS_LOCK:
            job=JOBS.get(job_id)
            if not job: return
            tmp=_job_file(job_id)+".tmp"
            with open(tmp,"w",encoding="utf-8") as f: json.dump(job,f,ensure_ascii=False,indent=2)
            os.replace(tmp,_job_file(job_id))
    except Exception as e:
        logger.warning(f"[Persist] {e}")
def add_message(job_id: str, text: str):
    with JOBS_LOCK:
        job=JOBS.get(job_id)
        if not job: return
        job['messages'].append(text)
        job['status_html']="<br>".join(job['messages'][-12:])
    persist_job(job_id)

# ... [Job helper functions] ...
LINKEDIN_PROFILE_RE = re.compile(r'(?:^|\.)linkedin\.com/(?:in|pub)/', re.I)
CLEAN_LINKEDIN_SUFFIX_RE = re.compile(r'\s*\|\s*LinkedIn.*$', re.I)
MULTI_SPACE_RE = re.compile(r'\s+')

def is_linkedin_profile(url: str) -> bool:
    return bool(url and LINKEDIN_PROFILE_RE.search(url))

def parse_linkedin_title(title: str):
    if not title: return None, None, None
    cleaned=CLEAN_LINKEDIN_SUFFIX_RE.sub('', title).strip()
    cleaned=cleaned.replace('–','-').replace('—','-')
    if '-' not in cleaned: return None, None, None
    name_part, rest = cleaned.split('-', 1)
    name=name_part.strip()
    if len(name.split())>9 or len(name)<2: return None, None, None
    if not re.search(r'[A-Za-z]', name): return None, None, None
    rest=rest.strip()
    company=""; jobtitle=rest
    at_idx=rest.lower().find(" at ")
    if at_idx!=-1:
        jobtitle=rest[:at_idx].strip()
        company=rest[at_idx+4:].strip()
    name=MULTI_SPACE_RE.sub(' ',name)
    jobtitle=MULTI_SPACE_RE.sub(' ',jobtitle)
    company=MULTI_SPACE_RE.sub(' ',company)
    return name or None, jobtitle or None, company or None

def google_cse_search_page(query: str, api_key: str, cx: str, num: int, start_index: int, gl_hint: str = None):
    if not api_key or not cx: return []
    endpoint="https://www.googleapis.com/customsearch/v1"
    params={"key":api_key,"cx":cx,"q":query,"num":min(num,10),"start":start_index}
    if gl_hint: params["gl"]=gl_hint
    try:
        r=requests.get(endpoint, params=params, timeout=30)
        r.raise_for_status()
        data=r.json()
        items=data.get("items",[]) or []
        out=[]
        for it in items:
            out.append({"link":it.get("link") or "","title":it.get("title") or "","snippet":it.get("snippet") or "","displayLink":it.get("displayLink") or ""})
        return out
    except Exception as e:
        logger.warning(f"[CSE] page fetch failed: {e}")
        return []

def get_linkedin_profile_picture(linkedin_url: str):
    """
    Retrieve LinkedIn profile picture URL using scraping and Google Custom Search Image API.
    Returns profile picture URL or None if not found.
    
    Priority:
    1. Try to fetch og:image meta tag from LinkedIn profile (most reliable for profile picture)
    2. Fall back to Google Custom Search for LinkedIn images
    3. Return None if no valid image found
    
    Security Note: Validates LinkedIn URLs to prevent SSRF attacks.
    """
    if not linkedin_url:
        return None
    
    # SECURITY: Validate LinkedIn URL to prevent SSRF
    # Must be a valid LinkedIn profile URL
    if not re.match(r'^https?://([a-z]+\.)?linkedin\.com/in/[a-zA-Z0-9\-]+/?$', linkedin_url, re.IGNORECASE):
        logger.warning(f"[Profile Pic] Invalid LinkedIn URL format: {linkedin_url}")
        return None
    
    profile_pic_url = None
    
    # Method 1: Try to fetch og:image meta tag directly from LinkedIn profile
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(linkedin_url, headers=headers, timeout=10)
        
        # Handle rate limiting and forbidden responses
        if response.status_code == 429:
            logger.warning(f"[Profile Pic] Rate limited by LinkedIn: {linkedin_url}")
            # Continue to fallback method
        elif response.status_code == 403:
            logger.warning(f"[Profile Pic] Forbidden by LinkedIn (may require auth): {linkedin_url}")
            # Continue to fallback method
        elif response.status_code == 200:
            # Parse HTML to find og:image meta tag
            # Note: LinkedIn may actively block scraping - this is best-effort
            og_image_match = re.search(r'<meta[^>]*property=["\']og:image["\'][^>]*content=["\']([^"\']+)["\']', response.text, re.IGNORECASE)
            if not og_image_match:
                # Try reverse order (content before property)
                og_image_match = re.search(r'<meta[^>]*content=["\']([^"\']+)["\'][^>]*property=["\']og:image["\']', response.text, re.IGNORECASE)
            
            if og_image_match:
                profile_pic_url = og_image_match.group(1)
                logger.info(f"[Profile Pic] Found og:image from LinkedIn profile: {profile_pic_url}")
                
                # Validate that it's not a placeholder or default image
                if profile_pic_url and not any(placeholder in profile_pic_url.lower() for placeholder in ['default', 'placeholder', 'ghost']):
                    return profile_pic_url
                else:
                    logger.info(f"[Profile Pic] og:image appears to be placeholder, trying fallback")
    except Exception as e:
        logger.warning(f"[Profile Pic] Failed to fetch og:image from LinkedIn (may be blocked): {e}")
    
    # Method 2: Fall back to Google Custom Search if og:image failed or not configured
    if not profile_pic_url and GOOGLE_CSE_API_KEY and GOOGLE_CSE_CX:
        try:
            # Extract profile name from LinkedIn URL
            # e.g., https://www.linkedin.com/in/john-doe -> john-doe
            match = re.search(r'linkedin\.com/in/([^/?]+)', linkedin_url)
            if not match:
                logger.warning(f"[Profile Pic] Could not extract profile name from URL: {linkedin_url}")
                return None
            
            profile_name = match.group(1)
            
            # Use Google Custom Search with image search targeting LinkedIn
            endpoint = "https://www.googleapis.com/customsearch/v1"
            params = {
                "key": GOOGLE_CSE_API_KEY,
                "cx": GOOGLE_CSE_CX,
                "q": f"site:linkedin.com/in/{profile_name} profile picture",
                "searchType": "image",
                "num": 3  # Get top 3 to have fallback options
            }
            
            response = requests.get(endpoint, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            
            items = data.get("items", [])
            if items:
                # Try each image in order, skip obvious non-profile images
                for item in items:
                    image_url = item.get("link")
                    image_context = item.get("image", {}).get("contextLink", "")
                    
                    # SECURITY: Validate that context link is actually from LinkedIn domain
                    # Check that linkedin.com is the domain, not just anywhere in the URL
                    if image_url and image_context:
                        # Parse and validate the domain
                        try:
                            from urllib.parse import urlparse
                            parsed = urlparse(image_context)
                            # Check that the domain ends with linkedin.com (handles www.linkedin.com, etc.)
                            if not parsed.netloc.endswith('linkedin.com'):
                                continue
                        except Exception:
                            continue
                        
                        # Skip banner/background images (usually larger dimensions)
                        width = item.get("image", {}).get("width", 0)
                        height = item.get("image", {}).get("height", 0)
                        
                        # Profile pictures are usually square or near-square, and not too large
                        if width and height:
                            aspect_ratio = width / height if height > 0 else 0
                            if 0.8 <= aspect_ratio <= 1.2 and width < 1000:  # Square-ish and reasonable size
                                profile_pic_url = image_url
                                logger.info(f"[Profile Pic] Found suitable image via CSE for {profile_name}: {image_url}")
                                break
                
                # If no suitable image found with size validation, use first result
                if not profile_pic_url and items:
                    profile_pic_url = items[0].get("link")
                    logger.info(f"[Profile Pic] Using first CSE result for {profile_name}: {profile_pic_url}")
            else:
                logger.info(f"[Profile Pic] No images found via CSE for {profile_name}")
        except Exception as e:
            logger.warning(f"[Profile Pic] Google CSE failed: {e}")
    
    # Final validation: ensure URL is not empty or broken
    if profile_pic_url:
        try:
            # Quick HEAD request to verify image exists
            head_response = requests.head(profile_pic_url, timeout=5)
            if head_response.status_code == 200:
                return profile_pic_url
            else:
                logger.warning(f"[Profile Pic] Image URL returned status {head_response.status_code}: {profile_pic_url}")
                return None
        except Exception as e:
            logger.warning(f"[Profile Pic] Failed to validate image URL: {e}")
            return None
    
    return None

def fetch_image_bytes_from_url(image_url: str, max_size_mb=5):
    """
    Fetch image bytes from a URL and return as bytes suitable for bytea storage.
    Returns bytes or None if fetch failed or image too large.
    
    Args:
        image_url: The URL of the image to fetch
        max_size_mb: Maximum allowed image size in MB (default: 5MB)
    """
    if not image_url:
        return None
    
    try:
        response = requests.get(image_url, timeout=15, stream=True)
        response.raise_for_status()
        
        # Check content type
        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('image/'):
            logger.warning(f"[Fetch Image Bytes] Invalid content type: {content_type}")
            return None
        
        # Check content length
        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > max_size_mb * 1024 * 1024:
            logger.warning(f"[Fetch Image Bytes] Image too large: {content_length} bytes")
            return None
        
        # Read the image data
        image_bytes = response.content
        
        # Verify size after download
        if len(image_bytes) > max_size_mb * 1024 * 1024:
            logger.warning(f"[Fetch Image Bytes] Downloaded image too large: {len(image_bytes)} bytes")
            return None
        
        logger.info(f"[Fetch Image Bytes] Successfully fetched {len(image_bytes)} bytes from {image_url}")
        return image_bytes
        
    except Exception as e:
        logger.warning(f"[Fetch Image Bytes] Failed to fetch image from {image_url}: {e}")
        return None

def _dedupe_links(records):
    seen=set(); out=[]
    for r in records:
        link=r.get("link")
        if not link or link in seen: continue
        seen.add(link); out.append(r)
    return out

def _infer_primary_job_title(job_titles):
    if job_titles and isinstance(job_titles,list) and job_titles:
        return job_titles[0]
    return ""

def _perform_cse_queries(job_id, queries, target_limit, country):
    results=[]
    # Assuming queries are distributed evenly
    m_cc=re.search(r'site:([a-z]{2})\.linkedin\.com/in', " ".join(queries), re.I)
    country_code_hint = m_cc.group(1).lower() if m_cc else None
    
    # Calculate a simple target per query
    total_queries = max(1, len(queries))
    per_query_target = max(1, target_limit // total_queries)

    for q in queries:
        gathered=0; start_index=1; pages_fetched=0
        add_message(job_id, f"Running CSE: {q} target={per_query_target}")
        
        while gathered < per_query_target:
            remaining = per_query_target - gathered
            page_size = min(CSE_PAGE_SIZE, remaining)
            
            page = google_cse_search_page(q, GOOGLE_CSE_API_KEY, GOOGLE_CSE_CX, page_size, start_index, gl_hint=country_code_hint)
            pages_fetched+=1
            
            if not page:
                add_message(job_id, f"  No results page start={start_index}")
                break
            
            results.extend(page); gathered+=len(page)
            if len(page) < page_size: break
            start_index += len(page)
            
            # Simple safety break
            if pages_fetched >= 20: break 
            time.sleep(CSE_PAGE_DELAY)
            
        add_message(job_id, f"CSE done (collected {gathered}). pages={pages_fetched}")
    return _dedupe_links(results)

def _infer_seniority_from_titles(job_titles):
    if not job_titles: return None
    joined=" ".join([t or "" for t in job_titles])
    if re.search(r"\bAssociate\b", joined, flags=re.I): return "Associate"
    if re.search(r"\bManager\b", joined, flags=re.I): return "Manager"
    if re.search(r"\bDirector\b", joined, flags=re.I): return "Director"
    return None

_SPECIALS = "<>àÀáÁâÂãÃäÄåÅæÆçÇèÈéÉêÊëËìÌíÍîÎïÏðÐñÑòÒóÓôÔõÖøØùÙúÚûÛüÜýÝÿŸšŠžŽłŁßþÞœŒ~"
_SPECIALS_RE = re.compile("[" + re.escape(_SPECIALS) + "]")

def _sanitize_for_excel(val: str) -> str:
    if not isinstance(val, str): return val or ""
    try:
        import unicodedata
        s=unicodedata.normalize("NFKC", val)
    except Exception:
        s=val
    s=(s.replace("–","-").replace("—","-").replace("’","'").replace("‘","'").replace("“",'"').replace("”",'"'))
    s=_SPECIALS_RE.sub("", s)
    try:
        import unicodedata
        s="".join(ch for ch in s if unicodedata.category(ch)[0]!="C" and ch not in {"\u200b","\u200c","\u200d","\ufeff"})
    except Exception:
        s=s.replace("\u200b","").replace("\u200c","").replace("\u200d","").replace("\ufeff","")
    s=re.sub(r"\s+"," ",s).strip()
    if len(s)>512: s=s[:512]
    return s

def _aggregate_company_dropdown(meta):
    if not isinstance(meta, dict):
        return []
    user = meta.get('user_companies') or []
    auto = meta.get('auto_suggest_companies') or []
    sectors = meta.get('selected_sectors') or []
    languages = meta.get('languages') or []
    sector_companies=[]
    try:
        if sectors:
            norm=[]
            for s in sectors:
                if not isinstance(s,str): continue
                parts=[p.strip() for p in re.split(r'>', s) if p.strip()]
                norm.append(parts[-1] if parts else s)
            sector_payload=_heuristic_multi_sector(norm,"","",languages)
            sector_companies = sector_payload.get('company',{}).get('related',[]) if sector_payload else []
    except Exception as e:
        logger.warning(f"[Dropdown] Sector heuristic failed: {e}")
    merged=[]; seen=set()
    for source in (user, auto, sector_companies):
        for c in source:
            if not isinstance(c,str): continue
            t=c.strip()
            if not t: continue
            k=t.lower()
            if k in seen: continue
            seen.add(k); merged.append(t)
            if len(merged)>=200: break
        if len(merged)>=200: break
    return merged

def _extract_company_from_jobtitle(job_title_raw: str, existing_company: str, company_list):
    if not job_title_raw or existing_company:
        return existing_company, job_title_raw
    seps=r"[\s\-\|,/@]"
    candidates=sorted([c for c in (company_list or []) if isinstance(c,str) and c.strip()], key=lambda x: len(x), reverse=True)
    for comp in candidates:
        pat=re.compile(rf"(^|{seps}+)" rf"({re.escape(comp)})" rf"({seps}+|$)", re.IGNORECASE)
        m=pat.search(job_title_raw)
        if not m: continue
        start_company,end_company=m.span(2)
        cleaned=job_title_raw[:start_company]+job_title_raw[end_company:]
        cleaned=re.sub(rf"({seps}+)", " ", cleaned).strip(" -|,/@").strip()
        return comp, cleaned
    return existing_company, job_title_raw

def _gemini_extract_company_from_jobtitle(job_title_raw: str, candidates=None):
    if not job_title_raw or not (GEMINI_API_KEY and genai): return None, job_title_raw
    try:
        model=genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
        context={"jobTitle": job_title_raw.strip(), "knownCandidates": (candidates or [])[:30]}
        prompt=("Extract inline employer/company from jobTitle strictly if present. "
                "Return JSON {\"company\":\"\",\"jobTitleWithoutCompany\":\"\"}. "
                f"INPUT:\n{json.dumps(context,ensure_ascii=False)}\nOUTPUT:")
        resp=model.generate_content(prompt)
        text=(resp.text or "").strip()
        start=text.find('{'); end=text.rfind('}')
        if start==-1 or end==-1 or end<=start: return None, job_title_raw
        obj=json.loads(text[start:end+1])
        company=(obj.get("company") or "").strip()
        jt_wo=(obj.get("jobTitleWithoutCompany") or "").strip()
        if not company: return None, job_title_raw
        if not jt_wo: jt_wo=job_title_raw
        return company, jt_wo
    except Exception as e:
        logger.warning(f"[Gemini Title->Company] {e}")
        return None, job_title_raw

def _job_runner(job_id, queries, fallback_queries, auto_expand, manual_urls, search_results_only, country, dynamic_target, job_titles):
    add_message(job_id, "Starting search pipeline...")
    rows=[]; urls=[]
    target_limit=dynamic_target if (isinstance(dynamic_target,int) and dynamic_target>0) else SEARCH_RESULTS_TARGET
    primary_job_title=_infer_primary_job_title(job_titles)
    try:
        executed_primary=False
        if (search_results_only or auto_expand):
            if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_CX:
                add_message(job_id, "ERROR: GOOGLE_CSE_API_KEY/CX not set. Cannot run search.")
            else:
                executed_primary=True
                cse_results=_perform_cse_queries(job_id, queries or ["site:linkedin.com/in"], target_limit, country)
                urls=[r["link"] for r in cse_results]
                with JOBS_LOCK:
                    JOBS[job_id]['urls']=urls
                    JOBS[job_id]['progress']['total']=len(urls)
                persist_job(job_id)
                processed=0
                for item in cse_results:
                    link=item.get("link","") or ""
                    title=item.get("title") or ""
                    domain_part=re.sub(r'^https?://','',link+'/').split('/')[0].lower()
                    if is_linkedin_profile(link):
                        name, jobtitle, company = parse_linkedin_title(title)
                        if name or jobtitle or company:
                            rows.append({"Name":name or "", "Company":company or "", "JobTitle":jobtitle or "", "Country":country or "", "LinkedInURL":link})
                    processed+=1
                    with JOBS_LOCK:
                        JOBS[job_id]['progress']['processed']=processed
                    if processed % 15 == 0: persist_job(job_id)
        if executed_primary and len(rows)==0 and fallback_queries:
            add_message(job_id,"Primary produced zero rows. Executing fallback queries...")
            cse_results=_perform_cse_queries(job_id, fallback_queries, target_limit, country)
            new_urls=[r["link"] for r in cse_results]
            with JOBS_LOCK:
                JOBS[job_id]['urls']=list(dict.fromkeys(JOBS[job_id]['urls']+new_urls))
                JOBS[job_id]['progress']['total']=len(JOBS[job_id]['urls'])
            persist_job(job_id)
            processed=0
            for item in cse_results:
                link=item.get("link"); title=item.get("title")
                domain_part=re.sub(r'^https?://','',link).split('/')[0].lower()
                if is_linkedin_profile(link):
                    name, jobtitle, company = parse_linkedin_title(title)
                    if name or jobtitle or company:
                        rows.append({"Name":name or "", "Company":company or "", "JobTitle":jobtitle or "", "Country":country or "", "LinkedInURL":link})
                processed+=1
                with JOBS_LOCK:
                    JOBS[job_id]['progress']['processed']=processed
                if processed % 15 == 0: persist_job(job_id)
        if manual_urls and not (search_results_only or auto_expand):
            add_message(job_id, f"Processing manual URLs: {len(manual_urls)}")
            for u in manual_urls:
                domain_part=re.sub(r'^https?://','',u).split('/')[0].lower()
                if is_linkedin_profile(u):
                    rows.append({"Name":"","Company":"","JobTitle":primary_job_title,"Country":country or "","LinkedInURL":u})
            with JOBS_LOCK:
                JOBS[job_id]['urls']=list(dict.fromkeys(JOBS[job_id]['urls']+manual_urls))
                JOBS[job_id]['progress']['total']=len(JOBS[job_id]['urls'])
                JOBS[job_id]['progress']['processed']=len(JOBS[job_id]['urls'])
            persist_job(job_id)
        dedup=[]; seen=set()
        for r in rows:
            key=(r.get("LinkedInURL",""), r.get("Name","").lower(), r.get("JobTitle","").lower())
            if key in seen: continue
            seen.add(key); dedup.append(r)
        rows=dedup
        with JOBS_LOCK:
            meta=(JOBS.get(job_id) or {}).get('meta',{})
        seniority_effective=meta.get('seniority') or _infer_seniority_from_titles(job_titles)
        if seniority_effective:
            before=len(rows)
            srules=(SEARCH_RULES or {}).get("seniority_rules") or {}
            excl_rule=None
            for k in srules.keys():
                if str(k).strip().lower()==seniority_effective.lower():
                    excl_rule=srules.get(k); break
            if excl_rule and isinstance(excl_rule,dict):
                tokens_raw=excl_rule.get("xrayExclusion","")
                tokens=[]
                m=re.search(r"\((.*?)\)", tokens_raw)
                if m:
                    tokens=[t.strip().strip('"\'') for t in m.group(1).split("OR")]
                elif tokens_raw:
                    m2=re.search(r"-\s*([A-Za-z ]+)", tokens_raw)
                    if m2: tokens=[m2.group(1).strip()]
                filtered=[]
                lowers=[t.lower() for t in tokens if t]
                if lowers:
                    for row in rows:
                        jt=(row.get("JobTitle") or "").lower()
                        if not any(x in jt for x in lowers):
                            filtered.append(row)
                    rows=filtered
            after=len(rows)
            add_message(job_id, f"Seniority filter '{seniority_effective}' applied: kept {after}/{before} rows.")
        csv_name,xlsx_name=_write_outputs(job_id, rows)
        # CSV/XLSX both saved to SEARCH_XLS_DIR now
        csv_ok=bool(csv_name) and os.path.exists(os.path.join(SEARCH_XLS_DIR, csv_name))
        xlsx_ok=bool(xlsx_name) and os.path.exists(os.path.join(SEARCH_XLS_DIR, xlsx_name))
        with JOBS_LOCK:
            job=JOBS.get(job_id)
            if job:
                job['output_csv']=csv_name
                job['output_xlsx']=xlsx_name
                job['done']=bool(csv_ok or xlsx_ok)
                job['messages'].append(f"Job complete. Wrote {len(rows)} rows.")
                job['status_html']="<br>".join(job['messages'][-12:])
        persist_job(job_id)
    except Exception as e:
        add_message(job_id, f"Pipeline error: {e}")
        with JOBS_LOCK:
            job=JOBS.get(job_id)
            if job:
                job['done']=False
                job['status_html']="<br>".join(job['messages'][-12:])
        persist_job(job_id)

def _write_outputs(job_id, rows):
    with JOBS_LOCK:
        job_meta=(JOBS.get(job_id) or {}).get('meta',{})
        job_top=(JOBS.get(job_id) or {})
    dropdown_companies=_aggregate_company_dropdown(job_meta)
    processed=[]
    for r in rows:
        link=r.get("LinkedInURL","")
        country_val=r.get("Country","")
        if not is_linkedin_profile(link): country_val=""
        raw_name=r.get("Name",""); raw_company=r.get("Company",""); raw_job=r.get("JobTitle","")
        moved_company, adjusted_job=_extract_company_from_jobtitle(raw_job, raw_company, dropdown_companies)
        if not moved_company:
            g_company,g_job=_gemini_extract_company_from_jobtitle(raw_job, dropdown_companies)
            if g_company:
                moved_company=g_company.strip()
                adjusted_job=(g_job or raw_job).strip()
        name_val=_sanitize_for_excel(raw_name)
        job_val=_sanitize_for_excel(adjusted_job)
        company_val=(moved_company or "").strip()
        processed.append({"Name":name_val,"Company":company_val,"JobTitle":job_val,"Country":country_val,"LinkedInURL":link})
    csv_name=f"{job_id}_results.csv"
    csv_path=os.path.join(SEARCH_XLS_DIR, csv_name)
    # Ensure target dir exists
    os.makedirs(SEARCH_XLS_DIR, exist_ok=True)
    with open(csv_path,"w",encoding="utf-8",newline="") as f:
        w=DictWriter(f, fieldnames=["Name","Company","JobTitle","Country","LinkedInURL"])
        w.writeheader()
        for pr in processed: w.writerow(pr)
        try:
            f.flush(); os.fsync(f.fileno())
        except Exception:
            pass
    xlsx_name=None
    try:
        import openpyxl
        from openpyxl import Workbook
        from openpyxl.worksheet.datavalidation import DataValidation
        wb=Workbook(); ws=wb.active
        ws.title="Results"
        ws.append(["Name","Company","JobTitle","Country","LinkedInURL"])
        for pr in processed: ws.append([pr["Name"],pr["Company"],pr["JobTitle"],pr["Country"],pr["LinkedInURL"]])
        company_dropdown=dropdown_companies
        if company_dropdown:
            cs=wb.create_sheet(title="Companies")
            cs.append(["Companies"])
            for c in company_dropdown: cs.append([c])
            last_row_comp=len(company_dropdown)+1
            dv=DataValidation(type="list", formula1=f"=Companies!$A$2:$A${last_row_comp}", allow_blank=True,
                              showErrorMessage=True, error="Select a company from the dropdown list.", errorTitle="Invalid Company")
            ws.add_data_validation(dv)
            last_results_row=ws.max_row
            dv.add(f"B2:B{last_results_row}")
        xlsx_name=f"{job_id}_results.xlsx"
        # Save to SEARCH_XLS_DIR
        xlsx_full=os.path.join(SEARCH_XLS_DIR, xlsx_name)
        wb.save(xlsx_full)
        for _ in range(20):
            try:
                if os.path.exists(xlsx_full) and os.path.getsize(xlsx_full)>0:
                    break
            except Exception:
                pass
            time.sleep(0.05)
        try:
            import psycopg2
            from psycopg2 import sql
            excel_path=xlsx_full
            wb_ing=openpyxl.load_workbook(excel_path, read_only=True, data_only=True)
            sheet=wb_ing["Results"]
            headers=[cell.value for cell in next(sheet.iter_rows(min_row=1, max_row=1))]
            expected=["Name","Company","JobTitle","Country","LinkedInURL"]
            if headers!=expected:
                logger.warning(f"[Ingest] Header mismatch. Expected {expected}, got {headers}. Skipping ingestion.")
            else:
                data_rows=[]
                for row in sheet.iter_rows(min_row=2, values_only=True):
                    if all((val is None or str(val).strip()=="" ) for val in row): continue
                    data_rows.append(row)
                if data_rows:
                    pg_host=os.getenv("PGHOST","localhost")
                    pg_port=int(os.getenv("PGPORT","5432"))
                    pg_user=os.getenv("PGUSER","postgres")
                    pg_password=os.getenv("PGPASSWORD","") or "orlha"
                    pg_db=os.getenv("PGDATABASE","candidate_db")
                    logger.info(f"[Ingest] Connecting to Postgres host={pg_host} port={pg_port} db={pg_db} user={pg_user}")
                    conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
                    conn.autocommit=False
                    cur=conn.cursor()
                    
                    # Check if pic column exists in sourcing table
                    cur.execute("""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_schema='public' AND table_name='sourcing' AND column_name='pic'
                    """)
                    has_pic_column = bool(cur.fetchone())
                    
                    active_userid=(job_meta.get('userid') or job_top.get('userid') or '').strip()
                    active_username=(job_meta.get('username') or job_top.get('username') or '').strip()
                    
                    if has_pic_column:
                        # Include pic column in insert
                        insert_stmt=sql.SQL("INSERT INTO sourcing (userid, username, name, company, jobtitle, country, linkedinurl, pic) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING")
                        batch_rows = []
                        for r in data_rows:
                            # r is a tuple: (name, company, jobtitle, country, linkedinurl)
                            # LinkedInURL is at index 4 (0-based indexing, 5th column)
                            linkedin_url = r[4]
                            # Retrieve profile picture and convert to bytea
                            pic_bytes = None
                            try:
                                pic_url = get_linkedin_profile_picture(linkedin_url) if linkedin_url else None
                                if pic_url:
                                    pic_bytes = fetch_image_bytes_from_url(pic_url)
                                    if pic_bytes:
                                        import psycopg2
                                        pic_bytes = psycopg2.Binary(pic_bytes)
                            except Exception as pic_err:
                                logger.warning(f"[Ingest] Failed to get profile pic for {linkedin_url}: {pic_err}")
                            batch_rows.append((active_userid, active_username, r[0], r[1], r[2], r[3], linkedin_url, pic_bytes))
                    else:
                        # No pic column, use original insert
                        insert_stmt=sql.SQL("INSERT INTO sourcing (userid, username, name, company, jobtitle, country, linkedinurl) VALUES (%s, %s, %s, %s, %s, %s, %s) ON CONFLICT DO NOTHING")
                        batch_rows=[(active_userid, active_username, r[0], r[1], r[2], r[3], r[4]) for r in data_rows]
                    
                    batch_size=500
                    total_inserted=0
                    for i in range(0,len(batch_rows),batch_size):
                        batch=batch_rows[i:i+batch_size]
                        cur.executemany(insert_stmt, batch)
                        total_inserted+=len(batch)
                    conn.commit()
                    cur.close(); conn.close()
                    logger.info(f"[Ingest] Inserted {total_inserted} rows into sourcing (userid='{active_userid}' username='{active_username}').")
                else:
                    logger.info(f"[Ingest] No data rows to insert from {xlsx_name}.")
        except Exception as e:
            logger.warning(f"[Ingest] PostgreSQL ingestion failed: {e}")
    except Exception as e:
        logger.warning(f"[Excel Dropdown] XLSX generation failed: {e}")
    return csv_name, xlsx_name

def _gemini_multi_sector(selected, user_job_title, user_company, languages=None):
    if not (GEMINI_API_KEY and genai): return None
    languages = languages or []
    region_hint="SG/APAC" if SINGAPORE_CONTEXT else None
    input_obj={"selectedSectors": selected, "userJobTitle": user_job_title or None, "userCompany": user_company or None, "languages": languages, "regionHint": region_hint}
    prompt=("SYSTEM:\nYou are a sourcing assistant. integrated into an application. Generate concise suggestions.\n"
            "Return ONLY JSON: {\"job\":{\"related\":[...]},\"company\":{\"related\":[...]}}\n"
            f"- Provide EXACTLY 15 real job titles (or fill with closest relevant if fewer) in job.related.\n"
            f"- Provide EXACTLY {COMPANY_SUGGESTIONS_LIMIT} real company names (brand-level) in company.related.\n"
            "- NO generic placeholders (e.g., 'Tech Company', 'Gaming Studio').\n"
            "- NO commentary or extra keys.\n"
            f"INPUT:\n{json.dumps(input_obj,ensure_ascii=False)}\nJSON:")
    try:
        model=genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
        resp=model.generate_content(prompt)
        parsed=_extract_json_object(resp.text or "")
        if not isinstance(parsed, dict): return None
        job=parsed.get("job",{}) if isinstance(parsed.get("job"),dict) else {}
        comp=parsed.get("company",{}) if isinstance(parsed.get("company"),dict) else {}
        jr=job.get("related") if isinstance(job.get("related"),list) else []
        cr=comp.get("related") if isinstance(comp.get("related"),list) else []
        jr_clean=_clean_list([s for s in jr if isinstance(s,str)], 15)
        if len(jr_clean) < 15:
            extra=_heuristic_job_suggestions(jr_clean, "Non-Gaming", languages, selected) or []
            for e in extra:
                if e not in jr_clean and len(jr_clean) < 15:
                    jr_clean.append(e)
        cr_enforced=_enforce_company_limit(cr, None, COMPANY_SUGGESTIONS_LIMIT)
        return {"job":{"related":jr_clean[:15]}, "company":{"related":cr_enforced[:COMPANY_SUGGESTIONS_LIMIT]}}
    except Exception as e:
        logger.warning(f"[Gemini Multi-Sector] {e}")
    return None

@app.post("/start_job")
def start_job():
    data=request.get_json(force=True, silent=True) or {}
    queries=data.get('queries') or []
    fallback_queries=data.get('fallbackQueries') or []
    auto_expand=bool(data.get('autoExpand'))
    manual_urls=data.get('manualUrls') or []
    search_results_only=bool(data.get('searchResultsOnly'))
    country=(data.get("country") or "").strip()
    languages=data.get("languages") or []
    language_query=(data.get("languageQuery") or "").strip()
    auto_suggest_companies=data.get("autoSuggestedCompanyNames") or []
    user_companies=data.get("companyNames") or []
    job_titles=data.get("jobTitles") or []
    current_role=bool(data.get("currentRole"))
    selected_sectors=data.get("selectedSectors") or data.get("sectors") or []
    seniority=(data.get("seniority") or "").strip()
    deep_mode=bool(data.get("deepMode"))
    xray_platform_queries=data.get("xrayPlatformQueries") or []
    channel_gaming=bool(data.get("channelGaming"))
    channel_media=bool(data.get("channelMedia"))
    channel_technology=bool(data.get("channelTechnology"))
    if deep_mode and xray_platform_queries:
        for q in xray_platform_queries:
            if q not in queries:
                queries.append(q)
    channel_count=int(channel_gaming)+int(channel_media)+int(channel_technology)
    platform_count=len(xray_platform_queries)
    
    # Check if user provided an explicit target limit
    user_target_raw = data.get("userTarget")
    dynamic_target = 0
    if user_target_raw is not None:
        try:
            dynamic_target = int(user_target_raw)
        except Exception:
            dynamic_target = 0
    
    if dynamic_target <= 0:
        dynamic_target=_compute_search_target(job_titles, country, user_companies, auto_suggest_companies,
                                              selected_sectors, languages, current_role, seniority or None,
                                              channel_count, platform_count)
                                          
    job_id=uuid.uuid4().hex[:10]
    userid=(data.get('userid') or '').strip()
    username=(data.get('username') or '').strip()

    # --- PATCH START: Automatically update role_tag in login table based on job_titles ---
    # The requirement is that autosourcing.html search title must pass automatically to login.role_tag.
    # While frontend calls updateRoleTagOnServer, doing it here guarantees it syncs with the actual search.
    try:
        if username and job_titles:
            # Construct the tag string, same logic as frontend: joined by commas
            # job_titles is a list of strings
            role_tag_val = ", ".join([str(t).strip() for t in job_titles if t]).strip()
            if role_tag_val:
                import psycopg2
                pg_host_l = os.getenv("PGHOST", "localhost")
                pg_port_l = int(os.getenv("PGPORT", "5432"))
                pg_user_l = os.getenv("PGUSER", "postgres")
                pg_password_l = os.getenv("PGPASSWORD", "") or "orlha"
                pg_db_l = os.getenv("PGDATABASE", "candidate_db")
                conn_l = psycopg2.connect(host=pg_host_l, port=pg_port_l, user=pg_user_l, password=pg_password_l, dbname=pg_db_l)
                cur_l = conn_l.cursor()
                # Update login table
                cur_l.execute("UPDATE login SET role_tag=%s WHERE username=%s", (role_tag_val, username))
                conn_l.commit()
                cur_l.close()
                conn_l.close()
                logger.info(f"[StartJob Auto-Update] Set role_tag='{role_tag_val}' for user='{username}'")
    except Exception as e_rt:
        logger.warning(f"[StartJob Auto-Update role_tag] Failed: {e_rt}")
    # --- PATCH END ---

    with JOBS_LOCK:
        JOBS[job_id]={
            'status_html':'Job created. Initializing...',
            'done':False,
            'output_csv':None,
            'output_xlsx':None,
            'progress':{'processed':0,'total':0},
            'urls':[],
            'messages':[],
            'started':time.time(),
            'userid':userid,
            'username':username,
            'meta':{
                'languages':languages,
                'language_query':language_query,
                'auto_suggest_companies':auto_suggest_companies,
                'user_companies':user_companies,
                'fallback_queries':fallback_queries,
                'selected_sectors':selected_sectors,
                'dynamic_target':dynamic_target,
                'seniority':seniority or None,
                'deepMode':deep_mode,
                'platform_queries':xray_platform_queries,
                'included_platforms':{
                    'gaming':channel_gaming,
                    'media':channel_media,
                    'technology':channel_technology
                },
                'channel_count':channel_count,
                'platform_count':platform_count,
                'userid':userid,
                'username':username
            }
        }
    persist_job(job_id)
    threading.Thread(target=_job_runner,
                     args=(job_id, queries, fallback_queries, auto_expand, manual_urls,
                           search_results_only, country, dynamic_target, job_titles),
                     daemon=True).start()
    return jsonify({'job_id': job_id}), 200

@app.get('/job_status/<job_id>')
def job_status(job_id):
    with JOBS_LOCK:
        job=JOBS.get(job_id)
    if not job:
        return jsonify({"error":"Unknown job id"}), 404
    return jsonify(job)

@app.get('/download/<filename>')
def download_file(filename):
    # Look in BASE_DIR first (legacy), then SEARCH_XLS_DIR
    file_path_base = os.path.join(BASE_DIR, filename)
    if os.path.exists(file_path_base):
        return send_from_directory(BASE_DIR, filename, as_attachment=True)
    file_path_search = os.path.join(SEARCH_XLS_DIR, filename)
    if os.path.exists(file_path_search):
        return send_from_directory(SEARCH_XLS_DIR, filename, as_attachment=True)
    return {'error':'File not found'}, 404

@app.get("/SourcingVerify.html")
def sourcing_verify_page():
    return send_from_directory(BASE_DIR, "SourcingVerify.html")

@app.get("/sourcing/list")
def sourcing_list():
    try:
        userid = (request.args.get("userid") or "").strip()
        if not userid:
            return jsonify({"rows": []})
        page = request.args.get("page", type=int)
        page_size = request.args.get("page_size", type=int) or request.args.get("pagesize", type=int)
        all_flag = (request.args.get("all") or "").strip().lower() in {"1","true","yes"}
        use_paging = (bool(page and page_size) and not all_flag)
        if use_paging:
            page = max(1, int(page))
            page_size = max(1, min(int(page_size), 1000))
            offset = (page - 1) * page_size
        import psycopg2
        from psycopg2 import sql
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()
        total = None
        if use_paging:
            cur.execute("SELECT COUNT(*) FROM sourcing WHERE userid=%s", (userid,))
            total = int(cur.fetchone()[0])
            # Check if pic column exists
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='sourcing' AND column_name='pic'")
            has_pic = cur.fetchone() is not None
            
            # LEFT JOIN with process table to get latest company value for Gemini inference
            # Priority: process.company > sourcing.company (CV-parsed data is source of truth)
            if has_pic:
                cur.execute(
                    sql.SQL("""
                        SELECT s.name, 
                               COALESCE(p.company, s.company) as company,
                               COALESCE(p.jobtitle, s.jobtitle) as jobtitle,
                               COALESCE(p.country, s.country) as country,
                               s.experience, 
                               s.linkedinurl, 
                               s.pic,
                               p.rating
                        FROM sourcing s
                        LEFT JOIN process p ON s.linkedinurl = p.linkedinurl
                        WHERE s.userid=%s 
                        ORDER BY s.name NULLS LAST 
                        LIMIT %s OFFSET %s
                    """),
                    (userid, page_size, offset)
                )
            else:
                cur.execute(
                    sql.SQL("""
                        SELECT s.name, 
                               COALESCE(p.company, s.company) as company,
                               COALESCE(p.jobtitle, s.jobtitle) as jobtitle,
                               COALESCE(p.country, s.country) as country,
                               s.experience, 
                               s.linkedinurl,
                               p.rating
                        FROM sourcing s
                        LEFT JOIN process p ON s.linkedinurl = p.linkedinurl
                        WHERE s.userid=%s 
                        ORDER BY s.name NULLS LAST 
                        LIMIT %s OFFSET %s
                    """),
                    (userid, page_size, offset)
                )
        else:
            # Check if pic column exists
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='sourcing' AND column_name='pic'")
            has_pic = cur.fetchone() is not None
            
            # LEFT JOIN with process table to get latest company value for Gemini inference
            # Priority: process.company > sourcing.company (CV-parsed data is source of truth)
            if has_pic:
                cur.execute("""
                    SELECT s.name, 
                           COALESCE(p.company, s.company) as company,
                           COALESCE(p.jobtitle, s.jobtitle) as jobtitle,
                           COALESCE(p.country, s.country) as country,
                           s.experience, 
                           s.linkedinurl, 
                           s.pic,
                           p.rating
                    FROM sourcing s
                    LEFT JOIN process p ON s.linkedinurl = p.linkedinurl
                    WHERE s.userid=%s 
                    ORDER BY s.name NULLS LAST
                """, (userid,))
            else:
                cur.execute("""
                    SELECT s.name, 
                           COALESCE(p.company, s.company) as company,
                           COALESCE(p.jobtitle, s.jobtitle) as jobtitle,
                           COALESCE(p.country, s.country) as country,
                           s.experience, 
                           s.linkedinurl,
                           p.rating
                    FROM sourcing s
                    LEFT JOIN process p ON s.linkedinurl = p.linkedinurl
                    WHERE s.userid=%s 
                    ORDER BY s.name NULLS LAST
                """, (userid,))
        
        # Process rows and convert bytea to base64 if needed
        import base64
        rows = []
        for r in cur.fetchall():
            row_dict = {
                "name": r[0] or "",
                "company": r[1] or "",
                "jobtitle": r[2] or "",
                "country": r[3] or "",
                "experience": r[4] or "",
                "linkedinurl": r[5] or "",
                "rating": "",  # Initialize rating field (full JSON)
                "rating_score": "",  # Convenience field: numeric score (e.g., "79")
                "rating_stars": "",  # Convenience field: star count (e.g., "4")
                "rating_level": ""   # Convenience field: assessment level (e.g., "L1")
            }
            # Add pic column if it exists
            if has_pic and len(r) > 6:
                pic_data = r[6]
                if pic_data:
                    # Convert bytea to base64 string for JSON transport
                    if isinstance(pic_data, (bytes, memoryview)):
                        row_dict["pic"] = base64.b64encode(bytes(pic_data)).decode('utf-8')
                    else:
                        # Already a string (legacy URL data)
                        row_dict["pic"] = str(pic_data)
                else:
                    row_dict["pic"] = ""
                # Rating is at index 7 when pic exists
                if len(r) > 7:
                    row_dict["rating"] = r[7] or ""
            else:
                row_dict["pic"] = ""
                # Rating is at index 6 when no pic
                if len(r) > 6:
                    row_dict["rating"] = r[6] or ""
            
            # Parse rating JSON and extract convenience fields for frontend
            if row_dict["rating"]:
                try:
                    rating_obj = None
                    if isinstance(row_dict["rating"], str):
                        rating_obj = json.loads(row_dict["rating"])
                    elif isinstance(row_dict["rating"], dict):
                        rating_obj = row_dict["rating"]
                    
                    if rating_obj:
                        # Extract total_score (e.g., "79%") and convert to numeric
                        total_score_str = rating_obj.get("total_score", "")
                        if total_score_str and "%" in total_score_str:
                            row_dict["rating_score"] = total_score_str.replace("%", "").strip()
                        
                        # Extract stars
                        stars = rating_obj.get("stars")
                        if stars is not None:
                            row_dict["rating_stars"] = str(stars)
                        
                        # Extract assessment level (e.g., "L1" from "L1 Assessment" or legacy "Level 1 Assessment")
                        assessment_level = rating_obj.get("assessment_level", "")
                        if "Level 1" in assessment_level or "L1" in assessment_level:
                            row_dict["rating_level"] = "L1"
                        elif "Level 2" in assessment_level or "L2" in assessment_level:
                            row_dict["rating_level"] = "L2"
                except Exception as e:
                    logger.warning(f"[Sourcing List] Failed to parse rating JSON for {row_dict['linkedinurl']}: {e}")
            
            rows.append(row_dict)
        cur.close(); conn.close()
        
        # Diagnostic logging to check if rating is included
        if rows:
            sample = rows[0]
            has_rating_field = "rating" in sample
            rating_value = sample.get("rating", "")
            has_rating_data = bool(rating_value and rating_value != "")
            rating_score = sample.get("rating_score", "")
            rating_stars = sample.get("rating_stars", "")
            rating_level = sample.get("rating_level", "")
            logger.info(f"[Sourcing List] Returning {len(rows)} rows | Has rating: {has_rating_data} | Convenience fields: score={rating_score}%, stars={rating_stars}, level={rating_level}")
        
        resp = {"rows": rows}
        if use_paging:
            resp.update({"page": page, "page_size": page_size, "total": total})
        
        # Add no-cache headers to ensure fresh data after assessments
        response = jsonify(resp)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response
    except Exception as e:
        logger.warning(f"[Sourcing List] {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/sourcing/update")
def sourcing_update():
    data=request.get_json(force=True, silent=True) or {}
    linkedinurl=(data.get("linkedinurl") or "").strip()
    field=(data.get("field") or "").strip().lower()
    value=(data.get("value") or "").strip()
    allowed_fields = {
        "name": "name",
        "company": "company",
        "jobtitle": "jobtitle",
        "country": "country",
        "appeal": "appeal",
        "experience": "experience"
    }
    if not linkedinurl or field not in allowed_fields:
        return jsonify({"error":"Invalid parameters"}), 400
    if field == "appeal" and len(value) > 500:
        value = value[:500]
    if field == "experience" and len(value) > 5000:
        value = value[:5000]
    try:
        import psycopg2
        from psycopg2 import sql
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()
        col_identifier = sql.Identifier(allowed_fields[field])
        query = sql.SQL("UPDATE sourcing SET {col}=%s WHERE linkedinurl=%s").format(col=col_identifier)
        cur.execute(query, (value, linkedinurl))
        affected=cur.rowcount
        conn.commit()
        cur.close(); conn.close()
        if affected==0:
            return jsonify({"error":"Row not found"}), 404
        return jsonify({"updated":affected,"field":field,"value":value})
    except Exception as e:
        logger.warning(f"[Sourcing Update] {e}")
        return jsonify({"error":str(e)}), 500

@app.post("/sourcing/delete")
def sourcing_delete():
    data=request.get_json(force=True, silent=True) or {}
    arr=data.get("linkedinurls")
    if not isinstance(arr,list) or not arr:
        return jsonify({"error":"linkedinurls list required"}), 400
    cleaned=[(x or "").strip() for x in arr if (x or "").strip()]
    if not cleaned:
        return jsonify({"error":"No valid linkedinurls"}), 400
    try:
        import psycopg2
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()
        
        # Delete from sourcing
        cur.execute("DELETE FROM sourcing WHERE linkedinurl = ANY(%s)", (cleaned,))
        deleted=cur.rowcount
        
        # Data Consistency: Delete from process as well (rebated/deleted)
        cur.execute("DELETE FROM process WHERE linkedinurl = ANY(%s)", (cleaned,))
        
        conn.commit()
        cur.close(); conn.close()
        return jsonify({"deleted":deleted})
    except Exception as e:
        logger.warning(f"[Sourcing Delete] {e}")
        return jsonify({"error":str(e)}), 500

@app.post("/process/delete")
def process_delete_entry():
    data = request.get_json(force=True, silent=True) or {}
    linkedinurls = data.get("linkedinurls")
    # Support single or list
    if not linkedinurls:
        single = data.get("linkedinurl")
        if single:
            linkedinurls = [single]
    
    username = (data.get("username") or "").strip()
    userid = (data.get("userid") or "").strip()

    if not linkedinurls or not isinstance(linkedinurls, list):
        return jsonify({"error": "linkedinurl or linkedinurls list required"}), 400

    cleaned = [str(x).strip() for x in linkedinurls if str(x).strip()]
    if not cleaned:
         return jsonify({"error": "No valid URLs"}), 400

    try:
        import psycopg2
        from psycopg2 import sql
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()

        deleted_total = 0

        # Check for normalized column
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='process' AND column_name='normalized_linkedin'")
        has_normalized = bool(cur.fetchone())

        for url in cleaned:
             # Try exact delete
             clause = "linkedinurl = %s"
             args = [url]
             if userid:
                 clause += " AND userid = %s"
                 args.append(userid)
             
             cur.execute(f"DELETE FROM process WHERE {clause}", tuple(args))
             cnt = cur.rowcount
             
             if cnt == 0 and has_normalized:
                 norm = _normalize_linkedin_to_path(url)
                 if norm:
                     clause_n = "normalized_linkedin = %s"
                     args_n = [norm]
                     if userid:
                         clause_n += " AND userid = %s"
                         args_n.append(userid)
                     cur.execute(f"DELETE FROM process WHERE {clause_n}", tuple(args_n))
                     cnt = cur.rowcount
             
             if cnt > 0:
                 deleted_total += 1

        new_token = 0
        if deleted_total > 0:
             # Update token
             cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='login' AND column_name='userid'")
             login_has_userid = bool(cur.fetchone())
             
             if login_has_userid and userid:
                 cur.execute("UPDATE login SET token = COALESCE(token,0) + %s WHERE userid = %s RETURNING COALESCE(token,0)", (deleted_total, userid))
                 row = cur.fetchone()
                 if row: new_token = row[0]
             elif username:
                 cur.execute("UPDATE login SET token = COALESCE(token,0) + %s WHERE username = %s RETURNING COALESCE(token,0)", (deleted_total, username))
                 row = cur.fetchone()
                 if row: new_token = row[0]
        else:
             # Just fetch current token
             if username:
                 cur.execute("SELECT COALESCE(token,0) FROM login WHERE username=%s", (username,))
                 r = cur.fetchone()
                 if r: new_token = r[0]

        conn.commit()
        cur.close(); conn.close()

        return jsonify({"deleted": deleted, "token_delta": deleted, "new_token": int(new_token)}), 200

    except Exception as e:
        logger.error(f"[Process Delete] {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/process/update")
def process_update():
    """
    Update process table fields. Accepts linkedinurl and any allowed field to update.
    Used for updating tenure and profile picture (pic column).
    """
    data = request.get_json(force=True, silent=True) or {}
    linkedinurl = (data.get("linkedinurl") or "").strip()
    
    if not linkedinurl:
        return jsonify({"error": "linkedinurl required"}), 400
    
    # Define allowed fields for update
    allowed_fields = {
        "tenure": "tenure",
        "pic": "pic",
        "name": "name",
        "company": "company",
        "jobtitle": "jobtitle",
        "country": "country"
    }
    
    # Collect fields to update
    updates = {}
    for key, col in allowed_fields.items():
        if key in data:
            updates[col] = data[key]
    
    if not updates:
        return jsonify({"error": "No valid fields to update"}), 400
    
    try:
        import psycopg2
        from psycopg2 import sql
        pg_host = os.getenv("PGHOST", "localhost")
        pg_port = int(os.getenv("PGPORT", "5432"))
        pg_user = os.getenv("PGUSER", "postgres")
        pg_password = os.getenv("PGPASSWORD", "") or "orlha"
        pg_db = os.getenv("PGDATABASE", "candidate_db")
        conn = psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur = conn.cursor()
        
        # Check which columns exist in the process table
        cur.execute("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema='public' AND table_name='process'
        """)
        existing_cols = {r[0].lower() for r in cur.fetchall()}
        
        # Filter updates to only existing columns
        valid_updates = {k: v for k, v in updates.items() if k.lower() in existing_cols}
        
        if not valid_updates:
            cur.close(); conn.close()
            return jsonify({"error": "No valid columns to update in process table"}), 400
        
        # Build UPDATE query
        set_parts = []
        params = []
        for col, value in valid_updates.items():
            set_parts.append(sql.SQL("{} = %s").format(sql.Identifier(col)))
            params.append(value)
        
        params.append(linkedinurl)
        
        query = sql.SQL("UPDATE process SET {} WHERE linkedinurl = %s").format(
            sql.SQL(", ").join(set_parts)
        )
        
        cur.execute(query, params)
        affected = cur.rowcount
        
        # If no rows affected, try with normalized_linkedin
        if affected == 0 and 'normalized_linkedin' in existing_cols:
            normalized = _normalize_linkedin_to_path(linkedinurl)
            if normalized:
                params[-1] = normalized
                query_norm = sql.SQL("UPDATE process SET {} WHERE normalized_linkedin = %s").format(
                    sql.SQL(", ").join(set_parts)
                )
                cur.execute(query_norm, params)
                affected = cur.rowcount
        
        conn.commit()
        cur.close(); conn.close()
        
        if affected == 0:
            return jsonify({"error": "No matching record found"}), 404
        
        return jsonify({
            "updated": affected,
            "fields": list(valid_updates.keys())
        }), 200
        
    except Exception as e:
        logger.error(f"[Process Update] {e}")
        return jsonify({"error": str(e)}), 500
        
@app.post("/sourcing/save_profile_json")
def sourcing_save_profile_json():
    data = request.get_json(force=True, silent=True) or {}
    linkedinurl = (data.get("linkedinurl") or "").strip()
    userid = (data.get("userid") or "").strip()
    
    if not linkedinurl:
        return jsonify({"error": "linkedinurl required"}), 400

    try:
        import psycopg2
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()
        
        # Determine query based on provided URL format
        # Try exact match first, then normalized if needed
        cur.execute("SELECT name FROM sourcing WHERE linkedinurl=%s", (linkedinurl,))
        row = cur.fetchone()
        
        if not row:
             # Try normalized lookup if strict match fails
             from urllib.parse import urlparse
             path = urlparse(linkedinurl).path.lower().rstrip('/')
             if path:
                 cur.execute("SELECT name FROM sourcing WHERE LOWER(linkedinurl) LIKE %s LIMIT 1", (f"%{path}%",))
                 row = cur.fetchone()

        candidate_name = None
        if row and row[0]:
            candidate_name = row[0].strip()

        # Fetch username from login table if userid is provided
        username_str = "unknown"
        if userid:
            cur.execute("SELECT username FROM login WHERE userid=%s", (userid,))
            u_row = cur.fetchone()
            if u_row and u_row[0]:
                username_str = u_row[0].strip()

        cur.close(); conn.close()

        if not candidate_name:
            return jsonify({"error": "Profile not found or name empty"}), 404
             
        # Sanitize filename components to "pname {username}.json"
        safe_username = re.sub(r'[\\/*?:"<>|]', "", username_str)
        filename = f"pname {safe_username}.json"
        out_path = os.path.join(OUTPUT_DIR, filename)
        
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"name": candidate_name, "linkedinurl": linkedinurl}, f, ensure_ascii=False, indent=2)
            
        return jsonify({"status": "ok", "file": filename}), 200
        
    except Exception as e:
        logger.error(f"[Save Profile JSON] {e}")
        return jsonify({"error": str(e)}), 500

def _normalize_linkedin_to_path(linkedin_url: str) -> str:
    if not linkedin_url:
        return ""
    s = linkedin_url.split('?', 1)[0].strip()
    path = re.sub(r'^https?://[^/]+', '', s, flags=re.I)
    path = path.lower().rstrip('/')
    return path

@app.post("/sourcing/market_analysis")
def sourcing_market_analysis():
    payload = request.get_json(force=True, silent=True) or {}
    records = payload.get("records")
    if not isinstance(records, list) or not records:
        return jsonify({"error": "records list required"}), 400

    normalized_records = []
    for r in records:
        if not isinstance(r, dict):
            continue
        name = (r.get("name") or "").strip()
        company = (r.get("organisation") or r.get("company") or "").strip()
        jobtitle_val = (r.get("jobtitle") or r.get("role") or "").strip()
        country = (r.get("country") or "").strip()
        linkedinurl = (r.get("snapshot_at") or r.get("linkedinurl") or "").strip()
        username = (r.get("username") or "").strip()
        userid = (r.get("userid") or "").strip()
        role_tag_val = (r.get("role_tag") or r.get("roleTag") or "").strip()
        experience_val = (r.get("experience") or "").strip()
        rating_val = (r.get("rating") or "").strip()
        
        normalized_linkedin = _normalize_linkedin_to_path(linkedinurl)
        normalized = {
            "name": name,
            "company": company,
            "jobtitle": jobtitle_val,
            "country": country,
            "linkedinurl": linkedinurl,
            "normalized_linkedin": normalized_linkedin,
            "username": username,
            "userid": userid,
            "role_tag": role_tag_val,
            "experience": experience_val,
            "rating": rating_val
        }
        normalized_records.append(normalized)

    valid_records = []
    for nr in normalized_records:
        if nr["name"] and nr["company"] and nr["jobtitle"] and nr["country"] and nr["linkedinurl"]:
            valid_records.append(nr)

    if not valid_records:
        return jsonify({"error": "No valid rows to insert into process table after normalization"}), 400

    inserted_process = 0
    try:
        import psycopg2
        from psycopg2 import sql
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()

        # Discover available columns in process table
        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'process'
        """)
        cols = {row[0].lower() for row in cur.fetchall()}

        preferred_title_col = None
        if 'jobtitle' in cols:
            preferred_title_col = 'jobtitle'
        elif 'role' in cols:
            preferred_title_col = 'role'
        else:
            cur.close()
            conn.close()
            msg = "Process table does not have 'jobtitle' or 'role' column."
            logger.error(msg)
            return jsonify({"error": msg}), 500

        role_col = None
        if 'role_tag' in cols:
            role_col = 'role_tag'
        elif 'roletag' in cols:
            role_col = 'roletag'

        experience_col = None
        if 'experience' in cols:
            experience_col = 'experience'

        rating_col = None
        if 'rating' in cols:
            rating_col = 'rating'

        normalized_col = 'normalized_linkedin' if 'normalized_linkedin' in cols else None

        # NEW: If process table exposes an 'id' column, we'll attempt to copy sourcing.id into process.id
        process_has_id = 'id' in cols

        # PATCH: Check for geographic column
        geo_col = 'geographic' if 'geographic' in cols else None
        
        # PATCH: Check for pic column
        pic_col = 'pic' if 'pic' in cols else None

        # Helper to lookup role_tag from login if missing
        def _get_role_tag_for_user(c, uname, uid):
            if not uname and not uid: return None
            try:
                # We need a fresh cursor or reuse 'cur' carefully
                if uid:
                    c.execute("SELECT role_tag FROM login WHERE userid=%s LIMIT 1", (uid,))
                else:
                    c.execute("SELECT role_tag FROM login WHERE username=%s LIMIT 1", (uname,))
                r = c.fetchone()
                return r[0] if r and r[0] else None
            except Exception as e_rt:
                logger.warning(f"Failed to lookup role_tag from login: {e_rt}")
                return None

        # Build list for INITIAL INSERT - strictly core identity fields only.
        # EXCLUDE rating and experience from here to prevent positional mismatches.
        # If process has id, include it so we can set process.id = sourcing.id
        field_list = []
        if process_has_id:
            field_list.append('id')
        field_list.extend(['name', 'company', preferred_title_col, 'country', 'linkedinurl', 'username', 'userid'])
        if role_col:
            field_list.append(role_col)
        if normalized_col:
            field_list.append(normalized_col)
        if geo_col:
            field_list.append(geo_col)
        if pic_col:
            field_list.append(pic_col)

        placeholders = sql.SQL(', ').join([sql.Placeholder() for _ in field_list])
        insert_sql = sql.SQL("INSERT INTO process ({fields}) VALUES ({placeholders})").format(
            fields=sql.SQL(', ').join([sql.Identifier(f) for f in field_list]),
            placeholders=placeholders
        )

        # We'll keep a small cache mapping linkedinurl -> sourcing.id to avoid repeated sourcing lookups
        sourcing_id_cache = {}

        def get_sourcing_id_by_linkedin(link):
            if not link: return None
            if link in sourcing_id_cache:
                return sourcing_id_cache[link]
            try:
                cur2 = conn.cursor()
                cur2.execute("SELECT id FROM sourcing WHERE linkedinurl = %s LIMIT 1", (link,))
                r = cur2.fetchone()
                if not r:
                    # fallback to LIKE normalized path
                    path = _normalize_linkedin_to_path(link)
                    if path:
                        cur2.execute("SELECT id FROM sourcing WHERE LOWER(linkedinurl) LIKE %s LIMIT 1", (f"%{path.lower()}",))
                        r = cur2.fetchone()
                cur2.close()
                sid = r[0] if r else None
                sourcing_id_cache[link] = sid
                return sid
            except Exception:
                sourcing_id_cache[link] = None
                return None

        batch_rows = []
        for nr in valid_records:
            # Check for existing record to avoid duplicate insert if ON CONFLICT DO NOTHING is not effective (no unique constraint)
            exists = False
            l_val = nr.get('linkedinurl')
            norm_val = nr.get('normalized_linkedin')

            # Attempt to find sourcing.id for this linkedin so we can set process.id accordingly
            sourcing_id = get_sourcing_id_by_linkedin(l_val)

            try:
                # First, try to find existing process row using multiple strategies
                if process_has_id and sourcing_id:
                    # Check by id
                    check_query = sql.SQL("SELECT 1 FROM process WHERE id = %s LIMIT 1")
                    cur.execute(check_query, (sourcing_id,))
                    if cur.fetchone():
                        exists = True
                if not exists:
                    # Next try exact linkedinurl match
                    check_query = sql.SQL("SELECT 1 FROM process WHERE linkedinurl = %s LIMIT 1")
                    cur.execute(check_query, (l_val,))
                    if cur.fetchone():
                        exists = True
                if not exists and normalized_col and norm_val:
                    # Fallback check by normalized
                    check_norm = sql.SQL("SELECT 1 FROM process WHERE {} = %s LIMIT 1").format(sql.Identifier(normalized_col))
                    cur.execute(check_norm, (norm_val,))
                    if cur.fetchone():
                        exists = True
            except Exception as e_check:
                logger.warning(f"[Market Analysis Existence Check] Failed for {l_val}: {e_check}")
                # Assume not exists or let insert handle it if DB error allows

            if not exists:
                vals = []
                # Pre-calculate geographic
                geo_val = None
                if geo_col:
                    country_input = nr.get('country', '')
                    if country_input:
                        geo_val = _infer_region_from_country(country_input)

                # Pre-calculate role_tag if missing
                role_val_final = nr.get('role_tag')
                if not role_val_final and role_col:
                    role_val_final = _get_role_tag_for_user(cur, nr.get('username'), nr.get('userid'))
                
                # Pre-calculate profile picture if needed
                pic_val = None
                if pic_col:
                    linkedin_url = nr.get('linkedinurl')
                    if linkedin_url:
                        try:
                            pic_url = get_linkedin_profile_picture(linkedin_url)
                            if pic_url:
                                pic_bytes = fetch_image_bytes_from_url(pic_url)
                                if pic_bytes:
                                    import psycopg2
                                    pic_val = psycopg2.Binary(pic_bytes)
                        except Exception as pic_err:
                            logger.warning(f"[Market Analysis] Failed to get profile pic for {linkedin_url}: {pic_err}")

                for f in field_list:
                    if f == 'id':
                        # Use sourcing_id if available; else None so DB can assign serial if permitted
                        vals.append(sourcing_id)
                    elif f == 'name':
                        vals.append(nr.get('name'))
                    elif f == 'company':
                        vals.append(nr.get('company'))
                    elif f == preferred_title_col:
                        vals.append(nr.get('jobtitle'))
                    elif f == 'country':
                        vals.append(nr.get('country'))
                    elif f == 'linkedinurl':
                        vals.append(nr.get('linkedinurl'))
                    elif f == 'username':
                        vals.append(nr.get('username'))
                    elif f == 'userid':
                        vals.append(nr.get('userid'))
                    elif role_col and f == role_col:
                        vals.append(role_val_final or None)
                    elif normalized_col and f == normalized_col:
                        vals.append(nr.get('normalized_linkedin') or None)
                    elif geo_col and f == geo_col:
                        vals.append(geo_val)
                    elif pic_col and f == pic_col:
                        vals.append(pic_val)
                    else:
                        vals.append(None)

                try:
                    cur.execute(insert_sql, tuple(vals))
                    inserted_process += cur.rowcount
                except Exception as e_ins:
                    # Insert failed; likely duplicate key or id collision. Rollback to keep session consistent and continue.
                    logger.warning(f"[Market Analysis Insert] Insert failed for {l_val}: {e_ins}")
                    conn.rollback()

        # Explicitly UPDATE core fields for every record.
        # This covers updates for existing records (where conflict occurred).
        update_fields = ['name', 'company', 'country', 'username', 'userid']
        if role_col:
            update_fields.append(role_col)
        update_fields.append(preferred_title_col)
        if geo_col:
            update_fields.append(geo_col)

        # Iterate over records to perform updates
        for nr in valid_records:
            l_val = nr.get('linkedinurl')
            norm_val = nr.get('normalized_linkedin')
            if not l_val: continue

            # Attempt to get sourcing id again (cache will make this cheap)
            sourcing_id = get_sourcing_id_by_linkedin(l_val)

            # Pre-calculate geographic for update
            geo_val = None
            if geo_col:
                country_input = nr.get('country', '')
                if country_input:
                    geo_val = _infer_region_from_country(country_input)

            # Prepare role_tag for update if missing in incoming data
            role_val_final = nr.get('role_tag')
            if not role_val_final and role_col:
                role_val_final = _get_role_tag_for_user(cur, nr.get('username'), nr.get('userid'))

            # Update core fields if record exists
            set_parts = []
            update_values = []

            for f in update_fields:
                val = None
                if f == 'name': val = nr.get('name')
                elif f == 'company': val = nr.get('company')
                elif f == 'country': val = nr.get('country')
                elif f == 'username': val = nr.get('username')
                elif f == 'userid': val = nr.get('userid')
                elif f == preferred_title_col: val = nr.get('jobtitle')
                elif role_col and f == role_col: val = role_val_final
                elif geo_col and f == geo_col: val = geo_val

                if val is not None:
                    set_parts.append(sql.SQL("{} = %s").format(sql.Identifier(f)))
                    update_values.append(val)

            if set_parts:
                try:
                    if process_has_id and sourcing_id:
                        # Prefer update by process.id when we have sourcing.id mapped
                        update_query = sql.SQL("UPDATE process SET {} WHERE id = %s").format(sql.SQL(', ').join(set_parts))
                        cur.execute(update_query, update_values + [sourcing_id])
                        if cur.rowcount == 0:
                            # fallback to linkedinurl/normalized
                            update_query = sql.SQL("UPDATE process SET {} WHERE linkedinurl = %s").format(sql.SQL(', ').join(set_parts))
                            cur.execute(update_query, update_values + [l_val])
                            if cur.rowcount == 0 and normalized_col and norm_val:
                                update_norm_query = sql.SQL("UPDATE process SET {} WHERE {} = %s").format(
                                    sql.SQL(', ').join(set_parts),
                                    sql.Identifier(normalized_col)
                                )
                                cur.execute(update_norm_query, update_values + [norm_val])
                    else:
                        update_query = sql.SQL("UPDATE process SET {} WHERE linkedinurl = %s").format(sql.SQL(', ').join(set_parts))
                        cur.execute(update_query, update_values + [l_val])
                        if cur.rowcount == 0 and normalized_col and norm_val:
                            update_norm_query = sql.SQL("UPDATE process SET {} WHERE {} = %s").format(
                                sql.SQL(', ').join(set_parts),
                                sql.Identifier(normalized_col)
                            )
                            cur.execute(update_norm_query, update_values + [norm_val])
                except Exception as e_upd_core:
                    logger.warning(f"[Market Analysis Core Update] Failed for {l_val}: {e_upd_core}")

            # Explicit update for Rating
            if rating_col:
                r_val = nr.get('rating')
                if r_val is not None:
                    try:
                        if process_has_id and sourcing_id:
                            query = sql.SQL("UPDATE process SET {} = %s WHERE id = %s").format(sql.Identifier(rating_col))
                            cur.execute(query, (r_val, sourcing_id))
                            # Fallbacks
                            if cur.rowcount == 0:
                                query = sql.SQL("UPDATE process SET {} = %s WHERE linkedinurl = %s").format(sql.Identifier(rating_col))
                                cur.execute(query, (r_val, l_val))
                                if cur.rowcount == 0 and normalized_col and norm_val:
                                    query_norm = sql.SQL("UPDATE process SET {} = %s WHERE {} = %s").format(sql.Identifier(rating_col), sql.Identifier(normalized_col))
                                    cur.execute(query_norm, (r_val, norm_val))
                        else:
                            query = sql.SQL("UPDATE process SET {} = %s WHERE linkedinurl = %s").format(sql.Identifier(rating_col))
                            cur.execute(query, (r_val, l_val))
                            if cur.rowcount == 0 and normalized_col and norm_val:
                                query_norm = sql.SQL("UPDATE process SET {} = %s WHERE {} = %s").format(sql.Identifier(rating_col), sql.Identifier(normalized_col))
                                cur.execute(query_norm, (r_val, norm_val))
                    except Exception as e_upd:
                        logger.warning(f"[Market Analysis Update Patch] Failed to update rating for {l_val}: {e_upd}")

            # Explicit update for Experience
            if experience_col:
                e_val = nr.get('experience')
                if e_val is not None:
                     try:
                        if process_has_id and sourcing_id:
                            query = sql.SQL("UPDATE process SET {} = %s WHERE id = %s").format(sql.Identifier(experience_col))
                            cur.execute(query, (e_val, sourcing_id))
                            if cur.rowcount == 0:
                                query = sql.SQL("UPDATE process SET {} = %s WHERE linkedinurl = %s").format(sql.Identifier(experience_col))
                                cur.execute(query, (e_val, l_val))
                                if cur.rowcount == 0 and normalized_col and norm_val:
                                    query_norm = sql.SQL("UPDATE process SET {} = %s WHERE {} = %s").format(sql.Identifier(experience_col), sql.Identifier(normalized_col))
                                    cur.execute(query_norm, (e_val, norm_val))
                        else:
                            query = sql.SQL("UPDATE process SET {} = %s WHERE linkedinurl = %s").format(sql.Identifier(experience_col))
                            cur.execute(query, (e_val, l_val))
                            if cur.rowcount == 0 and normalized_col and norm_val:
                                query_norm = sql.SQL("UPDATE process SET {} = %s WHERE {} = %s").format(sql.Identifier(experience_col), sql.Identifier(normalized_col))
                                cur.execute(query_norm, (e_val, norm_val))
                     except Exception as e_upd:
                        logger.warning(f"[Market Analysis Update Patch] Failed to update experience for {l_val}: {e_upd}")

        # PATCH: Sync sequence if we inserted IDs
        if process_has_id and inserted_process > 0:
             try:
                 cur.execute("SELECT setval(pg_get_serial_sequence('process', 'id'), (SELECT MAX(id) FROM process))")
                 conn.commit()
             except Exception:
                 conn.rollback()

        conn.commit()
        cur.close(); conn.close()

        return jsonify({
            "inserted_process": inserted_process,
            "received_process": len(batch_rows),
            "used_title_column": preferred_title_col,
            "used_role_column": role_col,
            "used_experience_column": experience_col,
            "used_rating_column": rating_col,
            "used_normalized_column": normalized_col,
            "process_id_mapped_from_sourcing": process_has_id
        }), 200
    except Exception as e:
        logger.warning(f"[Market Analysis Insert -> process] {e}")
        return jsonify({"error": str(e)}), 500

@app.get("/process/geography")
def process_geography():
    linkedin = (request.args.get("linkedin") or "").strip()
    if not linkedin:
        return jsonify({"error": "linkedin param required"}), 400
    linkedin_norm = linkedin.split('?')[0].rstrip('/')
    linkedin_path = _normalize_linkedin_to_path(linkedin)

    def _standardize_host(url_str: str) -> str:
        s = (url_str or "").strip()
        if not s:
            return s
        return re.sub(r'^https?://[^/]+', 'https://www.linkedin.com', s, flags=re.I)

    linkedin_norm_www = _standardize_host(linkedin_norm)

    try:
        import psycopg2
        from psycopg2 import sql
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()

        cur.execute("""
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name = 'process'
        """)
        available_cols = {r[0].lower() for r in cur.fetchall()}

        desired = [
            "id","name","company","jobtitle","country","linkedinurl","username","userid",
            "product","sector","jobfamily","geographic","seniority","skillset","sourcingstatus",
            "email","mobile","office","role_tag","experience","cv","exp","education","tenure","vskillset"
        ]
        selected = [c for c in desired if c in available_cols]
        if not selected:
            cur.close(); conn.close()
            return jsonify({"error": "Process table does not contain expected columns."}), 500

        fields_sql = sql.SQL(', ').join([sql.Identifier(c) for c in selected])

        row = None

        if 'normalized_linkedin' in available_cols and linkedin_path:
            q_norm_exact = sql.SQL("SELECT {fields} FROM process WHERE normalized_linkedin = %s LIMIT 1").format(fields=fields_sql)
            cur.execute(q_norm_exact, (linkedin_path,))
            row = cur.fetchone()

        if not row and 'linkedinurl' in available_cols:
            q_legacy_exact = sql.SQL("SELECT {fields} FROM process WHERE linkedinurl = %s LIMIT 1").format(fields=fields_sql)
            cur.execute(q_legacy_exact, (linkedin_norm,))
            row = cur.fetchone()

        if not row and 'linkedinurl' in available_cols and linkedin_norm_www and linkedin_norm_www != linkedin_norm:
            q_legacy_www = sql.SQL("SELECT {fields} FROM process WHERE linkedinurl = %s LIMIT 1").format(fields=fields_sql)
            cur.execute(q_legacy_www, (linkedin_norm_www,))
            row = cur.fetchone()

        if not row and 'snapshot_at' in available_cols:
            q_snap_exact = sql.SQL("SELECT {fields} FROM process WHERE snapshot_at = %s LIMIT 1").format(fields=fields_sql)
            cur.execute(q_snap_exact, (linkedin_norm,))
            row = cur.fetchone()
            if not row and linkedin_norm_www and linkedin_norm_www != linkedin_norm:
                cur.execute(q_snap_exact, (linkedin_norm_www,))
                row = cur.fetchone()

        if not row and linkedin_path and 'linkedinurl' in available_cols:
            suffix = linkedin_path
            q_like_legacy = sql.SQL("SELECT {fields} FROM process WHERE LOWER(linkedinurl) LIKE %s LIMIT 1").format(fields=fields_sql)
            cur.execute(q_like_legacy, (f"%{suffix.lower()}",))
            row = cur.fetchone()

        if not row and linkedin_path and 'snapshot_at' in available_cols:
            q_like_snap = sql.SQL("SELECT {fields} FROM process WHERE LOWER(snapshot_at) LIKE %s LIMIT 1").format(fields=fields_sql)
            cur.execute(q_like_snap, (f"%{linkedin_path.lower()}",))
            row = cur.fetchone()

        if not row and 'normalized_linkedin' in available_cols and linkedin_path:
            q_like_norm = sql.SQL("SELECT {fields} FROM process WHERE normalized_linkedin LIKE %s LIMIT 1").format(fields=fields_sql)
            cur.execute(q_like_norm, (f"%{linkedin_path}",))
            row = cur.fetchone()

        if not row:
            sourcing_fields = ["name","company","jobtitle","country","experience","linkedinurl"]
            q_src = sql.SQL("SELECT {fields} FROM sourcing WHERE LOWER(linkedinurl) LIKE %s LIMIT 1").format(
                fields=sql.SQL(', ').join([sql.Identifier(f) for f in sourcing_fields])
            )
            cur.execute(q_src, (f"%{linkedin_path.lower()}",))
            srow = cur.fetchone()
            if srow:
                result = {}
                for c in selected:
                    if c == "name": result[c] = srow[0] or ""
                    elif c == "company": result[c] = srow[1] or ""
                    elif c == "jobtitle": result[c] = srow[2] or ""
                    elif c == "country": result[c] = srow[3] or ""
                    elif c == "experience": result[c] = srow[4] or ""
                    elif c == "linkedinurl": result[c] = srow[5] or ""
                    else: result[c] = ""
                
                # PATCH: Infer geographic region if missing or matches country (legacy data)
                country_val = result.get("country", "")
                geo_val = result.get("geographic", "")
                if country_val and (not geo_val or geo_val.strip().lower() == country_val.strip().lower()):
                    inferred = _infer_region_from_country(country_val)
                    if inferred:
                        result["geographic"] = inferred

                cur.close(); conn.close()
                # Add no-cache headers to ensure fresh data after assessments
                response = jsonify(result)
                response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
                response.headers['Pragma'] = 'no-cache'
                response.headers['Expires'] = '0'
                return response, 200

        cur.close(); conn.close()
        if not row:
            # Add no-cache headers even for 404 responses
            response = jsonify(None)
            response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
            response.headers['Pragma'] = 'no-cache'
            response.headers['Expires'] = '0'
            return response, 404

        result = {}
        for idx, col in enumerate(selected):
            val = row[idx]
            if col == 'cv':
                 result[col] = bool(val)
            elif col == 'seniority':
                 # Add "-level" suffix for UI display if not already present
                 seniority_val = val if val is not None else ""
                 if seniority_val and not seniority_val.endswith('-level'):
                     result[col] = seniority_val + '-level'
                 else:
                     result[col] = seniority_val
            else:
                 result[col] = val if val is not None else ""

        # PATCH: Infer geographic region if missing or duplicate of country
        country_val = result.get("country", "")
        geo_val = result.get("geographic", "")
        
        # If geographic is missing OR matches country (e.g. "Singapore"=="Singapore"), try to infer "Asia"
        if country_val and (not geo_val or geo_val.strip().lower() == country_val.strip().lower()):
            inferred = _infer_region_from_country(country_val)
            if inferred:
                result["geographic"] = inferred

        # Add no-cache headers to ensure fresh data after assessments
        response = jsonify(result)
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
        return response, 200
    except Exception as e:
        logger.warning(f"[Process Geography] {e}")
        try:
            return jsonify({"error": str(e)}), 500
        except Exception:
            return jsonify({"error": "Internal error"}), 500

@app.post("/process/upload_cv")
def process_upload_cv():
    """
    Uploads a PDF CV file to the 'process' table, storing it in the 'cv' bytea column.
    Trigger analysis after upload.
    Also persists candidate name if provided.
    """
    try:
        if 'cv' not in request.files:
            return jsonify({"error": "No file part"}), 400
        file = request.files['cv']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400
        
        # AFFECTED: Extract name from request form
        candidate_name = (request.form.get('name') or '').strip()
        
        if file and file.filename.lower().endswith('.pdf'):
            linkedinurl = request.form.get('linkedinurl', '').strip()
            if not linkedinurl:
                 return jsonify({"error": "linkedinurl required"}), 400
            
            file_bytes = file.read()
            
            import psycopg2
            from psycopg2 import sql
            pg_host=os.getenv("PGHOST","localhost")
            pg_port=int(os.getenv("PGPORT","5432"))
            pg_user=os.getenv("PGUSER","postgres")
            pg_password=os.getenv("PGPASSWORD","") or "orlha"
            pg_db=os.getenv("PGDATABASE","candidate_db")
            
            conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
            cur=conn.cursor()
            
            binary_cv = psycopg2.Binary(file_bytes)
            normalized = _normalize_linkedin_to_path(linkedinurl)
            
            # --- PATCH START: Insert ID from sourcing into process if exists ---
            sourcing_id = None
            try:
                # Discover if we have 'id' column in process first
                cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='process' AND column_name='id'")
                has_process_id = bool(cur.fetchone())
                
                if has_process_id:
                    # Try to get id from sourcing table
                    cur.execute("SELECT id FROM sourcing WHERE linkedinurl = %s LIMIT 1", (linkedinurl,))
                    sid_row = cur.fetchone()
                    if not sid_row and normalized:
                        cur.execute("SELECT id FROM sourcing WHERE LOWER(linkedinurl) LIKE %s LIMIT 1", (f"%{normalized}%",))
                        sid_row = cur.fetchone()
                    
                    if sid_row:
                        sourcing_id = sid_row[0]
            except Exception as e_id:
                logger.warning(f"[Upload CV] Failed to lookup sourcing ID: {e_id}")
            # --- PATCH END ---

            # Try updating by existing ID first if we found one
            updated = False
            
            # AFFECTED: Prepare update fields including name if present
            update_fields = ["cv = %s"]
            update_values = [binary_cv]
            if candidate_name:
                update_fields.append("name = %s")
                update_values.append(candidate_name)
            
            update_sql_fragment = ", ".join(update_fields)

            if sourcing_id:
                try:
                    cur.execute(f"UPDATE process SET {update_sql_fragment} WHERE id = %s", tuple(update_values + [sourcing_id]))
                    if cur.rowcount > 0:
                        updated = True
                except Exception:
                    conn.rollback()

            if not updated:
                cur.execute(f"UPDATE process SET {update_sql_fragment} WHERE linkedinurl = %s", tuple(update_values + [linkedinurl]))
                if cur.rowcount > 0:
                    updated = True
            
            if not updated and normalized:
                try:
                    cur.execute(f"UPDATE process SET {update_sql_fragment} WHERE normalized_linkedin = %s", tuple(update_values + [normalized]))
                    if cur.rowcount > 0:
                        updated = True
                except Exception:
                    conn.rollback()
            
            if not updated:
                # Insert new record
                try:
                    cols = ["linkedinurl", "cv"]
                    vals = [linkedinurl, binary_cv]
                    placeholders = ["%s", "%s"]
                    
                    # AFFECTED: Include name in insert
                    if candidate_name:
                        cols.append("name")
                        vals.append(candidate_name)
                        placeholders.append("%s")
                    
                    if sourcing_id:
                        cols.append("id")
                        vals.append(sourcing_id)
                        placeholders.append("%s")
                    
                    if normalized:
                        # check if column exists
                        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='process' AND column_name='normalized_linkedin'")
                        if cur.fetchone():
                            cols.append("normalized_linkedin")
                            vals.append(normalized)
                            placeholders.append("%s")

                    insert_q = sql.SQL("INSERT INTO process ({}) VALUES ({})").format(
                        sql.SQL(', ').join(map(sql.Identifier, cols)),
                        sql.SQL(', ').join(map(sql.SQL, placeholders))
                    )
                    cur.execute(insert_q, tuple(vals))
                    
                    # PATCH: Sync sequence if we inserted IDs
                    if sourcing_id:
                        try:
                            cur.execute("SELECT setval(pg_get_serial_sequence('process', 'id'), (SELECT MAX(id) FROM process))")
                        except Exception:
                            pass

                except psycopg2.errors.UniqueViolation:
                    conn.rollback()
                    # Fallback update again just in case race condition
                    cur.execute(f"UPDATE process SET {update_sql_fragment} WHERE linkedinurl = %s", tuple(update_values + [linkedinurl]))
                except Exception as e:
                    conn.rollback()
                    return jsonify({"error": f"Database error on insert: {str(e)}"}), 500

            conn.commit()
            cur.close(); conn.close()
            
            # Fire and forget analysis in background
            threading.Thread(target=analyze_cv_background, args=(linkedinurl, file_bytes)).start()

            return jsonify({"status": "ok"}), 200
        else:
            return jsonify({"error": "Invalid file type, PDF required"}), 400
    except Exception as e:
        logger.error(f"[Upload CV] {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/process/upload_multiple_cvs")
def process_upload_multiple_cvs():
    """
    Accept multiple CV files from a browser upload (FormData 'files').
    For each file:
      - attempt to fuzzy-match filename to a sourcing row (by name -> linkedinurl)
      - if matched, ensure record exists in 'process' table (auto-accept)
      - update process.cv
      - spawn analyze_cv_background(linkedinurl, bytes) for analysis
    Returns: { uploaded_count: int, errors: [ ... ] }
    """
    try:
        if 'files' not in request.files and not request.files:
            files = []
            for k in request.files:
                files.extend(request.files.getlist(k))
        else:
            files = request.files.getlist('files')

        if not files:
            return jsonify({"uploaded_count": 0, "errors": ["No files provided"]}), 400

        # Normalize list, filter allowed extensions
        allowed_ext = ('.pdf', '.doc', '.docx')
        to_process = [f for f in files if f and f.filename and f.filename.lower().endswith(allowed_ext)]
        rejected = [f.filename for f in files if f and f.filename and not f.filename.lower().endswith(allowed_ext)]

        import psycopg2
        from psycopg2 import sql
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()

        # Fetch context from sourcing
        cur.execute("SELECT id, name, linkedinurl, company, jobtitle, country, username, userid FROM sourcing WHERE name IS NOT NULL AND name != ''")
        candidates = cur.fetchall()  # list of tuples

        def normalize_name(s):
            # Remove common suffixes like " _ LinkedIn" before normalization
            s = (s or '').lower()
            # Remove file extension
            s = re.sub(r'\.(pdf|doc|docx)$', '', s, flags=re.IGNORECASE)
            # Remove " _ linkedin" suffix (common in LinkedIn profile PDFs)
            s = re.sub(r'\s*_\s*linkedin\s*$', '', s)
            # Remove all non-alphanumeric characters
            return re.sub(r'[^a-z0-9]', '', s)
        
        def clean_name_for_display(s):
            """Clean special characters and artifacts from names for display.
            Removes non-printable characters, special Unicode artifacts, and non-Latin characters.
            Preserves Latin letters (including accented characters like José, François) and common name punctuation."""
            if not s:
                return s
            
            # Define Unicode ranges for allowed characters
            ASCII_MAX = 127  # Standard ASCII (0-127)
            LATIN_EXTENDED_MAX = 591  # Covers Latin-1 Supplement + Latin Extended A/B
            
            # Remove non-printable characters and non-Latin Unicode characters
            # Keep only Latin letters (ASCII + Latin-1 Supplement + Latin Extended blocks),
            # spaces, hyphens, periods, apostrophes, commas
            # This filters out Korean (님), special artifacts (δïÿ), etc.
            cleaned = []
            for char in s:
                if not char.isprintable():
                    continue  # Skip non-printable characters
                
                char_code = ord(char)
                # Allow common punctuation (works in all ranges)
                if char in ' -.\',':
                    cleaned.append(char)
                # Allow ASCII letters (A-Z, a-z)
                elif char_code <= ASCII_MAX and char.isalpha():
                    cleaned.append(char)
                # Allow Latin-1 Supplement and Latin Extended letters (e.g., À, É, ñ)
                elif ASCII_MAX < char_code <= LATIN_EXTENDED_MAX and char.isalpha():
                    cleaned.append(char)
                # Reject everything else (Korean, Chinese, Arabic, special symbols, etc.)
            
            result = ''.join(cleaned)
            # Normalize multiple spaces to single space
            result = re.sub(r'\s+', ' ', result)
            return result.strip()

        candidate_map = {}
        # Map: normalized_name -> list of records
        # record: {id, name, linkedinurl, company, jobtitle, country, username, userid}
        for row in candidates:
            sid, cname, clink, comp, job, ctry, uname, uid = row
            norm = normalize_name(cname)
            if len(norm) < 3: continue
            # Clean name for display to remove special characters
            clean_cname = clean_name_for_display(cname)
            entry = {
                "id": sid, "name": clean_cname, "linkedinurl": clink,
                "company": comp, "jobtitle": job, "country": ctry,
                "username": uname, "userid": uid
            }
            candidate_map.setdefault(norm, []).append(entry)

        uploaded_count = 0
        errors = []
        uploaded_profiles = []  # Track successfully uploaded profiles
        did_insert_explicit_id = False

        # Check process columns
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='process'")
        process_cols = {r[0].lower() for r in cur.fetchall()}
        
        has_process_id = 'id' in process_cols
        has_role_tag = 'role_tag' in process_cols
        
        user_role_tags = {}
        
        for f in to_process:
            fname = f.filename or "unnamed"
            fname_norm = normalize_name(fname)
            matched_entry = None

            # Substring match strategy
            possible = []
            for norm_name, entries in candidate_map.items():
                if norm_name in fname_norm:
                    for e in entries:
                        possible.append((len(norm_name), e))
            if possible:
                possible.sort(key=lambda x: x[0], reverse=True)
                matched_entry = possible[0][1]

            try:
                if matched_entry:
                    file_bytes = f.read()
                    binary_cv = psycopg2.Binary(file_bytes)
                    m_link = matched_entry['linkedinurl']
                    sourcing_id = matched_entry['id']
                    
                    # Logic: Try to find existing process row to update (by ID or URL)
                    pid = None
                    
                    # 1. Try match by ID (preferred if safe)
                    if has_process_id and sourcing_id:
                        cur.execute("SELECT id FROM process WHERE id=%s", (sourcing_id,))
                        r_id = cur.fetchone()
                        if r_id: pid = r_id[0]
                        
                    # 2. Try match by LinkedIn URL if no ID match
                    if not pid:
                        cur.execute("SELECT id FROM process WHERE linkedinurl=%s", (m_link,))
                        r_link = cur.fetchone()
                        if r_link: pid = r_link[0]

                    if pid:
                        # Update existing record - also update name to clean any special characters
                        cleaned_name = matched_entry['name']  # Already cleaned from clean_name_for_display
                        cur.execute("UPDATE process SET cv=%s, name=%s WHERE id=%s", (binary_cv, cleaned_name, pid))
                        conn.commit()
                        uploaded_count += 1
                        threading.Thread(target=analyze_cv_background, args=(m_link, file_bytes)).start()
                    else:
                        # Insert new record into process
                        r_tag = ""
                        u_name = matched_entry['username']
                        if u_name:
                            if u_name not in user_role_tags:
                                cur.execute("SELECT role_tag FROM login WHERE username=%s", (u_name,))
                                rt = cur.fetchone()
                                user_role_tags[u_name] = rt[0] if rt else ""
                            r_tag = user_role_tags[u_name]

                        # Infer Geographic from sourcing country
                        src_country = matched_entry['country'] or ""
                        geo_val = _infer_region_from_country(src_country)

                        # Base columns
                        ins_cols = ["linkedinurl", "name", "company", "jobtitle", "country", "username", "userid", "cv"]
                        ins_vals = [m_link, matched_entry['name'], matched_entry['company'], matched_entry['jobtitle'], src_country, matched_entry['username'], matched_entry['userid'], binary_cv]
                        
                        # Add Geographic if column exists
                        if 'geographic' in process_cols and geo_val:
                            ins_cols.append("geographic")
                            ins_vals.append(geo_val)

                        if has_role_tag:
                            ins_cols.append("role_tag")
                            ins_vals.append(r_tag)
                        
                        # Attempt Insert with Explicit ID
                        inserted = False
                        if has_process_id and sourcing_id:
                            try:
                                cols_id = ins_cols + ["id"]
                                vals_id = ins_vals + [sourcing_id]
                                placeholders = ["%s"] * len(vals_id)
                                
                                q = sql.SQL("INSERT INTO process ({}) VALUES ({})").format(
                                    sql.SQL(", ").join(map(sql.Identifier, cols_id)),
                                    sql.SQL(", ").join(map(sql.SQL, placeholders))
                                )
                                cur.execute(q, vals_id)
                                conn.commit()
                                inserted = True
                                did_insert_explicit_id = True
                            except psycopg2.errors.UniqueViolation:
                                conn.rollback()
                                # Fallback to no-ID insert below
                            except Exception as e:
                                conn.rollback()
                                logger.warning(f"[Bulk Upload] ID Insert failed for {m_link}: {e}")
                        
                        # Fallback: Insert without ID (auto-serial)
                        if not inserted:
                            try:
                                placeholders = ["%s"] * len(ins_vals)
                                q = sql.SQL("INSERT INTO process ({}) VALUES ({}) RETURNING id").format(
                                    sql.SQL(", ").join(map(sql.Identifier, ins_cols)),
                                    sql.SQL(", ").join(map(sql.SQL, placeholders))
                                )
                                cur.execute(q, ins_vals)
                                conn.commit()
                                inserted = True
                            except psycopg2.errors.UniqueViolation:
                                conn.rollback()
                                # Race condition: Row created in meantime? Try Update
                                try:
                                    cur.execute("UPDATE process SET cv=%s WHERE linkedinurl=%s", (binary_cv, m_link))
                                    conn.commit()
                                    inserted = True
                                except Exception:
                                    conn.rollback()

                        if inserted:
                            uploaded_count += 1
                            uploaded_profiles.append(m_link)  # Track uploaded profile
                            threading.Thread(target=analyze_cv_background, args=(m_link, file_bytes)).start()
                        else:
                            errors.append(f"Failed to insert/update for {fname}")

                else:
                    errors.append(f"No sourcing match for file {fname}")
            except Exception as e:
                conn.rollback()
                errors.append(f"Failed to process {fname}: {e}")

        # Final Sequence Fix if we manually inserted any IDs
        if did_insert_explicit_id and has_process_id:
             try:
                 # Sync sequence to max(id) to avoid future collisions
                 cur.execute("SELECT setval(pg_get_serial_sequence('process', 'id'), (SELECT MAX(id) FROM process))")
                 conn.commit()
             except Exception as e_seq:
                 conn.rollback()
                 # This can happen if id column is not SERIAL, safe to ignore
                 # logger.warning(f"[Bulk Upload] Sequence sync warning: {e_seq}")

        cur.close(); conn.close()

        result = {"uploaded_count": uploaded_count, "errors": errors, "uploaded_profiles": uploaded_profiles}
        if rejected:
            result["rejected_files"] = rejected

        return jsonify(result), 200

    except Exception as e:
        logger.exception("[Upload Multiple CVs] failed")
        return jsonify({"uploaded_count": 0, "errors": [str(e)]}), 500

@app.get("/process/download_cv")
def process_download_cv():
    linkedin = (request.args.get("linkedin") or "").strip()
    if not linkedin:
        return "LinkedIn URL required", 400

    linkedin_norm = linkedin.split('?')[0].rstrip('/')
    linkedin_path = _normalize_linkedin_to_path(linkedin)
    
    def _standardize_host(url_str: str) -> str:
        s = (url_str or "").strip()
        if not s: return s
        return re.sub(r'^https?://[^/]+', 'https://www.linkedin.com', s, flags=re.I)

    linkedin_norm_www = _standardize_host(linkedin_norm)

    try:
        import psycopg2
        from psycopg2 import sql
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()

        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='process' AND column_name='cv'")
        if not cur.fetchone():
             cur.close(); conn.close()
             return "CV column not found in database", 404

        row = None
        if linkedin_path:
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='process' AND column_name='normalized_linkedin'")
            if cur.fetchone():
                cur.execute("SELECT cv, name FROM process WHERE normalized_linkedin = %s AND cv IS NOT NULL LIMIT 1", (linkedin_path,))
                row = cur.fetchone()

        if not row:
            cur.execute("SELECT cv, name FROM process WHERE linkedinurl = %s AND cv IS NOT NULL LIMIT 1", (linkedin_norm,))
            row = cur.fetchone()

        if not row and linkedin_norm_www != linkedin_norm:
             cur.execute("SELECT cv, name FROM process WHERE linkedinurl = %s AND cv IS NOT NULL LIMIT 1", (linkedin_norm_www,))
             row = cur.fetchone()
        
        if not row and linkedin_path:
             cur.execute("SELECT cv, name FROM process WHERE LOWER(linkedinurl) LIKE %s AND cv IS NOT NULL LIMIT 1", (f"%{linkedin_path.lower()}",))
             row = cur.fetchone()

        cur.close(); conn.close()

        if row and row[0]:
            pdf_data = row[0]
            candidate_name = (row[1] or "candidate").strip().replace(" ", "_")
            candidate_name = re.sub(r'[^a-zA-Z0-9_]', '', candidate_name)
            
            from flask import make_response
            response = make_response(bytes(pdf_data))
            response.headers.set('Content-Type', 'application/pdf')
            response.headers.set('Content-Disposition', f'attachment; filename="{candidate_name}_CV.pdf"')
            return response
        else:
            return "CV not found", 404

    except Exception as e:
        logger.error(f"[Download CV] {e}")
        return f"Error: {str(e)}", 500

def _strip_level_suffix(seniority: str) -> str:
    """
    Strip '-level' suffix from seniority for database storage.
    Example: 'Mid-level' -> 'Mid', 'Senior-level' -> 'Senior'
    
    This ensures DB stores clean values without '-level' suffix.
    UI layer should add 'level' back when displaying.
    """
    if not seniority:
        return ""
    # Remove '-level' suffix (lowercase only, as normalized values use consistent casing)
    return re.sub(r'-level$', '', seniority).strip()

def _normalize_seniority_to_8_levels(seniority_text: str, total_experience_years=None) -> str:
    """
    Normalize freeform seniority to one of the 8 specified levels:
    1. Junior-level
    2. Mid-level
    3. Senior-level
    4. Lead-level
    5. Manager-level
    6. Expert-level
    7. Director-level
    8. Executive-level
    
    Rules:
    - Map based on keywords in the seniority text
    - Use experience years as fallback if provided
    - Return empty string if cannot determine
    """
    if not seniority_text:
        # Fallback to experience-based mapping
        if total_experience_years is not None:
            try:
                years = float(total_experience_years)
                if years < 2:
                    return "Junior-level"
                elif years < 5:
                    return "Mid-level"
                elif years < 8:
                    return "Senior-level"
                elif years < 12:
                    return "Lead-level"
                else:
                    return "Expert-level"
            except Exception:
                pass
        return ""
    
    s = str(seniority_text).strip().lower()
    
    # Exact matches first (case-insensitive)
    exact_matches = {
        "junior-level": "Junior-level",
        "mid-level": "Mid-level",
        "senior-level": "Senior-level",
        "lead-level": "Lead-level",
        "manager-level": "Manager-level",
        "expert-level": "Expert-level",
        "director-level": "Director-level",
        "executive-level": "Executive-level",
    }
    if s in exact_matches:
        return exact_matches[s]
    
    # Executive level - highest
    executive_keywords = ["executive", "ceo", "cto", "cfo", "coo", "cxo", "chief", "president", "vp", "vice president", "c-level"]
    for keyword in executive_keywords:
        if keyword in s:
            return "Executive-level"
    
    # Director level
    director_keywords = ["director", "head of", "group director"]
    for keyword in director_keywords:
        if keyword in s:
            return "Director-level"
    
    # Expert level - principal, staff, distinguished, fellow
    expert_keywords = ["expert", "principal", "staff", "distinguished", "fellow", "architect", "specialist"]
    for keyword in expert_keywords:
        if keyword in s:
            return "Expert-level"
    
    # Manager level - check before Lead to avoid "Senior Manager" being classified as Lead
    manager_keywords = ["manager", "mgr", "supervisor", "team lead"]
    for keyword in manager_keywords:
        if keyword in s:
            return "Manager-level"
    
    # Lead level - senior/lead but not manager
    lead_keywords = ["lead", "senior"]
    for keyword in lead_keywords:
        if keyword in s:
            return "Lead-level"
    
    # Mid level
    mid_keywords = ["mid", "intermediate", "associate"]
    for keyword in mid_keywords:
        if keyword in s:
            return "Mid-level"
    
    # Junior level
    junior_keywords = ["junior", "entry", "trainee", "intern", "graduate", "jr"]
    for keyword in junior_keywords:
        if keyword in s:
            return "Junior-level"
    
    # Fallback to experience-based mapping
    if total_experience_years is not None:
        try:
            years = float(total_experience_years)
            if years < 2:
                return "Junior-level"
            elif years < 5:
                return "Mid-level"
            elif years < 8:
                return "Senior-level"
            elif years < 12:
                return "Lead-level"
            else:
                return "Expert-level"
        except Exception:
            pass
    
    # Default to empty if cannot determine
    return ""

def _is_internship_role(job_title):
    """
    Check if a job title indicates an internship role.
    Returns True if the title contains 'intern' or 'internship' (case-insensitive).
    """
    if not job_title:
        return False
    return bool(re.search(r'\bintern\b|\binternship\b', job_title, re.IGNORECASE))

def _normalize_company_name(company_name):
    """
    Normalize company name for duplicate detection.
    Removes common suffixes and converts to lowercase for consistent matching.
    
    Note: Currently handles common US/UK company suffixes. International suffixes
    (GmbH, S.A., AG, etc.) are not normalized but can be added if needed.
    
    Returns normalized company name or None if input is empty.
    """
    if not company_name:
        return None
    
    # Convert to lowercase and remove common company suffixes
    normalized = company_name.lower().strip()
    normalized = re.sub(r'\s+(inc\.?|llc\.?|ltd\.?|corp\.?|corporation|company|co\.?|limited|group|plc)$', '', normalized, flags=re.IGNORECASE)
    normalized = normalized.strip()
    
    return normalized if normalized else None

def _recalculate_tenure_and_experience(experience_list):
    """
    Recalculate total_experience_years and tenure from experience list.
    
    This function enforces business rules:
    1. Internship roles are excluded from total_experience_years calculation
    2. Same company (regardless of job title) counts as ONE employer for tenure
    3. Internship roles are excluded from employer count for tenure
    4. Overlapping periods at the same company are merged (e.g., two roles at same company with same dates)
    
    Args:
        experience_list: List of experience strings in format "Job Title, Company, StartYear to EndYear|present"
                        or "Job Title, Company, Month YYYY to Month YYYY|present"
    
    Returns:
        dict: {
            "total_experience_years": float,  # Total years excluding internships and overlaps
            "tenure": float,  # Average tenure per unique employer (excluding internships)
            "employer_count": int,  # Number of unique employers (excluding internships)
            "total_roles": int  # Total number of roles (including internships)
        }
    """
    if not experience_list or not isinstance(experience_list, list):
        return {
            "total_experience_years": 0.0,
            "tenure": 0.0,
            "employer_count": 0,
            "total_roles": 0
        }
    
    current_year = datetime.now().year  # Note: Uses year only, timezone not critical for year calculation
    # Track periods per employer: company -> list of (start_year, end_year) tuples
    employer_periods = {}
    total_roles = len(experience_list)
    
    for entry in experience_list:
        if not entry or not isinstance(entry, str):
            continue
        
        # Expected format: "Job Title, Company, StartYear to EndYear|present"
        # or "Job Title, Company, Month YYYY to Month YYYY|present"
        parts = [p.strip() for p in entry.split(',')]
        
        if len(parts) < 3:
            # Cannot reliably parse, skip this entry
            continue
        
        job_title = parts[0]
        company = parts[1]
        duration_str = parts[2]
        
        # Check if this is an internship role
        is_intern = _is_internship_role(job_title)
        
        # Parse duration: Improved regex to handle month names
        # Matches: "Aug 2020 to present", "2020 to 2021", "Jan 2019 to Dec 2020", etc.
        duration_match = re.search(r'(?:\w+\s+)?(\d{4})\s*(?:to|[-–—])\s*(?:\w+\s+)?(present|\d{4})', duration_str, re.IGNORECASE)
        
        if not duration_match:
            continue
        
        start_year = int(duration_match.group(1))
        end_part = duration_match.group(2).lower()
        
        if end_part == 'present':
            end_year = current_year
        else:
            end_year = int(end_part)
        
        # Calculate duration in years
        # Note: This calculates full years only (2020 to 2021 = 1 year)
        # Partial years are not accounted for due to limited date precision in CV format
        if end_year >= start_year and not is_intern:
            # Track periods per employer to handle overlaps
            normalized_company = _normalize_company_name(company)
            if normalized_company:
                if normalized_company not in employer_periods:
                    employer_periods[normalized_company] = []
                employer_periods[normalized_company].append((start_year, end_year))
    
    # Merge overlapping/duplicate periods for each employer
    total_experience = 0.0
    for company, periods in employer_periods.items():
        # Sort periods by start year
        periods.sort()
        
        # Merge overlapping or adjacent periods
        merged = []
        for start, end in periods:
            if merged and start <= merged[-1][1]:
                # Overlapping or adjacent - extend the last period
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
            else:
                # Non-overlapping - add new period
                merged.append((start, end))
        
        # Calculate total duration for this employer after merging overlaps
        company_duration = sum(end - start for start, end in merged)
        total_experience += company_duration
    
    # Calculate tenure
    employer_count = len(employer_periods)
    
    if employer_count > 0:
        tenure = total_experience / employer_count
        tenure = round(tenure, 1)
    else:
        tenure = 0.0
    
    return {
        "total_experience_years": round(total_experience, 1),
        "tenure": tenure,
        "employer_count": employer_count,
        "total_roles": total_roles
    }

def _analyze_cv_bytes_sync(pdf_bytes):
    """
    Synchronous helper to parse PDF bytes via Gemini.
    Supports translation for non-English CVs before analysis.
    
    SOURCE OF TRUTH: CV Column
    - Gemini exclusively references the cv column in the process table (Postgres)
    - Parse employment history strictly in format: "Job Title, Company, StartYear to EndYear" 
      OR "Job Title, Company, StartYear to present" (for current positions)
    
    Returns structured dict or None.
    """
    if not (genai and GEMINI_API_KEY):
        return None
    import io
    try:
        from pypdf import PdfReader
    except ImportError:
        logger.warning("[CV Sync] pypdf not installed")
        return None

    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        text = ""
        for page in reader.pages:
            t = page.extract_text()
            if t: text += t + "\n"
        
        if not text.strip(): return None

        # Detect language and translate if non-English
        original_text = text
        try:
            # Use Gemini to detect if CV is in non-English language
            model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
            lang_detect_prompt = (
                "Analyze this text and determine if it's primarily in English or another language.\n"
                "Return JSON: {\"language\": \"<language_code>\", \"is_english\": true/false}\n"
                f"Text sample (first {LANG_DETECTION_SAMPLE_LENGTH} chars): {text[:LANG_DETECTION_SAMPLE_LENGTH]}"
            )
            lang_resp = model.generate_content(lang_detect_prompt)
            lang_obj = _extract_json_object((lang_resp.text or "").strip())
            
            if lang_obj and not lang_obj.get("is_english", True):
                # Translate to English using the translation pipeline
                logger.info(f"[CV Sync] Detected non-English CV, translating from {lang_obj.get('language', 'unknown')}")
                source_lang = lang_obj.get("language", "")
                # Use translate_text_pipeline for translation
                # Note: Translation is limited to CV_TRANSLATION_MAX_CHARS to balance API limits and processing time
                # If CV exceeds this limit, later portions may not be translated but will still be analyzed
                translated_result = translate_text_pipeline(text[:CV_TRANSLATION_MAX_CHARS], "english", source_lang)
                if translated_result and translated_result.get("translated"):
                    text = translated_result["translated"]
                    logger.info(f"[CV Sync] Translation completed using {translated_result.get('engine', 'unknown')}")
                    if len(original_text) > CV_TRANSLATION_MAX_CHARS:
                        logger.warning(f"[CV Sync] CV truncated for translation ({len(original_text)} > {CV_TRANSLATION_MAX_CHARS} chars). Later portions analyzed in original language.")
        except Exception as e:
            logger.warning(f"[CV Sync] Language detection/translation failed, proceeding with original: {e}")
            text = original_text

        model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
        prompt = (
            "SYSTEM:\n"
            "SOURCE OF TRUTH: You are analyzing a CV from the cv column in the process table (Postgres).\n"
            "This CV is the EXCLUSIVE source for all information. Do not infer or add skills not explicitly in the CV.\n\n"
            "Analyze the following CV text.\n"
            "Return STRICT JSON only with these keys:\n"
            "{\n"
            "  \"skillset\": [\"Skill1\", \"Skill2\", ...],\n"
            "  \"total_experience_years\": <number>,\n"
            "  \"tenure\": <number>,\n"
            "  \"experience\": [\"Job Title, Company, StartYear to EndYear|present\", ...],\n"
            "  \"education\": [\"University Name, Degree Type, Discipline\", ...],\n"
            "  \"product_list\": [\"Product1\", \"Product2\", ...],\n"
            "  \"company\": \"<Current/Latest Company Name>\",\n"
            "  \"job_title\": \"<Current/Latest Job Title>\",\n"
            "  \"country\": \"<Country or Location>\",\n"
            "  \"seniority\": \"<Seniority>\",\n"
            "  \"sector\": \"<Sector>\",\n"
            "  \"job_family\": \"<Job Family>\"\n"
            "}\n"
            "Rules:\n"
            "1. Skillset: Extract ONLY skills explicitly mentioned in the CV. Max 15 items. Do not infer or add skills.\n"
            "2. Total Experience: Calculate sum of all employment durations in years, EXCLUDING internships and intern positions. Only count full-time, part-time, and regular employment. Return a number rounded to 1 decimal place.\n"
            "3. Tenure: Calculate average tenure using the total_experience_years you calculated in step 2. IMPORTANT: Treat repeated employment at the same company as ONE employer. DO NOT count internships or intern positions in the employer count. Formula: total_experience_years / number of UNIQUE employers (excluding internships). Return a number rounded to 1 decimal place.\n"
            "   Example 1: If someone worked at 'Google' from 2015-2017 (2 years) and again from 2019-2021 (2 years), total_experience_years=4, unique employers=1 (Google), so tenure=4/1=4.0 years.\n"
            "   Example 2: If someone had 'Software Engineer at Google' for 3 years, 'Data Scientist at Amazon' for 2 years, and 'Intern at Microsoft' for 1 year, total_experience_years=5 (intern excluded), unique employers=2 (Google and Amazon; Microsoft intern excluded), so tenure=5/2=2.5 years.\n"
            "4. Experience: STRICTLY parse employment history in format 'Job Title, Company, StartYear to EndYear'. If current job, use 'present' instead of EndYear. MANDATORY: Include EVERY SINGLE employment entry from the CV - do not omit any job.\n"
            "5. Education: Format each entry as 'University Name, Degree Type, Discipline'. MANDATORY: Include ALL educational qualifications - degrees, certifications, diplomas. Do not omit any.\n"
            "6. Products: Identify the LATEST company in the employment history. Then, identify its full range of products or services.\n"
            "7. Identify CURRENT employment details (company, job_title, country).\n"
            "8. Infer Seniority, Sector, and Job Family based on the profile.\n"
            "9. CRITICAL REQUIREMENT: Parse COMPLETE employment history without ANY omissions. Every job mentioned must be in the experience array.\n"
            "10. CRITICAL REQUIREMENT: Parse COMPLETE education history without ANY omissions. Every degree/certification must be in the education array.\n"
            "11. IMPORTANT: If a field value cannot be determined, return an empty string \"\" instead of 'unknown', 'N/A', or similar placeholders.\n"
            "12. No commentary, no extra keys. Output only valid JSON.\n\n"
            f"CV TEXT:\n{text[:CV_ANALYSIS_MAX_CHARS]}\n\nJSON:"
        )
        resp = model.generate_content(prompt)
        raw = (resp.text or "").strip()
        obj = _extract_json_object(raw)
        
        # Clean company and job_title
        if obj:
             for field in ["company", "job_title", "seniority", "sector", "job_family", "country"]:
                 if obj.get(field):
                     # Remove quotes and surrounding whitespace
                     value = re.sub(r'^[\s"\'`]+|[\s"\'`]+$', '', str(obj[field])).strip()
                     # Replace "unknown" or variations with empty string
                     if value.lower() in ['unknown', 'n/a', 'na', 'not specified', 'not available']:
                         obj[field] = ''
                     else:
                         obj[field] = value
             
             # Normalize seniority to one of the 8 specified levels, then strip '-level' suffix for DB storage
             if obj.get('seniority'):
                 normalized_seniority = _normalize_seniority_to_8_levels(obj['seniority'], obj.get('total_experience_years'))
                 obj['seniority'] = _strip_level_suffix(normalized_seniority)
             
             # Post-process: Recalculate tenure and total_experience_years from experience list
             # This ensures consistent application of business rules regardless of Gemini's interpretation
             experience_list = obj.get('experience', [])
             if experience_list and isinstance(experience_list, list):
                 # Store original Gemini values before recalculation
                 gemini_total_exp = obj.get('total_experience_years', 0)
                 gemini_tenure = obj.get('tenure', 0)
                 
                 recalc = _recalculate_tenure_and_experience(experience_list)
                 
                 # Update the values with recalculated ones
                 obj['total_experience_years'] = recalc['total_experience_years']
                 obj['tenure'] = recalc['tenure']
                 
                 # Log if there's a significant difference from Gemini's calculation
                 if abs(recalc['total_experience_years'] - float(gemini_total_exp or 0)) > 0.5:
                     logger.info(f"[CV Sync] Recalculated total_experience_years: {recalc['total_experience_years']} (Gemini: {gemini_total_exp})")
                 if abs(recalc['tenure'] - float(gemini_tenure or 0)) > 0.5:
                     logger.info(f"[CV Sync] Recalculated tenure: {recalc['tenure']} (Gemini: {gemini_tenure}, employers: {recalc['employer_count']})")

        return obj
    except Exception as e:
        logger.warning(f"[CV Sync] Analysis failed: {e}")
        return None

def _core_assess_profile(data):
    """
    Core assessment logic separated from endpoint wrapper.
    Args:
        data (dict): contains keys:
            job_title, role_tag, company, country, seniority, sector,
            experience_text, target_skills (list), candidate_skills (list),
            custom_weights (dict, optional), linkedinurl (optional),
            assessment_level (str, optional): 'L1' or 'L2',
            tenure (float, optional): Average tenure per employer
    Returns:
        dict: assessment result object
    """
    job_title = data.get("job_title", "")
    role_tag = data.get("role_tag", "")
    company = data.get("company", "")
    country = data.get("country", "")
    seniority = data.get("seniority", "")
    sector = data.get("sector", "")
    experience_text = data.get("experience_text", "")
    target_skills = data.get("target_skills", []) or []
    candidate_skills = data.get("candidate_skills", []) or []
    process_skills = data.get("process_skills", []) or []
    custom_weights = data.get("custom_weights", {}) or {}
    linkedinurl = data.get("linkedinurl", "")
    assessment_level = data.get("assessment_level", "L1").upper()  # L1 or L2
    tenure = data.get("tenure")  # Average tenure per employer
    vskillset_results = data.get("vskillset_results")  # vskillset inference results for scoring
    product = data.get("product", []) or []  # Product list from CV analysis
    
    # Log assessment inputs for debugging
    logger.info(f"[CORE_ASSESS] LinkedIn: {linkedinurl}")
    logger.info(f"[CORE_ASSESS] Candidate skills count: {len(candidate_skills)}, first 10: {candidate_skills[:10] if candidate_skills else []}")
    logger.info(f"[CORE_ASSESS] Target skills count: {len(target_skills)}, first 10: {target_skills[:10] if target_skills else []}")

    # Base weights configuration
    # If custom weights are provided and valid, use them. Otherwise default.
    default_weights = {
        "jobtitle_role_tag": 30.0,
        "skillset": 20.0,
        "tenure": 15.0,  # Average tenure per employer
        "country": 10.0,
        "company": 10.0,
        "product": 5.0,  # Product experience
        "seniority": 5.0,
        "sector": 5.0
    }
    
    # Map frontend keys to internal keys
    weights = default_weights.copy()
    if custom_weights:
        try:
            # Safe parsing helper
            def _get_weight(keys, default_val):
                if isinstance(keys, str): keys = [keys]
                for k in keys:
                    if k in custom_weights:
                        return float(custom_weights[k])
                return float(default_val)

            cw = {
                "jobtitle_role_tag": _get_weight(["jobtitle_role_tag", "job_title", "jobTitle"], 30.0),
                "skillset": _get_weight(["skillset", "skills"], 20.0),
                "tenure": _get_weight(["tenure", "avg_tenure"], 15.0),
                "country": _get_weight(["country", "location"], 10.0),
                "company": _get_weight(["company"], 10.0),
                "product": _get_weight(["product"], 5.0),
                "seniority": _get_weight(["seniority"], 5.0),
                "sector": _get_weight(["sector", "industry"], 5.0)
            }
            # Verify sum roughly 100
            total_w = sum(cw.values())
            if 99.0 <= total_w <= 101.0:
                weights = cw
            else:
                pass # logger.warning(f"[Assess] Custom weights sum {total_w} out of range (99-101). Using defaults.")
        except Exception as e:
            pass # logger.warning(f"[Assess] Invalid custom weights: {e}")

    
    # Identify active criteria
    active_criteria = []
    if job_title and role_tag:
        active_criteria.append("jobtitle_role_tag")
    
    if country: active_criteria.append("country")
    if company: active_criteria.append("company")
    if seniority: active_criteria.append("seniority")
    if sector: active_criteria.append("sector")
    # Product active if we have product data
    if product and len(product) > 0:
        active_criteria.append("product")
    # Tenure active if value is provided and valid
    if tenure is not None and tenure != "":
        try:
            float(tenure)  # Validate it's numeric
            active_criteria.append("tenure")
        except (ValueError, TypeError):
            pass
    # Skillset active if we have target skills AND (candidate skills OR experience text to infer from)
    if target_skills and (candidate_skills or experience_text):
        active_criteria.append("skillset")
    
    if not active_criteria:
        return {
            "assessment_level": "Level 1",
            "is_level2": False,
            "stars": 0,
            "total_score": "0%",
            "criteria": {},
            "comments": "No data available for assessment."
        }

    # Determine Level 2 status
    is_level2 = False
    if "jobtitle_role_tag" in active_criteria and "skillset" in active_criteria and len(active_criteria) >= 5:
        is_level2 = True

    # Distribute missing weights evenly
    total_weight_target = 100.0
    active_base_sum = sum(weights[c] for c in active_criteria)
    
    if active_base_sum > 0:
        missing_weight = total_weight_target - active_base_sum
        bonus_per_active = missing_weight / len(active_criteria)
        
        final_weights = {}
        for c in active_criteria:
            final_weights[c] = weights[c] + bonus_per_active
    else:
        # Fallback if active criteria base weights sum to 0 (unlikely unless configured 0)
        final_weights = {c: (100.0 / len(active_criteria)) for c in active_criteria}

    assessment_results = {}
    
    if genai and GEMINI_API_KEY:
        try:
            # Set temperature=0 for deterministic, non-creative output
            generation_config = {
                "temperature": 0,
                "top_p": 1,
                "top_k": 1,
            }
            model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL, generation_config=generation_config)
            
            # Prepare skills context
            skills_context = ""
            if "skillset" in active_criteria:
                skills_context = (
                    f"Target Skills (required): {', '.join(target_skills)}\n"
                    f"Candidate Skills (found): {', '.join(candidate_skills) if candidate_skills else '(Analyze from experience text)'}\n"
                )
                if process_skills:
                    skills_context += f"Process-provided Skills (hints): {', '.join(process_skills)}\n"

            # Generate appropriate prompt based on assessment level
            if assessment_level == "L2":
                # L2 PROMPT: Contextual inference allowed
                prompt = (
                    "SYSTEM: You are an expert sourcing assessor performing LEVEL 2 ASSESSMENT with contextual inference.\n"
                    "Your task: Assess skills using BOTH explicit evidence AND conservative contextual inference.\n\n"
                    "LEVEL 2 SKILLSET ASSESSMENT RULES:\n"
                    "1. CONFIRMED SKILLS: Skills explicitly mentioned in experience text (quote exact phrase)\n"
                    "2. CONTEXTUAL INFERENCE: Apply high-probability inference based on:\n"
                    "   - Job Title + Company combination (e.g., Game Programmer at Epic Games → Unreal Engine, C++)\n"
                    "   - Industry domain knowledge (e.g., Senior Java Developer → Spring Framework, Maven)\n"
                    "   - Company's core products (e.g., AWS Engineer at Amazon → EC2, S3, Lambda)\n"
                    "   - Sector/Job Family context when direct skills cannot be determined\n"
                    "3. MINIMUM SKILLSET: Generate at least 10 distinct skills (combining confirmed and inferred) related to the profile's job title and company\n"
                    "4. FALLBACK STRATEGY: If direct skillsets cannot be generated, suggest skills based on:\n"
                    "   - Sector (industry-specific skills)\n"
                    "   - Job Family (role-specific skills)\n"
                    "   - Product/Company domain (company/product-specific skills)\n"
                    "5. CONSERVATIVE APPROACH: Only infer skills with >80% probability given the context\n"
                    "6. EVIDENCE REQUIREMENT: For inferred skills, cite the contextual basis (job title + company)\n"
                    "7. LABEL CLEARLY: Mark skills as 'confirmed' (explicit) or 'inferred' (contextual)\n\n"
                    "EXAMPLES OF VALID L2 INFERENCE:\n"
                    "- 'Game Programmer at Epic Games' → Infer: Unreal Engine (Epic's core product), C++ (game engine requirement)\n"
                    "- 'Machine Learning Engineer at Google' → Infer: TensorFlow (Google's ML framework), Python\n"
                    "- 'iOS Developer at Apple' → Infer: Swift, Xcode, iOS SDK\n\n"
                    "For each field present, classify as:\n"
                    "- 'match' (explicit evidence OR high-confidence inference)\n"
                    "- 'related' (partial evidence OR moderate inference)\n"
                    "- 'unrelated' (no evidence and cannot infer)\n\n"
                    "INPUT:\n"
                    f"Target Role Tag: {role_tag}\n"
                    f"Candidate Job Title: {job_title}\n"
                    f"Candidate Company: {company}\n"
                    f"Candidate Country: {country}\n"
                    f"Candidate Seniority: {seniority}\n"
                    f"Candidate Sector: {sector}\n"
                    f"{skills_context}"
                    f"Experience Text:\n{experience_text[:1000]}...\n\n"
                    "OUTPUT JSON with keys: 'jobtitle_role_tag', 'company', 'country', 'seniority', 'sector', 'skillset'.\n"
                    "Each value must be object: {{ \"status\": \"match\"|\"related\"|\"unrelated\", \"comment\": \"...\", \"evidence\": \"quoted text or contextual basis\" }}\n"
                    "If a field is empty in input, output status: \"not_assessed\".\n"
                    "SKILLSET ASSESSMENT:\n"
                    "- Match = Most/all target skills found (confirmed OR inferred with high confidence)\n"
                    "- Related = Some target skills found (confirmed OR inferred)\n"
                    "- Unrelated = Few/no target skills found or inferred\n"
                    "- Include 'confirmed_skills' array (explicitly mentioned)\n"
                    "- Include 'inferred_skills' array (contextually inferred with basis)\n"
                    "- Total combined skillset (confirmed + inferred) should contain at least 10 skills when possible\n"
                    "- Include 'missing_skills' array (neither confirmed nor inferable)\n"
                )
            else:
                # L1 PROMPT: Strictly extractive, NO inference (existing prompt)
                prompt = (
                    "SYSTEM: You are a strict, evidence-based sourcing assessor performing LEVEL 1 ASSESSMENT.\n"
                    "Your task is EXTRACTIVE ONLY - confirm skills ONLY if they appear verbatim or as clear synonyms in the parsed CV experience text.\n"
                    "DO NOT hallucinate or infer skills without explicit evidence from the CV.\n\n"
                    "LEVEL 1 SKILLSET ASSESSMENT RULES:\n"
                    "1. ONLY confirm a skill if you can quote the exact phrase or a direct synonym from the parsed CV experience text\n"
                    "2. Use word-boundary matching - 'Python' in text matches target skill 'Python'\n"
                    "3. If a target skill is NOT found in parsed CV experience text, mark as 'unrelated' - DO NOT guess\n"
                    "4. For each confirmed skill, provide the quoted snippet from experience as evidence\n"
                    "5. NO INFERENCE ALLOWED - only explicit mentions from parsed CV data count\n"
                    "6. ONLY use parsed information from the CV column in the database\n\n"
                    "For each field present, classify as:\n"
                    "- 'match' (exact/strong evidence with quote)\n"
                    "- 'related' (partial evidence with quote)\n"
                    "- 'unrelated' (no evidence found)\n\n"
                    "INPUT:\n"
                    f"Target Role Tag: {role_tag}\n"
                    f"Candidate Job Title: {job_title}\n"
                    f"Candidate Company: {company}\n"
                    f"Candidate Country: {country}\n"
                    f"Candidate Seniority: {seniority}\n"
                    f"Candidate Sector: {sector}\n"
                    f"{skills_context}"
                    f"Parsed CV Experience Text (ONLY source of truth for skills):\n{experience_text[:1000]}...\n\n"
                    "OUTPUT JSON with keys: 'jobtitle_role_tag', 'company', 'country', 'seniority', 'sector', 'skillset'.\n"
                    "Each value must be object: {{ \"status\": \"match\"|\"related\"|\"unrelated\", \"comment\": \"...\", \"evidence\": \"quoted text or empty\" }}\n"
                    "If a field is empty in input, output status: \"not_assessed\".\n"
                    "SKILLSET ASSESSMENT:\n"
                    "- Match = ALL target skills found with evidence from parsed CV (100%)\n"
                    "- Related = SOME target skills found with evidence from parsed CV (50%)\n"
                    "- Unrelated = NO or FEW target skills found in parsed CV (0%)\n"
                    "- Include 'confirmed_skills' array with skills that have evidence from parsed CV\n"
                    "- Include 'missing_skills' array with skills without evidence in parsed CV\n"
                )
            resp = model.generate_content(prompt)
            raw = (resp.text or "").strip()
            parsed = _extract_json_object(raw)
            if isinstance(parsed, dict):
                assessment_results = parsed
                
                # POST-VALIDATION: Validate extracted skills against source text using word-boundary regex
                if "skillset" in active_criteria and "skillset" in assessment_results:
                    skillset_result = assessment_results["skillset"]
                    confirmed_skills = []
                    inferred_skills = []
                    missing_skills = []
                    
                    exp_lower = (experience_text or "").lower()
                    
                    # Guard against zero target skills
                    if not target_skills or len(target_skills) == 0:
                        assessment_results["skillset"] = {
                            "status": "not_assessed",
                            "comment": "No target skills provided",
                            "confirmed_skills": [],
                            "inferred_skills": [],
                            "missing_skills": [],
                            "evidence": "0/0 skills confirmed"
                        }
                    else:
                        # Create lowercased Gemini confirmed list once for efficiency
                        # Guard: Ensure we only process string values (Gemini might return dicts)
                        gemini_confirmed_raw = skillset_result.get("confirmed_skills", [])
                        gemini_confirmed = [s for s in gemini_confirmed_raw if isinstance(s, str)]
                        gemini_confirmed_lower = [s.lower() for s in gemini_confirmed]
                        
                        gemini_inferred_raw = skillset_result.get("inferred_skills", [])
                        gemini_inferred = [s for s in gemini_inferred_raw if isinstance(s, str)] if gemini_inferred_raw else []
                        gemini_inferred_lower = [s.lower() for s in gemini_inferred] if gemini_inferred else []
                        
                        for target_skill in target_skills:
                            # Use word boundary regex to find exact matches
                            pattern = r'\b' + re.escape(target_skill.lower()) + r'\b'
                            if re.search(pattern, exp_lower):
                                confirmed_skills.append(target_skill)
                            else:
                                # Check if Gemini claimed it was found
                                if target_skill in gemini_confirmed or target_skill.lower() in gemini_confirmed_lower:
                                    # Gemini said it found it, but we can't verify - mark as inferred
                                    inferred_skills.append(target_skill)
                                # L2: Also accept Gemini's inferred skills (contextual inference)
                                elif assessment_level == "L2" and gemini_inferred and (target_skill in gemini_inferred or target_skill.lower() in gemini_inferred_lower):
                                    inferred_skills.append(target_skill)
                                else:
                                    missing_skills.append(target_skill)
                        
                        # Update skillset result with validated sets
                        total_skills = len(target_skills)
                        confirmed_count = len(confirmed_skills)
                        inferred_count = len(inferred_skills)
                        
                        # Recalculate status based on assessment level
                        if assessment_level == "L2":
                            # L2: Count both confirmed and inferred skills
                            total_found = confirmed_count + inferred_count
                            if total_found >= total_skills * 0.75:  # 75%+ found (confirmed OR inferred)
                                status = "match"
                            elif total_found >= total_skills * 0.4:  # 40%+ found
                                status = "related"
                            else:
                                status = "unrelated"
                        else:
                            # L1: Count only confirmed skills (strict)
                            if confirmed_count >= total_skills * 0.8:  # 80%+ confirmed
                                status = "match"
                            elif confirmed_count >= total_skills * 0.4:  # 40%+ confirmed
                                status = "related"
                            else:
                                status = "unrelated"
                        
                        # Update comment with validation results
                        comment_parts = []
                        if confirmed_skills:
                            comment_parts.append(f"Confirmed: {', '.join(confirmed_skills[:3])}{'...' if len(confirmed_skills) > 3 else ''}")
                        if inferred_skills:
                            label = "Inferred (contextual)" if assessment_level == "L2" else "Inferred (unverified)"
                            comment_parts.append(f"{label}: {', '.join(inferred_skills[:2])}{'...' if len(inferred_skills) > 2 else ''}")
                        if missing_skills:
                            comment_parts.append(f"Missing: {', '.join(missing_skills[:3])}{'...' if len(missing_skills) > 3 else ''}")
                        
                        evidence_text = f"{confirmed_count}/{total_skills} confirmed"
                        if assessment_level == "L2" and inferred_count > 0:
                            evidence_text += f", {inferred_count} inferred"
                        
                        assessment_results["skillset"] = {
                            "status": status,
                            "comment": " | ".join(comment_parts) if comment_parts else "No skills validated",
                            "confirmed_skills": confirmed_skills,
                            "inferred_skills": inferred_skills,
                            "missing_skills": missing_skills,
                            "evidence": evidence_text
                        }
        except Exception as e:
            logger.warning(f"[Gemini Assess Core] {e}")
            
    # Fallback heuristics
    def heuristic_status(key, val, target):
        if not val: return "not_assessed", ""
        v = str(val).lower(); t = str(target).lower()
        if t in v or v in t: return "match", "Heuristic match"
        return "related", "Heuristic related" 

    def skill_heuristic(targets, candidates, exp_text):
        """
        Compare skillset with jskillset using ratio-based scoring.
        Returns (status, comment, match_count, total_count, match_ratio) tuple.
        The match_ratio is used for weighted scoring calculation.
        """
        if not targets: 
            return "not_assessed", "", 0, 0, 0.0
        
        t_set = set(t.lower() for t in targets)
        c_source = (candidates or []) + re.split(r'\W+', (exp_text or "").lower())
        c_set = set(c.lower() for c in c_source if c)
        
        matches = t_set.intersection(c_set)
        match_count = len(matches)
        total_count = len(t_set)
        match_ratio = match_count / total_count if total_count else 0.0
        
        # Determine status based on match ratio (for star display and categorization)
        if match_ratio > 0.6: 
            status = "match"
            comment = f"Strong skill overlap ({match_count}/{total_count})"
        elif match_ratio > 0.2: 
            status = "related"
            comment = f"Partial skill overlap ({match_count}/{total_count})"
        else:
            status = "unrelated"
            comment = f"Low skill overlap ({match_count}/{total_count})"
        
        return status, comment, match_count, total_count, match_ratio

    for c in active_criteria:
        if c not in assessment_results or assessment_results[c].get("status") not in ["match", "related", "unrelated"]:
            if c == "jobtitle_role_tag":
                st, cm = heuristic_status(c, job_title, role_tag)
                assessment_results[c] = {"status": st, "comment": cm}
            elif c == "country":
                assessment_results[c] = {"status": "match", "comment": "Present"} 
            elif c == "company":
                 assessment_results[c] = {"status": "match", "comment": "Present"}
            elif c == "seniority":
                 st, cm = heuristic_status(c, seniority, job_title)
                 assessment_results[c] = {"status": st, "comment": cm}
            elif c == "sector":
                 assessment_results[c] = {"status": "related", "comment": "Sector present"}
            elif c == "product":
                 # Assess product: if products are listed, it's a match
                 product_count = len(product) if isinstance(product, list) else 0
                 if product_count >= 3:
                     st, cm = "match", f"{product_count} products listed"
                 elif product_count >= 1:
                     st, cm = "related", f"{product_count} product(s) listed"
                 else:
                     st, cm = "unrelated", "No products listed"
                 assessment_results[c] = {"status": st, "comment": cm}
            elif c == "tenure":
                 # Assess tenure: <2 years = weak, 2-4 years = related, >4 years = match
                 try:
                     tenure_val = float(tenure)
                     if tenure_val >= 4.0:
                         st, cm = "match", f"{tenure_val:.1f} years avg tenure"
                     elif tenure_val >= 2.0:
                         st, cm = "related", f"{tenure_val:.1f} years avg tenure"
                     else:
                         st, cm = "unrelated", f"{tenure_val:.1f} years avg tenure (short)"
                 except (ValueError, TypeError):
                     st, cm = "not_assessed", "Tenure data unavailable"
                 assessment_results[c] = {"status": st, "comment": cm}
            elif c == "skillset":
                 st, cm, match_count, total_count, match_ratio = skill_heuristic(target_skills, candidate_skills, experience_text)
                 assessment_results[c] = {
                     "status": st, 
                     "comment": cm,
                     "match_count": match_count,
                     "total_count": total_count,
                     "match_ratio": match_ratio
                 }

    # Helper function to generate recruiter-style narrative comments based on score and assessment
    def generate_recruiter_narrative(assessment_results, active_criteria, data, total_score_value):
        """Generate professional recruiter-style narrative aligned with assessment score"""
        narrative_parts = []
        
        # Determine tone based on score
        is_high_score = total_score_value >= 70  # Good/excellent range
        is_mid_score = 40 <= total_score_value < 70  # Moderate range
        is_low_score = total_score_value < 40  # Weak range
        
        # Analyze key categories for detailed feedback
        skillset_res = assessment_results.get("skillset", {})
        skillset_status = skillset_res.get("status", "")
        skillset_comment = skillset_res.get("comment", "")
        
        jobtitle_res = assessment_results.get("jobtitle_role_tag", {})
        jobtitle_status = jobtitle_res.get("status", "")
        
        company_res = assessment_results.get("company", {})
        seniority_res = assessment_results.get("seniority", {})
        seniority_status = seniority_res.get("status", "")
        
        sector_res = assessment_results.get("sector", {})
        sector_status = sector_res.get("status", "")
        
        country_res = assessment_results.get("country", {})
        country_status = country_res.get("status", "")
        
        # Build narrative based on score range
        if is_high_score:
            # Positive, recruiter-style appraisal
            narrative_parts.append("Strong alignment with role requirements.")
            
            if skillset_status in ["match", "related"]:
                narrative_parts.append(f"Skillset coverage is {skillset_status} - {skillset_comment}.")
            
            if sector_status in ["match", "related"] and "sector" in active_criteria:
                narrative_parts.append("Sector expertise aligns well.")
            
            if seniority_status in ["match", "related"] and "seniority" in active_criteria:
                tenure_info = data.get("tenure", "")
                if tenure_info:
                    narrative_parts.append(f"Seniority level appropriate with {tenure_info} average tenure.")
                else:
                    narrative_parts.append("Seniority level is appropriate for the role.")
            
            if country_status in ["match", "related"]:
                narrative_parts.append("Geographic alignment supports local market knowledge.")
            
            narrative_parts.append("Well-suited for senior roles in multinational companies. Recommend advancing.")
            
        elif is_mid_score:
            # Balanced, constructive feedback
            narrative_parts.append("Moderate fit with some alignment to requirements.")
            
            if skillset_status == "related":
                narrative_parts.append(f"Skillset shows partial coverage - {skillset_comment}.")
            elif skillset_status == "unrelated":
                narrative_parts.append(f"Skillset coverage is limited - {skillset_comment}.")
            else:
                narrative_parts.append("Skillset overlap exists but may have gaps.")
            
            if sector_status == "unrelated" and "sector" in active_criteria:
                narrative_parts.append("Limited sector-specific experience noted.")
            
            if seniority_status == "unrelated" and "seniority" in active_criteria:
                narrative_parts.append("Seniority level may not fully align with role expectations.")
            
            narrative_parts.append("Consider for further screening to assess specific competencies.")
            
        else:  # is_low_score
            # Constructive, gap-focused feedback
            narrative_parts.append("Limited alignment with role requirements.")
            
            # Identify key gaps
            gaps = []
            if skillset_status == "unrelated":
                gaps.append(f"skillset ({skillset_comment})")
            if sector_status == "unrelated" and "sector" in active_criteria:
                gaps.append("sector experience")
            if seniority_status == "unrelated" and "seniority" in active_criteria:
                gaps.append("seniority level")
            if jobtitle_status == "unrelated":
                gaps.append("job title match")
            
            if gaps:
                narrative_parts.append(f"Key gaps identified in: {', '.join(gaps)}.")
            
            tenure_info = data.get("tenure", "")
            if tenure_info:
                # Handle both string and numeric tenure values
                if isinstance(tenure_info, (int, float)):
                    tenure_val = float(tenure_info)
                else:
                    # String format - remove "Years"/"years" and parse
                    tenure_str = str(tenure_info).replace("Years", "").replace("years", "").strip()
                    try:
                        tenure_val = float(tenure_str)
                    except ValueError:
                        tenure_val = None
                
                if tenure_val is not None and tenure_val < 2:
                    narrative_parts.append("Short average tenure may indicate limited depth.")

            
            narrative_parts.append("Skillset coverage is partial, reducing fit for the role. Not proceeding recommended.")
        
        return " ".join(narrative_parts)

    total_score = 0.0
    breakdown = {}
    comments_list = []
    category_appraisals = {}
    
    missing_fields = [k for k in weights.keys() if k not in active_criteria]
    if missing_fields:
        nice_names = [k.replace("jobtitle_role_tag", "Role").capitalize() for k in missing_fields]
        comments_list.append(f"{', '.join(nice_names)} Not Assessed")

    # Helper function to convert status to qualitative descriptor
    def status_to_descriptor(status):
        """Convert match/related/unrelated to qualitative descriptors"""
        if status == "match":
            return "Strong"
        elif status == "related":
            return "Suitable"
        else:  # unrelated or not_assessed
            return "Weak"
    
    # Helper function to calculate skillset factor using vskillset or fallback to match_ratio
    def calculate_skillset_factor(vskillset_results, target_skills, match_ratio_fallback):
        """
        Calculate skillset scoring factor using vskillset High count.
        Formula: vskillset_high_count / jskillset_total_count
        Falls back to match_ratio if vskillset is not available.
        
        Returns: (factor, log_message)
        """
        if vskillset_results and isinstance(vskillset_results, list) and target_skills:
            # Count vskillset items with High probability only
            vskillset_high_count = sum(
                1 for item in vskillset_results 
                if isinstance(item, dict) and item.get("category") == "High"
            )
            jskillset_total_count = len(target_skills)
            
            if jskillset_total_count > 0:
                factor = vskillset_high_count / jskillset_total_count
                log_msg = f"Skillset scoring: {vskillset_high_count} High vskills / {jskillset_total_count} jskills = {factor:.2f}"
                return factor, log_msg
            else:
                return 0.0, "Skillset scoring: No jskills (target_skills) available"
        else:
            # Fallback to original ratio-based scoring
            return match_ratio_fallback, f"Skillset scoring (fallback): match_ratio = {match_ratio_fallback:.2f}"
    
    # Map internal category names to display names
    category_display_names = {
        "jobtitle_role_tag": "Job Title",
        "skillset": "Skillset",
        "seniority": "Seniority",
        "company": "Company",
        "sector": "Sector",
        "country": "Country",
        "tenure": "Tenure",
        "product": "Product"
    }

    for c in active_criteria:
        res = assessment_results.get(c, {})
        st = res.get("status", "unrelated")
        cm = res.get("comment", "")
        
        # Calculate factor based on status OR ratio (for skillset)
        factor = 0.0
        
        if c == "skillset":
            # NEW: Use vskillset-based scoring formula
            # Formula: (vskillset_high_count / jskillset_total_count) * weight = actual_points
            # jskillset = target_skills (from login/process table)
            # vskillset = inference results with High probability
            match_ratio = res.get("match_ratio", 0.0)
            factor, log_msg = calculate_skillset_factor(vskillset_results, target_skills, match_ratio)
            logger.info(f"[CORE_ASSESS] {log_msg}")
        else:
            # Standard status-based scoring for other categories
            if st == "match": factor = 1.0
            elif st == "related": factor = 0.5
            # unrelated or not_assessed = 0.0
        
        points = final_weights[c] * factor
        total_score += points
        
        breakdown[c] = round(points, 1)
        
        # Calculate stars for this category
        category_stars = 0
        if c == "skillset":
            # Stars based on the actual factor used for scoring (already calculated above)
            skill_factor = factor
            
            if skill_factor >= 0.8: category_stars = 5
            elif skill_factor >= 0.6: category_stars = 4
            elif skill_factor >= 0.4: category_stars = 3
            elif skill_factor >= 0.2: category_stars = 2
            elif skill_factor > 0: category_stars = 1
            # else category_stars = 0
        else:
            # Standard status-based stars for other categories
            if st == "match": category_stars = 5
            elif st == "related": category_stars = 3
            elif st == "unrelated": category_stars = 1
            # not_assessed = 0
        
        # Add category appraisal with weightage, stars, and status
        display_name = category_display_names.get(c, c.capitalize())
        descriptor = status_to_descriptor(st)
        weight_percent = int(round(final_weights[c]))
        
        # Generate star string
        star_string = "★" * category_stars + "☆" * (5 - category_stars)
        
        category_appraisals[display_name] = {
            "rating": descriptor,
            "status": st,
            "comment": cm if cm else descriptor,
            "weight_percent": weight_percent,
            "stars": category_stars,
            "star_string": star_string
        }

    final_percent = min(100, max(0, int(round(total_score))))
    stars = int(round(final_percent / 20.0))
    if stars > 5: stars = 5
    
    # Generate recruiter-style narrative aligned with score
    final_comments = generate_recruiter_narrative(assessment_results, active_criteria, data, total_score)
    
    # Generate overall comment (≤MAX_COMMENT_LENGTH chars, professional tone)
    if final_percent >= ASSESSMENT_EXCELLENT_THRESHOLD:
        overall_comment = "Excellent match for the role requirements"
    elif final_percent >= ASSESSMENT_GOOD_THRESHOLD:
        overall_comment = "Good fit with relevant experience"
    elif final_percent >= ASSESSMENT_MODERATE_THRESHOLD:
        overall_comment = "Moderate alignment, some gaps present"
    else:
        overall_comment = "Limited match to role requirements"
    
    # Ensure overall comment is ≤MAX_COMMENT_LENGTH chars
    if len(overall_comment) > MAX_COMMENT_LENGTH:
        overall_comment = overall_comment[:COMMENT_TRUNCATE_LENGTH] + "..."
    
    # Determine assessment level display
    if assessment_level == "L2":
        level_display = "L2 Assessment"
    else:
        level_display = "L1 Assessment"
    
    # DIAGNOSTIC LOGGING - Log assessment inputs and output
    logger.info(f"[CORE_ASSESS] LinkedIn: {linkedinurl[:50] if linkedinurl else 'N/A'}")
    logger.info(f"[CORE_ASSESS] Profile data keys: {', '.join([k for k, v in data.items() if v])}")
    logger.info(f"[CORE_ASSESS] Active criteria ({len(active_criteria)}): {', '.join(active_criteria)}")
    
    # Log skillset ratio calculation if skillset is in active criteria
    if "skillset" in active_criteria and "skillset" in assessment_results:
        skillset_res = assessment_results["skillset"]
        match_count = skillset_res.get("match_count", 0)
        total_count = skillset_res.get("total_count", 0)
        match_ratio = skillset_res.get("match_ratio", 0.0)
        skillset_weight = final_weights.get("skillset", 0)
        skillset_points = breakdown.get("skillset", 0)
        logger.info(f"[CORE_ASSESS] Skillset: {match_count}/{total_count} matched ({match_ratio:.1%}), weight={skillset_weight}%, points={skillset_points}")
    
    logger.info(f"[CORE_ASSESS] Final score: {final_percent}% | Stars: {stars} | Level: {assessment_level}")
    logger.info(f"[CORE_ASSESS] Breakdown: {breakdown}")
    
    out_obj = {
        "assessment_level": level_display,
        "is_level2": is_level2 or (assessment_level == "L2"),
        "stars": stars,
        "total_score": f"{final_percent}%",
        "criteria": breakdown,
        "comments": final_comments,
        "overall_comment": overall_comment,
        "category_appraisals": category_appraisals
    }

    if linkedinurl:
        safe_name = "assessment_" + hashlib.sha256(linkedinurl.encode("utf-8")).hexdigest()[:16] + ".json"
        assess_dir = os.path.join(OUTPUT_DIR, "assessments")
        os.makedirs(assess_dir, exist_ok=True)
        out_path = os.path.join(assess_dir, safe_name)
        
        try:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(out_obj, f, indent=2, ensure_ascii=False)
            out_obj["file"] = f"output/assessments/{safe_name}"
        except Exception as e:
            logger.warning(f"Failed to write assessment file: {e}")

    return out_obj

def analyze_cv_background(linkedinurl, pdf_bytes):
    """
    Extracts text from PDF bytes using synchronous helper,
    updates DB, and performs automatic assessment.
    
    SOURCE OF TRUTH: CV Column
    - Gemini exclusively references the cv column in the process table (Postgres)
    - No merging with existing process.skillset data
    - No reconciliation with login.jskillset
    - Employment history strictly follows format: "Job Title, Company, StartYear to EndYear"
      OR "Job Title, Company, StartYear to present" (for current positions)
    """
    try:
        obj = _analyze_cv_bytes_sync(pdf_bytes)
        if not obj:
            logger.warning(f"[CV BG] Analysis returned None for {linkedinurl}")
            return

        skillset = obj.get("skillset", [])
        total_exp = obj.get("total_experience_years", 0)
        tenure = obj.get("tenure", 0.0)  # Extract calculated tenure
        experience = obj.get("experience", [])
        education = obj.get("education", [])
        product_list = obj.get("product_list", [])
        seniority = obj.get("seniority", "")
        sector = obj.get("sector", "")
        job_family = obj.get("job_family", "")
        company = obj.get("company", "")
        job_title = obj.get("job_title", "")
        country = obj.get("country", "")
        
        # Log product extraction for debugging
        if product_list and len(product_list) > 0:
            truncated = '...' if len(product_list) > 3 else ''
            logger.info(f"[CV BG] Extracted {len(product_list)} products for {linkedinurl[:50]}: {product_list[:3]}{truncated}")
        else:
            logger.warning(f"[CV BG] No products extracted for {linkedinurl[:50]} (company: {company})")
        
        # Create skillset string without length limits (DB columns now TEXT type)
        skillset_raw = ",".join([str(s).strip() for s in skillset if str(s).strip()])
        skillset_str = skillset_raw
        experience_str = "\n".join([str(e).strip() for e in experience if str(e).strip()])
        education_str = "\n".join([str(e).strip() for e in education if str(e).strip()])
        product_str = ", ".join([str(p).strip() for p in product_list if str(p).strip()])
        
        import psycopg2
        pg_host=os.getenv("PGHOST","localhost"); pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres"); pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()
        
        normalized = _normalize_linkedin_to_path(linkedinurl)
        where_clause = "linkedinurl = %s"
        params = [linkedinurl]
        
        sourcing_id = None
        try:
            cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='process' AND column_name='id'")
            if cur.fetchone():
                cur.execute("SELECT id FROM sourcing WHERE linkedinurl = %s LIMIT 1", (linkedinurl,))
                row_sid = cur.fetchone()
                if row_sid: sourcing_id = row_sid[0]
        except Exception: pass
        
        if sourcing_id:
            where_clause = "id = %s"; params = [sourcing_id]
        
        # Check available columns to ensure 'product' is fetched
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='process'")
        all_cols = {r[0].lower() for r in cur.fetchall()}
        has_product = 'product' in all_cols
        
        # Build SELECT query dynamically based on columns
        select_fields = ["skillset"]
        if has_product: select_fields.append("product")
        # Include username if available so we can reconcile against login.jskillset
        if 'username' in all_cols:
            select_fields.append("username")
        
        cur.execute(f"SELECT {', '.join(select_fields)} FROM process WHERE {where_clause}", tuple(params))
        row = cur.fetchone()
        
        if not row and not sourcing_id and normalized:
             cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='process' AND column_name='normalized_linkedin'")
             if cur.fetchone():
                 where_clause = "normalized_linkedin = %s"; params = [normalized]
                 cur.execute(f"SELECT {', '.join(select_fields)} FROM process WHERE {where_clause}", (normalized,))
                 row = cur.fetchone()
        
        if row:
            # SOURCE OF TRUTH: CV Column - use extracted skillset exclusively (no merge with existing)
            # Gemini must exclusively reference the cv column in the process table (Postgres)
            # skillset_str was already set from CV extraction above (line 5315) - no modification needed
            # Merge product if available (index 1 if product was selected)
            if has_product:
                prod_idx = 1 if 'product' in select_fields else None
                if prod_idx is not None and len(row) > prod_idx and row[prod_idx]:
                    curr_prod = [p.strip() for p in row[prod_idx].split(',') if p.strip()]
                    prod_set = set(curr_prod)
                    prod_set.update([str(p).strip() for p in product_list if str(p).strip()])
                    product_str = ", ".join(list(prod_set))
        
        # PATCH: Infer geographic region
        geo_region = _infer_region_from_country(country) if country else ""

        # Use previously fetched columns info
        cols = all_cols
        
        update_parts = []
        update_vals = []
        # NOTE: Deliberately exclude 'skillset' here so we can reconcile against jskillset
        mapping = {
            "exp": str(total_exp), "experience": experience_str,
            "education": education_str, "product": product_str, "seniority": seniority,
            "sector": sector, "job_family": job_family, "company": company, "jobtitle": job_title,
            "country": country, 
            "geographic": geo_region, # Use inferred region, NOT country
            "tenure": float(tenure) # Numeric average tenure
        }
        
        for col_key, val in mapping.items():
            db_col = col_key
            if col_key == "job_family" and "jobfamily" in cols: db_col = "jobfamily"
            if col_key == "jobtitle":
                if "jobtitle" in cols: db_col = "jobtitle"
                elif "role" in cols: db_col = "role"
            
            # Check tenure specifically as it's a numeric type, ensure column exists
            if col_key == "tenure" and "tenure" not in cols:
                continue

            if db_col in cols and (val is not None and val != ""):
                update_parts.append(f"{db_col} = %s")
                update_vals.append(val)
            
        if update_parts:
            sql = f"UPDATE process SET {', '.join(update_parts)} WHERE {where_clause}"
            update_vals.extend(params)
            cur.execute(sql, tuple(update_vals))
            conn.commit()

        # Update Sourcing table if fields exist
        if company or job_title or country:
            s_upd = []; s_vals = []
            if company: s_upd.append("company = %s"); s_vals.append(company)
            if job_title: s_upd.append("jobtitle = %s"); s_vals.append(job_title)
            if country: s_upd.append("country = %s"); s_vals.append(country)
            
            s_vals.append(linkedinurl) # where clause
            if s_upd:
                try:
                    cur.execute(f"UPDATE sourcing SET {', '.join(s_upd)} WHERE linkedinurl = %s", tuple(s_vals))
                    conn.commit()
                except Exception as e_src:
                    logger.warning(f"[CV BG] Sourcing update failed: {e_src}")

        # Trigger Auto-Assessment
        fetch_cols_sql = "SELECT jobtitle, company, country, role_tag, userid, username, skillset FROM process WHERE " + where_clause
        cur.execute(fetch_cols_sql, tuple(params))
        ctx_row = cur.fetchone()

        if ctx_row:
            job_title_db, company_db, country_db, role_tag_db, userid_db, username_db, skillset_db = ctx_row
            if not role_tag_db and username_db:
                cur.execute("SELECT role_tag FROM login WHERE username=%s", (username_db,))
                rt_row = cur.fetchone()
                if rt_row: role_tag_db = rt_row[0]

            # SOURCE OF TRUTH: CV Column - use extracted skillset exclusively
            # Gemini must exclusively reference the cv column in the process table (Postgres)
            # No reconciliation with login.jskillset - CV is the single source of truth
            final_skillset_str = skillset_str
            
            try:
                # Persist CV-extracted skillset to process table (no reconciliation)
                if 'skillset' in cols:
                    up_where = where_clause
                    up_params = list(params)
                    cur.execute(f"UPDATE process SET skillset = %s WHERE {up_where}", tuple([final_skillset_str] + up_params))
                    conn.commit()
                    logger.info(f"[CV BG] CV-extracted skillset saved for linkedin='{linkedinurl}' (exclusive source)")
            except Exception as e_save:
                logger.warning(f"[CV BG] Failed to persist CV skillset: {e_save}")

            if role_tag_db:
                target_skills_final = _fetch_jskillset(username_db) if username_db else []
                c_skills_list = [s.strip() for s in (final_skillset_str if 'final_skillset_str' in locals() else (skillset_db or "")).split(',') if s.strip()]
                profile_data = {
                    "job_title": job_title_db or "",
                    "role_tag": role_tag_db,
                    "company": company_db or "",
                    "country": country_db or "",
                    "seniority": seniority or "",
                    "sector": sector or "",
                    "experience_text": experience_str,
                    "target_skills": target_skills_final,
                    "candidate_skills": c_skills_list,
                    "linkedinurl": linkedinurl
                }
                assessment_result = _core_assess_profile(profile_data)
                if 'rating' in cols:
                    rating_json = json.dumps(assessment_result, ensure_ascii=False)
                    upd_sql = f"UPDATE process SET rating = %s WHERE {where_clause}"
                    cur.execute(upd_sql, (rating_json, *params))
                    conn.commit()
                
                # --- NEW: Trigger role_tag -> jskill sync during background CV analysis ---
                if role_tag_db and "jskill" in cols:
                     upd_js_sql = f"UPDATE process SET jskill = %s WHERE {where_clause}"
                     cur.execute(upd_js_sql, (role_tag_db, *params))
                     conn.commit()
                
                # Now trigger jskillset sync from login to process
                _sync_login_jskillset_to_process(username_db, linkedinurl, normalized)
                # --------------------------------------------------------------------------

        cur.close(); conn.close()
    except Exception as e:
        logger.error(f"Error in CV analysis background task: {e}")

@app.post("/process/parse_cv_and_update")
def process_parse_cv_and_update():
    """
    Manually trigger CV analysis via sync helper.
    
    SOURCE OF TRUTH: CV Column
    - Fetches CV from process.cv column and analyzes with Gemini
    - Returns skillset and employment history strictly from CV
    - Employment history format: "Job Title, Company, StartYear to EndYear"
      OR "Job Title, Company, StartYear to present" (for current positions)
    """
    data = request.get_json(force=True, silent=True) or {}
    linkedinurl = (data.get("linkedinurl") or "").strip()
    if not linkedinurl: return jsonify({"error": "linkedinurl required"}), 400

    try:
        import psycopg2
        pg_host=os.getenv("PGHOST","localhost"); pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres"); pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()
        
        linkedin_norm = linkedinurl.split('?')[0].rstrip('/')
        linkedin_path = _normalize_linkedin_to_path(linkedinurl)
        
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='process' AND column_name='cv'")
        if not cur.fetchone():
             cur.close(); conn.close()
             return jsonify({"error": "No CV column in DB"}), 500

        row = None
        if linkedin_path:
             cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='process' AND column_name='normalized_linkedin'")
             if cur.fetchone():
                 cur.execute("SELECT cv FROM process WHERE normalized_linkedin = %s AND cv IS NOT NULL LIMIT 1", (linkedin_path,))
                 row = cur.fetchone()
        
        if not row:
             cur.execute("SELECT cv FROM process WHERE linkedinurl = %s AND cv IS NOT NULL LIMIT 1", (linkedin_norm,))
             row = cur.fetchone()
             
        if not row or not row[0]:
             cur.close(); conn.close()
             return jsonify({"error": "CV not found for this profile"}), 404
             
        pdf_bytes = bytes(row[0])
        cur.close(); conn.close()
        
        obj = _analyze_cv_bytes_sync(pdf_bytes)
        if not obj:
             return jsonify({"error": "Analysis returned no data"}), 500
             
        # Trigger persistence in background
        threading.Thread(target=analyze_cv_background, args=(linkedinurl, pdf_bytes)).start()
        
        return jsonify({
            "skillset": obj.get("skillset", []),
            "total_years": obj.get("total_experience_years", 0),
            "tenure": obj.get("tenure", 0.0), # Includes tenure in API response
            "experience": obj.get("experience", []),
            "education": obj.get("education", []),
            "product": obj.get("product_list", []),
            "company": obj.get("company", ""),
            "job_title": obj.get("job_title", ""),
            "country": obj.get("country", ""),
            "experience_text": "\n".join(obj.get("experience", [])),
            "education_text": "\n".join(obj.get("education", []))
        }), 200

    except Exception as e:
        logger.error(f"[Parse CV Update] {e}")
        return jsonify({"error": str(e)}), 500

@app.get("/process/pending_assessments")
def process_pending_assessments():
    userid = (request.args.get("userid") or "").strip()
    if not userid: return jsonify({"rows": []})
    page_size = int(request.args.get("page_size", 100))
    try:
        import psycopg2
        pg_host=os.getenv("PGHOST","localhost"); pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres"); pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()
        cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='process' AND column_name='normalized_linkedin'")
        has_normalized = bool(cur.fetchone())
        cols = ["name", "company", "jobtitle", "country", "linkedinurl", "experience", "rating"]
        if has_normalized: cols.append("normalized_linkedin")
        from psycopg2 import sql
        query = sql.SQL("SELECT {fields} FROM process WHERE userid=%s ORDER BY id DESC LIMIT %s").format(
            fields=sql.SQL(', ').join(map(sql.Identifier, cols)))
        cur.execute(query, (userid, page_size))
        rows = []
        for r in cur.fetchall():
            row_dict = {"name": r[0], "company": r[1], "jobtitle": r[2], "country": r[3], "linkedinurl": r[4], "experience": r[5], "rating": r[6]}
            if has_normalized: row_dict["normalized_linkedin"] = r[7]
            rows.append(row_dict)
        cur.close(); conn.close()
        return jsonify({"rows": rows})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _generate_vskillset_for_profile(linkedinurl, target_skills, experience_text="", cv_data=None):
    """
    Generate vskillset for a profile using Gemini inference.
    Returns list of skill evaluation results or None if failed.
    """
    if not (genai and GEMINI_API_KEY):
        return None
    
    if not target_skills or len(target_skills) == 0:
        return None
    
    try:
        import psycopg2
        pg_host = os.getenv("PGHOST", "localhost")
        pg_port = int(os.getenv("PGPORT", "5432"))
        pg_user = os.getenv("PGUSER", "postgres")
        pg_password = os.getenv("PGPASSWORD", "") or "orlha"
        pg_db = os.getenv("PGDATABASE", "candidate_db")
        
        # Use experience as primary context, cv as fallback
        profile_context = experience_text if experience_text else ""
        if not profile_context and cv_data:
            # Extract text from CV if needed
            try:
                if isinstance(cv_data, bytes):
                    from pypdf import PdfReader
                    import io
                    reader = PdfReader(io.BytesIO(cv_data))
                    text = ""
                    for page in reader.pages:
                        t = page.extract_text()
                        if t: text += t + "\n"
                    profile_context = text[:3000]
            except Exception:
                pass
        
        if not profile_context:
            logger.warning(f"[vskillset_gen] No experience or CV data for {linkedinurl}")
            return None
        
        # Call Gemini to evaluate skills
        model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
        
        prompt = f"""SYSTEM:
You are an expert technical recruiter evaluating candidate skillsets based on their work experience.

TASK:
For each skill in the list below, evaluate the candidate's likely proficiency based on their experience.
Assign a probability score (0-100) and categorize as Low (<40), Medium (40-74), or High (75-100).
Provide clear reasoning based on job titles, companies, and experience patterns.

CANDIDATE PROFILE:
{profile_context[:3000]}

SKILLS TO EVALUATE:
{json.dumps(target_skills, ensure_ascii=False)}

OUTPUT FORMAT (JSON):
{{
  "evaluations": [
    {{
      "skill": "skill_name",
      "probability": 0-100,
      "category": "Low|Medium|High",
      "reason": "Brief explanation based on companies and roles"
    }}
  ]
}}

Return ONLY the JSON object, no other text."""
        
        resp = model.generate_content(prompt)
        raw_text = (resp.text or "").strip()
        
        # Extract JSON from response
        parsed = _extract_json_object(raw_text)
        
        if not parsed or "evaluations" not in parsed:
            logger.warning(f"[vskillset_gen] Gemini returned invalid JSON")
            return None
        
        results = parsed["evaluations"]
        
        # Ensure all required fields are present
        for item in results:
            if "probability" not in item:
                item["probability"] = 50
            if "category" not in item:
                prob = item.get("probability", 50)
                if prob >= 75:
                    item["category"] = "High"
                elif prob >= 40:
                    item["category"] = "Medium"
                else:
                    item["category"] = "Low"
            if "reason" not in item:
                item["reason"] = "No reasoning provided"
        
        # Persist to database
        try:
            conn = psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
            cur = conn.cursor()
            
            vskillset_json = json.dumps(results, ensure_ascii=False)
            confirmed_skills = [item["skill"] for item in results if item["category"] == "High"]
            # Ensure all skills are strings before joining
            skillset_str = ", ".join([str(s) for s in confirmed_skills if s])
            
            # Check if vskillset column exists
            cur.execute("""
                SELECT column_name 
                FROM information_schema.columns
                WHERE table_schema='public' AND table_name='process' 
                  AND column_name IN ('vskillset', 'skillset')
            """)
            available_cols = {r[0] for r in cur.fetchall()}
            
            # Update vskillset if column exists
            if 'vskillset' in available_cols:
                cur.execute("UPDATE process SET vskillset = %s WHERE linkedinurl = %s", (vskillset_json, linkedinurl))
                logger.info(f"[vskillset_gen] Persisted vskillset for {linkedinurl[:50]}")
            
            # Update skillset with High skills only as comma-separated string
            if 'skillset' in available_cols:
                cur.execute("UPDATE process SET skillset = %s WHERE linkedinurl = %s", (skillset_str, linkedinurl))
                logger.info(f"[vskillset_gen] Persisted {len(confirmed_skills)} High skills to skillset for {linkedinurl[:50]}")
            
            conn.commit()
            cur.close()
            conn.close()
        except Exception as e_db:
            logger.warning(f"[vskillset_gen] Failed to persist for {linkedinurl}: {e_db}")
        
        return results
        
    except Exception as e:
        logger.error(f"[vskillset_gen] Error for {linkedinurl}: {e}")
        return None

@app.post("/process/bulk_assess")
def process_bulk_assess():
    """
    Accepts JSON payload like:
    {
      "userid": "9896945",
      "linkedinurls": ["https://...","https://..."],
      "async": true,
      "custom_weights": {...},
      "assessment_level": "L2"
    }
    If async is true returns job_id and starts background worker, otherwise runs synchronously and returns results.
    Uses existing _core_assess_profile to perform per-profile assessment and persists rating into process.rating (if column exists).
    """
    payload = request.get_json(force=True, silent=True) or {}
    userid = (payload.get("userid") or "").strip()
    linkedinurls = payload.get("linkedinurls") or []
    if isinstance(linkedinurls, str):
        linkedinurls = [linkedinurls]
    if not isinstance(linkedinurls, list) or not linkedinurls:
        return jsonify({"error": "linkedinurls list required"}), 400

    async_flag = bool(payload.get("async"))
    custom_weights = payload.get("custom_weights") or {}
    assessment_level = (payload.get("assessment_level") or payload.get("assessmentLevel") or "L1").strip().upper()
    username = (payload.get("username") or "").strip()

    # helper for single assess + persist
    def _assess_and_persist(linkedinurl):
        try:
            # Fetch profile data from database
            import psycopg2
            from psycopg2 import sql
            pg_host=os.getenv("PGHOST","localhost")
            pg_port=int(os.getenv("PGPORT","5432"))
            pg_user=os.getenv("PGUSER","postgres")
            pg_password=os.getenv("PGPASSWORD","") or "orlha"
            pg_db=os.getenv("PGDATABASE","candidate_db")
            conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
            cur=conn.cursor()
            
            # Fetch profile data from process table
            normalized = None
            try:
                normalized = _normalize_linkedin_to_path(linkedinurl)
            except Exception:
                normalized = None
            
            # Fetch by linkedinurl (normalized_linkedin column doesn't exist in all schemas)
            cur.execute("""
                SELECT jobtitle, company, country, seniority, sector, experience, skillset, username, role_tag, cv, tenure, product
                FROM process 
                WHERE linkedinurl = %s
                LIMIT 1
            """, (linkedinurl,))
            row = cur.fetchone()
            
            # Default values if profile not found
            job_title = ""
            company = ""
            country = ""
            seniority = ""
            sector = ""
            experience_text = ""
            candidate_skills = []
            username_db = ""
            role_tag = ""
            cv_data = None
            tenure = None
            product = []
            
            if row:
                job_title = row[0] or ""
                company = row[1] or ""
                country = row[2] or ""
                seniority = row[3] or ""
                sector = row[4] or ""
                experience_text = row[5] or ""
                skillset_str = row[6] or ""
                username_db = row[7] or ""
                role_tag = row[8] or ""
                # Safe extraction of CV data (cv column may not exist in all schemas)
                try:
                    cv_data = row[9] if len(row) >= 10 else None
                    tenure = row[10] if len(row) >= 11 else None
                    product_str = row[11] if len(row) >= 12 else ""
                except (IndexError, TypeError):
                    cv_data = None
                    tenure = None
                    product_str = ""
                
                # Parse skillset
                if skillset_str:
                    candidate_skills = [s.strip() for s in skillset_str.split(',') if s.strip()]
                
                # Parse product
                if product_str:
                    try:
                        # Product could be JSON array or comma-separated string
                        product = json.loads(product_str) if product_str.startswith('[') else [s.strip() for s in product_str.split(',') if s.strip()]
                    except:
                        product = [s.strip() for s in product_str.split(',') if s.strip()]
                    
                    # Log product loading regardless of whether list is empty
                    if product:
                        logger.info(f"[BULK_ASSESS] Loaded {len(product)} products from DB for {linkedinurl[:50]}")
                    else:
                        logger.info(f"[BULK_ASSESS] Product field exists but is empty for {linkedinurl[:50]}")
                else:
                    logger.info(f"[BULK_ASSESS] No product data in DB for {linkedinurl[:50]}")
            
            # Check if CV is uploaded - if not, skip assessment
            if not cv_data:
                logger.info(f"[BULK_ASSESS] Skipping {linkedinurl[:50]} - No CV uploaded (Assessment pending)")
                return {
                    "linkedinurl": linkedinurl,
                    "result": {
                        "error": "Assessment pending - No CV uploaded"
                    }
                }
            
            # If role_tag not in process, try to fetch from login
            if not role_tag and username_db:
                cur.execute("SELECT role_tag FROM login WHERE username = %s LIMIT 1", (username_db,))
                login_row = cur.fetchone()
                if login_row and login_row[0]:
                    role_tag = login_row[0]
            
            cur.close()
            conn.close()
            
            # Fetch target skills
            target_skills = _fetch_jskillset_from_process(linkedinurl) or []
            if not target_skills and username_db:
                target_skills = _fetch_jskillset(username_db) or []
            
            # NEW: For L2 assessment, generate vskillset if target_skills exist
            vskillset_results = None  # Initialize to None for passing to profile_data
            if assessment_level == "L2" and target_skills and len(target_skills) > 0:
                logger.info(f"[BULK_ASSESS] L2 mode - generating vskillset for {linkedinurl[:50]}")
                try:
                    # Call vskillset inference logic inline
                    vskillset_results = _generate_vskillset_for_profile(linkedinurl, target_skills, experience_text, cv_data)
                    if vskillset_results:
                        # Update candidate_skills to use High probability skills from vskillset
                        candidate_skills = [item["skill"] for item in vskillset_results if item.get("category") == "High"]
                        logger.info(f"[BULK_ASSESS] vskillset generated: {len(candidate_skills)} High skills")
                except Exception as e_vsk:
                    logger.warning(f"[BULK_ASSESS] vskillset generation failed for {linkedinurl}: {e_vsk}")
            
            # Build profile_data consistent with gemini_assess_profile expectations
            profile_data = {
                "job_title": job_title,
                "role_tag": role_tag,
                "company": company,
                "country": country,
                "seniority": seniority,
                "sector": sector,
                "experience_text": experience_text,
                "target_skills": target_skills,
                "candidate_skills": candidate_skills,
                "process_skills": [],
                "custom_weights": custom_weights,
                "linkedinurl": linkedinurl,
                "assessment_level": assessment_level,
                "tenure": tenure,
                "product": product,
                "vskillset_results": vskillset_results  # Pass vskillset_results for scoring
            }

            # run core assessment
            try:
                result = _core_assess_profile(profile_data)
                logger.info(f"[BULK_ASSESS] Assessment completed for {linkedinurl[:50]}: Score={result.get('total_score', 'N/A')}, Stars={result.get('stars', 0)}")
            except Exception as e:
                logger.error(f"[BULK_ASSESS] Assessment error for {linkedinurl}: {e}")
                result = {"error": f"assessment_error: {str(e)}"}

            # persist rating JSON into process.rating if column exists
            try:
                conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
                cur=conn.cursor()

                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_schema='public' AND table_name='process' AND column_name='rating'
                """)
                if cur.fetchone():
                    # store JSON as text
                    rating_payload = json.dumps(result, ensure_ascii=False)
                    logger.info(f"[BULK_PERSIST] Persisting rating for {linkedinurl[:50]}, payload size: {len(rating_payload)} bytes")

                    # Update by linkedinurl (normalized_linkedin column doesn't exist in all schemas)
                    cur.execute("UPDATE process SET rating = %s WHERE linkedinurl = %s", (rating_payload, linkedinurl))
                    updated = cur.rowcount
                    logger.info(f"[BULK_PERSIST] Updated {updated} rows by linkedinurl: {linkedinurl[:50]}")

                    if updated > 0:
                        logger.info(f"[BULK_PERSIST] Successfully persisted rating for {linkedinurl[:50]}")
                    else:
                        logger.warning(f"[BULK_PERSIST] No rows updated for {linkedinurl[:50]} - profile may not exist in process table")

                    # Also sync jskill if result contains role_tag (safe best-effort)
                    role_tag_val = result.get("role_tag") if isinstance(result, dict) else None
                    if role_tag_val:
                        cur.execute("""
                            SELECT column_name FROM information_schema.columns
                            WHERE table_schema='public' AND table_name='process' AND column_name='jskill'
                        """)
                        if cur.fetchone():
                            cur.execute("UPDATE process SET jskill = %s WHERE linkedinurl = %s", (role_tag_val, linkedinurl))
                    conn.commit()
                cur.close(); conn.close()
            except Exception as e_db:
                logger.warning(f"[BulkAssess->DB] Failed to write rating for {linkedinurl}: {e_db}")

            return {"linkedinurl": linkedinurl, "result": result}
        except Exception as e:
            logger.error(f"[BulkAssess] Error assessing {linkedinurl}: {e}")
            return {"linkedinurl": linkedinurl, "error": str(e)}

    # If async, spin thread worker and return job id
    if async_flag:
        job_id = "bulk_" + uuid.uuid4().hex[:10]
        
        # Initialize job status
        with JOBS_LOCK:
            JOBS[job_id] = {'status': 'running', 'processed': 0, 'total': len(linkedinurls), 'messages': [], 'errors': []}
        persist_job(job_id)
        
        def _bg_worker(urls, job_id):
            results = []
            processed = 0
            for u in urls:
                try:
                    out = _assess_and_persist(u)
                    results.append(out)
                except Exception as e:
                    results.append({"linkedinurl": u, "error": str(e)})
                finally:
                    processed += 1
                    with JOBS_LOCK:
                        JOBS[job_id]['processed'] = processed
                    persist_job(job_id)
                    
            # persist results file for later retrieval
            try:
                fname = f"{job_id}_results.json"
                path = os.path.join(OUTPUT_DIR, fname)
                with open(path, "w", encoding="utf-8") as fh:
                    json.dump(results, fh, ensure_ascii=False, indent=2)
                logger.info(f"[BulkAssess] Completed job {job_id} results saved to {path}")
            except Exception as e:
                logger.warning(f"[BulkAssess] Failed to write results for {job_id}: {e}")
            
            with JOBS_LOCK:
                JOBS[job_id]['status'] = 'done'
            persist_job(job_id)

        threading.Thread(target=_bg_worker, args=(linkedinurls, job_id), daemon=True).start()
        return jsonify({"ok": True, "job_id": job_id}), 202

    # synchronous: do all and return results
    all_results = []
    for u in linkedinurls:
        all_results.append(_assess_and_persist(u))
    return jsonify({"ok": True, "results": all_results}), 200


@app.get("/process/bulk_assess_status/<job_id>")
def process_bulk_status(job_id):
    with JOBS_LOCK: job = JOBS.get(job_id)
    if not job: return jsonify({"error": "Job not found"}), 404
    
    # Calculate progress percentage
    job_response = dict(job)
    if 'processed' in job and 'total' in job and job['total'] > 0:
        job_response['progress'] = int((job['processed'] / job['total']) * 100)
    else:
        job_response['progress'] = 0
    
    # If job is done, try to load and include the actual results from JSON file
    if job.get('status') == 'done':
        try:
            fname = f"{job_id}_results.json"
            path = os.path.join(OUTPUT_DIR, fname)
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as fh:
                    results = json.load(fh)
                # Include results in the response
                job_response['results'] = results
                job_response['progress'] = 100  # Ensure progress is 100% when done
                logger.info(f"[BulkAssessStatus] Loaded {len(results)} results from {fname}")
                return jsonify(job_response)
        except Exception as e:
            logger.warning(f"[BulkAssessStatus] Failed to load results for {job_id}: {e}")
    
    return jsonify(job_response)

@app.get("/process/bulk_assess_stream/<job_id>")
def process_bulk_assess_stream(job_id):
    """
    Server-Sent Events (SSE) endpoint for real-time bulk assessment progress.
    Streams progress updates to the client instead of requiring polling.
    """
    def generate_events():
        """Generator function that yields SSE-formatted messages."""
        last_status = None
        last_progress = -1
        
        while True:
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            
            if not job:
                yield f"event: error\ndata: {json.dumps({'error': 'Job not found'})}\n\n"
                break
            
            # Calculate current progress
            current_progress = 0
            if 'processed' in job and 'total' in job and job['total'] > 0:
                current_progress = int((job['processed'] / job['total']) * 100)
            
            current_status = job.get('status', 'pending')
            
            # Only send update if something changed
            if current_status != last_status or current_progress != last_progress:
                event_data = {
                    'status': current_status,
                    'progress': current_progress,
                    'processed': job.get('processed', 0),
                    'total': job.get('total', 0)
                }
                
                # If job is done, include results
                if current_status == 'done':
                    try:
                        fname = f"{job_id}_results.json"
                        path = os.path.join(OUTPUT_DIR, fname)
                        if os.path.exists(path):
                            with open(path, "r", encoding="utf-8") as fh:
                                results = json.load(fh)
                            event_data['results'] = results
                            logger.info(f"[SSE] Loaded {len(results)} results for {job_id}")
                    except Exception as e:
                        logger.warning(f"[SSE] Failed to load results for {job_id}: {e}")
                
                yield f"data: {json.dumps(event_data)}\n\n"
                
                last_status = current_status
                last_progress = current_progress
                
                # If job is done or failed, close the stream
                if current_status in ('done', 'failed'):
                    break
            
            # Wait before checking again (reduce CPU usage)
            time.sleep(0.5)
    
    return Response(
        stream_with_context(generate_events()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering
            'Connection': 'keep-alive'
        }
    )

@app.patch("/process/profile_assessment/<path:linkedinurl>")
def patch_profile_assessment(linkedinurl):
    """
    HTTP PATCH endpoint for updating individual profile assessments.
    Faster than full POST as it only updates specific fields.
    """
    import psycopg2
    from psycopg2 import sql
    
    try:
        data = request.get_json(force=True, silent=True) or {}
        if 'rating' not in data:
            return jsonify({"error": "rating field required"}), 400
        
        rating = data.get('rating')
        
        # Normalize LinkedIn URL
        normalized = _normalize_linkedin_to_path(linkedinurl)
        
        # Update only the rating field in database
        pg_host = os.getenv("PGHOST", "localhost")
        pg_port = int(os.getenv("PGPORT", "5432"))
        pg_user = os.getenv("PGUSER", "postgres")
        pg_password = os.getenv("PGPASSWORD", "") or "orlha"
        pg_db = os.getenv("PGDATABASE", "candidate_db")
        
        conn = None
        cur = None
        try:
            conn = psycopg2.connect(
                host=pg_host, port=pg_port, user=pg_user, 
                password=pg_password, dbname=pg_db
            )
            cur = conn.cursor()
            
            # Check if rating column exists
            cur.execute("""
                SELECT column_name FROM information_schema.columns 
                WHERE table_schema='public' AND table_name='process' AND column_name='rating'
            """)
            
            if cur.fetchone():
                rating_json = json.dumps(rating) if isinstance(rating, dict) else rating
                cur.execute(
                    sql.SQL("UPDATE process SET rating = %s WHERE linkedinurl = %s"),
                    (rating_json, normalized)
                )
                conn.commit()
                
                updated = cur.rowcount
                
                logger.info(f"[PATCH] Updated assessment for {normalized}")
                return jsonify({"success": True, "updated": updated}) if updated > 0 else (jsonify({"error": "Profile not found"}), 404)
            else:
                return jsonify({"error": "rating column does not exist"}), 500
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()
                
    except Exception as e:
        logger.error(f"[PATCH] Error updating assessment: {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/user/upload_jd")
def user_upload_jd():
    try:
        username = request.form.get("username", "").strip()
        if not username: return jsonify({"error": "username required"}), 400
        if 'file' not in request.files: return jsonify({"error": "No file part"}), 400
        file = request.files['file']
        if file.filename == '': return jsonify({"error": "No selected file"}), 400
        filename = file.filename.lower()
        file_bytes = file.read()
        extracted_text = ""
        if filename.endswith('.pdf'):
            import io
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(file_bytes))
                for page in reader.pages: extracted_text += (page.extract_text() or "") + "\n"
            except ImportError: return jsonify({"error": "pypdf not installed, cannot process PDF"}), 500
            except Exception as e: return jsonify({"error": f"PDF parsing error: {e}"}), 500
        elif filename.endswith('.docx'):
            import io
            try:
                import docx
                doc = docx.Document(io.BytesIO(file_bytes))
                for para in doc.paragraphs: extracted_text += para.text + "\n"
            except ImportError: 
                return jsonify({"error": "python-docx library not installed. Please install it with: pip install python-docx"}), 500
            except Exception as e: 
                return jsonify({"error": f"DOCX parsing error: {e}"}), 500
        elif filename.endswith('.doc'):
            # Note: Legacy .doc format requires the python-docx library (or alternatives like antiword, textract)
            # python-docx primarily supports .docx but can sometimes read .doc files
            import io
            try:
                import docx
                doc = docx.Document(io.BytesIO(file_bytes))
                for para in doc.paragraphs: extracted_text += para.text + "\n"
            except ImportError:
                return jsonify({"error": "python-docx library not installed. Please install it with: pip install python-docx"}), 500
            except Exception as e:
                # Legacy .doc format may not be fully supported by python-docx
                return jsonify({"error": "Legacy .doc format could not be processed. Please convert to .docx or .pdf format."}), 400
        else:
            try: extracted_text = file_bytes.decode('utf-8', errors='ignore')
            except Exception as e: return jsonify({"error": f"Text decoding error: {e}"}), 500
        extracted_text = extracted_text.strip()
        if not extracted_text: return jsonify({"error": "Could not extract text from file"}), 400
        try:
            import psycopg2
            pg_host=os.getenv("PGHOST","localhost"); pg_port=int(os.getenv("PGPORT","5432"))
            pg_user=os.getenv("PGUSER","postgres"); pg_password=os.getenv("PGPASSWORD","") or "orlha"
            pg_db=os.getenv("PGDATABASE","candidate_db")
            conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
            cur=conn.cursor()
            cur.execute("UPDATE login SET jd = %s WHERE username = %s", (extracted_text, username))
            updated = cur.rowcount
            conn.commit(); cur.close(); conn.close()
            if updated == 0: return jsonify({"error": "Username not found"}), 404
        except Exception as e: return jsonify({"error": f"DB error storing JD: {e}"}), 500
        try:
            from chat_gemini_review import analyze_job_description
            analysis_result = analyze_job_description(extracted_text)
            parsed = analysis_result.get("parsed", {})
            skills = parsed.get("skills", [])
            if skills: _persist_jskillset(username, skills)
        except Exception as e: logger.warning(f"Failed to auto-extract skills after upload: {e}")
        return jsonify({"status": "ok", "message": "JD uploaded and stored", "length": len(extracted_text)}), 200
    except Exception as e:
        logger.error(f"[Upload JD] {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/gemini/analyze_jd")
def gemini_jd_analyze():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    text_input = (data.get("text") or "").strip()
    sectors_data = data.get("sectors") or []
    jd_text = text_input
    if not jd_text and username:
        try:
            import psycopg2
            pg_host=os.getenv("PGHOST","localhost"); pg_port=int(os.getenv("PGPORT","5432"))
            pg_user=os.getenv("PGUSER","postgres"); pg_password=os.getenv("PGPASSWORD","") or "orlha"
            pg_db=os.getenv("PGDATABASE","candidate_db")
            conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
            cur=conn.cursor()
            cur.execute("SELECT jd FROM login WHERE username = %s", (username,))
            row = cur.fetchone()
            cur.close(); conn.close()
            if row and row[0]: jd_text = row[0]
        except Exception as e: return jsonify({"error": f"DB fetch error: {e}"}), 500
    if not jd_text: return jsonify({"error": "No JD text provided or found for user"}), 400
    try:
        from chat_gemini_review import analyze_job_description
        result = analyze_job_description(jd_text, sectors_data)
        parsed = result.get("parsed", {})
        skills = parsed.get("skills", [])
        if username and skills: _persist_jskillset(username, skills)
        response_obj = {
            "seniority": parsed.get("seniority"),
            "job_title": parsed.get("job_title"),
            "sectors": parsed.get("sectors") or ([parsed.get("sector")] if parsed.get("sector") else []),
            "country": parsed.get("country"),
            "summary": result.get("summary"),
            "skills": skills
        }
        return jsonify(response_obj), 200
    except Exception as e:
        logger.warning(f"[Gemini JD Analyze] {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/user/update_skills")
def user_update_skills():
    """
    Update user skills in the jskillset field.
    Syncs frontend skill additions/removals to the database immediately.
    """
    try:
        data = request.get_json(force=True, silent=True) or {}
        username = (data.get("username") or "").strip()
        skills = data.get("skills", [])
        
        if not username:
            return jsonify({"error": "username required"}), 400
        
        # Use existing _persist_jskillset function
        success, message = _persist_jskillset(username, skills)
        
        if success:
            return jsonify({"status": "ok", "message": message}), 200
        else:
            return jsonify({"error": message}), 400
    except Exception as e:
        logger.error(f"[Update Skills] {e}")
        return jsonify({"error": str(e)}), 500

@app.post("/process/scan_and_upload_cvs")
def process_scan_and_upload_cvs():
    data = request.get_json(force=True, silent=True) or {}
    directory_path = (data.get("directory_path") or "").strip()
    if not directory_path or not os.path.isdir(directory_path): return jsonify({"error": "Valid directory path is required"}), 400
    try:
        files = [f for f in os.listdir(directory_path) if f.lower().endswith('.pdf')]
        if not files: return jsonify({"uploaded_count": 0, "message": "No PDF files found in directory"}), 200
        import psycopg2
        pg_host=os.getenv("PGHOST","localhost"); pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres"); pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()
        cur.execute("SELECT id, name, linkedinurl FROM process WHERE name IS NOT NULL AND name != ''")
        candidates = cur.fetchall()
        uploaded_count = 0; errors = []
        def normalize(s): return re.sub(r'[^a-z0-9]', '', s.lower())
        candidate_map = {}
        for cid, cname, clink in candidates:
            norm = normalize(cname)
            if len(norm) < 3: continue
            if norm not in candidate_map: candidate_map[norm] = []
            candidate_map[norm].append((cid, clink, cname))
        for fname in files:
            fname_norm = normalize(fname)
            matched_candidate = None
            possible_matches = []
            for norm_name, entries in candidate_map.items():
                if norm_name in fname_norm:
                    for entry in entries: possible_matches.append((len(norm_name), entry))
            if possible_matches:
                possible_matches.sort(key=lambda x: x[0], reverse=True)
                matched_candidate = possible_matches[0][1]
            if matched_candidate:
                cid, clink, cname = matched_candidate
                full_path = os.path.join(directory_path, fname)
                try:
                    with open(full_path, "rb") as f: file_bytes = f.read()
                    binary_cv = psycopg2.Binary(file_bytes)
                    if cid: cur.execute("UPDATE process SET cv = %s WHERE id = %s", (binary_cv, cid))
                    else: cur.execute("UPDATE process SET cv = %s WHERE linkedinurl = %s", (binary_cv, clink))
                    if cur.rowcount > 0:
                        conn.commit()
                        uploaded_count += 1
                        threading.Thread(target=analyze_cv_background, args=(clink, file_bytes)).start()
                    else: errors.append(f"DB update failed for {fname} (Candidate: {cname})")
                except Exception as e:
                    conn.rollback(); errors.append(f"Error processing {fname}: {e}")
        cur.close(); conn.close()
        return jsonify({"uploaded_count": uploaded_count, "errors": errors, "message": f"Scanned {len(files)} files, matched and uploaded {uploaded_count}."}), 200
    except Exception as e:
        logger.error(f"[Batch Upload] {e}")
        return jsonify({"error": str(e)}), 500

@app.get("/")
def index():
    html_file=os.path.join(BASE_DIR, "AutoSourcing.html")
    if os.path.isfile(html_file): return send_from_directory(BASE_DIR, "AutoSourcing.html")
    return "AutoSourcing WebBridge is running! (AutoSourcing.html not found)", 200

@app.get("/AutoSourcing.html")
def autosourcing_explicit(): return send_from_directory(BASE_DIR, "AutoSourcing.html")

@app.get('/favicon.ico')
def favicon():
    path=os.path.join(BASE_DIR, 'favicon.ico')
    if not os.path.isfile(path): abort(404)
    return send_from_directory(BASE_DIR, 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# --- START: New Endpoint to serve data_sorter.json ---
@app.get("/data_sorter.json")
def get_data_sorter_json():
    """
    Serve data_sorter.json if present in static folder.
    This allows frontend or other services to access reference lists (JobFamilyRoles, GeoCountries)
    even when data_sorter.py is not active or directly reachable.
    """
    try:
        # Check standard static location relative to BASE_DIR
        static_folder = os.path.join(BASE_DIR, "static")
        filename = "data_sorter.json"
        file_path = os.path.join(static_folder, filename)
        
        if os.path.isfile(file_path):
            return send_from_directory(static_folder, filename, mimetype='application/json')
        else:
            # Fallback check in base dir just in case
            if os.path.isfile(os.path.join(BASE_DIR, filename)):
                return send_from_directory(BASE_DIR, filename, mimetype='application/json')
            
            return jsonify({"error": "data_sorter.json not found"}), 404
    except Exception as e:
        logger.warning(f"Failed to serve data_sorter.json: {e}")
        return jsonify({"error": str(e)}), 500
# --- END: New Endpoint ---

# --- START: Integration of data_sorter.py ---
try:
    import data_sorter
    if hasattr(data_sorter, 'app'):
        # Ensure session compatibility if needed by sharing secret key
        try:
            data_sorter.app.secret_key = app.secret_key
        except Exception:
            pass

        # Mount data_sorter app at /data_sorter prefix
        app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
            '/data_sorter': data_sorter.app.wsgi_app
        })
        logger.info("Integrated data_sorter app mounted at /data_sorter")
    else:
        logger.warning("data_sorter module found but has no 'app' attribute.")
except ImportError:
    logger.warning("data_sorter.py not found. Skipping integration.")
except Exception as e:
    logger.warning(f"Failed to integrate data_sorter: {e}")
# --- END: Integration ---

if __name__ == '__main__':
    port=int(os.getenv("PORT","8091"))
    logger.info(f"Starting AutoSourcing webbridge on :{port}")
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_CX:
        logger.warning("GOOGLE_CSE_API_KEY/CX not set. Search Results Only / Auto-expand may not produce rows.")
    
    # Using run_simple is implicitly handled by app.run when DispatcherMiddleware wraps the app
    # provided we monkeypatch app.wsgi_app correctly above.
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)