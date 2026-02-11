import json
import os
import time
import re

# Try to import Google generative AI client (Gemini). If unavailable, genai will be None.
try:
    import google.generativeai as genai
except Exception:
    genai = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_SOURCING_MODEL", "gemini-2.5-flash-lite")
if GEMINI_API_KEY and genai:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
    except Exception:
        pass


def _extract_json_fragment(s: str):
    """
    Extract a JSON object from arbitrary text by finding the first { ... } pair
    and attempting to parse it. Returns the parsed dict or None.
    """
    if not isinstance(s, str):
        return None
    s = s.strip()
    st = s.find("{")
    ed = s.rfind("}")
    if st == -1 or ed == -1 or ed <= st:
        return None
    frag = s[st:ed + 1]
    try:
        return json.loads(frag)
    except Exception:
        # try small repairs: remove trailing commas
        repaired = re.sub(r",\s*}", "}", frag)
        repaired = re.sub(r",\s*\]", "]", repaired)
        try:
            return json.loads(repaired)
        except Exception:
            return None


def gemini_json_extract(text):
    """
    Compatibility wrapper - attempts to extract JSON object from model output.
    """
    return _extract_json_fragment(text)


GEMINI_AVAILABLE = bool(GEMINI_API_KEY and genai)


def _pick_list(x):
    if isinstance(x, list):
        return [str(s).strip() for s in x if str(s).strip()]
    if isinstance(x, str) and x.strip():
        # try comma split
        if "," in x:
            return [s.strip() for s in x.split(",") if s.strip()]
        return [x.strip()]
    return []


# --- New helper: robust skill extraction heuristics ---
_COMMON_TECH_TERMS = [
    # Languages
    "python", "java", "c++", "c#", "c", "javascript", "typescript", "golang", "go", "rust", "scala", "kotlin", "swift",
    # Web / frameworks
    "react", "angular", "vue", "django", "flask", "spring", "express", "node", "next.js", "nextjs", "rails",
    # Data / analytics
    "sql", "postgresql", "postgres", "mysql", "mongodb", "redis", "cassandra", "hadoop", "spark", "kafka",
    # Cloud / infra
    "aws", "azure", "gcp", "google cloud", "docker", "kubernetes", "terraform", "ansible", "vmware",
    # ML / data science
    "tensorflow", "pytorch", "scikit-learn", "pandas", "numpy", "keras", "xgboost",
    # Tools / workflows
    "git", "jira", "confluence", "grpc", "rest", "graphql", "api", "selenium", "pytest", "junit",
    # Security / compliance
    "oauth", "saml", "iam", "oauth2", "cis", "iso27001",
    # Payments / domain terms often relevant for Product Manager in banking
    "payments", "clearing", "settlement", "card", "acquiring", "issuing", "risk", "fraud",
    # Misc
    "devops", "sre", "ci/cd", "microservices", "event-driven", "distributed systems", "containerization",
]

# normalize tokens to regex-friendly forms (escape + handle C++/C#)
_COMMON_TECH_PATTERNS = [re.escape(t) for t in _COMMON_TECH_TERMS]
# add some explicit tokens for C++/C#
_COMMON_TECH_PATTERNS += [r"\bc\+\+\b", r"\bc#\b"]


def extract_skills_heuristic(text: str, job_title: str = "", sector: str = "", company: str = ""):
    """
    Heuristic skill extractor:
      - Finds known technical tokens from a curated list
      - Extracts phrases following patterns like "experience with X", "proficient in X"
      - Picks up capitalized technology tokens and common abbreviations
      - Deduplicates and returns a prioritized list (technical first)
    """
    if not text:
        text = ""
    lower = (text or "").lower()
    found = []
    seen = set()

    def add_skill(s):
        k = (s or "").strip()
        if not k:
            return
        k_norm = k.strip().lower()
        if k_norm in seen:
            return
        seen.add(k_norm)
        found.append(k.strip())

    # 1) Scan curated token list
    try:
        for pat in _COMMON_TECH_PATTERNS:
            # use word-boundary-ish search, case-insensitive
            if re.search(rf"(?i){pat}", text or ""):
                # convert to readable form (lower-case token as canonical)
                token = re.sub(r'\\b', '', pat)
                # revert escaped characters for display; prefer original term from list if present
                # We will just use the matched substring for better fidelity
                m = re.search(rf"(?i){pat}", text or "")
                if m:
                    add_skill(m.group(0).strip())
    except Exception:
        pass

    # 2) Phrase patterns: experience with / proficient in / knowledge of / familiar with / using
    try:
        # capture up to 6 tokens or comma-separated list
        for m in re.finditer(r'(?i)(?:experience with|proficient in|knowledge of|familiar with|expert in|using|works with|worked with|experience in)\s+([A-Za-z0-9\+\#\.\-/,& ]{2,180})', text or ""):
            grp = m.group(1).strip()
            # split common separators
            parts = re.split(r'[;,/]| and | or ', grp)
            for p in parts:
                p = p.strip(" .;:,")
                if p:
                    add_skill(p)
    except Exception:
        pass

    # 3) Patterns like "X, Y and Z" after "skills:" or "technologies:" or "requirements:" headings
    try:
        for m in re.finditer(r'(?mi)(?:skills|technologies|tech stack|requirements|must have|responsibilities)[:\-\s]*([A-Za-z0-9\+\#\.\-/,&\s]{2,300})', text or ""):
            grp = m.group(1).strip()
            parts = re.split(r'[;,/]| and | or ', grp)
            for p in parts:
                p = p.strip(" .;:,")
                if p and len(p) < 120:
                    add_skill(p)
    except Exception:
        pass

    # 4) Look for capitalized/ProperCase tokens that are likely tech names (e.g., "Kubernetes", "TensorFlow")
    try:
        for m in re.finditer(r'\b([A-Z][A-Za-z0-9\+\#\.\-]{2,40})\b', text or ""):
            token = m.group(1).strip()
            if token and token.lower() not in {"the", "and", "for", "with", "from", "that", "which"}:
                # prefer tokens that appear in curated list ignoring case
                if token.lower() in map(str.lower, _COMMON_TECH_TERMS):
                    add_skill(token)
                else:
                    # add only if token looks like a tech (contains digits or mixes cases or common suffix)
                    if re.search(r'[A-Za-z0-9]', token) and len(token) <= 40 and token.isalpha():
                        # guard: avoid adding simple English words
                        if token.lower() not in {'product', 'manager', 'business', 'development', 'experience', 'team'}:
                            add_skill(token)
    except Exception:
        pass

    # 5) Use contextual hints: job_title, sector, company may indicate domain skills
    try:
        for src in (job_title, sector, company):
            if not src:
                continue
            for m in re.finditer(r'\b([A-Za-z0-9\+\#\.\-]{2,40})\b', src or ""):
                tok = m.group(1).strip()
                if tok and tok.lower() not in seen:
                    # add domain-specific short tokens (e.g., "payments", "risk")
                    if tok.lower() in {'payments', 'risk', 'fraud', 'compliance', 'card', 'settlement', 'leasing', 'estate', 'property'}:
                        add_skill(tok)
    except Exception:
        pass

    # Final cleanup: normalize some variants (e.g., "aws" -> "AWS", "c++" -> "C++")
    normalized = []
    for s in found:
        s_strip = s.strip()
        # simple canonicalization
        if s_strip.lower() in {'aws'}:
            s_strip = 'AWS'
        elif s_strip.lower() in {'gcp','google cloud'}:
            s_strip = 'GCP'
        elif s_strip.lower() in {'postgres','postgresql'}:
            s_strip = 'PostgreSQL'
        elif s_strip.lower() in {'mysql'}:
            s_strip = 'MySQL'
        elif s_strip.lower() in {'sql'}:
            s_strip = 'SQL'
        elif s_strip.lower() == 'k8s':
            s_strip = 'Kubernetes'
        normalized.append(s_strip)

    # Deduplicate preserving order
    out = []
    seen2 = set()
    for s in normalized:
        k = s.strip().lower()
        if k and k not in seen2:
            seen2.add(k)
            out.append(s.strip())

    # Cap to reasonable number
    return out[:40]


def review_and_flush_session_with_gemini(history_path, reset_func, max_turns=10):
    """
    Reads recent chat history, asks Gemini if session should be flushed.  
    If so, calls reset_func and returns a note for the user.
    """
    if not os.path.isfile(history_path):
        return None  # No history -- nothing to check

    try:
        with open(history_path, "r", encoding="utf-8") as f:
            hist = json.load(f)
    except Exception:
        return None

    last_msgs = hist[-max_turns:]
    transcript = "\n".join(f'{m["role"]}: {m["content"]}' for m in last_msgs)
    prompt = (
        "The following is a dialog between a user and an AI sourcing chatbot for recruiting.\n"
        "If the recent conversation appears incoherent (e.g. the bot repeats itself, fails to update suggestions after new user input, gets stuck, or ignores newly-asked questions/parameters), reply ONLY with: FLUSH SESSION.\n"
        "If the conversation is logical, relevant, and interactive, reply ONLY with: KEEP SESSION.\n"
        "----\n"
        + transcript + "\n"
        "----\n"
        "Your evaluation:"
    )

    if not genai or not GEMINI_API_KEY:
        return None

    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(prompt)
        reply = (resp.text or "").strip().upper()
        if "FLUSH SESSION" in reply:
            try:
                reset_func()
            except Exception:
                pass
            return "Detected a stuck or confusing conversation. Bot context has been reset for a fresh start!"
    except Exception:
        pass
    return None


# --- Job Description analysis helper using Gemini ---
def analyze_job_description(jd_text: str):
    """
    Analyze a job description text with Gemini and return a dict:
    {
      "parsed": {"seniority": str, "job_title": str, "sector": str or sectors:list, "country": str, "skills": [...]},
      "missing": [...],
      "summary": "Are you seeking a ...?",
      "suggestions": [...],
      "justification": "...",
      "raw": "<raw model output>",
      "observation": "Concise paragraph explaining model reasoning",
      "specific": "Yes"/"No",
      "skills": [...]
    }

    This function first attempts to call Gemini (if configured) with a strict JSON prompt.
    If Gemini isn't available or the model output cannot be parsed, it falls back to heuristics.
    """
    result = {
        "parsed": {"seniority": "", "job_title": "", "sector": "", "country": "", "skills": []},
        "missing": [],
        "summary": "",
        "raw": "",
        "suggestions": [],
        "justification": "",
        "observation": "",
        "specific": "No",
        "skills": []
    }

    if not jd_text or not isinstance(jd_text, str) or not jd_text.strip():
        result["missing"] = ["job_title", "sector", "country"]
        result["summary"] = "I couldn't analyze an empty job description."
        return result

    # Construct a careful prompt that asks for strict JSON including an "observation" field and "skills"
    prompt = (
        "You are a recruiting assistant that extracts structured sourcing tags from a Job Description and explains the reasoning.\n"
        "Return STRICT JSON ONLY. The JSON object must contain these keys exactly:\n"
        " - parsed: { seniority, job_title, sector, country, skills }\n"
        " - missing: array of strings from ['seniority','job_title','sector','country'] that could not be determined\n"
        " - summary: a one-line confirmation question following template: 'Are you seeking a (seniority) (job_title) in the (sector) based in (country)?' (omit empty parts gracefully)\n"
        " - suggestions: a short array (max 4) of alternative role titles that could fit this JD (strings)\n"
        " - justification: a 1-3 sentence explanation of which phrases in the JD led you to the parsed values\n"
        " - observation: a short (1-3 sentence) paragraph giving an interpretive observation connecting the JD content to candidate profile expectations (e.g., required expertise, likely team, recommended alternate titles)\n"
        " - skills: an array of technical competencies and tools derived from the JD and the detected job context. Prioritize programming languages, frameworks, cloud and data technologies, infrastructure tools, and domain-specific skills. Provide concise names only (strings).\n"
        " - specific: 'Yes' if job_title, sector and country are all confidently identified, otherwise 'No'\n"
        "Rules:\n"
        "- Output JSON ONLY and nothing else.\n"
        "- If a field is missing, return an empty string or empty list as appropriate.\n"
        "- For sector prefer hierarchical labels like 'Financial Services > Banking' where applicable.\n\n"
        f"JOB DESCRIPTION TEXT:\n{jd_text[:15000]}\n\nJSON:"
    )

    # Try using Gemini if available
    if genai and GEMINI_API_KEY:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            resp = model.generate_content(prompt)
            raw_out = (resp.text or "").strip()
            result["raw"] = raw_out

            parsed_json = _extract_json_fragment(raw_out)
            if isinstance(parsed_json, dict):
                parsed_block = parsed_json.get("parsed", {}) or {}
                seniority = (parsed_block.get("seniority") or "").strip()
                job_title = (parsed_block.get("job_title") or parsed_block.get("role") or "").strip()
                sector_val = parsed_block.get("sector") or parsed_block.get("sectors") or ""
                # normalize sectors into list when appropriate
                sectors_list = []
                if isinstance(sector_val, list):
                    sectors_list = [str(s).strip() for s in sector_val if str(s).strip()]
                elif isinstance(sector_val, str) and sector_val.strip():
                    if "," in sector_val and ">" not in sector_val:
                        sectors_list = [s.strip() for s in sector_val.split(",") if s.strip()]
                    else:
                        sectors_list = [sector_val.strip()]
                country = (parsed_block.get("country") or parsed_block.get("location") or "").strip()

                # Extract skills from parsed_block or top-level
                raw_skills = parsed_block.get("skills") or parsed_json.get("skills") or parsed_json.get("skillsets") or []
                skills = _pick_list(raw_skills)

                suggestions_raw = parsed_json.get("suggestions", []) or []
                suggestions = []
                if isinstance(suggestions_raw, str):
                    if "," in suggestions_raw:
                        suggestions = [x.strip() for x in suggestions_raw.split(",") if x.strip()]
                    elif suggestions_raw.strip():
                        suggestions = [suggestions_raw.strip()]
                elif isinstance(suggestions_raw, list):
                    suggestions = [str(x).strip() for x in suggestions_raw if str(x).strip()]

                summary = (parsed_json.get("summary") or "").strip()
                justification = (parsed_json.get("justification") or parsed_json.get("reason") or "").strip()
                observation = (parsed_json.get("observation") or "").strip()
                missing = parsed_json.get("missing") if isinstance(parsed_json.get("missing"), list) else []
                specific = (parsed_json.get("specific") or "").strip() or ("Yes" if (job_title and sectors_list and country) else "No")

                # If Gemini didn't return skills, try heuristic extraction quickly
                if not skills:
                    skills = extract_skills_heuristic(jd_text, job_title, sectors_list[0] if sectors_list else "", "")

                result["parsed"]["seniority"] = seniority
                result["parsed"]["job_title"] = job_title
                # Represent sector as the first sector string (for compatibility), and also provide sectors array if multiple
                result["parsed"]["sector"] = sectors_list[0] if sectors_list else ""
                # attach sectors array
                if sectors_list:
                    result["parsed"]["sectors"] = sectors_list
                result["parsed"]["country"] = country
                result["parsed"]["skills"] = skills

                result["summary"] = summary
                result["missing"] = missing
                result["suggestions"] = suggestions
                result["justification"] = justification
                result["observation"] = observation or ""
                result["raw"] = raw_out
                result["specific"] = specific or ("Yes" if (job_title and sectors_list and country) else "No")
                result["skills"] = skills

                # Ensure missing computed if model didn't provide
                if not isinstance(result["missing"], list) or result["missing"] is None:
                    computed = []
                    if not seniority: computed.append("seniority")
                    if not job_title: computed.append("job_title")
                    if not sectors_list: computed.append("sector")
                    if not country: computed.append("country")
                    result["missing"] = computed

                # Ensure suggestions fallback
                if not result["suggestions"] and job_title:
                    result["suggestions"] = _heuristic_suggestions(job_title)

                # If observation absent, synthesize short observation from justification + jd excerpt
                if not result["observation"]:
                    if justification:
                        result["observation"] = justification
                    else:
                        excerpt = jd_text.replace("\n", " ")[:600].strip()
                        if job_title or sectors_list or country:
                            pieces = []
                            if job_title:
                                pieces.append(f"{job_title}")
                            if sectors_list:
                                pieces.append(f"{', '.join(sectors_list)}")
                            if country:
                                pieces.append(f"{country}")
                            lead = "Based on the Job Description, this role appears to be " + ", ".join(pieces) + "."
                            if excerpt:
                                lead += f" Notable JD excerpt: \"{excerpt[:300]}...\""
                            result["observation"] = lead
                        else:
                            result["observation"] = excerpt or justification or ""
                return result
        except Exception:
            # any issue with Gemini generation should fall through to heuristics
            pass

    # --- Fallback heuristics if Gemini not available or parsing failed ---
    # Try to use lighter heuristics and produce an observation.

    # Attempt to use helper modules if present
    try:
        from chat_extract import detect_seniority, extract_sectors_regex, extract_country_hint
    except Exception:
        detect_seniority = None
        extract_sectors_regex = None
        extract_country_hint = None

    text = jd_text or ""
    lower = text.lower()

    # job title heuristic
    job_title = heuristic_job_title_from_text(text)

    # seniority heuristic
    seniority = ""
    try:
        if detect_seniority:
            seniority = detect_seniority(text) or ""
    except Exception:
        seniority = ""
    if not seniority and job_title:
        jtlow = job_title.lower()
        if re_search_word("senior", jtlow): seniority = "Senior"
        elif re_search_word("lead", jtlow): seniority = "Lead"
        elif re_search_word("manager", jtlow): seniority = "Manager"

    # sector heuristic
    sectors_list = []
    try:
        if extract_sectors_regex:
            sectors_list = extract_sectors_regex(text) or []
    except Exception:
        sectors_list = []
    if not sectors_list:
        sec = heuristic_sector_from_text(lower)
        if sec:
            sectors_list = [sec]

    # country heuristic
    country = ""
    try:
        if extract_country_hint:
            country = extract_country_hint(text) or ""
    except Exception:
        country = ""
    if not country:
        country = heuristic_country_from_text(lower) or ""

    # Build missing list
    missing = []
    if not seniority:
        missing.append("seniority")
    if not job_title:
        missing.append("job_title")
    if not sectors_list:
        missing.append("sector")
    if not country:
        missing.append("country")

    # skills extraction via heuristic
    skills = extract_skills_heuristic(text, job_title, sectors_list[0] if sectors_list else "", "")

    # summary
    if job_title and sectors_list and country:
        summary = f"Are you seeking a {seniority + ' ' if seniority else ''}{job_title} in the {sectors_list[0]} sector based in {country}?"
    elif job_title and country:
        summary = f"Are you seeking a {seniority + ' ' if seniority else ''}{job_title} based in {country}?"
    elif job_title and sectors_list:
        summary = f"Are you seeking a {seniority + ' ' if seniority else ''}{job_title} in the {sectors_list[0]} sector?"
    elif job_title:
        summary = f"Are you seeking a {seniority + ' ' if seniority else ''}{job_title}?"

    # suggestions heuristic
    suggestions = _heuristic_suggestions(job_title) if job_title else []

    # justification: short heuristic sentence
    justification_parts = []
    if "automation" in lower or "automate" in lower or "process optimization" in lower:
        justification_parts.append("JD emphasizes process automation and optimization.")
    if "product design" in lower or "product & design" in lower or "product team" in lower:
        justification_parts.append("Mentions Product & Design team involvement.")
    if "bank" in lower or "banking" in lower or "financial" in lower or "payments" in lower:
        justification_parts.append("References to financial products or banking context.")
    if country:
        justification_parts.append(f"Location hint: {country}.")
    justification = " ".join(justification_parts) or (text.replace("\n", " ")[:300].strip() or "")

    # observation: interpretative paragraph
    if job_title or sectors_list or country:
        obs_pieces = []
        if "process optimization" in lower or "automation" in lower:
            obs_pieces.append("the role emphasises process optimization and automation")
        if "product" in lower or "product design" in lower or "product & design" in lower:
            obs_pieces.append("it sits within Product & Design functions")
        if "bank" in lower or "banking" in lower or "financial" in lower:
            obs_pieces.append("the discipline aligns with Financial Services, particularly Banking")
        if obs_pieces:
            observation = ("Based on the Job Description, " + ", ".join(obs_pieces) +
                           (f". Candidate experience in automation and consulting would be preferred." if "automation" in lower or "consult" in lower else "."))
        else:
            # Compose a generic observation
            observation = f"Based on the Job Description, this appears to be a {seniority + ' ' if seniority else ''}{job_title or 'role'}" + (f" in {', '.join(sectors_list)}" if sectors_list else "") + (f" based in {country}" if country else "") + "."
        # Add suggested alternate title note (if any)
        if suggestions:
            observation += " Alternatives to consider: " + ", ".join(suggestions[:3]) + "."
    else:
        observation = justification or summary or (text.replace("\n", " ")[:300].strip())

    result["parsed"]["seniority"] = seniority or ""
    result["parsed"]["job_title"] = job_title or ""
    result["parsed"]["sector"] = sectors_list[0] if sectors_list else ""
    if sectors_list:
        result["parsed"]["sectors"] = sectors_list
    result["parsed"]["country"] = country or ""
    result["parsed"]["skills"] = skills or []
    result["missing"] = missing
    result["summary"] = summary or ""
    result["suggestions"] = suggestions
    result["justification"] = justification
    result["observation"] = observation
    result["raw"] = ""  # No model raw output in heuristic path
    result["specific"] = "Yes" if (job_title and sectors_list and country) else "No"
    result["skills"] = skills or []

    return result


# --- Clarify & Interpret helpers using Gemini (or robust fallback) ---

def clarify_jd_tags(jd_text: str, parsed: dict = None) -> str:
    """
    Produce a concise clarifying question that asks the user which extracted tags are incorrect.
    Tries Gemini first and falls back to deterministic phrasing.
    Returns a user-facing string question.
    """
    parsed = parsed or {}
    detected = {
        "job_title": parsed.get("job_title") or "",
        "seniority": parsed.get("seniority") or "",
        "sectors": parsed.get("sectors") or ([parsed.get("sector")] if parsed.get("sector") else []),
        "country": parsed.get("country") or "",
        "skills": parsed.get("skills") or []
    }

    def _fallback():
        parts = []
        if detected["job_title"]:
            parts.append(f"Job title: {detected['job_title']}")
        if detected["seniority"]:
            parts.append(f"Seniority: {detected['seniority']}")
        if detected["sectors"]:
            parts.append(f"Sector: {', '.join(detected['sectors'])}")
        if detected["country"]:
            parts.append(f"Country: {detected['country']}")
        if detected["skills"]:
            parts.append(f"Skills: {', '.join(detected['skills'][:8])}")
        if parts:
            return (
                "I detected the following from the Job Description:\n\n" +
                "\n".join(parts) +
                "\n\nWhich of these are incorrect or would you like to change? You can reply with corrections like:\n"
                "- Job title: <correct title>\n- Country: <correct country>\nOr simply type the corrected information."
            )
        return "I couldn't reliably extract tags from the Job Description. Please provide the Job Title and Country (e.g., Product Manager in Malaysia), or type which fields you'd like to correct."

    if GEMINI_AVAILABLE:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            # Build a minimal system + user prompt to ask for a short clarifying question
            prompt = (
                "SYSTEM:\nYou are a recruiting assistant. The user uploaded a Job Description and the system extracted tags.\n"
                "Produce a short, polite clarifying question that lists the detected tags and asks the user which ones are incorrect or need updating.\n"
                "Do NOT include any internal reasoning or extra explanation. Keep it concise (max 3 sentences).\n\n"
                f"DETECTED (JSON): {json.dumps(detected, ensure_ascii=False)}\n\n"
                "OUTPUT:"
            )
            resp = model.generate_content(prompt)
            out = (resp.text or "").strip()
            if out:
                # Basic safety: must ask which parts are incorrect or request corrections
                low = out.lower()
                if ("which" in low and ("incorrect" in low or "change" in low)) or "please" in low:
                    return out
        except Exception:
            pass
    return _fallback()


def interpret_jd_corrections(jd_text: str, prior_parsed: dict, user_correction_text: str) -> dict:
    """
    Interpret a user's free-text correction into structured tags:
    Returns a dict with keys: job_title (str), seniority (str), sectors (list), country (str)
    Attempts Gemini strict JSON first; falls back to regex heuristics and local helpers.
    """
    # Default result fallback uses prior_parsed values
    result = {
        "job_title": (prior_parsed.get("job_title") if isinstance(prior_parsed, dict) else "") or "",
        "seniority": (prior_parsed.get("seniority") if isinstance(prior_parsed, dict) else "") or "",
        "sectors": (prior_parsed.get("sectors") if isinstance(prior_parsed, dict) else None) or ([prior_parsed.get("sector")] if isinstance(prior_parsed, dict) and prior_parsed.get("sector") else []),
        "country": (prior_parsed.get("country") if isinstance(prior_parsed, dict) else "") or ""
    }

    # Try Gemini to parse into strict JSON
    if GEMINI_AVAILABLE:
        try:
            model = genai.GenerativeModel(GEMINI_MODEL)
            prompt = (
                "SYSTEM:\nYou are a concise JSON extractor for recruiting metadata corrections. "
                "The user will reply with corrections to previously extracted tags from a Job Description.\n"
                "Return STRICT JSON ONLY with EXACT keys: {job_title, seniority, sectors, country}.\n"
                "- job_title: string or empty\n"
                "- seniority: string or empty\n"
                "- sectors: array of strings (may be empty)\n"
                "- country: string or empty\n"
                "Rules:\n"
                "- Only emit the JSON object and nothing else.\n"
                "- If the user didn't correct a field, you may leave it as empty string or empty array; do not repeat prior values.\n\n"
                f"PRIOR_PARSED: {json.dumps(prior_parsed or {}, ensure_ascii=False)}\n"
                f"USER_REPLY: {user_correction_text}\n\nJSON:"
            )
            resp = model.generate_content(prompt)
            raw = (resp.text or "").strip()
            parsed_json = _extract_json_fragment(raw)
            if isinstance(parsed_json, dict):
                jt = (parsed_json.get("job_title") or "").strip()
                sen = (parsed_json.get("seniority") or "").strip()
                secs = parsed_json.get("sectors") or []
                if isinstance(secs, str):
                    secs = [s.strip() for s in secs.split(",") if s.strip()]
                elif not isinstance(secs, list):
                    secs = []
                country = (parsed_json.get("country") or "").strip()

                # If field empty in JSON, keep prior value; only replace when provided.
                if jt:
                    result["job_title"] = jt
                if sen:
                    result["seniority"] = sen
                if secs:
                    result["sectors"] = secs
                if country:
                    result["country"] = country
                return result
        except Exception:
            pass

    # Fallback: attempt simple structured parsing from user text
    try:
        txt = user_correction_text or ""
        # Look for explicit keyed fields first
        mJT = re.search(r'job\s*title\s*[:\-]\s*(.+)', txt, flags=re.I)
        if mJT:
            jtval = mJT.group(1).strip()
            # strip trailing "in <country>" if present
            m_in = re.search(r'(.+?)\s+in\s+([A-Za-z \-]+)$', jtval, flags=re.I)
            if m_in:
                result["job_title"] = m_in.group(1).strip()
                maybe_ct = m_in.group(2).strip()
                if maybe_ct:
                    result["country"] = maybe_ct
            else:
                result["job_title"] = jtval

        mCT = re.search(r'country\s*[:\-]\s*([A-Za-z \-]+)', txt, flags=re.I)
        if mCT:
            result["country"] = mCT.group(1).strip()

        mSec = re.search(r'(sector|industry)\s*[:\-]\s*(.+)', txt, flags=re.I)
        if mSec:
            sec = mSec.group(2).strip()
            result["sectors"] = [s.strip() for s in re.split(r'[;,/]', sec) if s.strip()]

        mSen = re.search(r'(seniority|level)\s*[:\-]\s*(.+)', txt, flags=re.I)
        if mSen:
            result["seniority"] = mSen.group(2).strip()

        # If none of the above matched, attempt to salvage from short freeform like "Senior Product Manager in Singapore"
        if not (mJT or mCT or mSec or mSen):
            # Try to detect country token
            country_guess = heuristic_country_from_text(txt.lower())
            if country_guess:
                result["country"] = country_guess
                # remove country from text for title extraction
                txt_for_title = re.sub(re.escape(country_guess), "", txt, flags=re.I).strip()
            else:
                txt_for_title = txt

            # Attempt to find seniority token
            sen_guess = None
            if re.search(r'\bsenior\b|\bsr\b|\bprincipal\b|\blead\b', txt_for_title, flags=re.I):
                if re.search(r'\bdirector\b|\bvp\b|\bvice\b', txt_for_title, flags=re.I):
                    sen_guess = "Director"
                elif re.search(r'\bsenior\b|\bsr\b', txt_for_title, flags=re.I):
                    sen_guess = "Senior"
                elif re.search(r'\blead\b', txt_for_title, flags=re.I):
                    sen_guess = "Lead"
            if sen_guess:
                result["seniority"] = sen_guess

            # Heuristic job title extraction: reuse heuristic_job_title_from_text
            maybe_title = heuristic_job_title_from_text(txt_for_title)
            if maybe_title:
                result["job_title"] = maybe_title

            # Heuristic sector extraction
            sec_guess = heuristic_sector_from_text(txt.lower())
            if sec_guess:
                result["sectors"] = [sec_guess]

    except Exception:
        # on any failure, return prior values unchanged
        return result

    # Normalize sectors to list
    if result.get("sectors") is None:
        result["sectors"] = []

    return result


# --- Helpers used by this module ---
import re as _re

def re_search_word(token, text):
    try:
        return bool(_re.search(rf"\b{_re.escape(token)}\b", text, flags=_re.I))
    except Exception:
        return token in text

def heuristic_job_title_from_text(text):
    # Look at first few lines for likely title phrases
    try:
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        role_keywords = ["engineer","developer","programmer","manager","director","designer","scientist","consultant","assistant","specialist","coordinator","analyst","artist","producer","product manager","product"]
        for ln in lines[:8]:
            low = ln.lower()
            for k in role_keywords:
                if re_search_word(k, low):
                    cleaned = _re.sub(r'^[\-\•\*\d\.\s]+', '', ln).strip()
                    # Truncate sensibly
                    return cleaned[:140]
        # fallback: first non-empty line
        return lines[0][:140] if lines else ""
    except Exception:
        return ""

def heuristic_sector_from_text(lower_text):
    common_sectors = {
        "pharma": "Healthcare > Pharma",
        "pharmaceutical": "Healthcare > Pharma",
        "medical device": "Healthcare > Medical Devices",
        "biotech": "Biotech",
        "gaming": "Gaming",
        "semiconductor": "Semiconductor",
        "automotive": "Automotive",
        "retail": "Retail",
        "ecommerce": "Ecommerce",
        "finance": "Financial Services",
        "bank": "Financial Services > Banking",
        "banking": "Financial Services > Banking",
        "fintech": "Financial Services > Fintech",
        "healthcare": "Healthcare",
        "media": "Media",
        "insurance": "Financial Services > Insurance"
    }
    try:
        for k, v in common_sectors.items():
            if k in lower_text:
                return v
    except Exception:
        pass
    return ""

def heuristic_country_from_text(lower_text):
    countries = ["singapore","malaysia","japan","south korea","korea","china","india","australia","united states","united kingdom","germany","france","italy","spain"]
    for c in countries:
        if c in lower_text:
            if c == "korea":
                return "South Korea"
            if c == "united states":
                return "United States"
            return c.title() if not c.startswith("south") else "South Korea"
    return ""

def _heuristic_suggestions(job_title):
    """
    Generate a short list of alternative titles from the base job title using simple rules.
    """
    if not job_title or not isinstance(job_title, str):
        return []
    jt = job_title.strip()
    jt_low = jt.lower()
    suggestions = []
    try:
        if "product manager" in jt_low or "product" in jt_low:
            suggestions = ["Senior Product Manager", "Product Owner", "Digital Product Manager", "Product Manager - Payments"]
        elif "consultant" in jt_low or "consulting" in jt_low:
            suggestions = ["Digital Transformation Consultant", "Automation Consultant", "Senior Consultant"]
        elif "engineer" in jt_low or "developer" in jt_low:
            suggestions = ["Senior " + jt, "Lead " + jt, "Principal " + jt]
        else:
            if not re_search_word("senior", jt_low):
                suggestions.append("Senior " + jt)
            if not re_search_word("lead", jt_low):
                suggestions.append("Lead " + jt)
            suggestions.append(jt + " Manager" if not jt_low.endswith("manager") else "Senior " + jt)
        # Deduplicate and cap
        out = []
        seen = set()
        for s in suggestions:
            key = s.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(s.strip())
            if len(out) >= 4:
                break
        return out
    except Exception:
        return []