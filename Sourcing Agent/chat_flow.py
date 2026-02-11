import re
import os
import json
import time
from chat_state import (
    CONV_CONTEXT, PENDING_CLARIFICATION, PENDING_EXTRACTION,
    PENDING_EXCEL_OR_REVIEW, PENDING_SECTOR_COMPANY,
    LAST_JOB_INFO, LAST_PROFILE_COUNT,
    ctx_update, set_anchor, get_role_tag, force_refresh_role_tag
)
from chat_intent import classify_intent, is_affirmative, is_negative
from chat_extract import prepare_source, middleware_normalize
from chat_utils import extract_country_hint, dedupe
from chat_sourcing import build_friendly_context_phrase, start_sourcing

# Optional Gemini helpers for JD clarification/interpretation
try:
    from chat_gemini_review import interpret_jd_corrections, clarify_jd_tags
except Exception:
    interpret_jd_corrections = None
    clarify_jd_tags = None

# --- Helper stubs and small heuristics to prevent runtime NameError ---
SECTOR_KEYWORDS = [
    "pharma", "pharmaceutical", "pharmaceuticals", "biotech", "gaming", "game", "media",
    "medical", "medical device", "healthcare", "life sciences", "fintech", "finance",
    "semiconductor", "chip", "automotive", "ecommerce", "retail", "clinical", "cardiovascular",
    "real estate", "real-estate", "estate"
]

def heuristic_job_title_from_text(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""
    m = re.search(r'(?mi)^(?:title|role|position)[:\s\-]+(.+)$', text)
    if m:
        return m.group(1).strip()
    for line in text.splitlines():
        s = line.strip()
        if not s:
            continue
        if len(s.split()) <= 10:
            return s
    return ""

def heuristic_sector_from_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    lower = text.lower()
    for kw in SECTOR_KEYWORDS:
        if kw in lower:
            return kw
    return ""

def heuristic_country_from_text(text: str) -> str:
    try:
        cc = extract_country_hint(text)
        return cc or ""
    except Exception:
        common = ["singapore", "malaysia", "china", "japan", "korea", "united states", "usa", "india"]
        lower = (text or "").lower()
        for c in common:
            if c in lower:
                return c.title()
    return ""

def _heuristic_suggestions(job_title: str):
    if not job_title:
        return []
    jt = job_title.strip()
    jt_low = jt.lower()
    suggestions = []
    if "senior" not in jt_low:
        suggestions.append("Senior " + jt)
    if "lead" not in jt_low:
        suggestions.append("Lead " + jt)
    if "manager" not in jt_low and not jt_low.endswith("manager"):
        suggestions.append(jt + " Manager")
    if any(k in jt_low for k in ["clinical", "research", "cra"]):
        suggestions += ["Clinical Trial Manager", "Regulatory Affairs Specialist"]
    seen = set()
    out = []
    for s in suggestions:
        k = s.lower().strip()
        if k and k not in seen:
            seen.add(k)
            out.append(s)
    return out[:15]
# --- End helper stubs ---

# Human-friendly JD clarification prompt builder
def build_jd_clarify_prompt(job_title: str, seniority: str, sector: str, country: str, example_fields=None) -> str:
    lines = []
    lines.append("Thanks — I pulled these details from the job description:")
    if job_title:
        lines.append(f"• Job title: {job_title}")
    if seniority:
        lines.append(f"• Seniority: {seniority}")
    if sector:
        lines.append(f"• Sector: {sector}")
    if country:
        lines.append(f"• Country: {country}")
    lines.append("")
    lines.append("Does that look right?")
    lines.append("If something's off, tell me which fields to change. For example:")
    if example_fields is None:
        example_fields = [
            "Job title: Senior Product Manager",
            "Country: Malaysia"
        ]
    for ex in example_fields:
        lines.append(f"- {ex}")
    lines.append("Or just type the corrected information and I'll update it.")
    return "\n".join(lines)

def get_username():
    try:
        import flask
        user = getattr(flask.g, "user", None)
        if user and hasattr(user, "username"):
            return user.username
        if 'username' in flask.session:
            return flask.session["username"]
    except Exception:
        pass
    return os.getenv("DEFAULT_USER","")

def need_country_first(normalized):
    return normalized.get("job_title") and not normalized.get("country")

def country_query_prompt(job_title):
    jt = (job_title or "").strip()
    return f"Where do you want to find your {jt}?"

def confirmation_prompt(job_title, companies, sectors, country):
    if isinstance(job_title, (list, tuple)):
        jt_text = ", ".join([str(x).strip() for x in job_title if str(x).strip()])
    else:
        jt_text = (job_title or "").strip()
    jt_text = jt_text or "this role"
    comp_list = []
    if isinstance(companies, (list, tuple)) and len(companies) > 0:
        comp_list = [str(c).strip() for c in companies if str(c).strip()][:6]
    sec_list = []
    if isinstance(sectors, (list, tuple)) and len(sectors) > 0:
        sec_list = [str(s).strip() for s in sectors if str(s).strip()][:6]
    country_text = (country or "").strip()

    try:
        from chat_utils import country_to_cc
        if country_text:
            if any(country_text.lower() == c.lower() for c in comp_list) and not country_to_cc(country_text):
                country_text = ""
            elif any(country_text.lower() == s.lower() for s in sec_list) and not country_to_cc(country_text):
                country_text = ""
    except Exception:
        pass

    def _fallback_text():
        sector_display = (sec_list[0] if sec_list else "")
        country_part = f" based in {country_text}" if country_text else ""
        sector_part = f" in the {sector_display} sector" if sector_display else ""
        lead_in = f"Before I proceed with the search for a {jt_text}{sector_part}{country_part}…"
        return (
            f"{lead_in}\n"
            "Would you like me to:\n"
            "1. Start sourcing candidates\n"
            "2. Suggest related job titles and companies to explore?\n"
            "Just click the button to confirm"
        )
    return _fallback_text()

def llm_orchestrate_dialog(user_msg, purpose, fallback_text):
    ctx_snapshot = {
        "job_titles": CONV_CONTEXT.get("job_titles"),
        "country": CONV_CONTEXT.get("country"),
        "companies": CONV_CONTEXT.get("companies"),
        "sectors": CONV_CONTEXT.get("sectors"),
        "role_tag": CONV_CONTEXT.get("role_tag")
    }
    return fallback_text

def build_sourcing_decision_block(jt, ct, companies, sectors):
    base_link=os.getenv("AUTOSOURCING_BASE","http://localhost:8091").rstrip("/")+"/SourcingVerify.html"
    return (f"Ready to launch sourcing for:\n"
            f"Role: {jt or '(missing)'} | Country: {ct or '(missing)'}\n"
            f"Companies: {', '.join(companies[:6]) if companies else '(none)'} | "
            f"Sectors: {', '.join(sectors[:6]) if sectors else '(none)'}\n\n"
            "Respond:\n"
            "  'yes' → start\n"
            "  'review' → live results page\n"
            "  'excel' → file after completion\n"
            f"Live review link (activates after start): {base_link}\n"
            "Or refine further before starting.")

def get_dynamic_company_suggestions(sector="", job_title=""):
    sector_map = {
        "gaming": ["Ubisoft", "Nintendo", "Activision Blizzard", "Epic Games", "Riot Games", "Bandai Namco", "EA", "Valve", "Sony Interactive Entertainment"],
        "media": ["Disney", "Warner Bros", "Universal", "Sony Pictures"],
        "pharma": ["Pfizer", "Novartis", "Merck", "Roche", "AstraZeneca", "Johnson & Johnson", "Sanofi", "Bayer", "GSK", "AbbVie"],
        "biotech": ["Genentech", "Biogen", "Amgen", "Gilead Sciences"],
        "technology": ["Apple", "Microsoft", "Google", "Amazon", "Meta"],
        "graphics": ["Nvidia", "AMD", "Autodesk", "Unreal", "Unity Technologies"],
        "healthcare": ["Siemens Healthineers", "GE Healthcare", "Philips Healthcare"],
        "real estate": ["CBRE", "JLL", "Colliers", "Savills", "Knight Frank"]
    }
    default = ["ACME", "Globex", "Initech", "Umbrella", "Wonka"]

    sector_synonyms = {
        "cardiovascular": "pharma",
        "cardiovascular drug": "pharma",
        "drug": "pharma",
        "medical": "pharma",
        "biotech": "biotech",
        "clinical": "pharma",
        "hospital": "healthcare",
        "medicine": "pharma",
        "healthcare": "healthcare",
        "pharmaceutical": "pharma",
        "pharmaceuticals": "pharma",
        "gaming": "gaming",
        "real estate": "real estate",
        "estate": "real estate"
    }

    remote_companies = []
    try:
        import requests
        base=os.getenv("AUTOSOURCING_BASE","http://localhost:8091").rstrip("/")
        url=f"{base}/suggest"
        country = CONV_CONTEXT.get("country","")
        payload = {
            "jobTitles": [job_title] if job_title else [],
            "companies": [],
            "languages": CONV_CONTEXT.get("languages", []),
            "sectors": [sector] if sector else [],
            "country": country
        }
        resp = requests.post(url, json=payload, timeout=5)
        if resp.status_code == 200:
            data = resp.json() or {}
            comp_block = (data.get("company") or {}).get("related") or []
            remote_companies = [c.strip() for c in comp_block if isinstance(c,str) and c.strip()]
            remote_companies = dedupe(remote_companies)
    except Exception:
        remote_companies = []

    if remote_companies:
        return remote_companies[:30]

    sec_key = (sector or "").lower()
    jt_key = (job_title or "").lower()

    for phrase, mapped in sector_synonyms.items():
        if phrase in sec_key:
            return sector_map.get(mapped, default)
    for phrase, mapped in sector_synonyms.items():
        if phrase in jt_key:
            return sector_map.get(mapped, default)

    if sec_key:
        for key in sector_map:
            if key in sec_key:
                return sector_map[key]
    if "gaming" in jt_key or "game" in jt_key:
        return sector_map["gaming"]
    if "graphics" in jt_key or "graphic" in jt_key:
        return sector_map["graphics"]
    if "media" in jt_key:
        return sector_map["media"]
    if "pharma" in jt_key or "medical" in jt_key:
        return sector_map["pharma"]
    if "biotech" in jt_key:
        return sector_map["biotech"]
    if "real estate" in jt_key or "estate" in jt_key or "property" in jt_key:
        return sector_map["real estate"]
    if "tech" in jt_key or "software" in jt_key or "developer" in jt_key:
        return sector_map["technology"]
    if "healthcare" in jt_key:
        return sector_map["healthcare"]
    return default

# Deduplicate guard for suggestions to avoid double-popup when duplicate requests arrive
def _should_skip_duplicate_suggestions(threshold_seconds: float = 2.0) -> bool:
    last = CONV_CONTEXT.get("last_suggestions_at")
    now = time.time()
    if last and (now - last) < threshold_seconds:
        return True
    CONV_CONTEXT["last_suggestions_at"] = now
    return False

def analyze_jd_summary(jd_text):
    if not jd_text:
        return ("Could not analyze empty job description.", {"job_title": "", "seniority": "", "sector": "", "country": ""})

    try:
        from chat_gemini_review import analyze_job_description as gemini_analyze
        res = gemini_analyze(jd_text)
        if isinstance(res, dict):
            parsed = res.get("parsed", {}) or {}
            out = {
                "job_title": parsed.get("job_title") or parsed.get("role") or "",
                "seniority": parsed.get("seniority") or "",
                "sector": parsed.get("sector") or (parsed.get("sectors")[0] if isinstance(parsed.get("sectors"), list) and parsed.get("sectors") else "") or "",
                "sectors": parsed.get("sectors") if isinstance(parsed.get("sectors"), list) and parsed.get("sectors") else ([parsed.get("sector")] if parsed.get("sector") else []),
                "country": parsed.get("country") or parsed.get("location") or "",
                "suggestions": res.get("suggestions", []) or [],
                "raw": res.get("raw", "") or "",
                "specific": res.get("specific", "No"),
                # Ensure skills are passed through for frontend/context use
                "skills": parsed.get("skills") or res.get("skills") or []
            }
            summary = res.get("summary") or ""
            if not summary:
                jt = out["job_title"]
                sen = out["seniority"]
                sec = out["sector"]
                ct = out["country"]
                if jt and sec and ct:
                    summary = f"Are you seeking a {sen + ' ' if sen else ''}{jt} in the {sec} sector based in {ct}?"
                elif jt and ct:
                    summary = f"Are you seeking a {sen + ' ' if sen else ''}{jt} based in {ct}?"
                elif jt and sec:
                    summary = f"Are you seeking a {sen + ' ' if sen else ''}{jt} in the {sec} sector?"
                elif jt:
                    summary = f"Are you seeking a {sen + ' ' if sen else ''}{jt}?"
            return (summary, out)
    except Exception:
        pass

    try:
        from chat_extract import detect_seniority, extract_sectors_regex, extract_country_hint
    except Exception:
        detect_seniority = None
        extract_sectors_regex = None
        extract_country_hint = None

    jt = heuristic_job_title_from_text(jd_text)
    seniority = ""
    try:
        if detect_seniority:
            seniority = detect_seniority(jd_text) or ""
    except Exception:
        seniority = ""
    if not seniority:
        if re.search(r'\bsenior\b', (jt or "").lower()): seniority = "Senior"
        elif re.search(r'\blead\b', (jt or "").lower()): seniority = "Lead"
        elif re.search(r'\bmanager\b', (jt or "").lower()): seniority = "Manager"

    sectors = []
    try:
        if extract_sectors_regex:
            sectors = extract_sectors_regex(jd_text) or []
    except Exception:
        sectors = []
    if not sectors:
        s = heuristic_sector_from_text(jd_text.lower())
        if s:
            sectors = [s]

    country = ""
    try:
        if extract_country_hint:
            country = extract_country_hint(jd_text) or ""
    except Exception:
        country = ""
    if not country:
        country = heuristic_country_from_text(jd_text.lower()) or ""

    out = {
        "job_title": jt or "",
        "seniority": seniority or "",
        "sector": sectors[0] if sectors else "",
        "sectors": sectors,
        "country": country or "",
        "suggestions": [],
        "raw": "",
        "specific": "Yes" if jt and sectors and country else "No",
        "skills": []
    }

    if out["job_title"] and out["sector"] and out["country"]:
        summary = f"Are you seeking a {out['seniority'] + ' ' if out['seniority'] else ''}{out['job_title']} in the {out['sector']} sector based in {out['country']}?"
    elif out["job_title"] and out["country"]:
        summary = f"Are you seeking a {out['seniority'] + ' ' if out['seniority'] else ''}{out['job_title']} based in {out['country']}?"
    elif out["job_title"] and out["sector"]:
        summary = f"Are you seeking a {out['seniority'] + ' ' if out['seniority'] else ''}{out['job_title']} in the {out['sector']} sector?"
    elif out["job_title"]:
        summary = f"Are you seeking a {out['seniority'] + ' ' if out['seniority'] else ''}{out['job_title']}?"
    else:
        summary = "I could not identify a clear job title from this Job Description."

    if out["job_title"]:
        out["suggestions"] = _heuristic_suggestions(out["job_title"])

    return (summary, out)

def infer_seniority_from_title(title: str, jd_text: str = None) -> str:
    if not title or not isinstance(title, str):
        return ""
    t = title.strip()
    t_low = t.lower()

    if re.search(r'\b(senior|sr\.|sr\b)\b', t_low): return "Senior"
    if re.search(r'\b(principal|director|head|vp|vice|vice-president|vice president)\b', t_low): return "Director"
    if re.search(r'\b(lead|manager|mgr|management)\b', t_low): return "Manager"
    if re.search(r'\b(associate|jr|junior|entry)\b', t_low): return "Associate"

    pm_keywords = ["product manager", "product owner"]
    for k in pm_keywords:
        if k in t_low:
            return "Manager"

    if any(k in t_low for k in ["engineer", "developer", "programmer", "sde"]):
        return "Engineer"

    return ""

def process_message(user_msg, hist_path):
    try:
        if CONV_CONTEXT.get("try_else_modal_active"):
            CONV_CONTEXT["try_else_modal_active"] = False
            try:
                CONV_CONTEXT["job_titles"] = []
                CONV_CONTEXT["companies"] = []
                CONV_CONTEXT["sectors"] = []
                CONV_CONTEXT["country"] = ""
                CONV_CONTEXT["languages"] = []
            except Exception:
                pass
            msg = (
                "Okay, let's start a fresh search.\n"
                "How would you like to search?\n"
                "1. Upload a Job Description\n"
                "2. Provide Job Title and Country manually"
            )
            return {"text": msg, "stage": "initial_choice_prompt"}
    except Exception:
        pass

    if CONV_CONTEXT.get("awaiting_initial_choice", True) and not CONV_CONTEXT.get("job_titles"):
        u_low = (user_msg or "").strip().lower()
        if "upload" in u_low or "job description" in u_low or "option a" in u_low:
            CONV_CONTEXT["awaiting_initial_choice"] = False
            CONV_CONTEXT["awaiting_jd_upload"] = True
            return {"text": "Please upload your Job Description file.", "stage": "jd_upload_request"}
        elif "title" in u_low or "country" in u_low or "option b" in u_low or "manual" in u_low:
            CONV_CONTEXT["awaiting_initial_choice"] = False
            CONV_CONTEXT["awaiting_jd_upload"] = False
        else:
            if len(user_msg.split()) > 2:
                CONV_CONTEXT["awaiting_initial_choice"] = False
                CONV_CONTEXT["awaiting_jd_upload"] = False
            else:
                return {"text": "How would you like to search?\n1. Upload a Job Description\n2. Provide Job Title and Country", "stage": "initial_choice_prompt"}

    if CONV_CONTEXT.get("awaiting_jd_upload"):
        if len(user_msg.split()) > 20:
            CONV_CONTEXT["awaiting_jd_upload"] = False
            summary, data = analyze_jd_summary(user_msg)

            try:
                if isinstance(data, dict) and data.get("job_title") and not data.get("seniority"):
                    inferred = infer_seniority_from_title(data.get("job_title"), user_msg)
                    if inferred:
                        data["seniority"] = inferred
            except Exception:
                pass

            ctx_update({
                "job_titles": [data.get("job_title")] if data.get("job_title") else [],
                "country": data.get("country", ""),
                "sectors": [data.get("sector")] if data.get("sector") else [],
                "seniority": data.get("seniority")
            })
            CONV_CONTEXT["jd_analysis_data"] = data

            missing = []
            if not data.get("job_title"): missing.append("job title")
            if not data.get("country"): missing.append("country")
            if not data.get("sector"): missing.append("sector")

            if missing:
                human_prompt = build_jd_clarify_prompt(
                    job_title=data.get("job_title") or "",
                    seniority=data.get("seniority") or "",
                    sector=data.get("sector") or "",
                    country=data.get("country") or "",
                    example_fields=["Job title: Senior Product Manager", "Country: Malaysia"]
                )
                CONV_CONTEXT["awaiting_jd_clarification"] = True
                return {
                    "text": human_prompt + f"\n\nCould you please specify which {', '.join(missing)} you are targeting?",
                    "stage": "jd_analysis_gap"
                }
            else:
                CONV_CONTEXT["awaiting_jd_confirmation"] = True
                human_confirm = build_jd_clarify_prompt(
                    job_title=data.get("job_title") or "",
                    seniority=data.get("seniority") or "",
                    sector=data.get("sector") or "",
                    country=data.get("country") or "",
                    example_fields=["(Click 'Start sourcing' or 'Suggest related job titles')"]
                )
                return {"text": human_confirm, "stage": "jd_analysis_confirm"}
        else:
            if is_negative(user_msg):
                CONV_CONTEXT["awaiting_jd_upload"] = False
                return {"text": "Okay, switching to manual mode. What Job Title are you looking for?", "stage": "ask_job_title"}
            return {"text": "Please upload the Job Description file or paste the text here.", "stage": "jd_upload_request"}

    if CONV_CONTEXT.get("awaiting_jd_clarification"):
        parsed_gap = prepare_source(user_msg, hist_path)
        applied = False

        try:
            if interpret_jd_corrections and isinstance(interpret_jd_corrections, type(interpret_jd_corrections)):
                prior = CONV_CONTEXT.get("jd_analysis_data") or {}
                try:
                    interpreted = interpret_jd_corrections("", prior, user_msg)
                    if isinstance(interpreted, dict):
                        updates = {}
                        if interpreted.get("job_title"):
                            updates["job_titles"] = [interpreted.get("job_title")]
                        if interpreted.get("country"):
                            updates["country"] = interpreted.get("country")
                        if interpreted.get("sectors"):
                            updates["sectors"] = interpreted.get("sectors")
                        if interpreted.get("seniority"):
                            updates["seniority"] = interpreted.get("seniority")
                        if updates:
                            try:
                                ctx_update(updates)
                                jd = CONV_CONTEXT.get("jd_analysis_data") or {}
                                if updates.get("job_titles"):
                                    jd["job_title"] = updates["job_titles"][0]
                                if updates.get("country"):
                                    jd["country"] = updates["country"]
                                if updates.get("sectors"):
                                    jd["sectors"] = updates["sectors"]
                                    jd["sector"] = (updates["sectors"][0] if isinstance(updates["sectors"], list) and updates["sectors"] else jd.get("sector",""))
                                if updates.get("seniority"):
                                    jd["seniority"] = updates.get("seniority")
                                CONV_CONTEXT["jd_analysis_data"] = jd
                                applied = True
                            except Exception:
                                applied = False
                except Exception:
                    applied = False
        except Exception:
            applied = False

        if not applied:
            try:
                if parsed_gap.get("job_title") or parsed_gap.get("country") or parsed_gap.get("sectors"):
                    ctx_update({
                        "job_titles": [parsed_gap.get("job_title")] if parsed_gap.get("job_title") else CONV_CONTEXT.get("job_titles"),
                        "country": parsed_gap.get("country") or CONV_CONTEXT.get("country"),
                        "sectors": parsed_gap.get("sectors") or CONV_CONTEXT.get("sectors")
                    })
                    jd = CONV_CONTEXT.get("jd_analysis_data") or {}
                    if parsed_gap.get("job_title"):
                        jd["job_title"] = parsed_gap.get("job_title")
                    if parsed_gap.get("country"):
                        jd["country"] = parsed_gap.get("country")
                    if parsed_gap.get("sectors"):
                        jd["sectors"] = parsed_gap.get("sectors")
                        jd["sector"] = (parsed_gap.get("sectors")[0] if isinstance(parsed_gap.get("sectors"), list) and parsed_gap.get("sectors") else jd.get("sector",""))
                    if parsed_gap.get("seniority"):
                        jd["seniority"] = parsed_gap.get("seniority")
                    CONV_CONTEXT["jd_analysis_data"] = jd
                    CONV_CONTEXT["awaiting_jd_clarification"] = False
                    CONV_CONTEXT["awaiting_jd_confirmation"] = True

                    jt = (CONV_CONTEXT.get("job_titles") or [""])[0]
                    ct = CONV_CONTEXT.get("country")
                    sec = (CONV_CONTEXT.get("sectors") or [""])[0]
                    sen = CONV_CONTEXT.get("jd_analysis_data", {}).get("seniority", "")

                    summary = build_jd_clarify_prompt(jt, sen, sec, ct, example_fields=["(Click 'Start sourcing' or 'Suggest related job titles')"])
                    return {"text": summary, "stage": "jd_analysis_confirm"}
            except Exception:
                pass

        return {"text": "Sorry, I didn't catch that. Which field would you like to change — Job title, Country, Sector or Seniority? Example: 'Job title: Senior Product Manager in Singapore'", "stage": "jd_analysis_gap"}

    if CONV_CONTEXT.get("awaiting_jd_confirmation"):
        try:
            low = (user_msg or "").strip().lower()
            def _is_suggest_command_local(txt: str) -> bool:
                if not txt: return False
                t = txt.strip().lower()
                return t in {"suggest","suggestions","ideas","more ideas","more options"} or "suggest" in t
            if _is_suggest_command_local(user_msg):
                # Dedupe: avoid double popup if suggestions were just returned
                if _should_skip_duplicate_suggestions():
                    return {"text": "I've just shown suggestions — please pick from the list above.", "stage": "suggestion_ack"}

                role = (CONV_CONTEXT.get("job_titles") or [""])[0]
                sectors = CONV_CONTEXT.get("sectors") or []
                country = CONV_CONTEXT.get("country") or ""
                seniority = CONV_CONTEXT.get("jd_analysis_data", {}).get("seniority", "")

                company_list = get_dynamic_company_suggestions(sector=sectors[0] if sectors else "", job_title=role)
                job_title_ideas = []
                try:
                    import requests
                    base=os.getenv("AUTOSOURCING_BASE","http://localhost:8091").rstrip("/")
                    url=f"{base}/suggest"
                    payload = {
                        "jobTitles": [role],
                        "companies": [],
                        "languages": CONV_CONTEXT.get("languages", []),
                        "sectors": sectors,
                        "country": country
                    }
                    r=requests.post(url, json=payload, timeout=5)
                    if r.status_code==200:
                        d=r.json() or {}
                        jblk=(d.get("job") or {}).get("related") or []
                        job_title_ideas=[t.strip() for t in jblk if isinstance(t,str) and t.strip()][:15]
                except Exception:
                    job_title_ideas = []

                suggestion_lines = []
                suggestion_lines.append("Company ideas:\n • " + "; ".join(company_list))
                if job_title_ideas:
                    suggestion_lines.append("Job Title ideas:\n • " + "; ".join(job_title_ideas))
                else:
                    if role:
                        jt_low = (role or "").lower()
                        fallback = []
                        if not re.search(r"\bsenior\b", jt_low): fallback.append("Senior " + role)
                        if not re.search(r"\blead\b", jt_low): fallback.append("Lead " + role)
                        if not re.search(r"\bmanager\b", jt_low) and not role.lower().endswith("manager"): fallback.append(role + " Manager")
                        job_title_ideas = fallback[:15]
                        suggestion_lines.append("Job Title ideas:\n • " + "; ".join(job_title_ideas))

                msg = "Here are some ideas:\n" + "\n".join(suggestion_lines)
                return {
                    "text": msg,
                    "stage": "suggestion",
                    "action": "show_suggestions",
                    "jobs": job_title_ideas,
                    "companies": company_list
                }
        except Exception:
            pass

        if is_affirmative(user_msg):
            CONV_CONTEXT["awaiting_jd_confirmation"] = False

            # Dedupe for the follow-up suggestions as well
            if _should_skip_duplicate_suggestions():
                return {"text": "I've already provided suggestions; you can use the list above.", "stage": "suggestion_ack"}

            role = (CONV_CONTEXT.get("job_titles") or [""])[0]
            sectors = CONV_CONTEXT.get("sectors") or []
            country = CONV_CONTEXT.get("country") or ""
            seniority = CONV_CONTEXT.get("jd_analysis_data", {}).get("seniority", "")
            company_list = get_dynamic_company_suggestions(sector=sectors[0] if sectors else "", job_title=role)
            job_title_ideas = []
            try:
                import requests
                base=os.getenv("AUTOSOURCING_BASE","http://localhost:8091").rstrip("/")
                url=f"{base}/suggest"
                payload = {
                    "jobTitles": [role],
                    "companies": [],
                    "languages": CONV_CONTEXT.get("languages", []),
                    "sectors": sectors,
                    "country": country
                }
                r=requests.post(url, json=payload, timeout=5)
                if r.status_code==200:
                    d=r.json() or {}
                    jblk=(d.get("job") or {}).get("related") or []
                    job_title_ideas=[t.strip() for t in jblk if isinstance(t,str) and t.strip()][:15]
            except Exception:
                job_title_ideas = []

            msg = "Great. Here are some relevant companies and job titles:\n" + "\n".join([
                "Company ideas:\n • " + "; ".join(company_list),
                "Job Title ideas:\n • " + "; ".join(job_title_ideas) if job_title_ideas else "Job Title ideas:\n • (no remote suggestions)"
            ]) + "\n\nWould you like to start sourcing with these?"
            pending = {
                "jt": role,
                "ct": country,
                "companies": company_list if isinstance(company_list, list) else [],
                "sectors": sectors,
                "seniority": seniority
            }
            return {
                "text": msg,
                "stage": "confirm_context",
                "pending_extraction": pending,
                "action": "show_suggestions",
                "jobs": job_title_ideas if 'job_title_ideas' in locals() else [],
                "companies": company_list if isinstance(company_list, list) else []
            }
        elif is_negative(user_msg):
            CONV_CONTEXT["awaiting_jd_confirmation"] = False
            try:
                jd = CONV_CONTEXT.get("jd_analysis_data") or {}
                if clarify_jd_tags:
                    try:
                        prompt = clarify_jd_tags("", jd)
                        if prompt:
                            CONV_CONTEXT["awaiting_jd_clarification"] = True
                            return {"text": prompt, "stage": "jd_analysis_gap"}
                    except Exception:
                        pass
            except Exception:
                pass
            return {"text": "Okay — let's refine it. Please tell me the correct Job Title and Country.", "stage": "ask_job_title"}
        else:
            return {"text": "I don’t understand. Are you seeking this role? (Yes/No)", "stage": "jd_analysis_confirm"}

    parsed = prepare_source(user_msg, hist_path)
    normalized = middleware_normalize(user_msg, hist_path)
    intent = classify_intent(user_msg)

    def _is_suggest_command(txt: str) -> bool:
        t = (txt or "").strip().lower()
        return t in {
            "suggest","suggestions","suggest jobs","suggest job titles",
            "suggest companies","more ideas","more options","ideas"
        }
    if _is_suggest_command(user_msg):
        try:
            intent["mode"] = "suggestion"
            intent["kind"] = intent.get("kind") or "both"
        except Exception:
            intent = {"mode": "suggestion", "kind": "both"}

    country_hint = extract_country_hint(user_msg)
    if country_hint and not parsed.get("country"):
        parsed["country"] = country_hint
        normalized["country"] = country_hint

    username = get_username()
    role_tag_fresh = (CONV_CONTEXT.get("role_tag") or "").strip()

    try:
        if CONV_CONTEXT.get("awaiting_role_tag_confirm"):
            role_tag = role_tag_fresh
            if not role_tag:
                CONV_CONTEXT["awaiting_role_tag_confirm"] = False
            else:
                if is_affirmative(user_msg):
                    if not parsed.get("job_title"):
                        parsed["job_title"] = role_tag
                    if not normalized.get("job_title"):
                        normalized["job_title"] = role_tag
                    CONV_CONTEXT["role_tag"] = role_tag
                    CONV_CONTEXT["awaiting_role_tag_confirm"] = False
                elif is_negative(user_msg):
                    CONV_CONTEXT["awaiting_role_tag_confirm"] = False
                    msg = llm_orchestrate_dialog(user_msg, "ask_job_title", "What Job Title are you trying to find?")
                    return {"text": msg, "stage": "ask_job_title"}
                else:
                    msg = llm_orchestrate_dialog(user_msg, "confirm_role_tag", f"Are you looking for a {role_tag} role?")
                    return {"text": msg, "stage": "confirm_role_tag"}
    except Exception:
        pass

    try:
        ctx_jt = ((CONV_CONTEXT.get("job_titles") or [None])[0] or "").strip()
        ctx_ct = (CONV_CONTEXT.get("country") or "").strip()
        ctx_companies_list = [c.strip() for c in (CONV_CONTEXT.get("companies") or []) if isinstance(c, str) and c.strip()]

        new_full_jt = parsed.get("job_title") or normalized.get("job_title")
        new_full_ct = parsed.get("country") or normalized.get("country")
        if new_full_jt and new_full_ct:
            if (new_full_jt.strip().lower() != ctx_jt.strip().lower()) or (new_full_ct.strip().lower() != ctx_ct.strip().lower()):
                ctx_update({
                    "job_titles": [new_full_jt],
                    "country": new_full_ct,
                    "companies": parsed.get("companies", []),
                    "sectors": parsed.get("sectors", [])
                })
                CONV_CONTEXT["role_tag"] = new_full_jt.strip()
                pending = {
                    "jt": new_full_jt,
                    "ct": new_full_ct,
                    "companies": parsed.get("companies", []),
                    "sectors": parsed.get("sectors", []),
                    "seniority": parsed.get("seniority")
                }
                return {"text": confirmation_prompt(new_full_jt, parsed.get("companies",[]), parsed.get("sectors",[]), new_full_ct),
                        "stage": "confirm_context", "pending_extraction": pending}

        if ctx_jt and ctx_ct:
            user_low = (user_msg or "").strip().lower()
            allowed_markers = ("yes","yep","yeah","yup","ok","okay","proceed","start","begin","review","excel","suggest")
            if any(tok in user_low for tok in allowed_markers):
                pass
            else:
                if "change" in user_low or "different" in user_low or "new" in user_low:
                    CONV_CONTEXT["job_titles"] = []
                    CONV_CONTEXT["companies"] = []
                    CONV_CONTEXT["sectors"] = []
                    if parsed.get("job_title"):
                        if parsed.get("country"):
                            ctx_update({
                                "job_titles":[parsed.get("job_title")],
                                "country": parsed.get("country"),
                                "companies": parsed.get("companies",[]),
                                "sectors": parsed.get("sectors",[])
                            })
                            CONV_CONTEXT["role_tag"] = parsed.get("job_title")
                            pending = {
                                "jt": parsed.get("job_title"),
                                "ct": parsed.get("country"),
                                "companies": parsed.get("companies", []),
                                "sectors": parsed.get("sectors", []),
                                "seniority": parsed.get("seniority")
                            }
                            return {
                                "text": confirmation_prompt(parsed.get("job_title"), parsed.get("companies",[]), parsed.get("sectors",[]), parsed.get("country")),
                                "stage": "confirm_context",
                                "pending_extraction": pending
                            }
                    ask_jt = llm_orchestrate_dialog(user_msg, "ask_job_title", "What Job Title are you trying to find?")
                    return {"text": ask_jt, "stage": "ask_job_title"}
    except Exception:
        pass

    if intent.get("mode") == "suggestion":
        # Dedupe remote suggestions to avoid double-popup from duplicate requests
        if _should_skip_duplicate_suggestions():
            return {"text": "I've just shown suggestions — please pick from the list above.", "stage": "suggestion_ack"}

        role = parsed.get("job_title") or normalized.get("job_title")
        if not role:
            role = (CONV_CONTEXT.get("job_titles") or [None])[0]
        if not role:
            role = (CONV_CONTEXT.get("role_tag") or "").strip()
        if not role and username:
            try:
                force_refresh_role_tag(username)
                role = (CONV_CONTEXT.get("job_titles") or [None])[0] or (CONV_CONTEXT.get("role_tag") or "").strip()
            except Exception:
                role = (CONV_CONTEXT.get("role_tag") or "").strip()

        sectors = [s.lower() for s in (parsed.get("sectors") or normalized.get("sectors") or [])]
        job_title_phrase = (role or "").lower()
        user_phrase = (user_msg or "").lower()
        country_for_suggest = normalized.get("country") or parsed.get("country") or CONV_CONTEXT.get("country","")

        best_sector = ""
        all_phrases_to_check = sectors + [job_title_phrase, user_phrase]
        for phrase in all_phrases_to_check:
            for sector_word in SECTOR_KEYWORDS:
                if sector_word in phrase:
                    best_sector = sector_word
                    break
            if best_sector:
                break

        suggestion_lines=[]
        job_title_ideas = []
        want = intent.get("kind", "both")

        remote_job_titles=[]
        if role:
            try:
                import requests
                base=os.getenv("AUTOSOURCING_BASE","http://localhost:8091").rstrip("/")
                url=f"{base}/suggest"
                payload = {
                    "jobTitles": [role],
                    "companies": [],
                    "languages": CONV_CONTEXT.get("languages", []),
                    "sectors": [best_sector] if best_sector else sectors,
                    "country": country_for_suggest
                }
                r=requests.post(url, json=payload, timeout=8)
                if r.status_code==200:
                    d=r.json() or {}
                    jblk=(d.get("job") or {}).get("related") or []
                    remote_job_titles=[t.strip() for t in jblk if isinstance(t,str) and t.strip()]
                    remote_job_titles=dedupe(remote_job_titles)
            except Exception:
                remote_job_titles=[]

        def _local_jobtitle_fallback(base_role: str, sector_list):
            ideas=[]
            b=(base_role or "").strip()
            if b:
                if not re.search(r"\bsenior\b", b, flags=re.I): ideas.append(f"Senior {b}")
                if not re.search(r"\blead\b", b, flags=re.I): ideas.append(f"Lead {b}")
                if not re.search(r"\bmanager\b", b, flags=re.I) and not b.lower().endswith("manager"): ideas.append(f"{b} Manager")
            sect_join=" ".join(sector_list or []).lower()
            bl=b.lower()
            if ("clinical" in bl) or ("clinical research" in bl) or ("cra" in bl) or any(k in sect_join for k in ["pharma","medical","healthcare","biotech"]):
                for jt in ["Clinical Research Coordinator","Clinical Trial Manager","Study Start-Up Specialist","Clinical Project Manager","Regulatory Affairs Specialist","Pharmacovigilance Specialist"]:
                    ideas.append(jt)
            return dedupe(ideas)[:15]

        if want in ("jobs","both") and role:
            base_ideas = remote_job_titles[:15] if remote_job_titles else _local_jobtitle_fallback(role, sectors)

            user_role = (role or "").strip()
            try:
                def _ci_in_list(val, lst):
                    if not val: return False
                    for x in lst or []:
                        if (x or "").strip().lower() == val.strip().lower():
                            return True
                    return False

                combined = list(base_ideas) if isinstance(base_ideas, list) else []
                if user_role and not _ci_in_list(user_role, combined):
                    combined.insert(0, user_role)

                seen = set()
                final_jobs = []
                for jt in combined:
                    key = (jt or "").strip().lower()
                    if not key or key in seen: continue
                    seen.add(key)
                    final_jobs.append((jt or "").strip())
                    if len(final_jobs) >= 15:
                        break
                job_title_ideas = final_jobs
            except Exception:
                job_title_ideas = base_ideas[:15] if base_ideas else []

            if job_title_ideas:
                suggestion_lines.append("Job Title ideas:\n • " + "; ".join(job_title_ideas))
            else:
                suggestion_lines.append("Job Title ideas:\n • (none returned for this role)")

        company_list = get_dynamic_company_suggestions(sector=best_sector or (sectors[-1] if sectors else ""), job_title=role)
        if want in ("companies", "both"):
            suggestion_lines.append("Company ideas:\n • " + "; ".join(company_list))

        if not suggestion_lines:
            suggestion_lines.append("Provide a role/company/sector or location to generate suggestions.")

        ctx_update({
            "job_titles":[role] if role else [],
            "companies": company_list,
            "sectors": [best_sector] if best_sector else sectors,
            "country": country_for_suggest
        })
        if role:
            CONV_CONTEXT["role_tag"] = role

        return {
            "text": "Here are some ideas:\n" + "\n".join(suggestion_lines),
            "stage": "suggestion",
            "action": "show_suggestions",
            "jobs": job_title_ideas,
            "companies": company_list
        }

    if need_country_first(normalized):
        msg = llm_orchestrate_dialog(user_msg, "ask_country", country_query_prompt(normalized["job_title"]))
        return {"text":msg, "stage":"ask_country"}

    if normalized["job_title"] and normalized["country"] and "ask_country" in user_msg.lower():
        return {"text":confirmation_prompt(normalized["job_title"], parsed.get("companies",[]), parsed.get("sectors",[]), normalized["country"]),
                "stage":"confirm_context"}

    if intent.get("mode") == "search":
        jt = parsed.get("job_title") or normalized.get("job_title")
        ct = parsed.get("country") or normalized.get("country")
        if not jt or not ct:
            if not jt:
                role_tag_offer = (CONV_CONTEXT.get("role_tag") or "").strip()
                if not role_tag_offer and username:
                    try:
                        force_refresh_role_tag(username)
                        role_tag_offer = (CONV_CONTEXT.get("role_tag") or "").strip()
                    except Exception:
                        role_tag_offer = ""
                if role_tag_offer:
                    CONV_CONTEXT["awaiting_role_tag_confirm"] = True
                    msg = llm_orchestrate_dialog(user_msg, "confirm_role_tag", f"Are you looking for a {role_tag_offer} role?")
                    return {"text": msg, "stage": "confirm_role_tag"}
            missing_prompt = llm_orchestrate_dialog(user_msg, "missing_parameters",
                                                   "Please provide both job title and country before sourcing. You can say: Product Manager in Malaysia")
            return {"text":missing_prompt, "stage":"need_more"}
        return {"text":build_sourcing_decision_block(jt, ct, parsed.get("companies",[]), parsed.get("sectors",[])),
                "stage":"decision",
                "pending_extraction":{"jt":jt,"ct":ct,"companies":parsed.get("companies",[]),"sectors":parsed.get("sectors",[]),"seniority":parsed.get("seniority")}}

    if parsed.get("job_title") and parsed.get("country"):
        ctx_update({
            "job_titles":[parsed.get("job_title")],
            "companies": parsed.get("companies",[]),
            "sectors": parsed.get("sectors",[]),
            "country": parsed.get("country")
        })
        CONV_CONTEXT["role_tag"] = parsed.get("job_title")
        pending = {
            "jt": parsed.get("job_title"),
            "ct": parsed.get("country"),
            "companies": parsed.get("companies", []),
            "sectors": parsed.get("sectors", []),
            "seniority": parsed.get("seniority")
        }
        return {"text":confirmation_prompt(parsed.get("job_title"), parsed.get("companies",[]), parsed.get("sectors",[]), parsed.get("country")),
                "stage":"confirm_context", "pending_extraction": pending}

    if parsed.get("job_title") and not parsed.get("country"):
        msg = llm_orchestrate_dialog(user_msg, "ask_country", country_query_prompt(parsed.get("job_title")))
        return {"text":msg, "stage":"ask_country"}

    role_tag_offer = (CONV_CONTEXT.get("role_tag") or "").strip()
    if not (parsed.get("job_title") or normalized.get("job_title")):
        if not role_tag_offer and username:
            try:
                force_refresh_role_tag(username)
                role_tag_offer = (CONV_CONTEXT.get("role_tag") or "").strip()
            except Exception:
                role_tag_offer = ""
        if role_tag_offer:
            CONV_CONTEXT["awaiting_role_tag_confirm"] = True
            msg = llm_orchestrate_dialog(user_msg, "confirm_role_tag", f"Are you looking for a {role_tag_offer} role?")
            return {"text": msg, "stage": "confirm_role_tag"}
        msg_uncertain = llm_orchestrate_dialog(user_msg, "uncertain",
                                               "- Just need the Job Title and Country to get started—mind entering them again?")
        return {"text":msg_uncertain, "stage":"uncertain"}

    msg_uncertain2 = llm_orchestrate_dialog(user_msg, "uncertain",
                                            "- Just need the Job Title and Country to get started—mind entering them again?")
    return {"text":msg_uncertain2, "stage":"uncertain"}

async def apply_decision(decision, user_msg, session_id=None):
    if not decision.get("pending_extraction"): return {"text":decision["text"]}
    jt = decision["pending_extraction"]["jt"]
    ct = decision["pending_extraction"]["ct"]
    companies = decision["pending_extraction"]["companies"]
    sectors = decision["pending_extraction"]["sectors"]
    seniority = decision["pending_extraction"]["seniority"]
    start = await start_sourcing(jt, ct, companies, seniority, sectors, session_id=session_id)
    payload, err = start
    if err: return {"text":err}
    if isinstance(payload.get("count"), (int, str)) and str(payload.get("count")).strip():
        try:
            CONV_CONTEXT["last_profile_count"] = int(payload["count"])
        except Exception:
            CONV_CONTEXT["last_profile_count"] = payload["count"]

    review_url = f"http://127.0.0.1:8091/login.html?next=%2FSourcingVerify.html"
    display_choice = (decision.get("display") or "").lower().strip()
    formatted = payload.get("formatted") or ""
    count_val = payload['count'] if isinstance(payload.get('count'), int) else 'collecting...'

    if display_choice == "review":
        reply = f"Live review: {review_url}\nProfiles initial count: {count_val}"
        return {"text":reply}
    if display_choice == "excel":
        lines = [ln for ln in (formatted.splitlines() if isinstance(formatted,str) else []) if ln.strip()]
        file_lines = [ln for ln in lines if ln.strip().lower().startswith("csv:") or ln.strip().lower().startswith("xlsx:")]
        if file_lines:
            reply = "\n".join(file_lines + [f"Profiles found: {count_val}"])
        else:
            reply = f"Export links are not ready yet. Please try again shortly.\nProfiles found: {count_val}"
            return {"text":reply}
        return {"text":reply}

    ctx_phrase = build_friendly_context_phrase()
    reply = (f"Search started for {ctx_phrase}.\nLive review: {review_url}\n"
           f"Profiles initial count: {count_val}\n"
           f"{formatted}")
    return {"text":reply}