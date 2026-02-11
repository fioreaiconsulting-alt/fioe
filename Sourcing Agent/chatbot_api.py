# file: chatbot_api.py
from fastapi import FastAPI, Request, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse, Response
import os
import re
import json

from chat_state import (
    history_file,
    append_history,
    reset_session_state,
    PENDING_EXTRACTION,
    PENDING_EXCEL_OR_REVIEW,
    PENDING_CLARIFICATION,
    CONV_CONTEXT,
    ctx_update,
    set_anchor,
    LAST_JOB_INFO,
    load_last_job_info,
    persist_last_job_info
)
from chat_flow import process_message, apply_decision, confirmation_prompt
from chat_intent import is_affirmative, is_negative
from chat_extract import prepare_source, analyze_job_description
from chat_utils import extract_country_hint, dedupe, country_to_cc
from sourcing_core import SourcingEngine

app = FastAPI()

# =========================
# Authentication + role_tag persistence helpers
# =========================
def _get_username_from_request(req: Request, payload: dict) -> str:
    uname = (payload.get("username") or "").strip()
    if not uname:
        cookie_header = req.headers.get("cookie") or ""
        m = re.search(r'(?:^|;\s*)username=([^;]+)', cookie_header)
        if m:
            try:
                uname = re.sub(r'[%]",?', '', m.group(1)).strip()
            except Exception:
                uname = m.group(1).strip()
    return uname

def _db_connect():
    import psycopg2
    pg_host = os.getenv("PGHOST", "localhost")
    pg_port = int(os.getenv("PGPORT", "5432"))
    pg_user = os.getenv("PGUSER", "postgres")
    pg_password = os.getenv("PGPASSWORD", "") or "orlha"
    pg_db = os.getenv("PGDATABASE", "candidate_db")
    return psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)

def _update_role_tag(username: str, job_title: str):
    """
    Persist the role_tag for a username.
    """
    if not username:
        return
    try:
        conn = _db_connect()
        cur = conn.cursor()
        # If job_title is falsy (None or empty after strip), set NULL in DB to clear value.
        if job_title is None or (isinstance(job_title, str) and job_title.strip() == ""):
            try:
                cur.execute("UPDATE login SET role_tag = NULL WHERE username=%s", (username,))
                conn.commit()
            except Exception:
                pass
        else:
            try:
                cur.execute("UPDATE login SET role_tag=%s WHERE username=%s", (job_title.strip(), username))
                conn.commit()
            except Exception:
                pass
        cur.close()
        conn.close()
    except Exception:
        pass

def _fetch_role_tag(username: str) -> str:
    if not username:
        return ""
    try:
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute("SELECT role_tag FROM login WHERE username=%s", (username,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row and row[0]:
            return (row[0] or "").strip()
    except Exception:
        pass
    return ""

# AFFECTED SECTION: Helper to look up userid by username (used before starting sourcing job)
def _lookup_userid(username: str) -> str:
    if not username:
        return ""
    try:
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute("SELECT userid FROM login WHERE username=%s", (username,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row and row[0]:
            return str(row[0]).strip()
    except Exception:
        pass
    return ""

# AFFECTED SECTION: Helpers to fetch jskillset and call Gemini for talent-pool suggestions
def _fetch_jskillset(username: str):
    """
    Attempt to retrieve a user's skillset from login table.
    Prefers jskillset (JSON), then skills (JSON or CSV/text), then skillset.
    Returns a list of skill strings (possibly empty).
    """
    if not username:
        return []
    try:
        conn = _db_connect()
        cur = conn.cursor()
        # Try resilient fetch by checking columns first or catching error
        # We will try a broad select and catch specific column errors if needed
        # Or just try/except the big query.
        try:
            cur.execute(
                "SELECT jskillset, jskills, skills, skillset FROM login WHERE username=%s LIMIT 1",
                (username,)
            )
            row = cur.fetchone()
        except Exception:
            # Fallback for simpler schema
            conn.rollback()
            try:
                cur.execute("SELECT skills FROM login WHERE username=%s LIMIT 1", (username,))
                row = cur.fetchone()
                # pad row to match structure if needed, or just process what we got
                if row: row = (None, None, row[0], None)
            except Exception:
                return []

        cur.close()
        conn.close()
        if not row:
            return []
        
        # row order: jskillset, jskills, skills, skillset
        for col in row:
            if not col:
                continue
            # if it's JSON-like (list serialized), try parse
            if isinstance(col, (list, tuple)):
                return [str(x).strip() for x in col if str(x).strip()]
            if isinstance(col, str):
                txt = col.strip()
                # try JSON parse when it starts with [ or {
                if txt.startswith("[") or txt.startswith("{"):
                    try:
                        v = json.loads(txt)
                        if isinstance(v, list):
                            return [str(x).strip() for x in v if str(x).strip()]
                    except Exception:
                        pass
                # comma separated fallback
                parts = [p.strip() for p in re.split(r'[,\n;]+', txt) if p.strip()]
                if parts:
                    return parts
        return []
    except Exception:
        return []

# Try to reuse existing gemini JSON extractor if available
try:
    from chat_gemini_review import gemini_json_extract as _gemini_json_extract
except Exception:
    _gemini_json_extract = None

# Configure Gemini parameters locally (best-effort, optional if not installed)
_GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "") or None
_GEMINI_MODEL = os.getenv("GEMINI_SOURCING_MODEL", "gemini-2.5-flash-lite")

def _call_gemini_for_talent_pools(skills_list):
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
    if _GEMINI_API_KEY:
        try:
            import google.generativeai as genai
            genai.configure(api_key=_GEMINI_API_KEY)
            model = genai.GenerativeModel(_GEMINI_MODEL)
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
            if _gemini_json_extract:
                parsed = _gemini_json_extract(raw)
            else:
                # Attempt simple JSON fragment extraction
                try:
                    st = raw.strip()
                    si = st.find("{")
                    ei = st.rfind("}")
                    parsed = json.loads(st[si:ei+1]) if si!=-1 and ei!=-1 else None
                except Exception:
                    parsed = None
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
                companies = dedupe(companies)[:20]
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
        company_pool = []
        if any(k in s_lower for k in ["gaming","game","graphics","render"]):
            company_pool += ["Ubisoft", "Electronic Arts", "Unity Technologies"]
        if any(k in s_lower for k in ["bank","payments","fintech"]):
            company_pool += ["DBS Bank", "OCBC Bank", "Standard Chartered", "Stripe", "Visa", "Mastercard"]
        if any(k in s_lower for k in ["pharma","biotech","clinical","medical"]):
            company_pool += ["Pfizer", "Roche", "Novartis", "GSK"]
        if any(k in s_lower for k in ["cloud","aws","azure","gcp","kubernetes"]):
            company_pool += ["Amazon", "Google", "Microsoft", "IBM", "Oracle"]
        if any(k in s_lower for k in ["retail","ecommerce","shop"]):
            company_pool += ["Amazon", "Shopify", "Sea Limited", "Shopee", "Lazada"]
        # add some generic tech companies for broad matches
        company_pool += ["Google", "Microsoft", "Amazon", "Facebook (Meta)", "Apple", "Nvidia", "Intel", "Accenture", "Capgemini"]

        companies = dedupe(company_pool)[:20]
        raw = json.dumps({"job_titles": job_titles, "companies": companies}, ensure_ascii=False)
        return job_titles, companies, raw
    except Exception:
        return [], [], ""

# END AFFECTED SECTION

# --- New Endpoint: Upload JD for Analysis (compatibility path for frontend) ---
@app.post("/user/upload_jd")
async def user_upload_jd_fastapi(username: str = Form(None), file: UploadFile = File(...)):
    """
    Compatibility endpoint so frontend can POST multipart/form-data with 'file' and 'username'
    similar to the Flask /user/upload_jd. Extract text (pdf/docx/txt), store into login.jd,
    and return a simple JSON acknowledging storage.
    """
    try:
        uname = (username or "").strip()
        if not uname:
            return JSONResponse({"error": "username required"}, status_code=400)

        if not file:
            return JSONResponse({"error": "No file part"}, status_code=400)

        fname = (file.filename or "").strip()
        if not fname:
            return JSONResponse({"error": "No selected file"}, status_code=400)

        filename = fname.lower()
        content = await file.read()
        extracted_text = ""

        if filename.endswith(".pdf"):
            import io
            try:
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(content))
                for page in reader.pages:
                    extracted_text += (page.extract_text() or "") + "\n"
            except ImportError:
                return JSONResponse({"error": "pypdf not installed, cannot process PDF"}, status_code=500)
            except Exception as e:
                return JSONResponse({"error": f"PDF parsing error: {e}"}, status_code=500)

        elif filename.endswith(".docx"):
            import io
            try:
                import docx
                doc = docx.Document(io.BytesIO(content))
                for para in doc.paragraphs:
                    extracted_text += para.text + "\n"
            except ImportError:
                return JSONResponse({"error": "python-docx not installed, cannot process DOCX"}, status_code=500)
            except Exception as e:
                return JSONResponse({"error": f"DOCX parsing error: {e}"}, status_code=500)

        elif filename.endswith(".doc"):
            return JSONResponse({"error": "Legacy .doc format not supported. Please save as .docx or .pdf"}, status_code=400)

        else:
            try:
                extracted_text = content.decode('utf-8', errors='ignore')
            except Exception as e:
                return JSONResponse({"error": f"Text decoding error: {e}"}, status_code=500)

        extracted_text = extracted_text.strip()
        if not extracted_text:
            return JSONResponse({"error": "Could not extract text from file"}, status_code=400)

        # Store JD into login.jd column
        try:
            conn = _db_connect()
            cur = conn.cursor()
            cur.execute("UPDATE login SET jd = %s WHERE username = %s", (extracted_text, uname))
            updated = cur.rowcount
            conn.commit()
            cur.close()
            conn.close()
            if updated == 0:
                return JSONResponse({"error": "Username not found"}, status_code=404)
        except Exception:
            # If DB unavailable or schema mismatch, return success but indicate storage skipped
            return JSONResponse({"status": "ok", "message": "JD extracted but DB update failed/skipped", "length": len(extracted_text)})

        return JSONResponse({"status": "ok", "message": "JD uploaded and stored", "length": len(extracted_text)})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
# End compatibility upload endpoint

@app.post("/chat/upload_jd")
async def upload_jd(req: Request, file: UploadFile = File(...)):
    """
    Handles Job Description file upload (PDF/Text).
    Extracts text, calls analysis, returns confirmation summary.
    Ensures extracted skillsets (if any) are returned to client and saved into login.skills column when possible.
    """
    try:
        # 1. Auth Check (Basic cookie/header based since this is Multipart)
        cookie_header = req.headers.get("cookie") or ""
        m = re.search(r'(?:^|;\s*)username=([^;]+)', cookie_header)
        username = ""
        if m:
            try:
                username = re.sub(r'[%]",?', '', m.group(1)).strip()
            except Exception:
                username = m.group(1).strip()
        
        if not username:
             return JSONResponse({"error": "Authentication required."}, status_code=401)
        
        # 2. Extract Text
        content = await file.read()
        filename = (file.filename or "").lower()
        text_content = ""
        
        if filename.endswith(".pdf"):
            try:
                import io
                from pypdf import PdfReader
                reader = PdfReader(io.BytesIO(content))
                for page in reader.pages:
                    text_content += (page.extract_text() or "") + "\n"
            except ImportError:
                 return JSONResponse({"error": "Server missing PDF library (pypdf)."}, status_code=500)
            except Exception as e:
                 return JSONResponse({"error": f"PDF parsing error: {e}"}, status_code=500)
        else:
            # Assume text/plain or DOCX handled at /user/upload_jd path; here accept text
            try:
                text_content = content.decode("utf-8", errors="ignore")
            except Exception:
                text_content = ""
            
        if not text_content.strip():
             return JSONResponse({"error": "Could not extract text from file."}, status_code=400)
             
        # 3. Store JD content (Optional based on requirements: 'stored in login table under jd column')
        try:
            conn = _db_connect()
            cur = conn.cursor()
            # Assuming 'jd' column exists in 'login' table as TEXT or VARCHAR
            cur.execute("UPDATE login SET jd = %s WHERE username = %s", (text_content, username))
            conn.commit()
            cur.close()
            conn.close()
        except Exception:
            pass
            
        # 4. Analyze
        # Use the higher-level analyzer from chat_gemini_review if present, otherwise fallback
        skills_list = []
        try:
            from chat_gemini_review import analyze_job_description as gemini_analyze
            res = gemini_analyze(text_content)
            # res may be dict or other; try to extract summary/missing/skills
            if isinstance(res, dict):
                summary = res.get("summary") or ""
                missing = res.get("missing") or []
                parsed_parsed = res.get("parsed", {}) if isinstance(res.get("parsed", {}), dict) else {}
                # extract skills from analyzer result (top-level or parsed)
                skills_list = res.get("skills") or parsed_parsed.get("skills") or res.get("parsed", {}).get("skills") or []
                # coerce to list of strings
                if isinstance(skills_list, str) and skills_list.strip():
                    # try CSV split
                    skills_list = [s.strip() for s in skills_list.split(",") if s.strip()] 
                elif not isinstance(skills_list, list):
                    skills_list = []
                # store some structured data into CONV_CONTEXT
                ctx_update({
                    "job_titles": [parsed_parsed.get("job_title")] if parsed_parsed.get("job_title") else [],
                    "country": parsed_parsed.get("country") or "",
                    "sectors": [parsed_parsed.get("sector")] if parsed_parsed.get("sector") else [],
                    "seniority": parsed_parsed.get("seniority")
                })
                CONV_CONTEXT["jd_analysis_data"] = {
                    "job_title": parsed_parsed.get("job_title") or "",
                    "seniority": parsed_parsed.get("seniority") or "",
                    "sector": parsed_parsed.get("sector") or "",
                    "sectors": parsed_parsed.get("sectors") or ([parsed_parsed.get("sector")] if parsed_parsed.get("sector") else []),
                    "country": parsed_parsed.get("country") or "",
                    "skills": skills_list
                }
            else:
                summary, missing = analyze_job_description(text_content)
        except Exception:
            # fallback to chat_extract analyzer
            summary, missing = analyze_job_description(text_content)
            # salvage parse and attempt to extract heuristics; chat_extract.analyze_job_description returns (summary, missing)
            try:
                hist_path = history_file("default")
                parsed_salvage = prepare_source(text_content, hist_path) or {}
                # detect skills heuristically via chat_gemini_review.extract_skills_heuristic if available
                # Try to import heuristic from chat_gemini_review
                try:
                    from chat_gemini_review import extract_skills_heuristic
                except Exception:
                    extract_skills_heuristic = None
                heur_skills = []
                try:
                    if extract_skills_heuristic:
                        heur_skills = extract_skills_heuristic(text_content, parsed_salvage.get("job_title",""), (parsed_salvage.get("sectors") or [""])[0] if parsed_salvage.get("sectors") else "", (parsed_salvage.get("companies") or [""])[0] if parsed_salvage.get("companies") else "")
                except Exception:
                    heur_skills = []
                skills_list = heur_skills or []
                ctx_update({
                    "job_titles": [parsed_salvage.get("job_title")] if parsed_salvage.get("job_title") else [],
                    "country": parsed_salvage.get("country") or "",
                    "sectors": parsed_salvage.get("sectors") or [],
                    "seniority": parsed_salvage.get("seniority")
                })
                CONV_CONTEXT["jd_analysis_data"] = {
                    "job_title": parsed_salvage.get("job_title") or "",
                    "seniority": parsed_salvage.get("seniority") or "",
                    "sector": (parsed_salvage.get("sectors") or [None])[0] or "",
                    "sectors": parsed_salvage.get("sectors") or [],
                    "country": parsed_salvage.get("country") or "",
                    "skills": skills_list
                }
            except Exception:
                pass
        
        # 4b. Persist skills into login.skills column (if analyzer found any)
        # Save as JSON array string for portability
        try:
            if skills_list:
                conn = _db_connect()
                cur = conn.cursor()
                try:
                    cur.execute("UPDATE login SET skills = %s WHERE username = %s", (json.dumps(skills_list, ensure_ascii=False), username))
                    conn.commit()
                except Exception:
                    # Try fallback: skills_text column or plain text column if 'skills' not present
                    try:
                        cur.execute("UPDATE login SET skills = %s WHERE username = %s", (", ".join(skills_list), username))
                        conn.commit()
                    except Exception:
                        pass
                cur.close()
                conn.close()
        except Exception:
            # Do not fail on DB write
            pass
        
        # 5. Set State
        CONV_CONTEXT["awaiting_jd_confirmation"] = True
        
        # 6. Append to History
        hist_path = history_file("default") # Or session specific if passed in header
        append_history(hist_path, "user", f"[Uploaded JD: {file.filename}]")
        
        response_text = summary if 'summary' in locals() else ""
        if 'missing' in locals() and missing:
             response_text += f"\n\nCould you please specify which {', '.join(missing)} you are targeting?"
             
        append_history(hist_path, "bot", response_text)
        
        # Return structured jd_analysis_data to help client render confirmation UI without parsing text
        # include skills explicitly
        jd_out = CONV_CONTEXT.get("jd_analysis_data", {}) or {}
        if skills_list:
            jd_out["skills"] = skills_list
        return JSONResponse({"response": response_text, "jd_analysis_data": jd_out})
        
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/chat")
async def chat(req: Request):
    global PENDING_EXTRACTION, PENDING_EXCEL_OR_REVIEW, LAST_JOB_INFO

    data = await req.json()
    user_msg = data.get("message", "")
    action = (data.get("action") or "").strip().lower()
    session_id = (data.get("session_id") or "default").strip()

    # Enforce authentication (must have valid username)
    username = _get_username_from_request(req, data)
    if not username:
        return JSONResponse({"error": "Authentication required. Please log in first."}, status_code=401)

    hist_path = history_file(session_id)
    append_history(hist_path, "user", user_msg if user_msg else (action or "[no message]"))

    # Ensure lower_msg is available early (many branches reference it)
    lower_msg = (user_msg or "").strip().lower()

    # Short-circuit action-based endpoints (from frontend buttons)
    if action == "use_suggestions":
        selected_jobs = data.get("selectedJobs") or []
        selected_companies = data.get("selectedCompanies") or []

        jt_primary = ""
        jt_all = []
        if isinstance(selected_jobs, list) and selected_jobs:
            jt_primary = str(selected_jobs[0]).strip()
            jt_all = [str(x).strip() for x in selected_jobs if str(x).strip()]
        elif CONV_CONTEXT.get("job_titles"):
            jt_primary = (CONV_CONTEXT["job_titles"] or [""])[0]
            jt_all = [jt_primary]

        try:
            user_provided_title = (data.get("originalJobTitle") or "").strip()
            if not user_provided_title:
                user_provided_title = (CONV_CONTEXT.get("role_tag") or "").strip()
            if not user_provided_title:
                user_provided_title = ((CONV_CONTEXT.get("job_titles") or [None])[0] or "").strip()

            if user_provided_title:
                if not jt_all:
                    jt_all = [user_provided_title]
                    jt_primary = user_provided_title
                else:
                    if not any(user_provided_title.lower() == s.lower() for s in jt_all):
                        jt_all.insert(0, user_provided_title)
                        jt_primary = jt_all[0]
        except Exception:
            pass

        ct = CONV_CONTEXT.get("country", "")

        if not jt_primary and not jt_all:
            msg = "What Job Title are you trying to find?"
            append_history(hist_path, "bot", msg)
            return JSONResponse({"response": msg})

        if jt_primary and not ct:
            msg = f"Which country do you wish to find your {jt_primary or 'role'} from?"
            append_history(hist_path, "bot", msg)
            return JSONResponse({"response": msg})
        if not jt_primary and not ct:
            msg = "Please provide both a job title and a country (e.g., Clinical Research Associate in Malaysia)."
            append_history(hist_path, "bot", msg)
            return JSONResponse({"response": msg})

        ctx_update({
            "job_titles": jt_all if jt_all else ([jt_primary] if jt_primary else []),
            "companies": selected_companies or [],
            "country": ct,
        })

        pending = {
            "jt": jt_primary,
            "ct": ct,
            "companies": selected_companies if isinstance(selected_companies, list) else [],
            "sectors": CONV_CONTEXT.get("sectors", []) or [],
            "seniority": None
        }
        try:
            sen = ""
            if isinstance(CONV_CONTEXT.get("jd_analysis_data"), dict):
                sen = CONV_CONTEXT.get("jd_analysis_data", {}).get("seniority") or ""
            if not sen:
                sen = (CONV_CONTEXT.get("seniority") or "") or ""
            if not sen:
                jt_low = (jt_primary or "").lower()
                if "director" in jt_low:
                    sen = "Director"
                elif "senior" in jt_low or re.search(r"\bsr\b", jt_low):
                    sen = "Senior"
                elif "lead" in jt_low:
                    sen = "Lead"
                elif "manager" in jt_low or "mgr" in jt_low:
                    sen = "Manager"
                elif "associate" in jt_low or "junior" in jt_low or "jr" in jt_low:
                    sen = "Associate"
            pending["seniority"] = sen or ""
        except Exception:
            pending["seniority"] = pending.get("seniority") or ""

        try:
            userid = _lookup_userid(username)
            pending["userid"] = userid
            pending["username"] = username
            CONV_CONTEXT["userid"] = userid
        except Exception:
            pending["userid"] = pending.get("userid") or ""
            pending["username"] = pending.get("username") or username

        try:
            PENDING_EXTRACTION = pending
        except Exception:
            pass

        confirmation = confirmation_prompt(
            job_title=jt_all if jt_all else jt_primary,
            companies=selected_companies,
            sectors=CONV_CONTEXT.get("sectors", []),
            country=ct
        )
        append_history(hist_path, "bot", confirmation)

        try:
            CONV_CONTEXT["try_else_modal_active"] = True
        except Exception:
            pass

        try:
            multi_role_tag = ", ".join(jt_all) if isinstance(jt_all, list) and len(jt_all) > 1 else jt_primary
            _update_role_tag(username, multi_role_tag)
        except Exception:
            _update_role_tag(username, jt_primary)

        try:
            if PENDING_EXTRACTION and isinstance(PENDING_EXTRACTION, dict):
                ident_path = os.path.join(os.getcwd(), ".chatbot_identity.json")
                import json as _json
                with open(ident_path, "w", encoding="utf-8") as f:
                    f.write(_json.dumps({"userid": PENDING_EXTRACTION.get("userid",""), "username": username}))
        except Exception:
            pass

        return JSONResponse({"response": confirmation, "pending_extraction": pending})

    if action == "try_something_else":
        PENDING_EXTRACTION = None
        PENDING_EXCEL_OR_REVIEW = None
        PENDING_CLARIFICATION = None
        CONV_CONTEXT.update({
            "job_titles": [],
            "companies": [],
            "sectors": [],
            "country": "",
            "languages": [],
            "role_tag": (CONV_CONTEXT.get("role_tag") or "").strip()
        })
        try:
            CONV_CONTEXT["try_else_modal_active"] = False
        except Exception:
            pass

        set_anchor("", [])
        msg = (
            "Okay, let's start a fresh search.\n"
            "How would you like to search for candidate profiles?\n"
            "1. Upload a Job Description\n"
            "2. Provide Job Title and Country"
        )
        append_history(hist_path, "bot", msg)
        return JSONResponse({"response": msg})

    if action == "highlight_talent_pools":
        try:
            # First attempt: skills saved in user profile (jskillset/skills)
            skills = _fetch_jskillset(username) or []

            # Second attempt: skills extracted from last JD analysis in CONV_CONTEXT
            if not skills:
                try:
                    jd_data = CONV_CONTEXT.get("jd_analysis_data") or {}
                    if jd_data and isinstance(jd_data.get("skills"), list) and jd_data.get("skills"):
                        skills = jd_data.get("skills")
                except Exception:
                    pass

            # Third attempt: fetch JD from DB and analyze it to extract skills (best-effort)
            if not skills:
                try:
                    conn = _db_connect()
                    cur = conn.cursor()
                    cur.execute("SELECT jd FROM login WHERE username=%s", (username,))
                    row = cur.fetchone()
                    cur.close()
                    conn.close()
                    jd_text = (row[0] or "").strip() if row and row[0] else ""
                except Exception:
                    jd_text = ""
                if jd_text:
                    # Try to use the richer analyzer if available
                    try:
                        from chat_gemini_review import analyze_job_description as gemini_analyze
                        res = gemini_analyze(jd_text)
                        if isinstance(res, dict):
                            skills = res.get("skills") or (res.get("parsed") or {}).get("skills") or []
                    except Exception:
                        # fallback to local heuristic analyzer
                        try:
                            _, _ = analyze_job_description(jd_text)
                            # analyze_job_description in chat_extract returns (summary, missing)
                            # but we can use prepare_source + chat_gemini_review.extract_skills_heuristic if available
                            parsed_salvage = prepare_source(jd_text, hist_path) if 'hist_path' in locals() else prepare_source(jd_text, history_file("default"))
                            try:
                                from chat_gemini_review import extract_skills_heuristic
                            except Exception:
                                extract_skills_heuristic = None
                            if extract_skills_heuristic:
                                skills = extract_skills_heuristic(jd_text, (parsed_salvage or {}).get("job_title",""), (parsed_salvage or {}).get("sectors",[])[0] if (parsed_salvage or {}).get("sectors") else "", "")
                        except Exception:
                            skills = skills or []

            # If still no skills, inform user
            if not skills:
                msg = "No skillset found in your profile or last JD. Highlighting is disabled until you upload or save skills."
                append_history(hist_path, "bot", msg)
                return JSONResponse({"response": msg, "error": "no_skills", "action": "highlight_disabled"}, status_code=200)

            # Optionally persist derived skills back to user profile if we extracted them from JD and DB write possible
            try:
                if skills:
                    conn = _db_connect()
                    cur = conn.cursor()
                    try:
                        cur.execute("UPDATE login SET jskillset = %s WHERE username = %s", (json.dumps(skills, ensure_ascii=False), username))
                        conn.commit()
                    except Exception:
                        try:
                            cur.execute("UPDATE login SET skills = %s WHERE username = %s", (json.dumps(skills, ensure_ascii=False), username))
                            conn.commit()
                        except Exception:
                            try:
                                cur.execute("UPDATE login SET skills = %s WHERE username = %s", (", ".join(skills), username))
                                conn.commit()
                            except Exception:
                                pass
                    cur.close()
                    conn.close()
            except Exception:
                pass

            # Generate talent pools from skills
            job_titles, companies, raw = _call_gemini_for_talent_pools(skills)

            if not job_titles and not companies:
                fallback = "Could not generate talent-pool suggestions automatically. Try adding more skills to your profile or ask for suggestions manually."
                append_history(hist_path, "bot", fallback)
                return JSONResponse({"response": fallback, "skills_count": len(skills)}, status_code=200)

            response_text = "Here are some talent pool ideas aligned to your last JD's skills:\n"
            if job_titles:
                response_text += "Job Title ideas:\n • " + "; ".join(job_titles) + "\n"
            if companies:
                response_text += "Company ideas:\n • " + "; ".join(companies)

            try:
                if job_titles:
                    CONV_CONTEXT["suggested_job_titles"] = job_titles
                if companies:
                    CONV_CONTEXT["suggested_companies"] = companies
            except Exception:
                pass

            append_history(hist_path, "bot", response_text)
            return JSONResponse({
                "response": response_text,
                "action": "show_suggestions",
                "jobs": job_titles,
                "companies": companies,
                "skills_count": len(skills),
                "raw": raw
            }, status_code=200)
        except Exception as e:
            append_history(hist_path, "bot", "Failed to generate talent pools.")
            return JSONResponse({"error": str(e)}, status_code=500)

    if action == "request_jd_clarify":
        try:
            jd_data = CONV_CONTEXT.get("jd_analysis_data") or {}
            detected_job = jd_data.get("job_title") if jd_data else ""
            detected_sen = jd_data.get("seniority") if jd_data else ""
            detected_sector = jd_data.get("sector") if jd_data else ""
            if not detected_sector and jd_data and isinstance(jd_data.get("sectors"), list) and jd_data.get("sectors"):
                detected_sector = jd_data.get("sectors")[0]
            detected_country = jd_data.get("country") if jd_data else ""
            
            if not jd_data or not (detected_job or detected_sector or detected_country):
                try:
                    conn = _db_connect()
                    cur = conn.cursor()
                    cur.execute("SELECT jd FROM login WHERE username=%s", (username,))
                    row = cur.fetchone()
                    cur.close()
                    conn.close()
                    jd_text = (row[0] or "").strip() if row and row[0] else ""
                except Exception:
                    jd_text = ""
                if jd_text:
                    parsed_salvage = prepare_source(jd_text, hist_path) or {}
                    detected_job = parsed_salvage.get("job_title") or detected_job
                    detected_country = parsed_salvage.get("country") or detected_country
                    detected_sector = (parsed_salvage.get("sectors") or [None])[0] if parsed_salvage.get("sectors") else detected_sector
                    detected_sen = parsed_salvage.get("seniority") or detected_sen
                    ctx_update({
                        "job_titles": [detected_job] if detected_job else [],
                        "country": detected_country or "",
                        "sectors": [detected_sector] if detected_sector else [],
                        "seniority": detected_sen or ""
                    })
                    CONV_CONTEXT["jd_analysis_data"] = {
                        "job_title": detected_job or "",
                        "seniority": detected_sen or "",
                        "sector": detected_sector or "",
                        "sectors": [detected_sector] if detected_sector else [],
                        "country": detected_country or ""
                    }
            
            parts = []
            if detected_job:
                parts.append(f"Job title: {detected_job}")
            if detected_sen:
                parts.append(f"Seniority: {detected_sen}")
            if detected_sector:
                parts.append(f"Sector: {detected_sector}")
            if detected_country:
                parts.append(f"Country: {detected_country}")
            if parts:
                q = ("I detected the following from the Job Description:\n\n" +
                     "\n".join(parts) +
                     "\n\nWhich of these are incorrect or would you like to change? You can reply with corrections like:\n- Job title: <correct title>\n- Country: <correct country>\nOr simply type the corrected information.")
            else:
                q = "I couldn't reliably extract tags from the Job Description. Please provide the Job Title and Country (e.g., Product Manager in Malaysia), or type the fields you'd like to change."
            CONV_CONTEXT["awaiting_jd_clarification"] = True
            append_history(hist_path, "bot", q)
            return JSONResponse({"response": q, "action": "show_jd_clarify"})
        except Exception as e:
            return JSONResponse({"error": f"Failed to prepare clarification: {e}"}, status_code=500)

    if CONV_CONTEXT.get("awaiting_jd_clarification") and (not action):
        try:
            parsed = prepare_source(user_msg, hist_path) or {}
            updates = {}
            changed = False
            if parsed.get("job_title"):
                updates["job_titles"] = [parsed.get("job_title")]
                changed = True
            if parsed.get("country"):
                updates["country"] = parsed.get("country")
                changed = True
            if parsed.get("sectors"):
                updates["sectors"] = parsed.get("sectors")
                changed = True
            if parsed.get("seniority"):
                updates["seniority"] = parsed.get("seniority")
                changed = True

            if not changed:
                mJT = re.search(r'job\s*title\s*[:\-]\s*(.+)', user_msg, flags=re.I)
                if mJT:
                    jtval = mJT.group(1).strip()
                    if jtval:
                        updates["job_titles"] = [jtval]
                        changed = True
                mCT = re.search(r'country\s*[:\-]\s*([A-Za-z \-]+)', user_msg, flags=re.I)
                if mCT:
                    ct = mCT.group(1).strip()
                    if ct:
                        updates["country"] = ct
                        changed = True
                mSec = re.search(r'(sector|industry)\s*[:\-]\s*(.+)', user_msg, flags=re.I)
                if mSec:
                    sec = mSec.group(2).strip()
                    if sec:
                        updates["sectors"] = [s.strip() for s in sec.split(",") if s.strip()]
                        changed = True
                mSen = re.search(r'(seniority|level)\s*[:\-]\s*(.+)', user_msg, flags=re.I)
                if mSen:
                    sen = mSen.group(2).strip()
                    if sen:
                        updates["seniority"] = sen
                        changed = True

            if not changed:
                ask = "I didn't detect a clear correction. Could you specify which field to correct (Job title / Country / Sector / Seniority) and provide the corrected value? Example: 'Job title: Senior Product Manager in Singapore'"
                append_history(hist_path, "bot", ask)
                return JSONResponse({"response": ask}, status_code=200)

            try:
                ctx_update(updates)
            except Exception:
                pass

            jd = CONV_CONTEXT.get("jd_analysis_data") or {}
            try:
                if updates.get("job_titles"):
                    jd["job_title"] = (updates.get("job_titles") or [""])[0] or jd.get("job_title","")
                if updates.get("country"):
                    jd["country"] = updates.get("country") or jd.get("country","")
                if updates.get("sectors"):
                    jd["sectors"] = updates.get("sectors") or jd.get("sectors") or []
                    jd["sector"] = jd["sectors"][0] if jd["sectors"] else jd.get("sector","")
                if updates.get("seniority"):
                    jd["seniority"] = updates.get("seniority") or jd.get("seniority","")
                CONV_CONTEXT["jd_analysis_data"] = jd
            except Exception:
                pass

            CONV_CONTEXT["awaiting_jd_clarification"] = False
            CONV_CONTEXT["awaiting_jd_confirmation"] = True

            jt = (CONV_CONTEXT.get("job_titles") or [""])[0]
            comps = CONV_CONTEXT.get("companies", [])
            secs = CONV_CONTEXT.get("sectors", []) or (jd.get("sectors") or [])
            country = CONV_CONTEXT.get("country") or jd.get("country","")
            confirm = confirmation_prompt(jt, comps, secs, country)
            append_history(hist_path, "bot", confirm)

            pending = {
                "jt": jt,
                "ct": country,
                "companies": comps,
                "sectors": secs,
                "seniority": jd.get("seniority","")
            }
            return JSONResponse({"response": confirm, "pending_extraction": pending})
        except Exception as e:
            return JSONResponse({"error": f"Clarification processing failed: {e}"}, status_code=500)

    if action == "start_sourcing":
        pending = data.get("pending_extraction") or PENDING_EXTRACTION
        if not pending or not isinstance(pending, dict):
            return JSONResponse({"error": "No pending extraction available. Please confirm role and country first."}, status_code=400)

        pending.setdefault("companies", pending.get("companies") or [])
        pending.setdefault("sectors", pending.get("sectors") or [])
        pending.setdefault("seniority", pending.get("seniority") or "")

        try:
            userid = pending.get("userid") or _lookup_userid(username)
            pending["userid"] = userid
            pending["username"] = username
            CONV_CONTEXT["userid"] = userid
        except Exception:
            pending["userid"] = pending.get("userid") or ""
            pending["username"] = pending.get("username") or username

        try:
            if not pending.get("seniority"):
                jt_low = (pending.get("jt") or "").lower()
                inferred = ""
                if "director" in jt_low:
                    inferred = "Director"
                elif "senior" in jt_low or re.search(r"\bsr\b", jt_low):
                    inferred = "Senior"
                elif "lead" in jt_low:
                    inferred = "Lead"
                elif "manager" in jt_low or "mgr" in jt_low:
                    inferred = "Manager"
                elif "associate" in jt_low or "junior" in jt_low or "jr" in jt_low:
                    inferred = "Associate"
                if inferred:
                    pending["seniority"] = inferred
        except Exception:
            pass

        missing_fields = []
        if not pending.get("jt"):
            missing_fields.append("job title")

        inferred_warnings = []
        ct_val = pending.get("ct") or pending.get("country") or CONV_CONTEXT.get("country") or ""
        if not ct_val:
            try:
                inferred_country = extract_country_hint(pending.get("jt") or "")
                if inferred_country:
                    ct_val = inferred_country
                    pending["ct"] = ct_val
                    inferred_warnings.append("country inferred from job title")
            except Exception:
                pass
        else:
            pending["ct"] = ct_val
        if not ct_val:
            missing_fields.append("country")

        has_sectors = False
        sectors_val = pending.get("sectors") or CONV_CONTEXT.get("sectors") or (CONV_CONTEXT.get("jd_analysis_data") or {}).get("sectors") or []
        if isinstance(sectors_val, str):
            sectors_val = [s.strip() for s in sectors_val.split(",") if s.strip()]
        if isinstance(sectors_val, list) and len(sectors_val) > 0:
            pending["sectors"] = sectors_val
            has_sectors = True
        else:
            pending["sectors"] = []
            has_sectors = False
            inferred_warnings.append("sector not provided; proceeding without sector may broaden results")

        critical_missing = [m for m in missing_fields if m in ("job title", "country")]
        if critical_missing:
            msg = f"Cannot start sourcing: missing {', '.join(critical_missing)}. Please specify them first."
            append_history(hist_path, "bot", msg)
            return JSONResponse({"error": msg, "missing": critical_missing, "inferred": {"ct": pending.get("ct"), "sectors": pending.get("sectors")}, "warnings": inferred_warnings}, status_code=400)

        decision = {"pending_extraction": pending, "display": data.get("display") or ""}
        try:
            PENDING_EXTRACTION = None
        except Exception:
            pass
        result = await apply_decision(decision, user_msg, session_id=session_id)
        append_history(hist_path, "bot", result["text"])
        resp_payload = {"response": result["text"], "warnings": inferred_warnings}
        if isinstance(result, dict):
            if result.get("action"):
                resp_payload["action"] = result.get("action")
            if result.get("jobs"):
                resp_payload["jobs"] = result.get("jobs")
            if result.get("companies"):
                resp_payload["companies"] = result.get("companies")
        return JSONResponse(resp_payload)

    if action == "status":
        stripped = user_msg.strip().lower()
        if (not isinstance(LAST_JOB_INFO, tuple) or len(LAST_JOB_INFO) != 2):
            recovered = load_last_job_info(session_id)
            if recovered:
                LAST_JOB_INFO = recovered
        if isinstance(LAST_JOB_INFO, tuple) and len(LAST_JOB_INFO) == 2:
            job_id, base = LAST_JOB_INFO
            engine = SourcingEngine(base_url=base)
            data_obj, err = engine._get(f"/job_status/{job_id}")
            if err or not isinstance(data_obj, dict):
                reply = f"Could not retrieve status right now: {err or 'unknown error'}"
                append_history(hist_path, "bot", reply)
                return JSONResponse({"response": reply})
            formatted = engine.format_status(data_obj, job_id)
            if not data_obj.get("done") and PENDING_EXCEL_OR_REVIEW:
                if data_obj.get("output_csv") or data_obj.get("output_xlsx"):
                    ocsv = data_obj.get("output_csv")
                    oxlsx = data_obj.get("output_xlsx")
                    lines = ["“Pulled the data from Excel—ready to go”"]
                    if ocsv:
                        lines.append(f"CSV: {engine.base}/download/{ocsv}")
                    if oxlsx:
                        lines.append(f"XLSX: {engine.base}/download/{oxlsx}")
                    formatted = "\n".join(lines)
                else:
                    formatted = "Export links are not ready yet. Please try again shortly."
            append_history(hist_path, "bot", formatted)
            return JSONResponse({"response": formatted})
        else:
            if CONV_CONTEXT.get("job_titles") and CONV_CONTEXT.get("country"):
                jt = (CONV_CONTEXT.get("job_titles") or [""])[0]
                ct = CONV_CONTEXT.get("country") or ""
                confirm = confirmation_prompt(
                    job_title=jt,
                    companies=CONV_CONTEXT.get("companies", []),
                    sectors=CONV_CONTEXT.get("sectors", []),
                    country=ct
                )
                append_history(hist_path, "bot", confirm)
                return JSONResponse({"response": confirm})
            msg = "No active sourcing job yet. Provide a role and country (e.g., Data Scientist in Singapore) or select suggestions."
            append_history(hist_path, "bot", msg)
            return JSONResponse({"response": msg})

    stripped = user_msg.strip().lower()
    if stripped in {"reset session", "end session", "clear session"}:
        reset_session_state()
        try:
            jf = f"jobinfo_{re.sub(r'[^A-Za-z0-9_.-]', '_', session_id or 'default')}.json"
            if os.path.isfile(jf):
                os.remove(jf)
        except Exception:
            pass

        try:
            try:
                _update_role_tag(username, None)
            except Exception:
                try:
                    conn = _db_connect()
                    cur = conn.cursor()
                    cur.execute("UPDATE login SET role_tag = NULL WHERE username=%s", (username,))
                    conn.commit()
                    cur.close()
                    conn.close()
                except Exception:
                    pass

            try:
                ident_path = os.path.join(os.getcwd(), ".chatbot_identity.json")
                if os.path.isfile(ident_path):
                    os.remove(ident_path)
            except Exception:
                pass
        except Exception:
            pass

        reset_msg = (
            "Session cleared. Ready for a fresh start.\n"
            "How would you like to search for candidate profiles?\n"
            "1. Upload a Job Description\n"
            "2. Provide Job Title and Country"
        )
        append_history(hist_path, "bot", reset_msg)
        
        resp = JSONResponse({"response": reset_msg})
        try:
            resp.set_cookie("role_tag", "", path="/", max_age=0, httponly=False, samesite="lax")
            resp.set_cookie("fullname", "", path="/", max_age=0, httponly=False, samesite="lax")
        except Exception:
            pass
        return resp

    stripped = user_msg.strip().lower()
    if stripped in {"status", "status?", "progress", "progress?", "job status"}:
        if (not isinstance(LAST_JOB_INFO, tuple) or len(LAST_JOB_INFO) != 2):
            recovered = load_last_job_info(session_id)
            if recovered:
                LAST_JOB_INFO = recovered
        if isinstance(LAST_JOB_INFO, tuple) and len(LAST_JOB_INFO) == 2:
            job_id, base = LAST_JOB_INFO
            engine = SourcingEngine(base_url=base)
            data_obj, err = engine._get(f"/job_status/{job_id}")
            if err or not isinstance(data_obj, dict):
                reply = f"Could not retrieve status right now: {err or 'unknown error'}"
                append_history(hist_path, "bot", reply)
                return JSONResponse({"response": reply})
            formatted = engine.format_status(data_obj, job_id)
            append_history(hist_path, "bot", formatted)
            return JSONResponse({"response": formatted})
        else:
            if CONV_CONTEXT.get("job_titles") and CONV_CONTEXT.get("country"):
                jt = (CONV_CONTEXT.get("job_titles") or [""])[0]
                ct = CONV_CONTEXT.get("country") or ""
                confirm = confirmation_prompt(
                    job_title=jt,
                    companies=CONV_CONTEXT.get("companies", []),
                    sectors=CONV_CONTEXT.get("sectors", []),
                    country=ct
                )
                append_history(hist_path, "bot", confirm)
                return JSONResponse({"response": confirm})
            msg = "No active sourcing job yet. Provide a role and country (e.g., Data Scientist in Singapore) or select suggestions."
            append_history(hist_path, "bot", msg)
            return JSONResponse({"response": msg})

    if PENDING_EXCEL_OR_REVIEW and (
        is_affirmative(user_msg) or "review" in lower_msg or "excel" in lower_msg
    ):
        if "review" in lower_msg:
            choice = "review"
        elif "excel" in lower_msg:
            choice = "excel"
        else:
            choice = "start"

        pe = PENDING_EXCEL_OR_REVIEW
        if not isinstance(pe, dict) or not pe.get("jt") or not pe.get("ct"):
            jt = ""
            if CONV_CONTEXT.get("job_titles"):
                jt = (CONV_CONTEXT["job_titles"] or [""])[0]
            ct = CONV_CONTEXT.get("country", "")
            companies = CONV_CONTEXT.get("companies", [])
            sectors = CONV_CONTEXT.get("sectors", [])
            if jt and ct:
                pe = {
                    "jt": jt,
                    "ct": ct,
                    "companies": companies,
                    "sectors": sectors,
                    "seniority": None
                    }
            else:
                missing = []
                if not jt:
                    missing.append("job title")
                if not ct:
                    missing.append("country")
                need = " and ".join(missing) if missing else "required parameters"
                msg = f"Please provide {need} before I can start. For example: Data Scientist in Singapore"
                append_history(hist_path, "bot", msg)
                return JSONResponse({"response": msg})

        try:
            userid = CONV_CONTEXT.get("userid") or _lookup_userid(username)
            CONV_CONTEXT["userid"] = userid
            ident_path = os.path.join(os.getcwd(), ".chatbot_identity.json")
            if userid or username:
                import json as _json
                with open(ident_path, "w", encoding="utf-8") as f:
                    f.write(_json.dumps({"userid": userid, "username": username}))
            if isinstance(pe, dict):
                pe["userid"] = userid
                pe["username"] = username
        except Exception:
            pass

        decision = {"pending_extraction": pe, "display": choice}
        PENDING_EXCEL_OR_REVIEW = None
        result = await apply_decision(decision, user_msg, session_id=session_id)
        if isinstance(LAST_JOB_INFO, tuple) and len(LAST_JOB_INFO) == 2:
            persist_last_job_info(session_id, LAST_JOB_INFO)
        append_history(hist_path, "bot", result["text"])
        resp = {"response": result["text"]}
        if isinstance(result, dict):
            if result.get("action"):
                resp["action"] = result.get("action")
            if result.get("jobs"):
                resp["jobs"] = result.get("jobs")
            if result.get("companies"):
                resp["companies"] = result.get("companies")
        return JSONResponse(resp)

    if PENDING_EXTRACTION:
        try:
            want_suggest = False
            if action in {"suggest", "use_suggestions", "suggestions"}:
                want_suggest = True
            if not want_suggest:
                t = (lower_msg or "").strip()
                if t in {"suggest", "suggestions", "ideas", "more ideas", "more options"} or "suggest" in t:
                    want_suggest = True

            if want_suggest:
                try:
                    decision_suggest = process_message("suggest", hist_path) or {}
                except Exception:
                    decision_suggest = {}

                bot_text = decision_suggest.get("text", "") or ""
                append_history(hist_path, "bot", bot_text)

                resp = {"response": bot_text}
                if decision_suggest.get("action"):
                    resp["action"] = decision_suggest.get("action")
                if decision_suggest.get("jobs"):
                    resp["jobs"] = decision_suggest.get("jobs")
                if decision_suggest.get("companies"):
                    resp["companies"] = decision_suggest.get("companies")
                return JSONResponse(resp)
        except Exception:
            pass

        if is_affirmative(user_msg):
            PENDING_EXCEL_OR_REVIEW = PENDING_EXTRACTION
            PENDING_EXTRACTION = None
            prompt = (
                "Perfect—getting started now.\n"
                "How would you like to receive the results?\n"
                "1. Review & Flag Panel (Highly Recommended)\n"
                "2. Export everything into an Excel file\n"
                "Just hit the button to confirm"
            )
            append_history(hist_path, "bot", prompt)
            return JSONResponse({"response": prompt})

        if is_negative(user_msg):
            PENDING_EXTRACTION = None
            msg = "Extraction cancelled. Provide new role/country or ask for suggestions."
            append_history(hist_path, "bot", msg)
            return JSONResponse({"response": msg})

        msg = "How can I help you?"
        append_history(hist_path, "bot", msg)
        return JSONResponse({"response": msg})

    decision = process_message(user_msg, hist_path)
    stage = decision.get("stage")

    if stage == "decision" and decision.get("pending_extraction"):
        pending = decision["pending_extraction"] or {}
        pending.setdefault("companies", pending.get("companies") or [])
        pending.setdefault("sectors", pending.get("sectors") or [])
        pending.setdefault("seniority", pending.get("seniority") or "")

        try:
            if not pending.get("seniority"):
                pending["seniority"] = CONV_CONTEXT.get("jd_analysis_data", {}).get("seniority") or CONV_CONTEXT.get("seniority") or ""
            if not pending.get("seniority"):
                jt_low = (pending.get("jt") or "").lower()
                if "director" in jt_low:
                    pending["seniority"] = "Director"
                elif "senior" in jt_low or re.search(r"\bsr\b", jt_low):
                    pending["seniority"] = "Senior"
                elif "lead" in jt_low:
                    pending["seniority"] = "Lead"
                elif "manager" in jt_low or "mgr" in jt_low:
                    pending["seniority"] = "Manager"
                elif "associate" in jt_low or "junior" in jt_low or "jr" in jt_low:
                    pending["seniority"] = "Associate"
        except Exception:
            pending["seniority"] = pending.get("seniority") or ""

        try:
            userid = pending.get("userid") or _lookup_userid(username)
            pending["userid"] = userid
            pending["username"] = username
        except Exception:
            pending["userid"] = pending.get("userid") or ""
            pending["username"] = pending.get("username") or username

        try:
            PENDING_EXTRACTION = pending
        except Exception:
            pass

        append_history(hist_path, "bot", decision["text"])
        try:
            parsed = prepare_source(user_msg, hist_path)
            jt_now = (parsed.get("job_title") or "").strip()
            _update_role_tag(username, jt_now)
        except Exception:
            pass

        resp = {"response": decision["text"], "pending_extraction": pending}
        if decision.get("action"):
            resp["action"] = decision.get("action")
        if decision.get("jobs"):
            resp["jobs"] = decision.get("jobs")
        if decision.get("companies"):
            resp["companies"] = decision.get("companies")
        return JSONResponse(resp)

    if stage == "confirm_context":
        if is_affirmative(user_msg):
            pending = decision.get("pending_extraction")
            if not (isinstance(pending, dict) and pending.get("jt") and pending.get("ct")):
                jt = ""
                if CONV_CONTEXT.get("job_titles"):
                    jt = (CONV_CONTEXT["job_titles"] or [""])[0]
                ct = CONV_CONTEXT.get("country", "")
                companies = CONV_CONTEXT.get("companies", [])
                sectors = CONV_CONTEXT.get("sectors", [])
                if jt and ct:
                    pending = {
                        "jt": jt,
                        "ct": ct,
                        "companies": companies,
                        "sectors": sectors,
                        "seniority": None
                    }
                else:
                    need = []
                    if not jt:
                        need.append("job title")
                    if not ct:
                        need.append("country")
                    msg = (
                        f"Please provide {(' and '.join(need))} before starting. "
                        "For example: Product Manager in Germany"
                    )
                    append_history(hist_path, "bot", msg)
                    return JSONResponse({"response": msg})
            
            try:
                userid = CONV_CONTEXT.get("userid") or _lookup_userid(username)
                CONV_CONTEXT["userid"] = userid
                if isinstance(pending, dict):
                    pending["userid"] = userid
                    pending["username"] = username
                ident_path = os.path.join(os.getcwd(), ".chatbot_identity.json")
                if userid or username:
                    import json as _json
                    with open(ident_path, "w", encoding="utf-8") as f:
                        f.write(_json.dumps({"userid": userid, "username": username}))
            except Exception:
                pass

            PENDING_EXCEL_OR_REVIEW = pending
            prompt = (
                "Perfect—getting started now.\n"
                "How would you like to receive the results?\n"
                "1. Review & Flag Panel (Advanced tools + token rebates for flagged errors)\n"
                "2. Export everything into an Excel file\n"
                "Just hit the button to confirm"
            )
            append_history(hist_path, "bot", prompt)
            return JSONResponse({"response": prompt})

        if is_negative(user_msg):
            options = (
                "Would you like suggestions instead (companies or job titles), "
                "or do you want to revise specific parameters?"
            )
            append_history(hist_path, "bot", options)
            return JSONResponse({"response": options})

    reply = decision["text"]
    append_history(hist_path, "bot", reply)
    try:
        parsed = prepare_source(user_msg, hist_path)
        jt_now = (parsed.get("job_title") or "").strip()

        if jt_now:
            _update_role_tag(username, jt_now)
        else:
            if not CONV_CONTEXT.get("job_titles"):
                fallback_jt = _fetch_role_tag(username)
                if fallback_jt:
                    ctx_update({"job_titles": [fallback_jt]})
    except Exception:
        pass

    resp = {"response": reply}
    if decision.get("action"):
        resp["action"] = decision.get("action")
    if decision.get("jobs"):
        resp["jobs"] = decision.get("jobs")
    if decision.get("companies"):
        resp["companies"] = decision.get("companies")

    return JSONResponse(resp)

# =========================
# Static File and Login HTML serving - login.html at default path!
# =========================

@app.get("/")
async def root(req: Request):
    username = _get_username_from_request(req, {})
    index_path = os.path.join(os.getcwd(), "index.html")
    login_path = os.path.join(os.getcwd(), "login.html")
    if not username and os.path.isfile(login_path):
        return FileResponse(login_path, media_type="text/html")
    if os.path.isfile(index_path):
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({
        "message": "Chatbot API running (modular version). Visit POST /chat to interact.",
        "auth_required": not bool(username)
    })

@app.get("/index.html")
async def index_html():
    index_path = os.path.join(os.getcwd(), "index.html")
    if os.path.isfile(index_path):
        return FileResponse(index_path, media_type="text/html")
    return JSONResponse({"error": "index.html not found"}, status_code=404)

@app.get("/AutoSourcing.html")
async def autosourcing_html():
    custom_path = r"F:\Recruiting Tools\Autosourcing\AutoSourcing.html"
    if os.path.isfile(custom_path):
        return FileResponse(custom_path, media_type="text/html")
    return JSONResponse({"error": "AutoSourcing.html not found"}, status_code=404)

@app.get("/login.html")
async def login_html():
    login_path = os.path.join(os.getcwd(), "login.html")
    if os.path.isfile(login_path):
        return FileResponse(login_path, media_type="text/html")
    return JSONResponse({"error": "login.html not found"}, status_code=404)

@app.post("/login")
async def login_account(req: Request):
    data = await req.json()
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    if not (username and password):
        return JSONResponse({"error": "username and password required"}, status_code=400)

    def _hash_password(p: str) -> str:
        import hashlib
        salt = os.getenv("PASSWORD_SALT", "")
        return hashlib.sha256((salt + p).encode("utf-8")).hexdigest()

    try:
        conn = _db_connect()
        cur = conn.cursor()
        cur.execute("SELECT password, userid, cemail, fullname, role_tag FROM login WHERE username=%s", (username,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if not row:
            return JSONResponse({"error": "Invalid credentials"}, status_code=401)
        stored_pw, userid, cemail, fullname, role_tag = row[0] or "", row[1] or "", row[2] or "", row[3] or "", (row[4] or "")
        hashed = _hash_password(password)
        if stored_pw != hashed and stored_pw != password:
            return JSONResponse({"error": "Invalid credentials"}, status_code=401)

        display_name = (fullname or "").strip() or username
        greeting = f"Hello {display_name}, it’s a pleasure to meet you! I’m an AI assistant designed to help source candidates."

        resp = JSONResponse({
            "ok": True,
            "userid": userid,
            "username": username,
            "cemail": cemail,
            "fullname": fullname,
            "role_tag": role_tag,
            "greeting": greeting
        })
        resp.set_cookie("username", username, path="/", max_age=2592000, httponly=False, samesite="lax")
        if userid:
            resp.set_cookie("userid", userid, path="/", max_age=2592000, httponly=False, samesite="lax")
        try:
            if fullname:
                resp.set_cookie("fullname", fullname, path="/", max_age=2592000, httponly=False, samesite="lax")
        except Exception:
            pass

        return resp
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/favicon.ico")
async def favicon():
    ico_path = os.path.join(os.getcwd(), "favicon.ico")
    if os.path.isfile(ico_path):
        return FileResponse(ico_path, media_type="image/vnd.microsoft.icon")
    return Response(status_code=204)

# AFFECTED SECTION: Robust /user/resolve endpoint with null checks
@app.get("/user/resolve")
async def user_resolve(username: str = ""):
    """
    Resolve user info (userid, fullname, role_tag, token).
    Handles cases where columns are missing or values are null.
    """
    username = (username or "").strip()
    if not username:
        return JSONResponse({"error": "username required"}, status_code=400)
    try:
        conn = _db_connect()
        cur = conn.cursor()
        
        # Try full query first
        row = None
        try:
            # Query attempts to coalesce null token to 0. 
            # Note: fullname might be null, handled below.
            cur.execute("SELECT userid, fullname, role_tag, COALESCE(token,0) FROM login WHERE username=%s", (username,))
            row = cur.fetchone()
        except Exception:
            # Fallback if 'token' column missing
            conn.rollback()
            try:
                cur.execute("SELECT userid, fullname, role_tag FROM login WHERE username=%s", (username,))
                r = cur.fetchone()
                if r:
                    row = (r[0], r[1], r[2], 0)
            except Exception:
                pass

        cur.close()
        conn.close()
        
        if not row:
            return JSONResponse({"error": "not found"}, status_code=404)
            
        return JSONResponse({
            "userid": row[0] or "",
            "fullname": row[1] or "",
            "role_tag": (row[2] or ""),
            "token": int(row[3] if len(row)>3 and row[3] is not None else 0)
        }, status_code=200)
    except Exception as e:
        # Return 500 but include error message for debugging if needed
        return JSONResponse({"error": f"DB Resolve Error: {str(e)}"}, status_code=500)
# End AFFECTED SECTION

# --- NEW: endpoint to update user's skillset into jskillset column ---
@app.post("/user/update_skills")
async def user_update_skills(req: Request):
    """
    Accepts JSON body: { "username": "...", "skills": [...] }
    Attempts to persist skills into login.jskillset column as JSON array.
    """
    try:
        try:
            data = await req.json()
        except Exception:
            data = {}
        username = (data.get("username") or "").strip()
        skills = data.get("skills") or data.get("skillset") or []

        if not username:
            return JSONResponse({"error": "username required"}, status_code=400)

        normalized = []
        if isinstance(skills, list):
            for s in skills:
                try:
                    if s is None:
                        continue
                    st = str(s).strip()
                    if st:
                        normalized.append(st)
                except Exception:
                    continue
        elif isinstance(skills, str) and skills.strip():
            if "," in skills:
                normalized = [s.strip() for s in skills.split(",") if s.strip()]
            else:
                normalized = [skills.strip()]
        else:
            normalized = []

        written_to = None
        try:
            conn = _db_connect()
            cur = conn.cursor()
            try:
                cur.execute("UPDATE login SET jskillset = %s WHERE username = %s", (json.dumps(normalized, ensure_ascii=False), username))
                if cur.rowcount > 0:
                    written_to = "jskillset"
                conn.commit()
            except Exception:
                conn.rollback()
                try:
                    cur.execute("UPDATE login SET skills = %s WHERE username = %s", (json.dumps(normalized, ensure_ascii=False), username))
                    if cur.rowcount > 0:
                        written_to = "skills"
                    conn.commit()
                except Exception:
                    conn.rollback()
                    try:
                        cur.execute("UPDATE login SET skills = %s WHERE username = %s", (", ".join(normalized), username))
                        if cur.rowcount > 0:
                            written_to = "skills_text"
                        conn.commit()
                    except Exception:
                        conn.rollback()
            cur.close()
            conn.close()
        except Exception as e:
            return JSONResponse({"error": f"DB error: {e}"}, status_code=500)

        if not written_to:
            return JSONResponse({"error": "No row updated (username not found or writable column missing)"} , status_code=404)

        return JSONResponse({"status": "ok", "written_to": written_to, "skills_count": len(normalized)}, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/gemini/analyze_jd")
async def gemini_analyze_jd(req: Request):
    """
    Compatibility endpoint for frontend to request JD analysis.
    """
    try:
        try:
            data = await req.json()
        except Exception:
            data = {}
        username = (data.get("username") or "").strip()
        text_input = (data.get("text") or "").strip()

        jd_text = text_input
        if not jd_text and username:
            try:
                conn = _db_connect()
                cur = conn.cursor()
                cur.execute("SELECT jd FROM login WHERE username = %s", (username,))
                row = cur.fetchone()
                cur.close()
                conn.close()
                if row and row[0]:
                    jd_text = row[0]
            except Exception as e:
                return JSONResponse({"error": f"DB fetch error: {e}"}, status_code=500)

        if not jd_text:
            return JSONResponse({"error": "No JD text provided or found for user"}, status_code=400)

        job_title = ""
        seniority = ""
        sectors = []
        country = ""
        summary = ""
        missing = []
        raw = ""
        observation = ""
        skills = []

        try:
            analyzer = None
            try:
                from chat_gemini_review import analyze_job_description as gemini_analyzer
                analyzer = gemini_analyzer
            except Exception:
                analyzer = None

            if analyzer:
                try:
                    res = analyzer(jd_text)
                    if isinstance(res, dict):
                        parsed = res.get("parsed", {}) or {}
                        job_title = (parsed.get("job_title") or parsed.get("role") or "").strip()
                        seniority = (parsed.get("seniority") or "").strip()
                        if isinstance(parsed.get("sectors"), list) and parsed.get("sectors"):
                            sectors = [str(s).strip() for s in parsed.get("sectors") if str(s).strip()]
                        else:
                            s_val = parsed.get("sector") or parsed.get("sectors") or ""
                            if isinstance(s_val, list):
                                sectors = [str(s).strip() for s in s_val if str(s).strip()]
                            elif isinstance(s_val, str) and s_val.strip():
                                if "," in s_val:
                                    sectors = [s.strip() for s in s_val.split(",") if s.strip()]
                                else:
                                    sectors = [s_val.strip()]
                        country = (parsed.get("country") or parsed.get("location") or "").strip()
                        summary = (res.get("summary") or "").strip()
                        missing = res.get("missing") if isinstance(res.get("missing"), list) else []
                        raw = (res.get("raw") or "") or (res.get("justification") or "")
                        observation = (res.get("observation") or res.get("justification") or res.get("raw") or "").strip()
                        skills = res.get("skills") or parsed.get("skills") or []
                        if isinstance(skills, str) and skills.strip():
                            if "," in skills:
                                skills = [s.strip() for s in skills.split(",") if s.strip()]
                            else:
                                skills = [skills.strip()]
                        elif not isinstance(skills, list):
                            skills = []
                except Exception:
                    pass

        except Exception:
            pass

        if not (job_title or sectors or country):
            try:
                try:
                    ssum, mm = analyze_job_description(jd_text)
                    if isinstance(ssum, str):
                        summary = ssum
                    if isinstance(mm, list):
                        missing = mm
                except Exception:
                    summary = (jd_text[:1000] + ("..." if len(jd_text) > 1000 else ""))
                    missing = []

                try:
                    hist_path = history_file("default")
                    parsed_salvage = prepare_source(jd_text, hist_path) or {}
                    job_title = (parsed_salvage.get("job_title") or "").strip()
                    seniority = (parsed_salvage.get("seniority") or "") or (parsed_salvage.get("seniority") or "")
                    if parsed_salvage.get("sectors"):
                        sectors = parsed_salvage.get("sectors") or []
                        if isinstance(sectors, str):
                            if "," in sectors:
                                sectors = [s.strip() for s in sectors.split(",") if s.strip()]
                            else:
                                sectors = [sectors]
                    elif parsed_salvage.get("sectors") is None:
                        sectors = []
                    country = (parsed_salvage.get("country") or "").strip()
                except Exception:
                    pass

                if not skills:
                    try:
                        from chat_gemini_review import extract_skills_heuristic
                        heur = extract_skills_heuristic(jd_text, job_title or "", sectors[0] if sectors else "", "")
                        if isinstance(heur, list):
                            skills = heur
                    except Exception:
                        skills = []
                computed_missing = []
                if not seniority:
                    computed_missing.append("seniority")
                if not job_title:
                    computed_missing.append("job_title")
                if not sectors:
                    computed_missing.append("sector")
                if not country:
                    computed_missing.append("country")
                if computed_missing:
                    if not missing:
                        missing = computed_missing
                else:
                    missing = []
            except Exception:
                summary = (jd_text[:1000] + ("..." if len(jd_text) > 1000 else ""))
                missing = []

        if not isinstance(sectors, list):
            try:
                if isinstance(sectors, str):
                    sectors = [s.strip() for s in sectors.split(",") if s.strip()]
                else:
                    sectors = []
            except Exception:
                sectors = []

        if not observation:
            if raw:
                observation = raw.strip()
                if len(observation) > 1600:
                    observation = observation[:1600] + "..."
            else:
                parts = []
                if job_title:
                    parts.append(f"{seniority + ' ' if seniority else ''}{job_title}")
                if sectors:
                    parts.append(f"in the {', '.join(sectors)} sector")
                if country:
                    parts.append(f"based in {country}")
                lead = ""
                if parts:
                    lead = "Based on the Job Description, this appears to be " + " ".join(parts) + "."
                jd_excerpt = (jd_text or "").strip()
                if jd_excerpt:
                    jd_excerpt = jd_excerpt.replace("\n", " ")[:600].strip()
                    observation = (lead + " " + ("Key JD notes: " + jd_excerpt if jd_excerpt else "")).strip()
                else:
                    observation = lead or summary or ""

        if username and skills:
            try:
                conn = _db_connect()
                cur = conn.cursor()
                try:
                    cur.execute("UPDATE login SET skills = %s WHERE username = %s", (json.dumps(skills, ensure_ascii=False), username))
                    conn.commit()
                except Exception:
                    try:
                        cur.execute("UPDATE login SET skills = %s WHERE username = %s", (", ".join(skills), username))
                        conn.commit()
                    except Exception:
                        pass
                cur.close()
                conn.close()
            except Exception:
                pass

        response = {
            "job_title": job_title or "",
            "seniority": seniority or "",
            "sectors": sectors or [],
            "country": country or "",
            "summary": summary or "",
            "missing": missing or [],
            "raw": raw or "",
            "observation": observation or "",
            "skills": skills or []
        }

        return JSONResponse(response, status_code=200)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "chatbot_api:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=False
    )