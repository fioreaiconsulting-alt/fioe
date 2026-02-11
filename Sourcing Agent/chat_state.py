import os, re, json
from datetime import datetime

# Conversation / global state containers
CONV_CONTEXT = {
    "job_titles": [],
    "companies": [],
    "sectors": [],
    "country": "",
    "languages": [],
    "role_tag": ""
}
ANCHOR_CONTEXT = {"job_title":"", "companies":[]}
LAST_JOB_CONTEXT = None
LAST_JOB_INFO = None
LAST_PROFILE_COUNT = None

PENDING_CLARIFICATION = None
PENDING_EXTRACTION = None
PENDING_EXCEL_OR_REVIEW = None
PENDING_SECTOR_COMPANY = None
PENDING_JD_UPLOAD = None  # New state for JD upload flow

SOURCING_FLOW_STATE = None
SOURCING_FLOW_CONTEXT = {}

# Convenience awaiting flags kept in CONV_CONTEXT as booleans,
# but we also ensure reset clears common ones.
CONV_FLAG_KEYS = [
    "awaiting_initial_choice",
    "awaiting_jd_upload",
    "awaiting_jd_clarification",
    "awaiting_jd_confirmation",
    "awaiting_role_tag_confirm",
    "try_else_modal_active"
]

def history_file(session_id: str):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id.strip() or "default")
    return f"history_{safe}.json"

def load_history(path: str):
    if os.path.isfile(path):
        try:
            with open(path,"r",encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_history(path: str, hist):
    try:
        with open(path,"w",encoding="utf-8") as f:
            json.dump(hist,f,ensure_ascii=False,indent=2)
    except Exception:
        pass

def append_history(path: str, role: str, content: str):
    hist=load_history(path)
    hist.append({
        "ts": datetime.utcnow().isoformat(timespec="seconds")+"Z",
        "role": role,
        "content": content
    })
    if len(hist)>300: hist=hist[-300:]
    save_history(path, hist)

def history_for_prompt(path: str, max_items=24, truncate_each=400):
    hist=load_history(path)[-max_items:]
    lines=[]
    for m in hist:
        c=m.get("content","")
        if len(c)>truncate_each: c=c[:truncate_each]+"…"
        lines.append(f"{m.get('role','?')}: {c}")
    return "\n".join(lines)

def ctx_update(slots: dict):
    if not isinstance(slots,dict): return
    if slots.get("job_titles"):
        val=slots["job_titles"]
        if isinstance(val,str): val=[val]
        CONV_CONTEXT["job_titles"]=[x.strip() for x in val if isinstance(x,str) and x.strip()]
    if slots.get("companies"):
        CONV_CONTEXT["companies"]=[x.strip() for x in slots["companies"] if isinstance(x,str) and x.strip()]
    if slots.get("sectors"):
        CONV_CONTEXT["sectors"]=[x.strip() for x in slots["sectors"] if isinstance(x,str) and x.strip()]
    if slots.get("languages"):
        CONV_CONTEXT["languages"]=[x.strip() for x in slots["languages"] if isinstance(x,str) and x.strip()]
    if isinstance(slots.get("country"), str) and slots.get("country").strip():
        CONV_CONTEXT["country"]=slots["country"].strip()
    # allow explicit flags like seniority
    if isinstance(slots.get("seniority"), str):
        CONV_CONTEXT["seniority"] = slots.get("seniority").strip() if slots.get("seniority") else ""

def set_anchor(job_title="", companies=None):
    if job_title: ANCHOR_CONTEXT["job_title"]=job_title.strip()
    if companies: ANCHOR_CONTEXT["companies"]=[c.strip() for c in companies if c.strip()]

def get_anchor():
    return {"job_title":ANCHOR_CONTEXT.get("job_title",""), "companies":ANCHOR_CONTEXT.get("companies",[])}

def reset_session_state(username=None):
    """
    Clears conversation-level and pending state to start a fresh session.
    If username is provided, will attempt to refresh role_tag from DB after reset.
    """
    global LAST_JOB_INFO, LAST_JOB_CONTEXT, LAST_PROFILE_COUNT
    global PENDING_CLARIFICATION, PENDING_EXTRACTION, PENDING_EXCEL_OR_REVIEW, PENDING_SECTOR_COMPANY
    global PENDING_JD_UPLOAD
    global SOURCING_FLOW_STATE, SOURCING_FLOW_CONTEXT

    LAST_JOB_INFO=None
    LAST_JOB_CONTEXT=None
    LAST_PROFILE_COUNT=None
    PENDING_CLARIFICATION=None
    PENDING_EXTRACTION=None
    PENDING_EXCEL_OR_REVIEW=None
    PENDING_SECTOR_COMPANY=None
    PENDING_JD_UPLOAD=None
    SOURCING_FLOW_STATE=None
    SOURCING_FLOW_CONTEXT={}

    # Reset conversation slots to empty defaults
    CONV_CONTEXT.update({
        "job_titles":[],
        "companies":[],
        "sectors":[],
        "country":"",
        "languages":[],
        "role_tag": ""
    })
    # Clear known awaiting flags if present
    for k in CONV_FLAG_KEYS:
        try:
            if k in CONV_CONTEXT:
                CONV_CONTEXT[k] = False
        except Exception:
            pass

    ANCHOR_CONTEXT.update({"job_title":"", "companies":[]})

    if username:
        force_refresh_role_tag(username)  # Ensure DB role_tag present immediately

# --- Affected Section: add flush context for suggestion intent ---
def ctx_flush_and_update(slots: dict, username=None):
    """
    Flush job_titles, companies, and sectors in CONV_CONTEXT before updating with slots.
    Use this for e.g. suggestion intent, to avoid persisting old context.

    Additionally, keep a conversational 'role_tag' in sync with the first job title
    encountered during the conversation, or force refresh from DB if no job title.
    """
    CONV_CONTEXT["job_titles"] = []
    CONV_CONTEXT["companies"] = []
    CONV_CONTEXT["sectors"] = []
    if slots.get("job_titles"):
        val = slots["job_titles"]
        if isinstance(val, str): val = [val]
        CONV_CONTEXT["job_titles"] = [x.strip() for x in val if isinstance(x, str) and x.strip()]
    if slots.get("companies"):
        CONV_CONTEXT["companies"] = [x.strip() for x in slots["companies"] if isinstance(x, str) and x.strip()]
    if slots.get("sectors"):
        CONV_CONTEXT["sectors"] = [x.strip() for x in slots["sectors"] if isinstance(x, str) and x.strip()]
    if isinstance(slots.get("country"), str) and slots.get("country").strip():
        CONV_CONTEXT["country"] = slots["country"].strip()
    if slots.get("languages"):
        CONV_CONTEXT["languages"] = [x.strip() for x in slots["languages"] if isinstance(x, str) and x.strip()]
    # Keep role_tag aligned OR force DB fetch
    try:
        if CONV_CONTEXT.get("job_titles"):
            CONV_CONTEXT["role_tag"] = (CONV_CONTEXT["job_titles"][0] or "").strip()
        else:
            if username:
                force_refresh_role_tag(username)
            else:
                CONV_CONTEXT["role_tag"] = CONV_CONTEXT.get("role_tag","").strip()  # retain if already set
    except Exception:
        pass

# --- Helper: Always pull role_tag from login DB (force each call) ---
def _fetch_role_tag(username: str) -> str:
    """
    Direct lookup of role_tag for the given username.
    Returns empty string on failure or if not found.
    """
    if not username:
        return ""
    try:
        import psycopg2
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD","") or "orlha"
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()
        cur.execute("SELECT role_tag FROM login WHERE username=%s", (username,))
        row=cur.fetchone()
        cur.close(); conn.close()
        if row and row[0]:
            return (row[0] or "").strip()
    except Exception:
        return ""
    return ""

def force_refresh_role_tag(username: str):
    """
    Force refresh role_tag from DB every time this is called.
    Overwrites any existing role_tag in CONV_CONTEXT.
    """
    try:
        db_tag = _fetch_role_tag(username)
        CONV_CONTEXT["role_tag"] = db_tag if db_tag else ""
    except Exception:
        # On failure, keep existing or empty
        if "role_tag" not in CONV_CONTEXT:
            CONV_CONTEXT["role_tag"] = ""

def get_role_tag(username=None) -> str:
    """
    Return current role_tag. If username provided, FORCE a refresh before returning.
    """
    try:
        if username:
            force_refresh_role_tag(username)
        return (CONV_CONTEXT.get("role_tag") or "").strip()
    except Exception:
        return ""

# --- JD upload helpers (affected section) ---
def set_pending_jd_upload(info: dict):
    """
    Set PENDING_JD_UPLOAD to a dict containing upload metadata or analysis result.
    """
    global PENDING_JD_UPLOAD
    try:
        if isinstance(info, dict):
            PENDING_JD_UPLOAD = info
        else:
            PENDING_JD_UPLOAD = {"note": str(info)}
    except Exception:
        PENDING_JD_UPLOAD = None

def get_pending_jd_upload():
    """
    Return current pending JD upload info or None.
    """
    return PENDING_JD_UPLOAD

def clear_pending_jd_upload():
    global PENDING_JD_UPLOAD
    PENDING_JD_UPLOAD = None

# --- Persistent storage for LAST_JOB_INFO per session ---
_JOB_INFO_PREFIX = "jobinfo_"
def job_info_file(session_id: str):
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", session_id.strip() or "default")
    return f"{_JOB_INFO_PREFIX}{safe}.json"

def persist_last_job_info(session_id: str, job_info: tuple):
    """
    Persist (job_id, base_url) so 'status' can recover after process restart.
    """
    if not isinstance(job_info, tuple) or len(job_info) != 2:
        return
    path = job_info_file(session_id)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"job_id": job_info[0], "base": job_info[1]}, f)
    except Exception:
        pass

def load_last_job_info(session_id: str):
    """
    Load persisted (job_id, base_url) tuple or None.
    """
    path = job_info_file(session_id)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f) or {}
        job_id = data.get("job_id")
        base = data.get("base")
        if job_id and base:
            return (job_id, base)
    except Exception:
        return None
    return None