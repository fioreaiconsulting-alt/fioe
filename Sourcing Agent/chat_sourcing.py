import os, re
from chat_state import (
    LAST_JOB_INFO, LAST_JOB_CONTEXT, LAST_PROFILE_COUNT,
    CONV_CONTEXT, set_anchor, ctx_update,
    # Affected: import persistence helpers
    persist_last_job_info
)
from sourcing_core import SourcingEngine

def build_friendly_context_phrase():
    ctx = LAST_JOB_CONTEXT or CONV_CONTEXT or {}
    jt = (ctx.get("job_titles") or [""])[0] if ctx.get("job_titles") else ""
    comps = ctx.get("companies") or []
    secs = ctx.get("sectors") or []
    country = ctx.get("country") or ""
    parts = []
    if jt: parts.append(jt)
    if comps: parts.append("at " + ", ".join(comps[:4]))
    if secs: parts.append("in " + ", ".join(secs[:3]))
    if country: parts.append(f"based in {country}")
    return " ".join(parts) if parts else "your current search"

async def start_sourcing(job_title, country, companies, seniority, sectors, session_id=None):
    engine = SourcingEngine()

    # AFFECTED SECTION: Persist identity file BEFORE starting the job (to ensure sourcing_core picks it up)
    try:
        userid = (CONV_CONTEXT.get("userid") or "").strip()
        username = (CONV_CONTEXT.get("username") or "").strip()
        if userid or username:
            ident_path = os.path.join(os.getcwd(), ".chatbot_identity.json")
            import json as _json
            with open(ident_path, "w", encoding="utf-8") as f:
                f.write(_json.dumps({"userid": userid, "username": username}))
    except Exception:
        pass
    # END AFFECTED SECTION

    # NOTE: Pass an explicit empty auto_suggested_companies list here; auto-suggestions
    # are produced in higher-level flows when available. Avoid passing the same companies
    # as both companies and auto-suggest to prevent duplication.
    job_id, err = engine.start_job(job_title, country, companies, seniority, sectors, [])
    if err:
        return None, f"Error starting job: {err}"
    status = await engine.poll_status(job_id, attempts=6, delay_sec=1.3)
    formatted = engine.format_status(status, job_id)
    count = engine.extract_profile_count(status)
    if isinstance(count, int):
        global LAST_PROFILE_COUNT
        LAST_PROFILE_COUNT = count
    global LAST_JOB_INFO, LAST_JOB_CONTEXT
    LAST_JOB_INFO = (job_id, engine.base)
    if session_id:
        persist_last_job_info(session_id, LAST_JOB_INFO)

    # AFFECTED SECTION (EXTENDED): propagate userid/username post-ingestion if missing (DB patch)
    try:
        userid = (CONV_CONTEXT.get("userid") or "").strip()
        username = (CONV_CONTEXT.get("username") or "").strip()
        if userid or username:
            urls = status.get("urls") if isinstance(status, dict) else []
            if isinstance(urls, list) and urls and (userid or username):
                try:
                    import psycopg2
                    pg_host = os.getenv("PGHOST", "localhost")
                    pg_port = int(os.getenv("PGPORT", "5432"))
                    pg_user = os.getenv("PGUSER", "postgres")
                    pg_password = os.getenv("PGPASSWORD", "") or "orlha"
                    pg_db = os.getenv("PGDATABASE", "candidate_db")
                    conn = psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
                    cur = conn.cursor()
                    for u in urls:
                        lu = (u or "").strip()
                        if not lu:
                            continue
                        cur.execute(
                            """
                            UPDATE sourcing
                               SET userid = COALESCE(NULLIF(%s,''), userid),
                                   username = COALESCE(NULLIF(%s,''), username)
                             WHERE linkedinurl = %s
                               AND (COALESCE(userid,'') = '' OR COALESCE(username,'') = '')
                            """,
                            (userid, username, lu)
                        )
                    conn.commit()
                    cur.close()
                    conn.close()
                except Exception:
                    pass
    except Exception:
        pass
    # --- End Affected Section ---

    LAST_JOB_CONTEXT = {
        "job_titles": [job_title] if job_title else [],
        "companies": companies or [],
        "sectors": sectors or [],
        "country": country or "",
        "languages": []
    }
    ctx_update(LAST_JOB_CONTEXT)
    set_anchor(job_title, companies or [])
    return {
        "formatted": formatted,
        "job_id": job_id,
        "base": engine.base,
        "count": count,
        "status": status
    }, None