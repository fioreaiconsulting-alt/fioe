import logging
import os

# Structured activity logger (writes daily .txt / JSONL files to log dir)
try:
    from app_logger import (
        log_identity, log_infrastructure, log_financial,
        log_security, log_error, log_approval, read_all_logs,
    )
    _APP_LOGGER_AVAILABLE = True
except ImportError:
    _APP_LOGGER_AVAILABLE = False
    def log_identity(**_kw): pass
    def log_infrastructure(**_kw): pass
    def log_financial(**_kw): pass
    def log_security(**_kw): pass
    def log_error(**_kw): pass
    def log_approval(**_kw): pass
    def read_all_logs(**_kw): return {}

# Load .env file using python-dotenv if available, otherwise fall back to a
# simple built-in parser so DB credentials work without any extra packages.
def _load_dotenv():
    try:
        from dotenv import load_dotenv
        load_dotenv()
        return
    except ImportError:
        pass
    # Built-in fallback: look for .env in the same directory as this script
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if not _line or _line.startswith("#") or "=" not in _line:
                continue
            _key, _, _val = _line.partition("=")
            _key = _key.strip()
            _val = _val.strip().strip('"').strip("'")
            if _key and _key not in os.environ:
                os.environ[_key] = _val

_load_dotenv()
import secrets
import threading
import time
import uuid
from csv import DictWriter
from datetime import datetime
from functools import wraps
import re
import json
import requests
import io
import hashlib
import heapq
import difflib
from flask import Flask, request, send_from_directory, jsonify, abort, Response, stream_with_context
try:
    from flask_limiter import Limiter
    from flask_limiter.util import get_remote_address
    _LIMITER_AVAILABLE = True
except ImportError:
    _LIMITER_AVAILABLE = False

# Import sector and product mappings from separate configuration file
from sector_mappings import (
    PRODUCT_TO_DOMAIN_KEYWORDS, 
    GENERIC_ROLE_KEYWORDS,
    BUCKET_COMPANIES,
    BUCKET_JOB_TITLES
)

# Import DispatcherMiddleware to mount the second app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

app = Flask(__name__, static_url_path='', static_folder='.')

# Set a secret key for session security (shared with data_sorter if integrated)
_flask_secret = os.getenv("FLASK_SECRET_KEY", "")
_is_production = os.getenv("FLASK_ENV", "").lower() in ("production", "prod") or \
                 os.getenv("PRODUCTION", "0") == "1"
if not _flask_secret or _flask_secret == "change-me-in-production-webbridge":
    _flask_secret = secrets.token_hex(32)
    if _is_production:
        # In production, a missing secret key is a critical security failure —
        # sessions will not survive restarts and HMAC signatures will change.
        # Set FLASK_SECRET_KEY in your environment before starting the server:
        #   python -c "import secrets; print(secrets.token_hex(32))"
        logging.critical(
            "FATAL: FLASK_SECRET_KEY is not set in a production environment. "
            "Refusing to start with an ephemeral key. "
            "Set FLASK_SECRET_KEY to a persistent strong random value and restart."
        )
        raise SystemExit(1)
    logging.warning(
        "FLASK_SECRET_KEY is not set (or is the default placeholder). "
        "A random key has been generated for this session — sessions will not "
        "persist across restarts. "
        "Set FLASK_SECRET_KEY in your .env file to a strong random value: "
        "python -c \"import secrets; print(secrets.token_hex(32))\""
    )
app.secret_key = _flask_secret

# Session cookie security flags.
# SESSION_COOKIE_SECURE is True by default (required for HTTPS deployments).
# Set DISABLE_SECURE_COOKIES=1 only in a local HTTP development environment.
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['SESSION_COOKIE_SECURE'] = os.getenv("DISABLE_SECURE_COOKIES", "0") != "1"

# Global upload limit raised to 80 MB to support bulk CV uploads.
# Single-file endpoints enforce their own 6 MB per-file check below.
app.config['MAX_CONTENT_LENGTH'] = 80 * 1024 * 1024  # 80 MB
_SINGLE_FILE_MAX = 6 * 1024 * 1024  # 6 MB per-file limit for single uploads

# Rate limiting (requires flask-limiter: pip install flask-limiter)
if _LIMITER_AVAILABLE:
    _limiter = Limiter(
        get_remote_address,
        app=app,
        default_limits=["200 per hour", "30 per minute"],
        storage_uri="memory://",
    )
    def _rate(limit_string):
        """Return a flask_limiter limit decorator."""
        return _limiter.limit(limit_string)
else:
    import functools
    def _rate(limit_string):
        """No-op when flask-limiter is not installed."""
        def decorator(f):
            return f
        return decorator


def _is_pdf_bytes(b: bytes) -> bool:
    """Return True only if b starts with the PDF magic bytes (%PDF-)."""
    return isinstance(b, (bytes, bytearray)) and len(b) >= 5 and b[:5] == b'%PDF-'


# Semaphore to cap concurrent background CV analysis threads (prevent CPU/memory exhaustion)
_CV_ANALYZE_SEMAPHORE = threading.Semaphore(4)

# Allowlist for credentialed CORS. Override with ALLOWED_ORIGINS env var (comma-separated).
_ALLOWED_ORIGINS = {
    o.strip().lower()
    for o in (os.getenv("ALLOWED_ORIGINS") or
              "http://localhost:3000,http://127.0.0.1:3000,http://localhost:4000,http://127.0.0.1:4000,http://localhost:8091,http://127.0.0.1:8091").split(",")
    if o.strip()
}

def _is_origin_allowed(origin: str) -> bool:
    if not origin:
        return False
    return origin.strip().lower() in _ALLOWED_ORIGINS

# ── Per-user rate limiter ──────────────────────────────────────────────────────
# Loads per-user rate limit overrides from rate_limits.json (same directory).
# Falls back to defaults defined in that file when no user-specific override exists.
# Both webbridge.py and server.js read from this shared file.

_RATE_LIMITS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rate_limits.json")

def _load_rate_limits() -> dict:
    """Return the parsed rate_limits.json; returns empty defaults on any error."""
    try:
        with open(_RATE_LIMITS_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {"defaults": {}, "users": {}}

def _save_rate_limits(config: dict) -> None:
    """Atomically write rate_limits.json."""
    tmp = _RATE_LIMITS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2)
    os.replace(tmp, _RATE_LIMITS_PATH)

_NO_LIMIT = 999999  # sentinel: effectively no rate limit when feature has no config entry

class _UserRateLimiter:
    """Simple per-(username, feature) sliding-window rate limiter."""
    def __init__(self):
        self._state: dict = {}
        self._lock = threading.Lock()

    def is_allowed(self, username: str, feature: str) -> bool:
        if not username:
            return True  # no identity → fall through to global limiter
        config = _load_rate_limits()
        user_limits = config.get("users", {}).get(username, {})
        default_limits = config.get("defaults", {})
        limit_cfg = user_limits.get(feature) or default_limits.get(feature)
        if not limit_cfg:
            return True
        max_req = int(limit_cfg.get("requests", _NO_LIMIT))
        window  = int(limit_cfg.get("window_seconds", 60))
        now = time.time()
        key = (username, feature)
        with self._lock:
            history = [t for t in self._state.get(key, []) if now - t < window]
            if len(history) >= max_req:
                self._state[key] = history
                return False
            history.append(now)
            self._state[key] = history
            return True

_user_limiter = _UserRateLimiter()

def _check_user_rate(feature: str):
    """Decorator that enforces the per-user rate limit for *feature*."""
    def decorator(f):
        import functools
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            # Best-effort username resolution from cookies or JSON body
            username = (
                request.cookies.get("username")
                or (request.get_json(force=True, silent=True) or {}).get("username")
                or ""
            )
            username = username.strip()
            if username and not _user_limiter.is_allowed(username, feature):
                _ip = request.headers.get("X-Forwarded-For", request.remote_addr or "")
                log_security("rate_limit_triggered", username=username, ip_address=_ip,
                             detail=f"Feature: {feature}", severity="warning")
                return jsonify({"error": f"Rate limit exceeded for feature '{feature}'"}), 429
            return f(*args, **kwargs)
        return wrapper
    return decorator

def _require_admin(f):
    """Decorator: reject request with 403 unless the caller is an admin."""
    import functools
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        username = (request.cookies.get("username") or "").strip()
        if not username:
            return jsonify({"error": "Authentication required"}), 401
        try:
            import psycopg2
            conn = psycopg2.connect(
                host=os.getenv("PGHOST", "localhost"),
                port=int(os.getenv("PGPORT", "5432")),
                user=os.getenv("PGUSER", "postgres"),
                password=os.getenv("PGPASSWORD", ""),
                dbname=os.getenv("PGDATABASE", "candidate_db"),
            )
            cur = conn.cursor()
            cur.execute("SELECT useraccess FROM login WHERE username=%s LIMIT 1", (username,))
            row = cur.fetchone()
            cur.close(); conn.close()
            if not row or (row[0] or "").strip().lower() != "admin":
                return jsonify({"error": "Admin access required"}), 403
        except Exception as e:
            return jsonify({"error": f"Auth check failed: {e}"}), 500
        return f(*args, **kwargs)
    return wrapper

def _csrf_required(f):
    """Reject state-changing requests that don't carry X-Requested-With or X-CSRF-Token.
    This is a lightweight CSRF mitigation for XHR/fetch clients; browsers cannot set
    these custom headers in cross-site form submissions, so the check is effective."""
    @wraps(f)
    def wrapped(*args, **kwargs):
        if request.method in ("POST", "PUT", "PATCH", "DELETE"):
            if not (request.headers.get("X-Requested-With") or request.headers.get("X-CSRF-Token")):
                return jsonify({"error": "Missing required header (X-Requested-With or X-CSRF-Token)"}), 403
        return f(*args, **kwargs)
    return wrapped

# ── Admin: rate-limit management API ──────────────────────────────────────────

def _pg_connect():
    """Return a new psycopg2 connection using environment variables."""
    import psycopg2
    return psycopg2.connect(
        host=os.getenv("PGHOST", "localhost"),
        port=int(os.getenv("PGPORT", "5432")),
        user=os.getenv("PGUSER", "postgres"),
        password=os.getenv("PGPASSWORD", ""),
        dbname=os.getenv("PGDATABASE", "candidate_db"),
    )

def _ensure_admin_columns(cur):
    """Idempotently add columns used by admin endpoints.

    Each DDL is wrapped in a savepoint so that a failure (e.g. column already
    exists with a different type, permission error, lock timeout) does NOT
    abort the surrounding psycopg2 transaction.  Without savepoints, a failed
    ALTER TABLE leaves the connection in an 'InFailedSqlTransaction' state and
    every subsequent statement in the same transaction also fails.
    """
    ddls = [
        "ALTER TABLE login ADD COLUMN IF NOT EXISTS target_limit INTEGER DEFAULT 10",
        "ALTER TABLE login ADD COLUMN IF NOT EXISTS last_result_count INTEGER",
        "ALTER TABLE login ADD COLUMN IF NOT EXISTS last_deducted_role_tag TEXT",
        "ALTER TABLE login ADD COLUMN IF NOT EXISTS session TIMESTAMPTZ",
        "ALTER TABLE login ADD COLUMN IF NOT EXISTS google_refresh_token TEXT",
        "ALTER TABLE login ADD COLUMN IF NOT EXISTS google_token_expires TIMESTAMP",
        "ALTER TABLE login ADD COLUMN IF NOT EXISTS corporation TEXT",
        "ALTER TABLE login ADD COLUMN IF NOT EXISTS useraccess TEXT",
        "ALTER TABLE login ADD COLUMN IF NOT EXISTS cse_query_count INTEGER DEFAULT 0",
        "ALTER TABLE login ADD COLUMN IF NOT EXISTS price_per_query NUMERIC(10,4) DEFAULT 0",
    ]
    for i, ddl in enumerate(ddls):
        sp = f"_adm_col_{i}"
        try:
            cur.execute(f"SAVEPOINT {sp}")
            cur.execute(ddl)
            cur.execute(f"RELEASE SAVEPOINT {sp}")
        except Exception:
            try:
                cur.execute(f"ROLLBACK TO SAVEPOINT {sp}")
            except Exception:
                pass

def _build_users_select(avail):
    """Return a SELECT … FROM login query built from the actual available columns.

    avail must be a dict {column_name: data_type} (from information_schema.columns).
    Every expression falls back to a safe literal (empty string / 0 / NULL) when
    the corresponding column does not exist, so the query never fails due to a
    missing column regardless of how the login table was originally created.
    """
    def _ts(c):
        if c not in avail:
            return f"NULL::text AS {c}"
        # Only use to_char for actual date/timestamp types; TEXT columns are returned as-is.
        dtype = avail[c]
        if 'timestamp' in dtype or dtype == 'date':
            return f"to_char({c}, 'YYYY-MM-DD HH24:MI') AS {c}"
        return f"COALESCE({c}::text, '') AS {c}"
    def _txt(c):
        return f"COALESCE({c}, '') AS {c}" if c in avail else f"'' AS {c}"
    def _int(c, default=0):
        return f"COALESCE({c}, {default}) AS {c}" if c in avail else f"{default} AS {c}"
    def _num(c, default=0):
        return f"COALESCE({c}::numeric, {default}) AS {c}" if c in avail else f"{default} AS {c}"
    # userid may be named 'id' on older schemas
    if 'userid' in avail:
        uid_expr = "userid::text AS userid"
    elif 'id' in avail:
        uid_expr = "id::text AS userid"
    else:
        uid_expr = "NULL AS userid"
    # role_tag may be named 'roletag'
    if 'role_tag' in avail:
        role_expr = "COALESCE(role_tag, '') AS role_tag"
    elif 'roletag' in avail:
        role_expr = "COALESCE(roletag, '') AS role_tag"
    else:
        role_expr = "'' AS role_tag"
    # jskillset may be stored as 'skills' or 'skillset'
    jsk_col = next((c for c in ('jskillset', 'skills', 'skillset') if c in avail), None)
    jsk_expr = f"COALESCE({jsk_col}, '') AS jskillset" if jsk_col else "'' AS jskillset"
    # jd preview
    jd_expr = ("CASE WHEN jd IS NOT NULL AND jd != '' THEN LEFT(jd, 120) ELSE '' END AS jd"
               if 'jd' in avail else "'' AS jd")
    # google_refresh_token: mask the value, only show Set/empty
    grt_expr = ("CASE WHEN google_refresh_token IS NOT NULL AND google_refresh_token != ''"
                "     THEN 'Set' ELSE '' END AS google_refresh_token"
                if 'google_refresh_token' in avail else "'' AS google_refresh_token")
    return f"""
        SELECT
            {uid_expr},
            username,
            {_txt('cemail')},
            {_txt('password')},
            {_txt('fullname')},
            {_txt('corporation')},
            {_ts('created_at')},
            {role_expr},
            {_int('token')},
            {jd_expr},
            {jsk_expr},
            {grt_expr},
            {_ts('google_token_expires')},
            {_int('last_result_count')},
            {_txt('last_deducted_role_tag')},
            {_ts('session')},
            {_txt('useraccess')},
            {_int('target_limit', 10)},
            {_int('cse_query_count')},
            {_num('price_per_query')}
        FROM login ORDER BY username
    """


@app.get("/admin/rate-limits")
@_require_admin
def admin_get_rate_limits():
    """Return current rate_limits.json content plus full user details."""
    config = _load_rate_limits()
    users_list = []
    db_err = None
    try:
        conn = _pg_connect()
        cur = conn.cursor()
        _ensure_admin_columns(cur)
        conn.commit()
        # Discover actual columns (with their data types) so the SELECT is
        # resilient to schema differences and to columns stored as TEXT instead
        # of TIMESTAMPTZ (to_char only works on date/timestamp types).
        cur.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_schema='public' AND table_name='login'"
        )
        avail = {r[0].lower(): r[1].lower() for r in cur.fetchall()}
        cur.execute(_build_users_select(avail))
        cols = [d[0] for d in cur.description]
        users_list = [dict(zip(cols, row)) for row in cur.fetchall()]
        cur.close(); conn.close()
    except Exception as e:
        logger.error(f"[admin/rate-limits] DB error fetching users: {e}")
        db_err = True
    result = {"config": config, "users": users_list}
    if db_err:
        result["db_error"] = "Failed to load users from database. Check server logs for details."
    return jsonify(result), 200

@app.post("/admin/rate-limits")
@_csrf_required
@_require_admin
def admin_save_rate_limits():
    """Replace rate_limits.json with the POSTed body."""
    body = request.get_json(force=True, silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "JSON object required"}), 400
    defaults = body.get("defaults")
    users = body.get("users")
    if not isinstance(defaults, dict) or not isinstance(users, dict):
        return jsonify({"error": "'defaults' and 'users' keys required"}), 400
    # Validate structure: each limit must have requests (int ≥ 1) and window_seconds (int ≥ 1)
    for scope_label, scope in [("defaults", defaults)] + [("users." + u, v) for u, v in users.items()]:
        for feat, cfg in scope.items():
            if not isinstance(cfg, dict):
                return jsonify({"error": f"Invalid config at {scope_label}.{feat}"}), 400
            if not (isinstance(cfg.get("requests"), int) and cfg["requests"] >= 1):
                return jsonify({"error": f"'{scope_label}.{feat}.requests' must be int ≥ 1"}), 400
            if not (isinstance(cfg.get("window_seconds"), int) and cfg["window_seconds"] >= 1):
                return jsonify({"error": f"'{scope_label}.{feat}.window_seconds' must be int ≥ 1"}), 400
    _save_rate_limits({"defaults": defaults, "users": users})
    return jsonify({"ok": True}), 200

@app.post("/admin/update-token")
@_csrf_required
@_require_admin
def admin_update_token():
    """Set the token balance for a specific user."""
    body = request.get_json(force=True, silent=True) or {}
    username = (body.get("username") or "").strip()
    token_val = body.get("token")
    if not username or token_val is None:
        return jsonify({"error": "username and token required"}), 400
    try:
        token_int = int(token_val)
        if token_int < 0:
            return jsonify({"error": "token must be >= 0"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "token must be an integer"}), 400
    try:
        conn = _pg_connect()
        cur = conn.cursor()
        # Read current balance before update to compute transaction delta
        cur.execute("SELECT COALESCE(token,0) FROM login WHERE username = %s", (username,))
        prev_row = cur.fetchone()
        token_before = int(prev_row[0]) if prev_row else None
        cur.execute("UPDATE login SET token = %s WHERE username = %s RETURNING token", (token_int, username))
        row = cur.fetchone()
        conn.commit(); cur.close(); conn.close()
        if not row:
            return jsonify({"error": "User not found"}), 404
        token_after = int(row[0])
        # Log the admin credit adjustment
        if _LOGGER_AVAILABLE:
            delta = token_after - token_before if token_before is not None else None
            if delta is None:
                txn_type = "adjustment"
            elif delta > 0:
                txn_type = "credit"
            elif delta < 0:
                txn_type = "spend"
            else:
                txn_type = "adjustment"
            log_financial(
                username=username,
                feature="admin_token_adjustment",
                transaction_type=txn_type,
                token_before=token_before,
                token_after=token_after,
                transaction_amount=abs(delta) if delta is not None else None,
            )
        return jsonify({"ok": True, "username": username, "token": token_after}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/admin/update-target-limit")
@_csrf_required
@_require_admin
def admin_update_target_limit():
    """Set the per-user default result target limit."""
    body = request.get_json(force=True, silent=True) or {}
    username = (body.get("username") or "").strip()
    limit_val = body.get("target_limit")
    if not username or limit_val is None:
        return jsonify({"error": "username and target_limit required"}), 400
    try:
        limit_int = int(limit_val)
        if limit_int < 1:
            return jsonify({"error": "target_limit must be >= 1"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "target_limit must be an integer"}), 400
    try:
        conn = _pg_connect()
        cur = conn.cursor()
        _ensure_admin_columns(cur)
        cur.execute(
            "UPDATE login SET target_limit = %s WHERE username = %s RETURNING target_limit",
            (limit_int, username)
        )
        row = cur.fetchone()
        conn.commit(); cur.close(); conn.close()
        if not row:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"ok": True, "username": username, "target_limit": row[0]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/admin/update-price-per-query")
@_csrf_required
@_require_admin
def admin_update_price_per_query():
    """Set the price-per-CSE-query for a specific user."""
    body = request.get_json(force=True, silent=True) or {}
    username = (body.get("username") or "").strip()
    price_val = body.get("price_per_query")
    if not username or price_val is None:
        return jsonify({"error": "username and price_per_query required"}), 400
    try:
        price_float = float(price_val)
        if price_float < 0:
            return jsonify({"error": "price_per_query must be >= 0"}), 400
    except (TypeError, ValueError):
        return jsonify({"error": "price_per_query must be a number"}), 400
    try:
        conn = _pg_connect()
        cur = conn.cursor()
        _ensure_admin_columns(cur)
        cur.execute(
            "UPDATE login SET price_per_query = %s WHERE username = %s RETURNING price_per_query",
            (price_float, username)
        )
        row = cur.fetchone()
        conn.commit(); cur.close(); conn.close()
        if not row:
            return jsonify({"error": "User not found"}), 404
        return jsonify({"ok": True, "username": username, "price_per_query": float(row[0])}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def _increment_cse_query_count(username, count):
    """Increment cse_query_count in the login table for the given user."""
    if not username or count is None or count < 1:
        return
    try:
        conn = _pg_connect()
        cur = conn.cursor()
        _ensure_admin_columns(cur)
        cur.execute(
            "UPDATE login SET cse_query_count = COALESCE(cse_query_count, 0) + %s WHERE username = %s",
            (int(count), username)
        )
        conn.commit(); cur.close(); conn.close()
    except Exception as e:
        logger.warning(f"[CSE count] Failed to update cse_query_count for '{username}': {e}")

@app.get("/admin/appeals")
@_require_admin
def admin_get_appeals():
    """Return sourcing rows that have a non-empty appeal value."""
    try:
        conn = _pg_connect()
        cur = conn.cursor()
        # Ensure appeal column exists in sourcing table
        cur.execute("ALTER TABLE sourcing ADD COLUMN IF NOT EXISTS appeal TEXT")
        conn.commit()
        cur.execute("""
            SELECT s.linkedinurl,
                   COALESCE(s.name, '') AS name,
                   COALESCE(s.jobtitle, '') AS jobtitle,
                   COALESCE(s.company, '') AS company,
                   s.appeal,
                   COALESCE(s.username, '') AS username,
                   COALESCE(s.userid, '') AS userid
            FROM sourcing s
            WHERE s.appeal IS NOT NULL AND s.appeal != ''
            ORDER BY s.linkedinurl
        """)
        cols = [d[0] for d in cur.description]
        rows = [dict(zip(cols, row)) for row in cur.fetchall()]
        cur.close(); conn.close()
        return jsonify({"appeals": rows}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/admin/appeal-action")
@_csrf_required
@_require_admin
def admin_appeal_action():
    """Approve or reject a user appeal.

    Body: { "linkedinurl": "...", "username": "...", "action": "approve"|"reject" }
    Approve: adds 1 token to the user's login record, then deletes the sourcing row.
    Reject: deletes the sourcing row without adding a token.
    """
    body = request.get_json(force=True, silent=True) or {}
    linkedinurl = (body.get("linkedinurl") or "").strip()
    username = (body.get("username") or "").strip()
    action = (body.get("action") or "").strip().lower()
    if not linkedinurl or action not in ("approve", "reject"):
        return jsonify({"error": "linkedinurl and action ('approve'|'reject') required"}), 400
    try:
        conn = _pg_connect()
        cur = conn.cursor()
        new_token = None
        if action == "approve" and username:
            cur.execute(
                "UPDATE login SET token = COALESCE(token, 0) + 1 WHERE username = %s RETURNING token",
                (username,)
            )
            row = cur.fetchone()
            if row:
                new_token = row[0]
        # Delete the sourcing row (appeal handled)
        cur.execute("DELETE FROM sourcing WHERE linkedinurl = %s", (linkedinurl,))
        deleted = cur.rowcount
        conn.commit(); cur.close(); conn.close()
        return jsonify({"ok": True, "action": action, "deleted": deleted, "new_token": new_token}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.get("/admin/logs")
@_require_admin
def admin_get_logs():
    """Return structured log entries for the System Logs dashboard tab.

    Query params (both optional):
      from  — start date  YYYY-MM-DD (inclusive)
      to    — end date    YYYY-MM-DD (inclusive)

    Response: { identity: [...], infrastructure: [...], agentic: [...],
                financial: [...], security: [...], approval: [...], errors: [...] }
    """
    from_date = request.args.get("from") or None
    to_date   = request.args.get("to")   or None
    # Validate date format and value using datetime.strptime
    from datetime import datetime as _dt
    def _valid_date(s):
        try:
            _dt.strptime(s, "%Y-%m-%d")
            return True
        except (ValueError, TypeError):
            return False
    if from_date and not _valid_date(from_date):
        return jsonify({"error": "Invalid 'from' date; expected YYYY-MM-DD"}), 400
    if to_date and not _valid_date(to_date):
        return jsonify({"error": "Invalid 'to' date; expected YYYY-MM-DD"}), 400
    logs = read_all_logs(from_date=from_date, to_date=to_date)
    return jsonify(logs), 200


@app.post("/admin/client-error")
@_csrf_required
def admin_client_error():
    """Accept a client-side error report from webbridge_client.js and write it
    to the Error Capture log.  No admin role required — any authenticated user
    (or the browser global handler) can submit errors."""
    body = request.get_json(force=True, silent=True) or {}
    message  = str(body.get("message",  "") or "")[:2000]
    source   = str(body.get("source",   "") or "")[:200]
    severity = str(body.get("severity", "") or "error")
    username = str(body.get("username", "") or "")[:200]
    if severity not in ("info", "warning", "warn", "error", "critical"):
        severity = "error"
    if message:
        log_error(source=source or "client", message=message, severity=severity,
                  username=username, endpoint="client-side")
    return jsonify({"ok": True}), 200


@app.post("/admin/logs/analyse-error")
@_csrf_required
@_require_admin
def admin_analyse_error():
    """Use Gemini to explain an error message and generate a Copilot-ready fix prompt.

    Body: { "error_message": "...", "source": "..." }
    Response: { "explanation": "...", "suggested_fix": "...", "copilot_prompt": "..." }
    """
    body = request.get_json(force=True, silent=True) or {}
    error_message = str(body.get("error_message", "") or "")[:3000]
    source        = str(body.get("source",        "") or "")[:200]
    if not error_message:
        return jsonify({"error": "error_message required"}), 400

    # Require Gemini to be available
    if not (genai and GEMINI_API_KEY):
        return jsonify({"error": "Gemini API not configured on server"}), 503

    prompt = f"""You are an expert software engineer and debugger.
A production error was captured from the AutoSourcing platform:

Source: {source or "unknown"}
Error:
{error_message}

Provide a JSON response with exactly four keys:
1. "explanation" — a clear, plain-language explanation of what this error means and why it occurs (2-4 sentences, no markdown).
2. "suggested_fix" — a concrete, developer-ready description of how to fix it (bullet list, no markdown code fences).
3. "test_case" — a short, developer-ready test case or verification step that proves the fix worked. For example: "Send a POST request to the endpoint with the corrected payload and confirm a 200 OK response." or "Run `pytest tests/test_endpoint.py::test_update_role_tag` and verify it passes." Keep it concise (1-3 steps, no markdown code fences).
4. "copilot_prompt" — a ready-to-paste prompt for GitHub Copilot that includes the raw error, the explanation, the suggested fix, the test case, and asks Copilot to generate the corrected implementation.

Respond ONLY with valid JSON. No extra commentary."""

    try:
        model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
        resp  = model.generate_content(prompt)
        raw   = (resp.text or "").strip()

        # Strip markdown code fences if present
        raw = re.sub(r'^```[a-z]*\s*', '', raw, flags=re.MULTILINE)
        raw = re.sub(r'\s*```$', '', raw, flags=re.MULTILINE)

        parsed = json.loads(raw)
        explanation   = str(parsed.get("explanation",   "") or "")
        suggested_fix = str(parsed.get("suggested_fix", "") or "")
        test_case     = str(parsed.get("test_case",     "") or "")
        copilot_prompt = str(parsed.get("copilot_prompt", "") or "")
        if not copilot_prompt:
            copilot_prompt = (
                f"// GitHub Copilot — Error Fix Request\n"
                f"// Source: {source}\n//\n"
                f"// === ERROR ===\n{error_message}\n\n"
                f"// === EXPLANATION ===\n{explanation}\n\n"
                f"// === SUGGESTED FIX ===\n{suggested_fix}\n\n"
                f"// === TEST CASE ===\n{test_case}\n\n"
                f"// Please suggest a corrected implementation that passes the above test case."
            )
        return jsonify({
            "ok": True,
            "explanation":    explanation,
            "suggested_fix":  suggested_fix,
            "test_case":      test_case,
            "copilot_prompt": copilot_prompt,
        }), 200
    except Exception as exc:
        logger.warning(f"[admin/analyse-error] Gemini call failed: {exc}")
        return jsonify({"error": f"Gemini analysis failed: {exc}"}), 500



# A wildcard ACAO is never sent — that would bypass credential isolation.
def _apply_cors_headers(response):
    try:
        origin = request.headers.get('Origin', '')
        if origin and _is_origin_allowed(origin):
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Access-Control-Allow-Credentials'] = 'true'
            response.headers['Vary'] = 'Origin'
        # Non-allowlisted origins: deliberately omit ACAO so the browser blocks
        # the cross-origin read.  Do NOT fall back to '*'.
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, OPTIONS, PATCH, PUT, DELETE'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
    except Exception:
        pass
    return response

# ── Security response headers ───────────────────────────────────────────────
# Note: 'unsafe-inline' and 'unsafe-eval' are required because the current
# HTML files use extensive inline <script> blocks and eval-adjacent patterns
# (e.g. Chart.js).  Removing them requires migrating all inline JS to external
# files — tracked as a follow-up hardening task.  This CSP still provides
# meaningful protection by locking down allowed external script/style sources
# and blocking clickjacking via frame-ancestors.
_CSP = (
    "default-src 'self'; "
    # Inline scripts / eval needed until inline JS is moved to external files.
    "script-src 'self' 'unsafe-inline' 'unsafe-eval' "
    "https://cdn.jsdelivr.net https://unpkg.com https://cdnjs.cloudflare.com; "
    # Inline styles needed until inline style blocks are moved to stylesheets.
    "style-src 'self' 'unsafe-inline' "
    "https://fonts.googleapis.com https://unpkg.com "
    "https://cdnjs.cloudflare.com https://cdn.jsdelivr.net; "
    "font-src 'self' https://fonts.gstatic.com; "
    # img-src includes https: for Leaflet map tiles (loaded from tile CDNs).
    "img-src 'self' data: blob: https:; "
    # connect-src: all API calls go through the same origin (WB_BASE_URL).
    "connect-src 'self'; "
    "worker-src 'self' blob:; "
    # frame-ancestors 'self' is consistent with X-Frame-Options: SAMEORIGIN.
    "frame-ancestors 'self';"
)

@app.after_request
def _apply_cors(response):
    # HSTS — only sent over HTTPS; instructs browsers to always use HTTPS.
    if os.getenv("FORCE_HTTPS", "0") == "1":
        response.headers.setdefault(
            "Strict-Transport-Security",
            "max-age=31536000; includeSubDomains"
        )
    # Content-Security-Policy: restrict what the browser can load/execute.
    response.headers.setdefault("Content-Security-Policy", _CSP)
    # Prevent MIME-type sniffing attacks.
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    # Block the page from being framed (clickjacking protection); consistent
    # with frame-ancestors 'self' in the CSP above.
    response.headers.setdefault("X-Frame-Options", "SAMEORIGIN")
    # Control the Referer header sent with outbound requests.
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")

    # ── HTTP error capture: log 4xx as warning, 5xx as critical ──────────────
    # Skip OPTIONS pre-flight and the logging/static endpoints themselves.
    _skip_paths = ("/admin/client-error", "/admin/logs", "/favicon.ico")
    if (response.status_code >= 400
            and request.method != "OPTIONS"
            and not any(request.path.startswith(p) for p in _skip_paths)):
        _sc = response.status_code
        _sev = "critical" if _sc >= 500 else "warning"
        _username = (request.cookies.get("username") or "").strip()
        _ip = (request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
               or request.remote_addr or "")
        # Best-effort: read JSON error message without consuming the body stream
        _body_msg = ""
        try:
            _ct = response.content_type or ""
            if "json" in _ct:
                _body_msg = response.get_data(as_text=True)[:500]
        except Exception:
            pass
        log_error(
            source="webbridge.py",
            message=f"{request.method} {request.path} → HTTP {_sc}",
            severity=_sev,
            username=_username,
            endpoint=request.path,
            http_status=_sc,
            ip_address=_ip,
            detail=_body_msg,
        )

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

SEARCH_RESULTS_TARGET = int(os.getenv("SEARCH_RESULTS_TARGET") or 0)
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

CITY_TO_COUNTRY_PATH = os.path.join(BASE_DIR, "static", "city_to_country.json")
if not os.path.isfile(CITY_TO_COUNTRY_PATH):
    CITY_TO_COUNTRY_PATH = os.path.join(BASE_DIR, "city_to_country.json")

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

def _load_city_to_country():
    try:
        if os.path.isfile(CITY_TO_COUNTRY_PATH):
            with open(CITY_TO_COUNTRY_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                logger.info(f"[CityToCountry] Loaded from {CITY_TO_COUNTRY_PATH}")
                return data
    except Exception as e:
        logger.warning(f"[CityToCountry] Failed to load {CITY_TO_COUNTRY_PATH}: {e}")
    return {}

CITY_TO_COUNTRY_DATA = _load_city_to_country()

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

def _normalize_seniority_single(seniority_text: str) -> str:
    """
    Collapse compound seniority labels (e.g. 'Mid-Senior', 'Senior Manager') to a single
    canonical token. Picks the *highest* seniority in the compound so the search query
    is not under-targeted.
    """
    if not seniority_text:
        return seniority_text
    s = seniority_text.strip()
    sl = s.lower()
    # Already a clean single level from the canonical set
    if sl in {"junior", "mid", "senior", "manager", "director", "associate", "intern",
              "entry-level", "entry level", "lead", "principal", "vp", "staff", "expert",
              "c-suite", "head"}:
        return s
    # Compound: pick highest level present
    if any(tok in sl for tok in ["director", "vp", "vice president", "principal", "head", "chief", "staff", "expert"]):
        return "Director"
    if any(tok in sl for tok in ["manager", "lead", "supervisor"]):
        return "Manager"
    if "senior" in sl:
        return "Senior"
    if any(tok in sl for tok in ["mid", "middle"]):
        return "Mid"
    if any(tok in sl for tok in ["junior", "entry", "intern", "trainee", "graduate", "associate"]):
        return "Junior"
    # Return first word as best guess if still compound
    parts = re.split(r'[-/\s]+', s)
    return parts[0] if parts else s

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
SECTORS_TOKEN_INDEX = []  # list of (label, token_set) pairs, built after SECTORS_INDEX is loaded

# Minimum Jaccard score required for a sector label match (configurable via env var)
MIN_SECTOR_JACCARD = float(os.getenv("MIN_SECTOR_JACCARD", "0.12"))

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
    # Rebuild pre-tokenized index if already defined (handles runtime reloads).
    # _build_sectors_token_index is defined later in the module; this guard avoids a
    # NameError on the initial _load_sectors_index() call made at module startup before
    # that function is defined, while still keeping SECTORS_TOKEN_INDEX in sync on
    # any subsequent runtime reloads of sectors.json.
    _rebuild = globals().get('_build_sectors_token_index')
    if callable(_rebuild):
        _rebuild()

# immediately load sectors index (best-effort)
_load_sectors_index()

# Helper functions for sector matching (new)
def _token_set(s):
    if not s: return set()
    # Normalize & and &amp; to "and" so label tokens match consistently
    normalized = re.sub(r'&amp;|&', 'and', s.lower())
    return set(re.findall(r'\w+', normalized))

def _build_sectors_token_index():
    """Pre-tokenize all sector labels so _find_best_sector_match_for_text avoids repeated tokenization."""
    global SECTORS_TOKEN_INDEX
    SECTORS_TOKEN_INDEX = [(label, _token_set(label)) for label in SECTORS_INDEX]

_build_sectors_token_index()

def _find_best_sector_match_for_text(candidate):
    """
    Given an arbitrary candidate string (e.g., "Air Conditioning / HVAC"),
    find the best-matching label from SECTORS_INDEX by token overlap.
    Uses Jaccard similarity (intersection/union) to normalize for label length.
    Requires a minimum Jaccard score (MIN_SECTOR_JACCARD) to reject weak matches.
    Returns the matched label (exact wording from sectors.json) or None.
    """
    try:
        if not candidate or not SECTORS_INDEX:
            return None
        cand_tokens = _token_set(candidate)
        if not cand_tokens:
            return None
        best = None
        best_score = 0.0
        best_abs = 0
        top_candidates = []
        for label, label_tokens in SECTORS_TOKEN_INDEX:
            if not label_tokens:
                continue
            intersection = cand_tokens & label_tokens
            abs_overlap = len(intersection)
            if abs_overlap == 0:
                continue
            # Jaccard similarity: intersection / union (normalizes for label length)
            score = abs_overlap / len(cand_tokens | label_tokens)
            top_candidates.append((score, abs_overlap, label))
            # Prefer highest Jaccard score; tie-break by abs overlap, then shorter label
            if (score > best_score or
                    (score == best_score and abs_overlap > best_abs) or
                    (score == best_score and abs_overlap == best_abs and best and len(label) < len(best))):
                best_score = score
                best_abs = abs_overlap
                best = label
        # Require a minimum Jaccard threshold to avoid weak matches.
        # Exception: accept absolute overlap >= 1 for short candidate strings (<=2 tokens)
        # where Jaccard can underestimate match quality (e.g. single-token "cloud").
        match_ok = best and (
            best_score >= MIN_SECTOR_JACCARD or
            (len(cand_tokens) <= 2 and best_abs >= 1)
        )
        if match_ok:
            top3 = heapq.nlargest(3, top_candidates, key=lambda x: (x[0], x[1]))
            logger.debug(
                "_find_best_sector_match_for_text top-3 for %r: %s",
                candidate,
                top3
            )
            return best
        logger.debug(
            "_find_best_sector_match_for_text: no strong match for %r (best_score=%.4f, top-3=%s)",
            candidate,
            best_score,
            heapq.nlargest(3, top_candidates, key=lambda x: (x[0], x[1])) if top_candidates else []
        )
        return None
    except Exception:
        return None

# Small explicit keyword -> sectors.json label mapping to handle cases like HVAC -> Machinery
# Keys are lowercase keywords; values are exact labels expected to exist (or closely match) in SECTORS_INDEX
# NOTE: pharma/clinical mapping removed per user request (do not auto-apply pharma heuristics)
_KEYWORD_TO_SECTOR_LABEL = {
    "aircon": "Industrial & Manufacturing > Machinery",
    "air-con": "Industrial & Manufacturing > Machinery",
    "hvac": "Industrial & Manufacturing > Machinery",
    "air conditioning": "Industrial & Manufacturing > Machinery",
    "air solutions": "Industrial & Manufacturing > Machinery",
    "refrigeration": "Industrial & Manufacturing > Machinery",
    "chiller": "Industrial & Manufacturing > Machinery",
    "ventilation": "Industrial & Manufacturing > Machinery",
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
    Uses word-boundary regex to avoid false substring matches (e.g., "ai" inside "training").
    """
    try:
        txt = (text or "").lower()
        for kw, label in _KEYWORD_TO_SECTOR_LABEL.items():
            if re.search(r'\b' + re.escape(kw) + r'\b', txt):
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


def _extract_confirmed_skills(profile_context: str, target_skills: list) -> list:
    """
    Extractive pass: find target skills that are explicitly mentioned in
    profile_context using word-boundary regex (case-insensitive).
    Returns list of confirmed skill names (preserving original casing).
    """
    if not profile_context or not target_skills:
        return []
    exp_lower = profile_context.lower()
    confirmed = []
    for skill in target_skills:
        if not skill or not isinstance(skill, str):
            continue
        pattern = r'\b' + re.escape(skill.lower()) + r'\b'
        if re.search(pattern, exp_lower):
            confirmed.append(skill)
    return confirmed


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
@_rate("10 per minute")
@_check_user_rate("gemini")
def gemini_analyze_jd():
    """
    Implements workflow:
    1. Identify companies mentioned in JD
    2. Determine sectors from identified companies (using sectors.json)
    3. If no companies found, infer sector from JD content
    4. Filter companies by legal entity in specified country
    5. Always identify at least one sector (using sectors.json)
    6. Derive second sector from skillset, job title, and JD text (if applicable)
    7. Enforce maximum of 2 sectors
    8. Generate at least 2 job titles (original + suggested variant)
    
    Returns JSON:
    {
      "job_title": "...",  # Single title for backward compatibility
      "job_titles": [...],  # Array of at least 2 job titles (original + suggestions)
      "seniority": "...",
      "sectors": [...],  # Always mapped to sectors.json, maximum 2 sectors
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
            conn = psycopg2.connect(host=os.getenv("PGHOST","localhost"), port=int(os.getenv("PGPORT","5432")), user=os.getenv("PGUSER","postgres"), password=os.getenv("PGPASSWORD", ""), dbname=os.getenv("PGDATABASE","candidate_db"))
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

    # Word-count guard: reject JDs that are too long for reliable Gemini analysis
    JD_MAX_WORDS = 700
    jd_word_count = len(text_input.split())
    if jd_word_count > JD_MAX_WORDS:
        return jsonify({
            "error": "jd_too_long",
            "word_count": jd_word_count,
            "max_words": JD_MAX_WORDS,
            "message": f"The uploaded JD is too long ({jd_word_count:,} words). Please reduce it to {JD_MAX_WORDS:,} words or fewer and re-upload."
        }), 413

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
            "You are a recruiting assistant. Analyze the following job description and identify:\n"
            "1. The PRIMARY job title being hired for (the exact role name, e.g. 'Cloud Engineer', 'Site Activation Manager', 'Sales Manager')\n"
            "2. ALL company names explicitly mentioned\n"
            "3. ALL product/technology/service names mentioned (e.g., 'Aircon', 'HVAC systems', 'cloud platforms', 'ERP software')\n"
            "Return STRICT JSON with this structure:\n"
            "{ \"job_title\": \"Exact Role Title\", \"companies\": [\"Company Name 1\", ...], \"products\": [\"Product1\", \"Product2\", ...] }\n"
            "Rules:\n"
            "- job_title: extract the SPECIFIC role title from the JD (e.g., 'Cloud Engineer', not 'Gaming Professional' or 'Technology Professional')\n"
            "- Include the hiring company if explicitly mentioned\n"
            "- Include client companies, partner companies, or competitor companies mentioned\n"
            "- Use official company names (e.g., 'Microsoft' not 'MS', 'Johnson & Johnson' not 'J&J')\n"
            "- Do NOT include generic industry terms (e.g., 'tech companies', 'pharma firms')\n"
            "- For products: include tangible product categories (e.g., 'Aircon', 'air conditioning', 'HVAC', 'refrigerators', 'mobile phones', 'electric vehicles')\n"
            "- Return empty string/array if none found\n"
            "\nJOB DESCRIPTION:\n" + (text_input[:15000]) + "\n\nJSON:"
        )
        
        _jd_gen_config = genai.GenerationConfig(temperature=0.1, max_output_tokens=2048)
        model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
        company_resp = model.generate_content(company_prompt, generation_config=_jd_gen_config)
        company_raw = (company_resp.text or "").strip()
        company_obj = _extract_json_object(company_raw) or {}
        
        # Extract job title identified in Step 1 as a strong signal for the main analysis
        step1_job_title = (company_obj.get("job_title") or "").strip()
        
        raw_companies = company_obj.get("companies") or []
        if isinstance(raw_companies, list):
            for c in raw_companies:
                if isinstance(c, str) and c.strip():
                    identified_companies.append(c.strip())

        # Extract products/technologies mentioned in the JD
        identified_products = []
        raw_products = company_obj.get("products") or []
        if isinstance(raw_products, list):
            for p in raw_products:
                if isinstance(p, str) and p.strip():
                    identified_products.append(p.strip())
        
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
        elif identified_products:
            # When no companies are found, use product references to infer sector
            companies_context = f"\n\nIDENTIFIED PRODUCTS/TECHNOLOGIES: {', '.join(identified_products)}\n"
            companies_context += (
                "No specific companies were mentioned in this JD, but these products/technologies were identified. "
                "Use them to:\n"
                "1. Identify the correct industry sector (e.g., 'Aircon'/'HVAC' → Industrial & Manufacturing > Machinery)\n"
                "2. Classify the role correctly — do NOT assign Gaming/Technology sectors to physical product roles\n"
                "3. The company suggestions should be direct competitors that manufacture or sell these products\n"
                "IMPORTANT: 'Aircon', 'air conditioning', 'HVAC', 'refrigeration' are Industrial/Manufacturing products, NOT gaming or technology."
            )

        # Build strict JSON request to Gemini
        # Include Step 1 job title as an anchor to prevent misclassification
        job_title_hint = ""
        if step1_job_title:
            job_title_hint = (
                f"\n\nPRE-IDENTIFIED JOB TITLE: \"{step1_job_title}\"\n"
                "Use this as the job_title in your response unless the JD text strongly contradicts it."
            )
        prompt = (
            "You are a recruiting assistant. Analyze the job description and return STRICT JSON with keys:\n"
            "{ parsed: { job_title, seniority, sector, country, skills }, missing: [...], summary: string, suggestions: [...], justification: string, observation: string, raw: string }\n"
            "IMPORTANT:\n"
            "- job_title: extract the EXACT role name from the JD (e.g., 'Cloud Engineer', 'Site Activation Manager'). NEVER use generic labels like 'Gaming Professional' or 'Technology Professional'.\n"
            "- seniority: return EXACTLY ONE single-word or two-word level (e.g. 'Junior', 'Mid', 'Senior', 'Manager', 'Director'). Do NOT combine levels (e.g. do NOT return 'Mid-Senior' or 'Senior-Manager'). Choose the closest single level.\n"
            "- You MUST identify at least one sector. Use your best judgment if unclear.\n"
            "- Multiple sectors may be assigned if the role spans multiple domains.\n"
            "- Match sectors to the AVAILABLE SECTORS list provided below.\n"
            "- CRITICAL: Physical product roles (e.g., Aircon, HVAC, manufacturing) belong to Industrial & Manufacturing, NOT Gaming or Technology. "
            "These roles involve physical supply chains, mechanical engineering, and industrial processes that are fundamentally different from software or gaming industries.\n"
            + job_title_hint
            + companies_context
            + sectors_list
            + "\nJOB DESCRIPTION:\n" + (text_input[:15000]) + "\n\nJSON:"
        )

        resp = model.generate_content(prompt, generation_config=_jd_gen_config)
        raw_out = (resp.text or "").strip()
        parsed_obj = _extract_json_object(raw_out) or {}
        parsed = parsed_obj.get("parsed", {})
        
        # Normalize output
        # Use Step 1 job title as fallback when Step 2 model returns empty
        job_title = (parsed.get("job_title") or parsed.get("role") or step1_job_title or "").strip()
        seniority = _normalize_seniority_single((parsed.get("seniority") or "").strip())
        sector = parsed.get("sector") or ""
        sectors = parsed.get("sectors") or ([sector] if sector else [])
        if not country:  # Use country from analysis if not provided in request
            country = (parsed.get("country") or parsed.get("location") or "").strip()
        skills = parsed.get("skills") or parsed_obj.get("skills") or []
        if isinstance(skills, str) and skills.strip():
            skills = [s.strip() for s in skills.split(",") if s.strip()]

        # Filter out skill strings that are clearly sentence fragments, not skill keywords
        skills = [s.strip() for s in skills if _is_valid_skill_token(s)]
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
        def derive_sector_from_skills_and_title(skills_list, job_title_text, jd_text, existing_sectors):
            """
            Derive a sector from the skillset, job title, and job description that is different from existing sectors.
            Uses the same hierarchical validation logic as webbridgepro.py with additional product/domain validation:
              1. Try to match exact or long-form labels from sectors.json (longest match wins)
              2. Try token overlap matching via _find_best_sector_match_for_text()
              3. Try keyword mapping via _map_keyword_to_sector_label()
              4. Validate that the product/domain mentioned in job title exists in the derived sector
              5. Return None if no match (do NOT return freeform labels)
            
            Args:
                skills_list (list): List of skill strings extracted from JD
                job_title_text (str): Job title from JD
                jd_text (str): Full job description text for additional context
                existing_sectors (list): List of already determined sector labels
            
            Returns:
                tuple: (sector_label or None, note_string)
                    - sector_label: A sectors.json validated label or None if no match
                    - note_string: Description of matched keywords or empty string
            
            Example: 
                skills_list = ["AWS", "Cloud", "Kubernetes"]
                job_title_text = "Cloud Engineer"
                jd_text = "Tencent is seeking a Cloud Solutions Developer..."
                existing_sectors = ["Media, Gaming & Entertainment > Gaming"]
                Returns: ("Technology > Cloud & Infrastructure", "Matched sectors.json label via token overlap")
            """
            if not skills_list and not job_title_text and not jd_text:
                return None, ""
            
            try:
                # Combine skills, job title, and JD text for comprehensive analysis
                skills_text = " ".join([str(s).lower() for s in skills_list if s])
                title_text = (job_title_text or "").lower()
                jd_lower = (jd_text or "").lower()
                combined_text = f"{skills_text} {title_text} {jd_lower}"
                
                # Helper function to validate product/domain in sector label
                def validate_product_in_sector(sector_label, job_title):
                    """
                    Validate that the product/domain mentioned in job title exists within the sector.
                    This is an exception rule when company name doesn't exist or cannot be mapped.
                    
                    Examples:
                    - "Product Manager, Mobile Phone" + "Consumer & Retail > Consumer Electronics" 
                      → "mobile phone" matches "consumer electronics" ✓
                    - "Cloud Engineer" + "Technology > Cloud & Infrastructure"
                      → "cloud" matches "cloud & infrastructure" ✓
                    """
                    if not sector_label or not job_title:
                        return True  # No validation needed if inputs missing
                    
                    # Extract domain part from sector label (e.g., "Cloud & Infrastructure" from "Technology > Cloud & Infrastructure")
                    parts = sector_label.split(" > ")
                    if len(parts) < 2:
                        return True  # No domain to validate against
                    
                    domain = parts[-1].lower()  # Get the last part (domain)
                    job_title_lower = job_title.lower()
                    
                    # Check if any product keyword in job title matches the domain (using word boundaries)
                    product_keyword_found = False
                    for product_keyword, valid_domains in PRODUCT_TO_DOMAIN_KEYWORDS.items():
                        # Use word boundary regex for exact word matching
                        pattern = r'\b' + re.escape(product_keyword) + r'\b'
                        if re.search(pattern, job_title_lower):
                            product_keyword_found = True
                            # Check if the sector domain matches any valid domain for this product
                            for valid_domain in valid_domains:
                                if valid_domain in domain:
                                    return True
                            # Product keyword found but doesn't match this domain - reject
                            return False
                    
                    # If no specific product keyword found, allow generic roles only when
                    # there is strong token overlap between combined_text and sector label (>= 0.4).
                    # This prevents generic titles like "Engineer" from validating unrelated sectors.
                    if not product_keyword_found:
                        combined_tokens = _token_set(combined_text)
                        label_tokens = _token_set(sector_label)
                        if combined_tokens and label_tokens:
                            overlap_ratio = len(combined_tokens & label_tokens) / len(label_tokens)
                        else:
                            overlap_ratio = 0.0
                        if overlap_ratio >= 0.4:
                            for role in GENERIC_ROLE_KEYWORDS:
                                pattern = r'\b' + re.escape(role) + r'\b'
                                if re.search(pattern, job_title_lower):
                                    return True
                    
                    return False
                
                # 1) Try token overlap matching on job title (most precise, short text signal)
                mapped_from_title = _find_best_sector_match_for_text(title_text) if title_text else None
                if mapped_from_title and mapped_from_title not in existing_sectors:
                    if validate_product_in_sector(mapped_from_title, job_title_text):
                        return mapped_from_title, "Matched sectors.json label via title token overlap with product validation"
                
                # 2) Try keyword mapping on title, skills, then combined text
                kw_map = _map_keyword_to_sector_label(title_text) if title_text else None
                if not kw_map and skills_text:
                    kw_map = _map_keyword_to_sector_label(skills_text)
                if not kw_map:
                    kw_map = _map_keyword_to_sector_label(combined_text)
                if kw_map and kw_map not in existing_sectors:
                    if validate_product_in_sector(kw_map, job_title_text):
                        return kw_map, "Mapped via keyword to sectors.json label with product validation"
                
                # 3) Try token overlap on combined text as a broader signal
                mapped_from_combined = _find_best_sector_match_for_text(combined_text)
                if mapped_from_combined and mapped_from_combined not in existing_sectors:
                    if validate_product_in_sector(mapped_from_combined, job_title_text):
                        return mapped_from_combined, "Matched sectors.json label via token overlap with product validation"
                
                # 4) Substring label match as strict fallback (only when label phrase truly appears in text)
                best_match = ""
                best_orig = ""
                for label in SECTORS_INDEX:
                    lbl_low = label.lower()
                    if lbl_low and lbl_low in combined_text:
                        # prefer the longest matched label (more specific)
                        if len(lbl_low) > len(best_match):
                            best_match = lbl_low
                            best_orig = label
                if best_orig and best_orig not in existing_sectors:
                    if validate_product_in_sector(best_orig, job_title_text):
                        return best_orig, "Matched sectors.json label with product validation"
                
                # 5) Do NOT return freeform labels; instead return None to indicate no strict sectors.json match
                return None, ""
            except Exception:
                return None, ""
        
        # Apply skillset-based sector derivation if we have skills, job title, or JD text, and at least one existing sector
        if (skills or job_title or text_input) and sectors:
            skillset_sector, skillset_note = derive_sector_from_skills_and_title(skills, job_title, text_input, sectors)
            if skillset_sector:
                sectors.append(skillset_sector)
                heuristic_notes.append(f"second sector: {skillset_note}")
        
        # -------------------------
        # STEP 4.6: Enforce maximum of 2 sectors
        # -------------------------
        if len(sectors) > 2:
            # Keep only the first 2 sectors (company-based + skillset-based priority)
            sectors = sectors[:2]
            heuristic_notes.append("limited to maximum 2 sectors")
        
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
                    country=country,
                    products=identified_products  # Pass extracted product references for competitor context
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
                # No job title in JD — try a dedicated fast extraction before using generic placeholders
                try:
                    _title_extract_prompt = (
                        "Extract the job title being hired for from this job description. "
                        "Return ONLY the job title as plain text (e.g. 'Cloud Engineer', 'Product Manager'). "
                        "If not determinable, return an empty string.\n\nJOB DESCRIPTION:\n"
                        + (text_input[:3000])
                    )
                    _title_extract_resp = model.generate_content(
                        _title_extract_prompt,
                        generation_config=genai.GenerationConfig(temperature=0.05, max_output_tokens=64)
                    )
                    _extracted_title = (_title_extract_resp.text or "").strip().strip('"').strip()
                    # Reject clearly generic/unhelpful responses
                    # Reject titles that are PURELY generic labels (only when the entire title is a single generic word)
                    _bad_patterns = re.compile(r'^(professional|specialist|expert|associate|general|worker|employee)$', re.I)
                    if _extracted_title and len(_extracted_title) < 80 and not _bad_patterns.match(_extracted_title.strip()):
                        job_titles = [_extracted_title, f"Senior {_extracted_title}"]
                    else:
                        job_titles = ["Professional", "Senior Professional"]
                except Exception:
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

        # Build a fallback summary from extracted fields when Gemini didn't return one
        if not summary:
            parts = []
            if job_title:
                parts.append(job_title)
            if seniority:
                parts.append(seniority)
            if sectors:
                sector_names = sectors if isinstance(sectors, list) else ([sectors] if sectors else [])
                parts.append(", ".join(str(s) for s in sector_names if s))
            if country:
                parts.append(country)
            if parts:
                summary = " · ".join(parts)

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
            "products": identified_products,  # Product references extracted from JD (for competitor suggestions)
            "raw": raw_out
        }
        return jsonify(out), 200
    except Exception as e:
        logger.exception("Gemini analyze_jd failed")
        return jsonify({"error": str(e)}), 500

@app.post("/chat/upload_jd")
@_rate("20 per minute")
@_check_user_rate("upload_jd")
def chat_upload_jd():
    """
    POST /chat/upload_jd  (multipart/form-data)
    Accepts a Job Description file upload (PDF / DOCX / plain text) from the chat
    interface. Extracts text, stores it in login.jd, and triggers skill extraction
    via Gemini if available.

    Form fields:
      - username (str): authenticated user's username
      - file      (file): the JD document

    Response: { "status": "ok", "message": "...", "length": <int> }
    """
    conn = None
    cur = None
    try:
        import io as _io
        username = (request.form.get("username") or "").strip()
        if not username:
            return jsonify({"error": "username required"}), 400
        if "file" not in request.files:
            return jsonify({"error": "No file part in request"}), 400
        file = request.files["file"]
        if not file or file.filename == "":
            return jsonify({"error": "No file selected"}), 400

        # Size guard — check content-length before reading the whole body
        if (request.content_length or 0) > _SINGLE_FILE_MAX:
            return jsonify({"error": "File too large (max 6 MB)"}), 413
        file_bytes = file.read()
        if len(file_bytes) > _SINGLE_FILE_MAX:
            return jsonify({"error": "File too large (max 6 MB)"}), 413

        filename = (file.filename or "").lower()
        extracted_text = ""

        if filename.endswith(".pdf"):
            if not _is_pdf_bytes(file_bytes):
                return jsonify({"error": "Uploaded file is not a valid PDF"}), 400
            try:
                from pypdf import PdfReader
                reader = PdfReader(_io.BytesIO(file_bytes))
                for page in reader.pages:
                    extracted_text += (page.extract_text() or "") + "\n"
            except ImportError:
                return jsonify({"error": "pypdf not installed; cannot process PDF"}), 500
            except Exception as pdf_err:
                return jsonify({"error": f"PDF parsing error: {pdf_err}"}), 500
        elif filename.endswith((".docx", ".doc")):
            try:
                import docx
                doc = docx.Document(_io.BytesIO(file_bytes))
                for para in doc.paragraphs:
                    extracted_text += para.text + "\n"
            except ImportError:
                return jsonify({"error": "python-docx not installed; cannot process DOCX"}), 500
            except Exception as docx_err:
                return jsonify({"error": f"DOCX parsing error: {docx_err}"}), 500
        else:
            try:
                extracted_text = file_bytes.decode("utf-8", errors="ignore")
            except Exception as txt_err:
                return jsonify({"error": f"Text decoding error: {txt_err}"}), 500

        extracted_text = extracted_text.strip()
        if not extracted_text:
            return jsonify({"error": "Could not extract text from the uploaded file"}), 400

        # Persist JD text to login table
        import psycopg2
        conn = psycopg2.connect(
            host=os.getenv("PGHOST", "localhost"),
            port=int(os.getenv("PGPORT", "5432")),
            user=os.getenv("PGUSER", "postgres"),
            password=os.getenv("PGPASSWORD", ""),
            dbname=os.getenv("PGDATABASE", "candidate_db"),
        )
        cur = conn.cursor()
        cur.execute("UPDATE login SET jd = %s WHERE username = %s", (extracted_text, username))
        if cur.rowcount == 0:
            conn.rollback()
            return jsonify({"error": "User not found"}), 404
        conn.commit()

        # Best-effort: auto-extract skills via Gemini and persist
        try:
            from chat_gemini_review import analyze_job_description
            analysis = analyze_job_description(extracted_text)
            skills = (analysis.get("parsed") or {}).get("skills") or []
            if skills:
                _persist_jskillset(username, skills)
        except Exception as skill_err:
            logger.warning(f"[chat/upload_jd] Skill extraction skipped: {skill_err}")

        logger.info(f"[chat/upload_jd] JD uploaded for user='{username}', length={len(extracted_text)}")
        return jsonify({"status": "ok", "message": "JD uploaded and stored",
                        "length": len(extracted_text)}), 200

    except Exception as e:
        logger.exception(f"[chat/upload_jd] Unexpected error for user='{request.form.get('username', '')}': {e}")
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return jsonify({"error": str(e)}), 500
    finally:
        if cur:
            try:
                cur.close()
            except Exception:
                pass
        if conn:
            try:
                conn.close()
            except Exception:
                pass

# ---------------------------------------------------------------------------
# Module-level skill token validator
# Rejects sentence fragments extracted by Gemini as skill strings.
# Used by both _persist_jskillset and gemini_analyze_jd.
# ---------------------------------------------------------------------------
_SKILL_MAX_WORDS = 5
_SKILL_INVALID_PREFIXES = re.compile(
    r'^(and|or|the|a|an|but|with|of|to|in|for|by|at|we|be|is|are|our|its)\b|\d+[\.\)]',
    re.I
)

def _is_valid_skill_token(s: str) -> bool:
    """Return True only for short keyword-style skill strings, False for sentence fragments."""
    if not isinstance(s, str):
        return False
    s = s.strip()
    if not s:
        return False
    if len(s.split()) > _SKILL_MAX_WORDS:
        return False
    if _SKILL_INVALID_PREFIXES.match(s):
        return False
    return True

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

    # Filter out sentence fragments — keep only keyword-style skill tokens
    deduped = [s for s in deduped if _is_valid_skill_token(s)]

    # Ensure a JSON-serializable list
    final_skills = [str(s) for s in deduped]

    try:
        import psycopg2
        from psycopg2 import sql
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD", "")
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
        pg_password=os.getenv("PGPASSWORD", "")
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
        from psycopg2 import sql as pgsql
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD", "")
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
                cur.execute(pgsql.SQL("SELECT {} FROM process WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl))=%s").format(pgsql.Identifier(col)), (normalized_url,))
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
        pg_password=os.getenv("PGPASSWORD", "")
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
@_rate("10 per minute")
@_check_user_rate("gemini")
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
@_rate("10 per minute")
@_check_user_rate("gemini")
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
            pg_password=os.getenv("PGPASSWORD", "")
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
@_rate("10 per minute")
@_check_user_rate("gemini")
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

def _should_overwrite_existing(existing_meta, incoming_level="L2", force=False):
    """
    Decide whether a new assessment should overwrite an existing one.
    existing_meta: dict with keys: level (str "L1"/"L2" or ""), updated_at, version; or None.
    incoming_level: "L1" or "L2"
    force: caller explicitly requests overwrite
    Returns: (bool, reason_str)
    """
    try:
        if force:
            return True, "force_reassess=True"
        if not existing_meta:
            return True, "no existing rating"
        existing_level = (existing_meta.get("level") or "").upper()
        if not existing_level:
            return True, "no existing level metadata"
        if incoming_level == "L2" and existing_level == "L1":
            return True, "upgrade L1 -> L2"
        if incoming_level == existing_level:
            return False, "same level existing"
        if incoming_level == "L1" and existing_level == "L2":
            return False, "incoming L1 would downgrade existing L2"
        return True, "default-allow"
    except Exception:
        return True, "error-eval-allow"


def _ensure_search_indexes(cur, conn):
    """Idempotently create full-text search and trigram indexes on sourcing/process tables.

    Sets up:
    - pg_trgm extension (for fuzzy / similarity matching)
    - search_vector TSVECTOR column on sourcing and process tables
    - GIN index on search_vector for fast full-text queries
    - pg_trgm GIN indexes on key text columns (name, jobtitle, company) for fuzzy matching
    - Trigger functions + BEFORE INSERT/UPDATE triggers to keep search_vector current
    - Backfill of search_vector for existing rows (runs only when column is first added)
    """
    ddls = []
    try:
        # 1. Enable pg_trgm extension (idempotent)
        cur.execute("SAVEPOINT _si_ext")
        cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")
        cur.execute("RELEASE SAVEPOINT _si_ext")
        conn.commit()
    except Exception as e:
        try:
            cur.execute("ROLLBACK TO SAVEPOINT _si_ext")
            conn.commit()
        except Exception:
            pass
        logger.warning(f"[SearchIdx] pg_trgm extension install failed (non-fatal): {e}")

    # 2. Add search_vector column to sourcing if absent
    sourcing_col_added = False
    try:
        cur.execute("SAVEPOINT _si_sv_s")
        cur.execute("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='sourcing'
              AND column_name='search_vector'
        """)
        if cur.fetchone() is None:
            cur.execute("ALTER TABLE sourcing ADD COLUMN search_vector TSVECTOR")
            sourcing_col_added = True
        cur.execute("RELEASE SAVEPOINT _si_sv_s")
        conn.commit()
    except Exception as e:
        try:
            cur.execute("ROLLBACK TO SAVEPOINT _si_sv_s")
            conn.commit()
        except Exception:
            pass
        logger.warning(f"[SearchIdx] sourcing.search_vector column add failed (non-fatal): {e}")

    # 3. Add search_vector column to process if absent
    process_col_added = False
    try:
        cur.execute("SAVEPOINT _si_sv_p")
        cur.execute("""
            SELECT 1 FROM information_schema.columns
            WHERE table_schema='public' AND table_name='process'
              AND column_name='search_vector'
        """)
        if cur.fetchone() is None:
            cur.execute("ALTER TABLE process ADD COLUMN search_vector TSVECTOR")
            process_col_added = True
        cur.execute("RELEASE SAVEPOINT _si_sv_p")
        conn.commit()
    except Exception as e:
        try:
            cur.execute("ROLLBACK TO SAVEPOINT _si_sv_p")
            conn.commit()
        except Exception:
            pass
        logger.warning(f"[SearchIdx] process.search_vector column add failed (non-fatal): {e}")

    # 4. Create GIN indexes on search_vector columns
    gin_ddls = [
        ("_si_gin_s", "CREATE INDEX IF NOT EXISTS idx_sourcing_search_vector ON sourcing USING GIN(search_vector)"),
        ("_si_gin_p", "CREATE INDEX IF NOT EXISTS idx_process_search_vector ON process USING GIN(search_vector)"),
        # Trigram GIN indexes for fuzzy matching on key text columns
        ("_si_trgm_s_name",     "CREATE INDEX IF NOT EXISTS idx_sourcing_name_trgm ON sourcing USING GIN(name gin_trgm_ops)"),
        ("_si_trgm_s_jobtitle", "CREATE INDEX IF NOT EXISTS idx_sourcing_jobtitle_trgm ON sourcing USING GIN(jobtitle gin_trgm_ops)"),
        ("_si_trgm_s_company",  "CREATE INDEX IF NOT EXISTS idx_sourcing_company_trgm ON sourcing USING GIN(company gin_trgm_ops)"),
        ("_si_trgm_p_jobtitle", "CREATE INDEX IF NOT EXISTS idx_process_jobtitle_trgm ON process USING GIN(jobtitle gin_trgm_ops)"),
        ("_si_trgm_p_company",  "CREATE INDEX IF NOT EXISTS idx_process_company_trgm ON process USING GIN(company gin_trgm_ops)"),
        ("_si_trgm_p_skillset", "CREATE INDEX IF NOT EXISTS idx_process_skillset_trgm ON process USING GIN(skillset gin_trgm_ops)"),
    ]
    for sp, ddl in gin_ddls:
        try:
            cur.execute(f"SAVEPOINT {sp}")
            cur.execute(ddl)
            cur.execute(f"RELEASE SAVEPOINT {sp}")
            conn.commit()
        except Exception as e:
            try:
                cur.execute(f"ROLLBACK TO SAVEPOINT {sp}")
                conn.commit()
            except Exception:
                pass
            logger.warning(f"[SearchIdx] Index creation failed (non-fatal): {sp}: {e}")

    # 5. Trigger function + trigger for sourcing table
    try:
        cur.execute("SAVEPOINT _si_trig_s")
        cur.execute("""
            CREATE OR REPLACE FUNCTION sourcing_search_vector_update()
            RETURNS TRIGGER LANGUAGE plpgsql AS $$
            BEGIN
                NEW.search_vector :=
                    setweight(to_tsvector('english', coalesce(NEW.jobtitle, '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(NEW.company,  '')), 'B') ||
                    setweight(to_tsvector('english', coalesce(NEW.name,     '')), 'C') ||
                    setweight(to_tsvector('english', coalesce(NEW.experience,'')), 'D');
                RETURN NEW;
            END;
            $$
        """)
        cur.execute("""
            DROP TRIGGER IF EXISTS trg_sourcing_search_vector ON sourcing
        """)
        cur.execute("""
            CREATE TRIGGER trg_sourcing_search_vector
            BEFORE INSERT OR UPDATE OF name, jobtitle, company, experience
            ON sourcing
            FOR EACH ROW EXECUTE FUNCTION sourcing_search_vector_update()
        """)
        cur.execute("RELEASE SAVEPOINT _si_trig_s")
        conn.commit()
    except Exception as e:
        try:
            cur.execute("ROLLBACK TO SAVEPOINT _si_trig_s")
            conn.commit()
        except Exception:
            pass
        logger.warning(f"[SearchIdx] sourcing trigger creation failed (non-fatal): {e}")

    # 6. Trigger function + trigger for process table
    try:
        cur.execute("SAVEPOINT _si_trig_p")
        cur.execute("""
            CREATE OR REPLACE FUNCTION process_search_vector_update()
            RETURNS TRIGGER LANGUAGE plpgsql AS $$
            BEGIN
                NEW.search_vector :=
                    setweight(to_tsvector('english', coalesce(NEW.jobtitle,  '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(NEW.company,   '')), 'B') ||
                    setweight(to_tsvector('english', coalesce(NEW.skillset,  '')), 'B') ||
                    setweight(to_tsvector('english', coalesce(NEW.name,      '')), 'C') ||
                    setweight(to_tsvector('english', coalesce(NEW.experience,'')), 'D');
                RETURN NEW;
            END;
            $$
        """)
        cur.execute("""
            DROP TRIGGER IF EXISTS trg_process_search_vector ON process
        """)
        cur.execute("""
            CREATE TRIGGER trg_process_search_vector
            BEFORE INSERT OR UPDATE OF name, jobtitle, company, skillset, experience
            ON process
            FOR EACH ROW EXECUTE FUNCTION process_search_vector_update()
        """)
        cur.execute("RELEASE SAVEPOINT _si_trig_p")
        conn.commit()
    except Exception as e:
        try:
            cur.execute("ROLLBACK TO SAVEPOINT _si_trig_p")
            conn.commit()
        except Exception:
            pass
        logger.warning(f"[SearchIdx] process trigger creation failed (non-fatal): {e}")

    # 7. Backfill search_vector for existing rows (only when column was just added)
    if sourcing_col_added:
        try:
            cur.execute("SAVEPOINT _si_backfill_s")
            cur.execute("""
                UPDATE sourcing SET search_vector =
                    setweight(to_tsvector('english', coalesce(jobtitle,  '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(company,   '')), 'B') ||
                    setweight(to_tsvector('english', coalesce(name,      '')), 'C') ||
                    setweight(to_tsvector('english', coalesce(experience,'')), 'D')
                WHERE search_vector IS NULL
            """)
            cur.execute("RELEASE SAVEPOINT _si_backfill_s")
            conn.commit()
        except Exception as e:
            try:
                cur.execute("ROLLBACK TO SAVEPOINT _si_backfill_s")
                conn.commit()
            except Exception:
                pass
            logger.warning(f"[SearchIdx] sourcing backfill failed (non-fatal): {e}")

    if process_col_added:
        try:
            cur.execute("SAVEPOINT _si_backfill_p")
            cur.execute("""
                UPDATE process SET search_vector =
                    setweight(to_tsvector('english', coalesce(jobtitle,  '')), 'A') ||
                    setweight(to_tsvector('english', coalesce(company,   '')), 'B') ||
                    setweight(to_tsvector('english', coalesce(skillset,  '')), 'B') ||
                    setweight(to_tsvector('english', coalesce(name,      '')), 'C') ||
                    setweight(to_tsvector('english', coalesce(experience,'')), 'D')
                WHERE search_vector IS NULL
            """)
            cur.execute("RELEASE SAVEPOINT _si_backfill_p")
            conn.commit()
        except Exception as e:
            try:
                cur.execute("ROLLBACK TO SAVEPOINT _si_backfill_p")
                conn.commit()
            except Exception:
                pass
            logger.warning(f"[SearchIdx] process backfill failed (non-fatal): {e}")


def _ensure_rating_metadata_columns(cur, conn):
    """Add rating_level, rating_updated_at, rating_version columns to process table if absent."""
    try:
        cur.execute("""
            SELECT column_name FROM information_schema.columns
            WHERE table_schema='public' AND table_name='process'
              AND column_name IN ('rating_level','rating_updated_at','rating_version')
        """)
        existing = {r[0] for r in cur.fetchall()}
        stmts = []
        if 'rating_level' not in existing:
            stmts.append("ADD COLUMN IF NOT EXISTS rating_level TEXT")
        if 'rating_updated_at' not in existing:
            stmts.append("ADD COLUMN IF NOT EXISTS rating_updated_at TIMESTAMPTZ")
        if 'rating_version' not in existing:
            stmts.append("ADD COLUMN IF NOT EXISTS rating_version INTEGER DEFAULT 1")
        if stmts:
            cur.execute("ALTER TABLE process " + ", ".join(stmts))
        conn.commit()
    except Exception as e_mig:
        try:
            conn.rollback()
        except Exception:
            pass
        logger.warning(f"[RatingMeta] Column migration failed (non-fatal): {e_mig}")


@app.post("/gemini/assess_profile")
@_rate("10 per minute")
@_check_user_rate("gemini")
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
    assessment_level = (data.get("assessment_level") or "L2").strip().upper()  # L2 by default
    tenure = data.get("tenure")  # Average tenure value
    force_reassess = bool(data.get("force_reassess") or False)

    # --- Idempotency pre-check: skip assessment if a rating already exists and policy forbids overwrite ---
    if linkedinurl:
        try:
            import psycopg2 as _psycopg2_idem
            _idem_conn = _psycopg2_idem.connect(
                host=os.getenv("PGHOST","localhost"), port=int(os.getenv("PGPORT","5432")),
                user=os.getenv("PGUSER","postgres"), password=os.getenv("PGPASSWORD", ""),
                dbname=os.getenv("PGDATABASE","candidate_db")
            )
            try:
                _idem_cur = _idem_conn.cursor()
                _ensure_rating_metadata_columns(_idem_cur, _idem_conn)
                _normalized_idem = None
                try:
                    _normalized_idem = _normalize_linkedin_to_path(linkedinurl)
                except Exception:
                    pass
                _idem_cur.execute("""
                    SELECT rating, rating_level, rating_updated_at, rating_version
                    FROM process
                    WHERE linkedinurl = %s OR (%s IS NOT NULL AND normalized_linkedin = %s)
                    LIMIT 1
                """, (linkedinurl, _normalized_idem, _normalized_idem))
                _row_idem = _idem_cur.fetchone()
                _existing_meta = None
                if _row_idem and _row_idem[0]:
                    _existing_meta = {
                        "rating": _row_idem[0],
                        "level": (_row_idem[1] or "").upper(),
                        "updated_at": _row_idem[2],
                        "version": _row_idem[3],
                    }
                _idem_cur.close()
            finally:
                _idem_conn.close()
            _allow, _reason = _should_overwrite_existing(_existing_meta, assessment_level, force_reassess)
            if not _allow:
                logger.info(f"[Assess] Skipping assessment for {linkedinurl}: {_reason}")
                _existing_obj = _existing_meta.get("rating") if _existing_meta else None
                if isinstance(_existing_obj, str):
                    try:
                        _existing_obj = json.loads(_existing_obj)
                    except Exception:
                        _existing_obj = {"raw": _existing_obj}
                if isinstance(_existing_obj, dict):
                    _existing_obj["_skipped"] = True
                    _existing_obj["_note"] = f"skipped - existing rating present ({_reason})"
                    return jsonify(_existing_obj), 200
                return jsonify({"_skipped": True, "error": "assessment skipped - existing rating", "reason": _reason}), 200
        except Exception as _e_idem:
            logger.warning(f"[Assess] Idempotency pre-check failed (continuing): {_e_idem}")

    # Resolve role_tag: sourcing table is authoritative; fallback to process, then login.
    # After resolution, write back to sourcing table so it is available for future assessments.
    if not role_tag and (linkedinurl or username):
        try:
            import psycopg2
            _pg_conn = psycopg2.connect(
                host=os.getenv("PGHOST","localhost"), port=int(os.getenv("PGPORT","5432")),
                user=os.getenv("PGUSER","postgres"), password=os.getenv("PGPASSWORD", ""),
                dbname=os.getenv("PGDATABASE","candidate_db")
            )
            try:
                _pg_cur = _pg_conn.cursor()
                # 1. Try sourcing by linkedinurl first, then by username (authoritative source)
                if linkedinurl:
                    _pg_cur.execute("SELECT role_tag FROM sourcing WHERE linkedinurl=%s AND role_tag IS NOT NULL AND role_tag != '' LIMIT 1", (linkedinurl,))
                    _r = _pg_cur.fetchone()
                    if _r and _r[0]: role_tag = _r[0]
                if not role_tag and username:
                    _pg_cur.execute("SELECT role_tag FROM sourcing WHERE username=%s AND role_tag IS NOT NULL AND role_tag != '' LIMIT 1", (username,))
                    _r = _pg_cur.fetchone()
                    if _r and _r[0]: role_tag = _r[0]
                # 2. Fallback to process table
                if not role_tag and linkedinurl:
                    _pg_cur.execute("SELECT role_tag FROM process WHERE linkedinurl=%s AND role_tag IS NOT NULL AND role_tag != '' LIMIT 1", (linkedinurl,))
                    _r = _pg_cur.fetchone()
                    if _r and _r[0]: role_tag = _r[0]
                if not role_tag and username:
                    _pg_cur.execute("SELECT role_tag FROM process WHERE username=%s AND role_tag IS NOT NULL AND role_tag != '' LIMIT 1", (username,))
                    _r = _pg_cur.fetchone()
                    if _r and _r[0]: role_tag = _r[0]
                # 3. Fallback to login table
                if not role_tag and username:
                    _pg_cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='login' AND column_name='role_tag'")
                    if _pg_cur.fetchone():
                        _pg_cur.execute("SELECT role_tag FROM login WHERE username=%s AND role_tag IS NOT NULL AND role_tag != '' LIMIT 1", (username,))
                        _r = _pg_cur.fetchone()
                        if _r and _r[0]: role_tag = _r[0]
                # 4. Persist resolved role_tag into sourcing table so it is available for future assessments.
                # This mirrors the bulk path and eliminates the discrepancy where individual assessments
                # could not find role_tag in sourcing even though it existed in login.
                if role_tag and username:
                    try:
                        _pg_cur.execute("SELECT column_name FROM information_schema.columns WHERE table_schema='public' AND table_name='sourcing' AND column_name='role_tag'")
                        if not _pg_cur.fetchone():
                            _pg_cur.execute("ALTER TABLE sourcing ADD COLUMN role_tag TEXT DEFAULT ''")
                        _pg_cur.execute(
                            "UPDATE sourcing SET role_tag=%s WHERE username=%s AND (role_tag IS NULL OR role_tag='')",
                            (role_tag, username)
                        )
                        _pg_conn.commit()
                        logger.info(f"[Assess] Synced role_tag='{role_tag}' from login→sourcing for user='{username}'")
                    except Exception as _e_sync_rt:
                        logger.warning(f"[Assess] Failed to sync role_tag to sourcing: {_e_sync_rt}")
                _pg_cur.close()
            finally:
                _pg_conn.close()
        except Exception as _e_rt:
            logger.warning(f"[Assess] Failed to resolve role_tag from sourcing/process: {_e_rt}")

    # 1. Sync jskillset from login → process (always, before fetch — mirrors bulk path).
    # This ensures target_skills is always populated with up-to-date recruiter skills.
    if username and linkedinurl:
        try:
            normalized_for_sync = _normalize_linkedin_to_path(linkedinurl)
            _sync_login_jskillset_to_process(username, linkedinurl, normalized_for_sync or "")
        except Exception as e_jsk_sync:
            logger.warning(f"[Gemini Assess] jskillset sync failed: {e_jsk_sync}")

    # 2. Fetch Target Skillset (jskillset) from process table (after sync)
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
            pg_password=os.getenv("PGPASSWORD", "")
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
            from psycopg2 import sql as pgsql
            pg_host=os.getenv("PGHOST","localhost")
            pg_port=int(os.getenv("PGPORT","5432"))
            pg_user=os.getenv("PGUSER","postgres")
            pg_password=os.getenv("PGPASSWORD", "")
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
                    cur.execute(pgsql.SQL("SELECT {} FROM process WHERE normalized_linkedin = %s").format(pgsql.Identifier(col_to_use)), (normalized,))
                    row = cur.fetchone()
                if not row:
                    cur.execute(pgsql.SQL("SELECT {} FROM process WHERE linkedinurl = %s").format(pgsql.Identifier(col_to_use)), (linkedinurl,))
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

    # If target_skills is still empty, build a conservative fallback from
    # role_tag, job_title, and any skills provided in the request body.
    # This ensures the vskillset inference is never skipped even when jskillset columns
    # are missing or linkedin URL normalization didn't match the DB row.
    if not target_skills:
        fallbacks = []
        if role_tag:
            fallbacks += [s.strip() for s in re.split(r'[,;/|]+', role_tag) if s.strip()]
        if job_title:
            fallbacks += [s.strip() for s in re.split(r'[,;/|]+', job_title) if s.strip()]
        parsed_sk = data.get('skills') or data.get('skillset')
        if parsed_sk:
            if isinstance(parsed_sk, list):
                fallbacks += [s.strip() for s in parsed_sk if isinstance(s, str) and s.strip()]
            elif isinstance(parsed_sk, str):
                fallbacks += [s.strip() for s in re.split(r'[,;/|]+', parsed_sk) if s.strip()]
        target_skills = dedupe([t for t in fallbacks if t])[:40]
        if target_skills:
            logger.info(f"[Assess] target_skills fallback built ({len(target_skills)}) from role_tag/job_title/request")

    # Read missing assessment fields from process table to mirror _assess_and_persist (bulk path).
    # The individual path frontend sends data from the UI table row / namecard cache, which may
    # be incomplete.  Filling in from the DB ensures all criteria (especially product, seniority,
    # sector, tenure) are evaluated — not just the ones the UI happened to have in cache.
    product = []
    if linkedinurl:
        try:
            import psycopg2 as _psycopg2_fill
            _pg_fill = _psycopg2_fill.connect(
                host=os.getenv("PGHOST","localhost"), port=int(os.getenv("PGPORT","5432")),
                user=os.getenv("PGUSER","postgres"), password=os.getenv("PGPASSWORD", ""),
                dbname=os.getenv("PGDATABASE","candidate_db")
            )
            try:
                _cur_fill = _pg_fill.cursor()
                _cur_fill.execute(
                    "SELECT seniority, sector, experience, tenure, product FROM process WHERE linkedinurl=%s LIMIT 1",
                    (linkedinurl,)
                )
                _row_fill = _cur_fill.fetchone()
                if _row_fill:
                    _db_seniority, _db_sector, _db_experience, _db_tenure, _db_product_raw = _row_fill
                    if not seniority and _db_seniority:
                        seniority = _db_seniority
                    if not sector and _db_sector:
                        sector = _db_sector
                    if not experience_text and _db_experience:
                        experience_text = (_db_experience or "").strip()
                    if tenure is None and _db_tenure is not None:
                        try:
                            tenure = float(_db_tenure)
                        except (ValueError, TypeError):
                            pass
                    if _db_product_raw:
                        try:
                            _p = json.loads(_db_product_raw) if isinstance(_db_product_raw, str) else None
                            if isinstance(_p, list):
                                product = _p
                            else:
                                product = [s.strip() for s in str(_db_product_raw).split(',') if s.strip()]
                        except Exception:
                            product = [s.strip() for s in str(_db_product_raw).split(',') if s.strip()]
                    logger.info(f"[Assess] DB fallback: seniority='{seniority}' sector='{sector}' product={len(product)} tenure={tenure}")
                _cur_fill.close()
            finally:
                _pg_fill.close()
        except Exception as _e_fill:
            logger.warning(f"[Assess] Failed to read fallback fields from process: {_e_fill}")

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
            from psycopg2 import sql as pgsql
            pg_host = os.getenv("PGHOST", "localhost")
            pg_port = int(os.getenv("PGPORT", "5432"))
            pg_user = os.getenv("PGUSER", "postgres")
            pg_password = os.getenv("PGPASSWORD", "")
            pg_db = os.getenv("PGDATABASE", "candidate_db")
            
            conn = psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
            cur = conn.cursor()
            
            # Normalize linkedin URL
            normalized = linkedinurl.lower().strip().rstrip('/')
            if not normalized.startswith('http'):
                normalized = 'https://' + normalized
            
            # Fetch experience and existing skillset from process table
            # Use a local variable to avoid overwriting the outer experience_text (from request)
            _db_experience_text = ""
            existing_skillset = []
            
            cur.execute("""
                SELECT experience, skillset 
                FROM process 
                WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl)) = %s
                LIMIT 1
            """, (normalized,))
            row = cur.fetchone()
            
            if row:
                _db_experience_text = (row[0] or "").strip()
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
            
            # Use experience as profile context; prefer DB value, fall back to request value
            profile_context = _db_experience_text or experience_text
            
            if not profile_context:
                logger.info(f"[Gemini Assess -> vskillset] Skipped: No experience data for linkedin='{linkedinurl}'")
            elif not genai or not GEMINI_API_KEY:
                logger.warning(f"[Gemini Assess -> vskillset] Skipped: Gemini not configured")
            else:
                # Idempotency guard: if vskillset already exists in DB, reuse it without re-running Gemini.
                _existing_vskillset = None
                try:
                    cur.execute("""
                        SELECT vskillset FROM process
                        WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl)) = %s
                        LIMIT 1
                    """, (normalized,))
                    _vs_row = cur.fetchone()
                    if _vs_row and _vs_row[0]:
                        _vs_val = _vs_row[0]
                        if isinstance(_vs_val, str):
                            _vs_val = json.loads(_vs_val)
                        if isinstance(_vs_val, list) and len(_vs_val) > 0:
                            _existing_vskillset = _vs_val
                except Exception as _e_vs_guard:
                    logger.warning(f"[Gemini Assess -> vskillset] Idempotency check failed ({_e_vs_guard}); will regenerate")

                if _existing_vskillset is not None:
                    # Reuse persisted vskillset — do not call Gemini again
                    high_skills = [i["skill"] for i in _existing_vskillset if isinstance(i, dict) and i.get("category") == "High"]
                    candidate_skills = list({s: None for s in (existing_skillset + high_skills)}.keys())  # deduplicate, preserve order
                    vskillset_results = _existing_vskillset
                    logger.info(f"[Gemini Assess -> vskillset] Reusing existing vskillset ({len(_existing_vskillset)} items) for {linkedinurl[:50]}")
                else:
                    # STEP 1: Extractive pass - find skills explicitly in experience text
                    explicitly_confirmed = _extract_confirmed_skills(profile_context, target_skills)
                    confirmed_set = set(s.lower() for s in explicitly_confirmed)
                    confirmed_results = [
                        {
                            "skill": skill,
                            "probability": 100,
                            "category": "High",
                            "reason": "Explicitly mentioned in experience text",
                            "source": "confirmed"
                        }
                        for skill in explicitly_confirmed
                    ]
                    logger.info(f"[Gemini Assess -> vskillset] Extractive pass: {len(confirmed_results)}/{len(target_skills)} skills confirmed from text")

                    # STEP 2: Only send unconfirmed skills to Gemini for inference
                    unconfirmed_skills = [s for s in target_skills if s.lower() not in confirmed_set]

                    inferred_results = []
                    if unconfirmed_skills:
                        # Call Gemini only for unconfirmed/missing skills
                        model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)

                        prompt = f"""SYSTEM:
You are an expert technical recruiter evaluating candidate skillsets based on their work experience.

TASK:
For each skill in the list below, evaluate the candidate's likely proficiency based on their experience.
These skills were NOT found explicitly in the experience text, so use contextual inference from
job titles, companies, products, sector, and experience patterns.
Assign a probability score (0-100) and categorize as Low (<40), Medium (40-74), or High (75-100).

CANDIDATE PROFILE:
{profile_context[:3000]}

SKILLS TO INFER (not found explicitly in experience text):
{json.dumps(unconfirmed_skills, ensure_ascii=False)}

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

                        parsed = _extract_json_object(raw_text)

                        if parsed and "evaluations" in parsed:
                            inferred_results = parsed["evaluations"]

                        # Ensure all required fields are present and annotate source
                        for item in inferred_results:
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
                            item["source"] = "inferred"

                    # STEP 3: Merge confirmed + inferred results
                    results = confirmed_results + inferred_results
                    logger.info(f"[Gemini Assess -> vskillset] Merged: {len(confirmed_results)} confirmed + {len(inferred_results)} inferred = {len(results)} total")

                    # Persist vskillset to database
                    vskillset_json = json.dumps(results, ensure_ascii=False)
                    
                    # Get High-confidence skills for skillset column (confirmed always High; inferred High ≥75%)
                    high_skills = [item["skill"] for item in results if item["category"] == "High"]
                    
                    # MERGE with existing skillset (not replace)
                    # Preserve order: keep existing skills first, then add new ones (avoiding duplicates)
                    existing_set = set(existing_skillset)
                    merged_skillset = existing_skillset + [skill for skill in high_skills if skill not in existing_set]
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
                        update_sql = pgsql.SQL("UPDATE process SET {} WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl)) = %s").format(pgsql.SQL(", ".join(updates)))
                        
                        update_values = []
                        if 'vskillset' in available_cols:
                            update_values.append(vskillset_json)
                        if 'skillset' in available_cols:
                            update_values.append(skillset_str)
                        update_values.append(normalized)
                        
                        cur.execute(update_sql, tuple(update_values))
                        conn.commit()
                        
                        logger.info(f"[Gemini Assess -> vskillset] Populated vskillset and merged {len(high_skills)} High skills into skillset for linkedin='{linkedinurl}'")
                        logger.info(f"[Gemini Assess -> vskillset] Merged skillset has {len(merged_skillset)} total skills: {merged_skillset[:10]}")
                        
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
        "vskillset_results": vskillset_results,  # vskillset inference results for scoring
        "product": product,  # Product list from DB (mirrors _assess_and_persist)
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
        pg_password=os.getenv("PGPASSWORD", "")
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

            # Ensure metadata columns exist before writing
            _ensure_rating_metadata_columns(cur, conn)

            updated = 0
            if normalized:
                try:
                    cur.execute(
                        "UPDATE process SET rating = %s, rating_level = %s, rating_updated_at = NOW(), "
                        "rating_version = COALESCE(rating_version, 0) + 1 WHERE normalized_linkedin = %s",
                        (rating_payload, assessment_level, normalized)
                    )
                    updated = cur.rowcount
                    conn.commit()
                except Exception:
                    conn.rollback()
                    updated = 0
            if updated == 0:
                try:
                    cur.execute(
                        "UPDATE process SET rating = %s, rating_level = %s, rating_updated_at = NOW(), "
                        "rating_version = COALESCE(rating_version, 0) + 1 WHERE linkedinurl = %s",
                        (rating_payload, assessment_level, linkedinurl)
                    )
                    updated = cur.rowcount
                    conn.commit()
                except Exception:
                    conn.rollback()
                    updated = 0
            logger.info(f"[Gemini Assess -> DB rating] Updated rating for linkedin='{linkedinurl}' normalized='{normalized}' updated_rows={updated} level={assessment_level}")
        
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

                # Robust role_tag → process sync (sourcing is authoritative; mirrors bulk path).
                # Tries normalized_linkedin first, then LOWER/TRIM linkedinurl fallback to handle
                # normalization mismatches.  Unconditional update overwrites stale values.
                try:
                    # Mirror bulk path: re-read role_tag from sourcing (authoritative source)
                    # before syncing to process, ensuring process receives the stored sourcing
                    # value rather than a potentially stale request-supplied value.
                    try:
                        _sourcing_rt = None
                        if linkedinurl:
                            cur.execute(
                                "SELECT role_tag FROM sourcing WHERE linkedinurl=%s AND role_tag IS NOT NULL AND role_tag != '' LIMIT 1",
                                (linkedinurl,)
                            )
                            _sr = cur.fetchone()
                            if _sr and _sr[0]:
                                _sourcing_rt = _sr[0]
                        if not _sourcing_rt and username:
                            cur.execute(
                                "SELECT role_tag FROM sourcing WHERE username=%s AND role_tag IS NOT NULL AND role_tag != '' LIMIT 1",
                                (username,)
                            )
                            _sr = cur.fetchone()
                            if _sr and _sr[0]:
                                _sourcing_rt = _sr[0]
                        if _sourcing_rt:
                            role_tag = _sourcing_rt
                    except Exception as _e_src_rt:
                        logger.warning(f"[Assess] Failed to re-read role_tag from sourcing: {_e_src_rt}")
                    if not normalized:
                        try:
                            normalized = _normalize_linkedin_to_path(linkedinurl)
                        except Exception:
                            normalized = None
                    cur.execute("""
                        SELECT column_name FROM information_schema.columns
                        WHERE table_schema='public' AND table_name='process' AND column_name='role_tag'
                    """)
                    if cur.fetchone():
                        rt_updated = 0
                        if normalized:
                            cur.execute(
                                "UPDATE process SET role_tag = %s WHERE normalized_linkedin = %s",
                                (role_tag, normalized)
                            )
                            rt_updated = cur.rowcount
                        if rt_updated == 0:
                            cur.execute(
                                "UPDATE process SET role_tag = %s WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl)) = %s OR linkedinurl = %s",
                                (role_tag, linkedinurl.lower().rstrip('/'), linkedinurl)
                            )
                            rt_updated = cur.rowcount
                        conn.commit()
                        if rt_updated:
                            logger.info(f"[Assess] Synced role_tag='{role_tag}' into process for linkedin='{linkedinurl[:80]}' updated_rows={rt_updated}")
                        else:
                            logger.info(f"[Assess] role_tag sync attempted but no matching process row found for linkedin='{linkedinurl[:80]}' (normalized='{normalized}')")
                except Exception as e_up:
                    conn.rollback()
                    logger.warning(f"[Assess] Failed to update process.role_tag for {linkedinurl}: {e_up}")

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
                        sql_update = sql.SQL("UPDATE process SET {} WHERE normalized_linkedin = %s OR linkedinurl = %s").format(sql.SQL(", ".join(update_parts)))
                        params.extend([norm, linkedinurl])
                    else:
                        # fallback: update by linkedinurl only
                        sql_update = sql.SQL("UPDATE process SET {} WHERE linkedinurl = %s").format(sql.SQL(", ".join(update_parts)))
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
@_rate("10 per minute")
@_check_user_rate("login")
@_csrf_required
def login_account():
    data = request.get_json(force=True, silent=True) or {}
    username = (data.get("username") or "").strip()
    password = (data.get("password") or "").strip()
    _ip = request.headers.get("X-Forwarded-For", request.remote_addr or "")
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
        pg_password=os.getenv("PGPASSWORD", "")
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()
        cur.execute("SELECT password, userid, cemail, fullname, role_tag, COALESCE(token,0) FROM login WHERE username=%s", (username,))
        row=cur.fetchone()
        cur.close(); conn.close()
        if not row:
            log_security("login_failed", username=username, ip_address=_ip,
                         detail="User not found", severity="warning")
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
                log_security("login_failed", username=username, ip_address=_ip,
                             detail="Password mismatch", severity="warning")
                return jsonify({"error":"Invalid credentials"}), 401
        else:
            def _local_hash_password(p: str) -> str:
                import hashlib
                salt = os.getenv("PASSWORD_SALT", "")
                return hashlib.sha256((salt + p).encode("utf-8")).hexdigest()
            hashed = hash_password_fn(password) if hash_password_fn else _local_hash_password(password)
            if stored_pw != hashed and stored_pw != password:
                log_security("login_failed", username=username, ip_address=_ip,
                             detail="Password mismatch", severity="warning")
                return jsonify({"error":"Invalid credentials"}), 401

        log_identity(userid=str(userid or ""), username=username,
                     ip_address=_ip, mfa_status="N/A")
        resp = jsonify({"ok": True, "userid": userid or "", "username": username, "cemail": cemail or "", "fullname": fullname or "", "role_tag": role_tag or "", "token": int(token_val or 0)})
        # httponly=False: AutoSourcing.html (and other pages) read the username
        # cookie via document.cookie to identify the logged-in user.  This
        # matches the behaviour of chatbot_api.py which also sets httponly=False.
        _cookie_opts = dict(max_age=2592000, path="/", httponly=False, samesite="lax",
                            secure=os.getenv("FORCE_HTTPS", "0") == "1")
        resp.set_cookie("username", username, **_cookie_opts)
        resp.set_cookie("userid", str(userid or ""), **_cookie_opts)
        return resp, 200
    except Exception as e:
        log_error(source="login", message=str(e), severity="error",
                  username=username, endpoint="/login")
        return jsonify({"error": str(e)}), 500

@app.post("/logout")
@_csrf_required
def logout_account():
    resp = jsonify({"ok": True})
    resp.delete_cookie("username", path="/")
    resp.delete_cookie("userid", path="/")
    return resp

@app.post("/register")
@_rate("10 per minute")
@_check_user_rate("register")
@_csrf_required
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
        from psycopg2 import sql as pgsql
        pg_host=os.getenv("PGHOST","localhost")
        pg_port=int(os.getenv("PGPORT","5432"))
        pg_user=os.getenv("PGUSER","postgres")
        pg_password=os.getenv("PGPASSWORD", "")
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

        col_sql = pgsql.SQL(", ").join(pgsql.Identifier(c) for c in insert_cols)
        placeholders = pgsql.SQL(", ".join(["%s"] * len(insert_cols)))
        cur.execute(pgsql.SQL("INSERT INTO login ({}) VALUES ({})").format(col_sql, placeholders), insert_vals)
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
        pg_password=os.getenv("PGPASSWORD", "")
        pg_db=os.getenv("PGDATABASE","candidate_db")
        conn=psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur=conn.cursor()
        cur.execute("SELECT userid, fullname, role_tag, COALESCE(token,0), COALESCE(target_limit,10), COALESCE(useraccess,'') FROM login WHERE username=%s", (username,))
        row = cur.fetchone()
        if not row:
            cur.close(); conn.close()
            return jsonify({"error":"not found"}), 404
        userid, fullname, login_role_tag, token_val, target_limit_val, useraccess_val = row
        # Prefer role_tag from sourcing table (authoritative source) over login table
        resolved_role_tag = login_role_tag or ""
        try:
            cur.execute("SELECT role_tag FROM sourcing WHERE username=%s AND role_tag IS NOT NULL AND role_tag != '' LIMIT 1", (username,))
            src_row = cur.fetchone()
            if src_row and src_row[0]:
                resolved_role_tag = src_row[0]
        except Exception:
            pass
        cur.close(); conn.close()
        return jsonify({"userid": userid or "", "fullname": fullname or "", "role_tag": resolved_role_tag, "token": int(token_val or 0), "target_limit": int(target_limit_val or 10), "useraccess": (useraccess_val or "").strip()}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Module-level flag: ALTER TABLE to add last_result_count runs at most once per
# server process so the idempotency guard column is created without per-request DDL.
_token_guard_column_ensured = False

# Module-level flag: ALTER TABLE to add role_tag_session column runs at most once
# per server process for both login and sourcing tables.
_role_tag_session_column_ensured = False

@app.post("/user/token_update")
@_csrf_required
def user_token_update():
    """
    POST /user/token_update
    Sets the token column in the login table to the supplied value.
    Used to persist the current "tokens left" figure after each
    token-consuming operation so the login table always reflects the
    most up-to-date balance.

    Body JSON: { "userid": "<id>", "token": <number>, "result_count": <int|optional> }

    When result_count is supplied the endpoint acts as an idempotent guard:
    the update only fires if result_count differs from the stored
    last_result_count, preventing the feedback loop where the same search
    result count is deducted repeatedly on every page refresh or new-tab load.
    """
    global _token_guard_column_ensured, _role_tag_session_column_ensured
    data = request.get_json(force=True, silent=True) or {}
    userid = (data.get("userid") or "").strip()
    token_val = data.get("token")
    if not userid or token_val is None:
        return jsonify({"error": "userid and token are required"}), 400
    try:
        token_int = int(token_val)
    except (TypeError, ValueError):
        return jsonify({"error": "token must be a number"}), 400
    result_count_int = None
    rc = data.get("result_count")
    if rc is not None:
        try:
            result_count_int = int(rc)
        except (TypeError, ValueError):
            pass
    role_tag = (data.get("role_tag") or "").strip()
    _token_before_tx: int | None = None  # captured before the UPDATE for financial logging
    _username_tx: str = ""               # captured for financial logging
    try:
        import psycopg2
        pg_host = os.getenv("PGHOST", "localhost")
        pg_port = int(os.getenv("PGPORT", "5432"))
        pg_user = os.getenv("PGUSER", "postgres")
        pg_password = os.getenv("PGPASSWORD", "")
        pg_db = os.getenv("PGDATABASE", "candidate_db")
        conn = psycopg2.connect(
            host=pg_host, port=pg_port, user=pg_user,
            password=pg_password, dbname=pg_db
        )
        cur = conn.cursor()
        try:
            if result_count_int is not None:
                # Ensure idempotency columns exist — run at most once per process
                if not _token_guard_column_ensured:
                    cur.execute(
                        "ALTER TABLE login ADD COLUMN IF NOT EXISTS last_result_count INTEGER"
                    )
                    cur.execute(
                        "ALTER TABLE login ADD COLUMN IF NOT EXISTS last_deducted_role_tag TEXT"
                    )
                    _token_guard_column_ensured = True
                # Ensure session tracking columns exist — run at most once per process
                if not _role_tag_session_column_ensured:
                    cur.execute("ALTER TABLE login ADD COLUMN IF NOT EXISTS session TIMESTAMPTZ")
                    cur.execute("ALTER TABLE sourcing ADD COLUMN IF NOT EXISTS session TIMESTAMPTZ")
                    cur.execute("ALTER TABLE sourcing ALTER COLUMN session DROP DEFAULT")
                    _role_tag_session_column_ensured = True
                # Read current token, stored result count, stored role_tag, login role_tag, session, and username.
                # role_tag and role_tag_session are read so that we can auto-generate the session
                # timestamp for rows where role_tag is already set but role_tag_session is NULL
                # (e.g. rows that pre-existed before the role_tag_session column was added).
                cur.execute(
                    "SELECT token, last_result_count, last_deducted_role_tag,"
                    " role_tag, session, username FROM login WHERE userid = %s",
                    (userid,)
                )
                existing = cur.fetchone()
                if not existing:
                    conn.commit()
                    return jsonify({"error": "user not found"}), 404
                current_token, stored_count, _stored_role_tag_raw, login_role_tag, login_session_ts, login_username = existing
                stored_role_tag = (_stored_role_tag_raw or "").strip()
                _token_before_tx = int(current_token) if current_token is not None else None
                _username_tx = login_username or ""
                # Auto-backfill: if role_tag is already set in login but role_tag_session is NULL,
                # generate a session timestamp now and transfer it to sourcing where role_tag matches.
                # This ensures every role_tag entry is tied to a valid session reference even for
                # rows that existed before the role_tag_session column was introduced.
                if (login_role_tag or "").strip() and login_session_ts is None:
                    cur.execute(
                        "UPDATE login SET session = NOW() WHERE userid = %s RETURNING session",
                        (userid,)
                    )
                    ts_row = cur.fetchone()
                    login_session_ts = ts_row[0] if ts_row else None
                    if login_session_ts is not None and login_username:
                        cur.execute(
                            "UPDATE sourcing SET session = %s WHERE username = %s AND role_tag = %s",
                            (login_session_ts, login_username, login_role_tag)
                        )
                        logger.info(
                            f"[TokenUpdate] Auto-backfilled role_tag_session='{login_session_ts}' "
                            f"for user='{login_username}' (role_tag='{login_role_tag}')"
                        )
                # Backend idempotency guard: skip if same result_count was already persisted.
                # When role_tag is also provided, require that the stored role_tag also matches;
                # a NULL/empty stored role_tag with a provided role_tag is treated as a new session.
                if stored_count is not None and stored_count == result_count_int:
                    if (not role_tag) or (stored_role_tag and stored_role_tag == role_tag):
                        conn.commit()
                        return jsonify({"ok": True, "token": int(current_token) if current_token is not None else 0, "skipped": True}), 200
                # New deduction — persist updated balance, result count, and role_tag
                if role_tag:
                    cur.execute(
                        "UPDATE login SET token = %s, last_result_count = %s, last_deducted_role_tag = %s WHERE userid = %s RETURNING token",
                        (token_int, result_count_int, role_tag, userid)
                    )
                else:
                    cur.execute(
                        "UPDATE login SET token = %s, last_result_count = %s WHERE userid = %s RETURNING token",
                        (token_int, result_count_int, userid)
                    )
            else:
                # Legacy path: no result_count supplied.
                # Uses a session+role_tag guard: if login.session == sourcing.session
                # AND role_tags match, the deduction for this session was already
                # processed — skip to prevent repeated deductions on page refresh.
                if not _role_tag_session_column_ensured:
                    cur.execute("ALTER TABLE login ADD COLUMN IF NOT EXISTS session TIMESTAMPTZ")
                    cur.execute("ALTER TABLE sourcing ADD COLUMN IF NOT EXISTS session TIMESTAMPTZ")
                    cur.execute("ALTER TABLE sourcing ALTER COLUMN session DROP DEFAULT")
                    _role_tag_session_column_ensured = True
                cur.execute(
                    "SELECT role_tag, session, username, token FROM login WHERE userid = %s",
                    (userid,)
                )
                _legacy_row = cur.fetchone()
                if _legacy_row:
                    _legacy_role_tag, _legacy_session_ts, _legacy_username, _legacy_token = _legacy_row
                    _token_before_tx = int(_legacy_token) if _legacy_token is not None else None
                    _username_tx = _legacy_username or ""
                    # Auto-backfill: if role_tag is set but session is NULL, generate now
                    if (_legacy_role_tag or "").strip() and _legacy_session_ts is None:
                        cur.execute(
                            "UPDATE login SET session = NOW() WHERE userid = %s RETURNING session",
                            (userid,)
                        )
                        _legacy_ts_row = cur.fetchone()
                        _legacy_new_ts = _legacy_ts_row[0] if _legacy_ts_row else None
                        if _legacy_new_ts is not None:
                            _legacy_session_ts = _legacy_new_ts
                            if _legacy_username:
                                cur.execute(
                                    "UPDATE sourcing SET session = %s WHERE username = %s AND role_tag = %s",
                                    (_legacy_new_ts, _legacy_username, _legacy_role_tag)
                                )
                                logger.info(
                                    f"[TokenUpdate] Auto-backfilled role_tag_session='{_legacy_new_ts}' "
                                    f"for user='{_legacy_username}' (role_tag='{_legacy_role_tag}') via legacy path"
                                )
                    # Session+role_tag guard: skip deduction when both tables have
                    # the same session timestamp and role_tag (already processed).
                    if (_legacy_session_ts is not None and (_legacy_role_tag or "").strip()
                            and _legacy_username):
                        cur.execute(
                            "SELECT session FROM sourcing"
                            " WHERE username = %s AND role_tag = %s LIMIT 1",
                            (_legacy_username, _legacy_role_tag)
                        )
                        _src_row = cur.fetchone()
                        _src_session = _src_row[0] if _src_row else None
                        if _src_session is not None and _src_session == _legacy_session_ts:
                            conn.commit()
                            return jsonify({"ok": True,
                                            "token": int(_legacy_token) if _legacy_token is not None else 0,
                                            "skipped": True}), 200
                cur.execute(
                    "UPDATE login SET token = %s WHERE userid = %s RETURNING token",
                    (token_int, userid)
                )
            row = cur.fetchone()
            conn.commit()
        finally:
            cur.close()
            conn.close()
        if not row:
            return jsonify({"error": "user not found"}), 404
        new_token = int(row[0])
        # Log token spend/credit transaction
        if _LOGGER_AVAILABLE:
            delta = (new_token - _token_before_tx) if _token_before_tx is not None else None
            if delta is None:
                txn_type = "adjustment"
            elif delta < 0:
                txn_type = "spend"
            elif delta > 0:
                txn_type = "credit"
            else:
                txn_type = "adjustment"
            log_financial(
                username=_username_tx,
                userid=userid,
                feature="token_update",
                transaction_type=txn_type,
                token_before=_token_before_tx,
                token_after=new_token,
                transaction_amount=abs(delta) if delta is not None else None,
            )
        return jsonify({"ok": True, "token": new_token}), 200
    except Exception as e:
        logger.error(f"[TokenUpdate] {e}")
        return jsonify({"error": str(e)}), 500


# ==================== Role Tag Update Endpoint ====================

@app.route("/user/update_role_tag", methods=["POST", "GET"])
def user_update_role_tag():
    """
    POST/GET /user/update_role_tag
    Updates role_tag in both login and sourcing tables for the given username.
    The sourcing table is the authoritative source for role-based job title assessment.

    Session tracking:
    - A timestamp (role_tag_session) is generated and stored in login when role_tag is set.
    - The same timestamp is transferred to sourcing only after validating that the
      role_tag value matches in both tables, ensuring cross-table traceability.
    """
    global _role_tag_session_column_ensured
    if request.method == "POST":
        data = request.get_json(force=True, silent=True) or {}
        username = (data.get("username") or "").strip()
        role_tag = (data.get("role_tag") or "").strip()
    else:
        username = (request.args.get("username") or "").strip()
        role_tag = (request.args.get("role_tag") or "").strip()
    if not username:
        return jsonify({"error": "username required"}), 400
    conn = None
    cur = None
    try:
        import psycopg2
        pg_host = os.getenv("PGHOST", "localhost")
        pg_port = int(os.getenv("PGPORT", "5432"))
        pg_user = os.getenv("PGUSER", "postgres")
        pg_password = os.getenv("PGPASSWORD", "")
        pg_db = os.getenv("PGDATABASE", "candidate_db")
        conn = psycopg2.connect(
            host=pg_host, port=pg_port, user=pg_user,
            password=pg_password, dbname=pg_db
        )
        cur = conn.cursor()
        # Ensure role_tag_session column exists in login and sourcing (once per process).
        # NOTE: This flag mirrors the _token_guard_column_ensured pattern; it is intentionally
        # not protected by a lock for the same reason — IF NOT EXISTS makes the DDL idempotent,
        # so concurrent first-time executions are safe.
        if not _role_tag_session_column_ensured:
            cur.execute("ALTER TABLE login ADD COLUMN IF NOT EXISTS session TIMESTAMPTZ")
            cur.execute("ALTER TABLE sourcing ADD COLUMN IF NOT EXISTS session TIMESTAMPTZ")
            cur.execute("ALTER TABLE sourcing ALTER COLUMN session DROP DEFAULT")
            _role_tag_session_column_ensured = True
        # Step 1: Update login — set role_tag and generate session timestamp atomically
        cur.execute(
            "UPDATE login SET role_tag=%s, session=NOW() WHERE username=%s",
            (role_tag, username)
        )
        if cur.rowcount == 0:
            conn.rollback()
            return jsonify({"error": "User not found"}), 404
        # Step 2: Read back the persisted role_tag and session timestamp from login
        cur.execute(
            "SELECT role_tag, session FROM login WHERE username=%s",
            (username,)
        )
        login_row = cur.fetchone()
        login_role_tag = login_row[0] if login_row else None
        login_session_ts = login_row[1] if login_row else None
        # Step 3: Update sourcing role_tag for all records of this user
        cur.execute("ALTER TABLE sourcing ADD COLUMN IF NOT EXISTS role_tag TEXT DEFAULT ''")
        cur.execute("UPDATE sourcing SET role_tag=%s WHERE username=%s", (role_tag, username))
        # Step 4: Validate that role_tag matches in both login and sourcing, then transfer
        # the session timestamp from login to sourcing for consistency and traceability.
        if login_role_tag == role_tag and login_session_ts is not None:
            cur.execute(
                "UPDATE sourcing SET session=%s WHERE username=%s AND role_tag=%s",
                (login_session_ts, username, role_tag)
            )
        conn.commit()
        logger.info(
            f"[UpdateRoleTag] Set role_tag='{role_tag}' session_ts='{login_session_ts}' "
            f"for user='{username}' in login and sourcing tables"
        )
        return jsonify({"ok": True, "username": username, "role_tag": role_tag,
                        "session": login_session_ts.isoformat() if login_session_ts else None}), 200
    except Exception as e:
        logger.exception(f"[UpdateRoleTag] Failed for user='{username}': {e}")
        if conn:
            try:
                conn.rollback()
            except Exception:
                pass
        return jsonify({"error": str(e)}), 500
    finally:
        if cur:
            try:
                cur.close()
            except Exception:
                pass
        if conn:
            try:
                conn.close()
            except Exception:
                pass

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
@_rate("5 per minute; 100 per day")
@_check_user_rate("vskillset_infer")
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
    assessment_level = (data.get("assessment_level") or "L2").upper()
    username = (data.get("username") or "").strip()
    force_regen = bool(data.get("force", False))
    
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
        pg_password = os.getenv("PGPASSWORD", "")
        pg_db = os.getenv("PGDATABASE", "candidate_db")
        
        conn = psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
        cur = conn.cursor()
        
        # Normalize linkedin URL
        normalized = linkedinurl.lower().strip().rstrip('/')
        if not normalized.startswith('http'):
            normalized = 'https://' + normalized
        
        # Idempotency guard: if vskillset already exists in DB, return it without
        # re-running Gemini. Pass force=true in the request body to override this.
        if not force_regen:
            try:
                cur.execute("""
                    SELECT vskillset FROM process
                    WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl)) = %s
                       OR normalized_linkedin = %s
                    LIMIT 1
                """, (normalized, normalized))
                vs_row = cur.fetchone()
                if vs_row and vs_row[0]:
                    existing_vs = vs_row[0]
                    if isinstance(existing_vs, str):
                        existing_vs = json.loads(existing_vs)
                    if isinstance(existing_vs, list) and len(existing_vs) > 0:
                        high_skills = [i["skill"] for i in existing_vs if isinstance(i, dict) and i.get("category") == "High"]
                        cur.close()
                        conn.close()
                        logger.info(f"[vskillset_infer] Returning existing vskillset ({len(existing_vs)} items) for {linkedinurl[:50]} — use force=true to regenerate")
                        return jsonify({
                            "results": existing_vs,
                            "persisted": True,
                            "skipped": True,
                            "high_skills": high_skills,
                            "confirmed_skills": [i["skill"] for i in existing_vs if isinstance(i, dict) and i.get("source") == "confirmed"],
                            "inferred_skills":  [i["skill"] for i in existing_vs if isinstance(i, dict) and i.get("source") == "inferred"],
                        }), 200
            except Exception as _e:
                logger.warning(f"[vskillset_infer] Idempotency check failed ({_e}); proceeding with generation")
        
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
        
        # STEP 1: Extractive pass - mark skills explicitly mentioned in experience text as confirmed/High
        explicitly_confirmed = _extract_confirmed_skills(profile_context, skills)
        confirmed_set = set(s.lower() for s in explicitly_confirmed)
        confirmed_results = [
            {
                "skill": skill,
                "probability": 100,
                "category": "High",
                "reason": "Explicitly mentioned in experience text",
                "source": "confirmed"
            }
            for skill in explicitly_confirmed
        ]
        logger.info(f"[vskillset_infer] Extractive pass: {len(confirmed_results)}/{len(skills)} skills confirmed from text")

        # STEP 2: Only send unconfirmed skills to Gemini for inference
        unconfirmed_skills = [s for s in skills if s.lower() not in confirmed_set]
        inferred_results = []

        if unconfirmed_skills:
            # Call Gemini only for unconfirmed/missing skills
            model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)

            prompt = f"""SYSTEM:
You are an expert technical recruiter evaluating candidate skillsets based on their work experience.

TASK:
For each skill in the list below, evaluate the candidate's likely proficiency based on their experience.
These skills were NOT found explicitly in the experience text, so use contextual inference from
job titles, companies, products, sector, and experience patterns.
Assign a probability score (0-100) and categorize as Low (<40), Medium (40-74), or High (75-100).

CANDIDATE PROFILE:
{profile_context[:3000]}

SKILLS TO INFER (not found explicitly in experience text):
{json.dumps(unconfirmed_skills, ensure_ascii=False)}

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

            parsed = _extract_json_object(raw_text)

            if not parsed or "evaluations" not in parsed:
                logger.warning(f"[vskillset_infer] Gemini returned invalid JSON: {raw_text[:200]}")
                # Fallback: create basic inferred results for unconfirmed skills
                for skill in unconfirmed_skills:
                    inferred_results.append({
                        "skill": skill,
                        "probability": 50,
                        "category": "Medium",
                        "reason": "Unable to parse Gemini response",
                        "source": "inferred"
                    })
            else:
                inferred_results = parsed["evaluations"]

            # Ensure all required fields are present and annotate source
            for item in inferred_results:
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
                item["source"] = "inferred"

        # STEP 3: Merge confirmed + inferred results
        results = confirmed_results + inferred_results
        
        # Persist to database
        # 1. Store full annotated results in vskillset column (JSON)
        # 2. Store only High skills in skillset column as comma-separated string
        
        vskillset_json = json.dumps(results, ensure_ascii=False)
        high_skills = [item["skill"] for item in results if item["category"] == "High"]
        # Ensure all skills are strings before joining
        skillset_str = ", ".join([str(s) for s in high_skills if s])
        
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
        
        if updates:
            update_sql = sql.SQL("UPDATE process SET {} WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl)) = %s").format(sql.SQL(", ".join(updates)))
            update_values = []
            if 'vskillset' in available_cols:
                update_values.append(vskillset_json)
            update_values.append(normalized)
            cur.execute(update_sql, tuple(update_values))
        
        # Skillset: merge new High skills into existing value (add only; never remove or replace)
        if 'skillset' in available_cols and high_skills:
            cur.execute(
                "SELECT skillset FROM process WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl)) = %s",
                (normalized,)
            )
            _sk_row = cur.fetchone()
            _existing_sk = (_sk_row[0] or "") if _sk_row else ""
            _existing_parts = [s.strip() for s in _existing_sk.split(",") if s.strip()]
            _existing_set = {s.lower() for s in _existing_parts}
            _new_high = [s for s in high_skills if s.lower() not in _existing_set]
            if _new_high:
                _merged_sk = ", ".join(_existing_parts + _new_high)
                cur.execute(
                    "UPDATE process SET skillset = %s"
                    " WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl)) = %s",
                    (_merged_sk, normalized)
                )
                logger.info(f"[vskillset_infer] Merged {len(_new_high)} new High skills into skillset for {linkedinurl[:50]}")
            else:
                logger.info(f"[vskillset_infer] No new High skills for {linkedinurl[:50]} — skillset unchanged")
        
        conn.commit()
        
        cur.close()
        conn.close()
        
        return jsonify({
            "results": results,
            "persisted": True,
            "confirmed_skills": [item["skill"] for item in results if item.get("source") == "confirmed"],
            "inferred_skills": [item["skill"] for item in results if item.get("source") == "inferred"],
            "high_skills": high_skills
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
        from psycopg2 import sql as pgsql
        pg_host = os.getenv("PGHOST", "localhost")
        pg_port = int(os.getenv("PGPORT", "5432"))
        pg_user = os.getenv("PGUSER", "postgres")
        pg_password = os.getenv("PGPASSWORD", "")
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
        
        query = pgsql.SQL("SELECT {} FROM process WHERE LOWER(TRIM(TRAILING '/' FROM linkedinurl)) = %s LIMIT 1").format(pgsql.SQL(", ").join(pgsql.Identifier(c) for c in select_cols))
        cur.execute(query, (normalized,))
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

# Country/region words that Gemini appends to company names (e.g. "Electronic Arts China")
# Strip these trailing tokens so search results are based on the clean brand name.
_COMPANY_COUNTRY_SUFFIX_RE = re.compile(
    r'\s+(?:china|india|japan|korea|taiwan|singapore|malaysia|indonesia|thailand|vietnam|'
    r'philippines|australia|germany|france|uk|us|usa|emea|apac|latam|anz|mea|'
    r'asia(?:\s+pacific)?|pacific|americas|europe|international|global|limited|ltd\.?|'
    r'pte\.?\s*ltd\.?|inc\.?|corp(?:oration)?\.?|llc\.?|co\.?\s*ltd\.?|holdings?)$',
    re.IGNORECASE
)

# Parenthetical regional/status suffixes e.g. "(Japan)", "(Asia Pacific)", "(Merged)"
_COMPANY_PAREN_SUFFIX_RE = re.compile(r'\s*\([^)]+\)\s*$')

# Trailing "&" company-form patterns e.g. "& Co., Inc.", "& Co.", "& Sons"
_COMPANY_AMPERSAND_SUFFIX_RE = re.compile(
    r'\s*&\s*(?:co\.?\s*(?:,\s*)?(?:inc\.?|ltd\.?|llc\.?|plc\.?)?|sons?|partners?|associates?)\s*$',
    re.IGNORECASE
)

# Industry/entity-type descriptor words that Gemini appends to brand names
# e.g. "Takeda Pharmaceutical Company" → "Takeda", "Roche Diagnostics" → "Roche"
# Deliberately narrow: only strip words that are clearly generic descriptors, not brand differentiators.
_COMPANY_INDUSTRY_SUFFIX_RE = re.compile(
    r'\s+(?:pharmaceutical(?:s|(?:\s+company)?)?|diagnostics|biotech(?:nology)?|'
    r'life\s+sciences?|healthcare|health\s*care)$',
    re.IGNORECASE
)

def _strip_company_country_suffix(name: str) -> str:
    """
    Remove trailing suffixes from a company name to return the core brand name.
    Steps applied in order:
      1. Parenthetical regional/status suffixes: "(Japan)", "(Asia Pacific)"
      2. Ampersand company-form patterns: "& Co., Inc.", "& Sons"
      3. Industry/entity-type descriptors: "Pharmaceutical Company", "Diagnostics", "HealthCare"
         (applied up to 2 passes to handle chains like "Pharmaceutical Company")
      4. Country/region/legal-entity suffixes: "China", "Japan", "Ltd", "Inc", "Holdings"
         (applied up to 2 passes)
    Falls back to the original name if stripping reduces it to < 3 chars.
    """
    if not name:
        return name
    original = name.strip()
    s = original

    # Step 1: parenthetical suffixes
    s = _COMPANY_PAREN_SUFFIX_RE.sub('', s).strip()

    # Step 2: ampersand company-form patterns
    s = _COMPANY_AMPERSAND_SUFFIX_RE.sub('', s).strip()

    # Steps 3+4: interleave industry-type and country/legal suffix stripping.
    # Up to _MAX_SUFFIX_STRIP_PASSES passes so chained descriptors like
    # "Pharmaceuticals Corporation" are fully unwound in a single call.
    _MAX_PASSES = 3
    for _ in range(_MAX_PASSES):
        prev = s
        s2 = _COMPANY_INDUSTRY_SUFFIX_RE.sub('', s).strip()
        if s2 != s:
            s = s2
        s2 = _COMPANY_COUNTRY_SUFFIX_RE.sub('', s).strip()
        if s2 != s:
            s = s2
        if s == prev:
            break

    # Keep original if stripping makes it too short (< 3 chars)
    return s if len(s) >= 3 else original

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
        t=_strip_company_country_suffix(c.strip())
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

def _gemini_suggestions(job_titles, companies, industry, languages=None, sectors=None, country: str = None, products: list = None):
    if not (GEMINI_API_KEY and genai): return None
    languages = languages or []
    sectors = sectors or []
    products = products or []
    locality_hint = "Prioritize Singapore/APAC relevance where naturally applicable." if SINGAPORE_CONTEXT else ""
    
    # Add country-specific filtering instruction
    country_filter_hint = ""
    if country:
        country_filter_hint = f"\n- When suggesting companies, ONLY recommend companies with a legal entity or registered presence in {country}.\n- Exclude companies that do not operate in {country}.\n"

    # Add strict sector rule when sectors are provided to prevent cross-sector leakage
    sector_strict_hint = ""
    if sectors:
        sector_strict_hint = (
            "\n- STRICT SECTOR RULE for company.related: ONLY include companies whose PRIMARY BUSINESS and CORE"
            " OPERATIONS are direct competitors in the specified sector(s). EXCLUDE any company from a different"
            " industry that merely uses or purchases products/services in those sectors. Examples of what to exclude:\n"
            "  * For Gaming / Technology sectors: do NOT include pharma, healthcare, finance, insurance, or"
            " manufacturing companies, even if they use software or hire engineers internally.\n"
            "  * For Healthcare / Clinical Research sectors: do NOT include gaming, tech, or retail companies.\n"
            "  * For Industrial & Manufacturing sectors: do NOT include pure software, gaming, or financial services companies.\n"
            "  Competitors must share the same product/service focus as the job context.\n"
        )

    # Add product-based competitor hint when products are present.
    # Only applied when no companies are provided: when companies exist, they already
    # give Gemini strong competitor context, and adding products could create conflicting signals.
    product_hint = ""
    if products and not companies:
        product_hint = (
            f"\n- PRODUCT CONTEXT: The JD references these products/technologies: {', '.join(products[:10])}.\n"
            "  When no companies are explicitly mentioned, prioritize direct competitors that manufacture or sell these SAME products.\n"
            "  For example: if the JD mentions 'Aircon' or 'HVAC', suggest companies like Daikin, Carrier, Trane, Mitsubishi Electric, LG Electronics, etc.\n"
            "  Do NOT suggest companies from unrelated industries just to fill the list.\n"
        )

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
        "- Company names MUST be real, brand-level entities (e.g., 'Ubisoft', 'Electronic Arts', 'Epic Games').\n"
        "- DO NOT output generic placeholders (e.g., 'Gaming Studio', 'Tech Company', 'Pharma Company', 'Consulting Firm', 'Marketing Agency').\n"
        + country_filter_hint
        + sector_strict_hint
        + product_hint
        + "- No duplicates, no commentary, no extra keys.\n"
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
    products = data.get("products") or []  # Product references extracted from JD
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

    gem=_gemini_suggestions(job_titles, companies, industry, languages, sectors, country, products)
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
    if not api_key or not cx: return [], 0
    endpoint="https://www.googleapis.com/customsearch/v1"
    params={"key":api_key,"cx":cx,"q":query,"num":min(num,10),"start":start_index}
    if gl_hint: params["gl"]=gl_hint
    try:
        r=requests.get(endpoint, params=params, timeout=30)
        r.raise_for_status()
        data=r.json()
        items=data.get("items",[]) or []
        total_str=(data.get("searchInformation") or {}).get("totalResults","0") or "0"
        try:
            estimated_total=int(str(total_str).replace(",",""))
        except (ValueError, TypeError):
            estimated_total=0
        out=[]
        for it in items:
            out.append({"link":it.get("link") or "","title":it.get("title") or "","snippet":it.get("snippet") or "","displayLink":it.get("displayLink") or ""})
        return out, estimated_total
    except Exception as e:
        logger.warning(f"[CSE] page fetch failed: {e}")
        return [], 0

def _is_private_host(url: str) -> bool:
    """Return True if the URL resolves to a private/loopback/reserved IP — used to block SSRF."""
    import socket
    import ipaddress
    from urllib.parse import urlparse
    try:
        host = urlparse(url).hostname
        if not host:
            return True
        for addrinfo in socket.getaddrinfo(host, None):
            ip = ipaddress.ip_address(addrinfo[4][0])
            if (ip.is_private or ip.is_loopback or ip.is_reserved
                    or ip.is_link_local or ip.is_multicast):
                return True
        return False
    except Exception:
        return True


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
                            # Check that the domain is linkedin.com or a proper subdomain.
                            if not (parsed.netloc == 'linkedin.com' or parsed.netloc.endswith('.linkedin.com')):
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
            # SECURITY: reject URLs that resolve to private/loopback addresses (SSRF)
            if _is_private_host(profile_pic_url):
                logger.warning(f"[Profile Pic] SSRF: blocked private-host URL: {profile_pic_url}")
                return None
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
        # SECURITY: block SSRF — reject URLs that resolve to private/loopback addresses
        if _is_private_host(image_url):
            logger.warning(f"[Fetch Image Bytes] SSRF: blocked private-host URL: {image_url}")
            return None
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
    m_cc=re.search(r'site:([a-z]{2})\.linkedin\.com/in', " ".join(queries), re.I)
    country_code_hint = m_cc.group(1).lower() if m_cc else None

    global_collected = 0

    for q in queries:
        # Global stop-loss: target already reached, no need to fire more queries.
        still_needed = target_limit - global_collected
        if still_needed <= 0:
            add_message(job_id, f"Target reached: {global_collected}/{target_limit} — skipping remaining queries")
            break

        # Each query tries to collect however many are still needed to reach the
        # overall target, so shortfalls from earlier queries are automatically filled.
        gathered=0; start_index=1; pages_fetched=0
        effective_target = still_needed
        add_message(job_id, f"Running CSE: {q} target={effective_target} (need {still_needed} more to reach {target_limit})")

        while gathered < effective_target:
            remaining = effective_target - gathered
            page_size = min(CSE_PAGE_SIZE, remaining)

            page, estimated_total = google_cse_search_page(q, GOOGLE_CSE_API_KEY, GOOGLE_CSE_CX, page_size, start_index, gl_hint=country_code_hint)
            pages_fetched+=1

            # Per-query stop-loss: if Google reports fewer total results than we
            # are requesting from this query, cap the query target to what Google
            # says is actually available.  This prevents wasting API quota on
            # pages that will always return empty, but does NOT prevent subsequent
            # queries from running to make up the shortfall.
            if pages_fetched == 1 and estimated_total > 0 and estimated_total < effective_target:
                effective_target = estimated_total
                add_message(job_id, f"  Stop-loss: Google reports ~{estimated_total} results for this query — capping to {effective_target}")

            if not page:
                add_message(job_id, f"  No results page start={start_index}")
                break

            results.extend(page); gathered+=len(page); global_collected+=len(page)
            if len(page) < page_size: break
            start_index += len(page)

            # Safety break — prevents runaway pagination on unexpectedly large indices
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


# ---------------------------------------------------------------------------
# Import second-half routes (job runner, sourcing, CV processing, bulk assess).
# webbridge_cv.py is a sibling module that imports shared state from this file;
# the circular import is safe because all names below are defined before this
# import statement is reached.
import webbridge_cv  # registers routes with `app`
from webbridge_cv import _gemini_multi_sector, _core_assess_profile  # backward refs

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

def _startup_backfill_role_tag_session():
    """
    One-time startup backfill: for every login row where role_tag is set but
    role_tag_session is NULL, generate a timestamp (NOW()) and transfer it to
    all matching sourcing rows (WHERE username matches AND role_tag matches).

    This handles rows that existed before the role_tag_session column was
    introduced via ALTER TABLE … ADD COLUMN IF NOT EXISTS (which sets NULL for
    pre-existing rows).  Called once when the server process starts.
    """
    try:
        import psycopg2
        pg_host = os.getenv("PGHOST", "localhost")
        pg_port = int(os.getenv("PGPORT", "5432"))
        pg_user = os.getenv("PGUSER", "postgres")
        pg_password = os.getenv("PGPASSWORD", "")
        pg_db = os.getenv("PGDATABASE", "candidate_db")
        conn = psycopg2.connect(
            host=pg_host, port=pg_port, user=pg_user,
            password=pg_password, dbname=pg_db
        )
        cur = conn.cursor()
        try:
            # Ensure columns exist before touching them
            cur.execute("ALTER TABLE login ADD COLUMN IF NOT EXISTS session TIMESTAMPTZ")
            cur.execute("ALTER TABLE sourcing ADD COLUMN IF NOT EXISTS session TIMESTAMPTZ")
            cur.execute("ALTER TABLE sourcing ALTER COLUMN session DROP DEFAULT")
            # Find all login rows with role_tag set but role_tag_session NULL
            cur.execute(
                "SELECT username, role_tag FROM login"
                " WHERE role_tag IS NOT NULL AND role_tag <> '' AND session IS NULL"
            )
            rows = cur.fetchall()
            count = 0
            for username, role_tag in rows:
                if not username:
                    continue
                # Generate a timestamp for this row and write it to login
                cur.execute(
                    "UPDATE login SET session = NOW()"
                    " WHERE username = %s AND role_tag = %s AND session IS NULL"
                    " RETURNING session",
                    (username, role_tag)
                )
                ts_row = cur.fetchone()
                if ts_row and ts_row[0] is not None:
                    # Transfer the same timestamp to sourcing for matching rows
                    cur.execute(
                        "UPDATE sourcing SET session = %s"
                        " WHERE username = %s AND role_tag = %s",
                        (ts_row[0], username, role_tag)
                    )
                    count += 1
            conn.commit()
            if count:
                logger.info(f"[Startup] Backfilled role_tag_session for {count} user(s) missing a session timestamp.")
            else:
                logger.info("[Startup] role_tag_session backfill: no rows needed backfilling (all sessions already set or no role_tag entries found).")
            global _role_tag_session_column_ensured
            _role_tag_session_column_ensured = True
        finally:
            cur.close()
            conn.close()
    except Exception as e:
        logger.error(f"[Startup] role_tag_session backfill failed: {e}")


_startup_backfill_role_tag_session()

# ── API Porting routes ─────────────────────────────────────────────────────────
import re as _re

_PORTING_INPUT_DIR = os.path.normpath(
    os.getenv("PORTING_INPUT_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "porting_input"))
)
_PORTING_MAPPINGS_DIR = os.path.normpath(
    os.getenv("PORTING_MAPPINGS_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "porting_mappings"))
)
_PROCESS_TABLE_FIELDS = [
    'id','name','company','jobtitle','country','linkedinurl','username','userid',
    'product','sector','jobfamily','geographic','seniority','skillset',
    'sourcingstatus','email','mobile','office','role_tag','experience','cv',
    'education','exp','rating','pic','tenure','comment','vskillset',
    'compensation','lskillset','jskillset',
]

def _porting_safe_name(s):
    return _re.sub(r'[^a-zA-Z0-9_\-]', '_', str(s))

def _porting_get_key() -> bytes:
    """Return a stable 32-byte encryption key.

    Priority:
    1. PORTING_SECRET env var (set by the operator for production use).
    2. Persisted key file  <porting_input>/porting.key  (auto-created on first run).
    """
    secret = os.getenv("PORTING_SECRET", "").strip()
    if secret:
        return (secret + "!" * 32)[:32].encode()[:32]
    # Auto-generate / reuse a persistent random key so restarts stay compatible.
    key_path = os.path.join(_PORTING_INPUT_DIR, "porting.key")
    os.makedirs(_PORTING_INPUT_DIR, exist_ok=True)
    if os.path.exists(key_path):
        with open(key_path, "rb") as fh:
            raw = fh.read()
        if len(raw) >= 32:
            return raw[:32]
        logger.warning("[porting] porting.key is shorter than 32 bytes — regenerating.")
    raw = os.urandom(32)
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=_PORTING_INPUT_DIR)
    try:
        os.write(fd, raw)
        os.close(fd)
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, key_path)
    except Exception:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return raw


def _porting_encrypt(data: bytes) -> bytes:
    """AES-256-GCM encrypt.  Returns nonce(12) + ciphertext + tag(16).
    Auto-installs the 'cryptography' package if it is not already present."""
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError:
        import subprocess
        import sys as _sys
        logger.info("[porting] 'cryptography' not found — installing…")
        result = subprocess.run(
            [_sys.executable, "-m", "pip", "install", "cryptography"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error("[porting] pip install cryptography failed: %s", result.stderr)
            raise RuntimeError(
                "The 'cryptography' package is required for encryption. "
                "Install it with: pip install cryptography"
            ) from None
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    key = _porting_get_key()
    nonce = os.urandom(12)
    ct = AESGCM(key).encrypt(nonce, data, None)
    return nonce + ct

def _porting_login_required():
    """Return (username, None) or (None, error_response)."""
    username = (request.cookies.get("username") or "").strip()
    if not username:
        return None, (jsonify({"error": "Authentication required"}), 401)
    return username, None

@app.post("/api/porting/upload")
def porting_upload():
    username, err = _porting_login_required()
    if err:
        return err
    try:
        body = request.get_json(silent=True) or {}
        upload_type = body.get("type", "")
        content = body.get("content", "")
        filename = body.get("filename", "")
        if not upload_type or not content:
            return jsonify({"error": "Missing type or content"}), 400
        if upload_type not in ("file", "text"):
            return jsonify({"error": 'type must be "file" or "text"'}), 400
        import base64
        if upload_type == "file":
            raw = base64.b64decode(content)
        else:
            raw = content.encode("utf-8")
        if len(raw) > 1024 * 1024:
            return jsonify({"error": "Content too large (max 1 MB)"}), 413
        safe_fname = os.path.basename(str(filename)).replace(" ", "_") if filename else (
            "upload.env" if upload_type == "file" else "api_keys.txt"
        )
        safe_fname = _re.sub(r'[^a-zA-Z0-9_\-\.]', '_', safe_fname)
        safe_fname = f"{_porting_safe_name(username)}_{int(__import__('time').time()*1000)}_{safe_fname}"
        os.makedirs(_PORTING_INPUT_DIR, exist_ok=True)
        encrypted = _porting_encrypt(raw)
        dest = os.path.join(_PORTING_INPUT_DIR, safe_fname + ".enc")
        with open(dest, "wb") as fh:
            fh.write(encrypted)
        return jsonify({"ok": True, "stored": safe_fname + ".enc"})
    except Exception as exc:
        logger.exception("[porting/upload]")
        return jsonify({"error": "Upload failed", "detail": str(exc)}), 500

@app.post("/api/porting/map")
def porting_map():
    username, err = _porting_login_required()
    if err:
        return err
    try:
        body = request.get_json(silent=True) or {}
        names = body.get("names", [])
        if not isinstance(names, list) or not names:
            return jsonify({"error": "names must be a non-empty array"}), 400
        if not genai:
            return jsonify({"error": "Gemini API key not configured."}), 500
        fields_str = ", ".join(_PROCESS_TABLE_FIELDS)
        names_str = ", ".join(f'"{str(n)}"' for n in names)
        prompt = (
            f'You are a database field mapping assistant.\n'
            f'Available target fields (PostgreSQL "process" table): {fields_str}\n\n'
            f'Map each of the following external API field names to the SINGLE best-matching target field.\n'
            f'If there is no reasonable match, use null.\n'
            f'Return ONLY a JSON object (no markdown, no explanation) where each key is the input name and '
            f'each value is the matching target field name or null.\n\n'
            f'Input names: {names_str}'
        )
        model = genai.GenerativeModel(GEMINI_SUGGEST_MODEL)
        result = model.generate_content(prompt)
        raw = result.text.strip()
        raw = _re.sub(r'^```(?:json)?\s*', '', raw, flags=_re.IGNORECASE)
        raw = _re.sub(r'\s*```$', '', raw).strip()
        try:
            mapping = json.loads(raw)
        except Exception:
            return jsonify({"error": "Gemini returned invalid JSON", "raw": raw}), 500
        cleaned = {k: (v if v and v in _PROCESS_TABLE_FIELDS else None) for k, v in mapping.items()}
        return jsonify({"ok": True, "mapping": cleaned})
    except Exception as exc:
        logger.exception("[porting/map]")
        return jsonify({"error": "Mapping failed", "detail": str(exc)}), 500

@app.post("/api/porting/confirm")
def porting_confirm():
    username, err = _porting_login_required()
    if err:
        return err
    try:
        body = request.get_json(silent=True) or {}
        mapping = body.get("mapping")
        if not mapping or not isinstance(mapping, dict):
            return jsonify({"error": "mapping is required"}), 400
        for k, v in mapping.items():
            if v is not None and v not in _PROCESS_TABLE_FIELDS:
                return jsonify({"error": f"Invalid target field: {v}"}), 400
        os.makedirs(_PORTING_MAPPINGS_DIR, exist_ok=True)
        path_out = os.path.join(_PORTING_MAPPINGS_DIR, _porting_safe_name(username) + ".json")
        with open(path_out, "w", encoding="utf-8") as fh:
            json.dump({"username": username, "mapping": mapping}, fh, indent=2)
        return jsonify({"ok": True})
    except Exception as exc:
        logger.exception("[porting/confirm]")
        return jsonify({"error": "Confirm failed", "detail": str(exc)}), 500

@app.get("/api/porting/mapping")
def porting_get_mapping():
    username, err = _porting_login_required()
    if err:
        return err
    try:
        path_in = os.path.join(_PORTING_MAPPINGS_DIR, _porting_safe_name(username) + ".json")
        if not os.path.isfile(path_in):
            return jsonify({"mapping": None})
        with open(path_in, encoding="utf-8") as fh:
            data = json.load(fh)
        return jsonify({"mapping": data.get("mapping")})
    except Exception as exc:
        logger.exception("[porting/mapping]")
        return jsonify({"error": "Could not load mapping", "detail": str(exc)}), 500

@app.post("/api/porting/export")
def porting_export():
    username, err = _porting_login_required()
    if err:
        return err
    try:
        path_map = os.path.join(_PORTING_MAPPINGS_DIR, _porting_safe_name(username) + ".json")
        if not os.path.isfile(path_map):
            return jsonify({"error": "No confirmed mapping found. Please complete the mapping step first."}), 400
        with open(path_map, encoding="utf-8") as fh:
            mapping = json.load(fh).get("mapping", {})
        cols = [c for c in _PROCESS_TABLE_FIELDS if c not in ("cv", "pic")]
        conn = _pg_connect()
        try:
            cur = conn.cursor()
            col_sql = ", ".join(f'"{c}"' for c in cols)
            cur.execute(f'SELECT {col_sql} FROM "process" WHERE username = %s', (username,))
            rows = cur.fetchall()
            cur.close()
        finally:
            conn.close()
        if not rows:
            return jsonify({"error": "No data found for this user in the process table."}), 404
        reverse_map = {proc: ext for ext, proc in mapping.items() if proc}
        exported = [
            {reverse_map.get(col, col): (row[i] if row[i] is not None else None) for i, col in enumerate(cols)}
            for row in rows
        ]
        json_str = json.dumps(exported, indent=2, default=str)
        body_req = request.get_json(silent=True) or {}
        target_url = body_req.get("targetUrl", "")
        if target_url:
            try:
                import urllib.parse as _up
                import urllib.request as _ur
                parsed = _up.urlparse(target_url)
                if parsed.scheme not in ("http", "https"):
                    raise ValueError("targetUrl must use http or https scheme")
                req_obj = _ur.Request(
                    target_url,
                    data=json_str.encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with _ur.urlopen(req_obj, timeout=15):
                    pass
            except Exception as push_err:
                logger.warning(f"[porting/export] push to {target_url} failed: {push_err}")
        from flask import make_response
        resp = make_response(json_str)
        resp.headers["Content-Type"] = "application/json"
        resp.headers["Content-Disposition"] = f'attachment; filename="porting_export_{_porting_safe_name(username)}.json"'
        log_approval(action="export_pdf_triggered", username=username,
                     detail=f"Data export triggered; {len(exported)} row(s)")
        return resp
    except Exception as exc:
        logger.exception("[porting/export]")
        log_error(source="porting_export", message=str(exc), severity="error",
                  username=username, endpoint="/api/porting/export")
        return jsonify({"error": "Export failed", "detail": str(exc)}), 500


# ── BYOK (Bring Your Own Keys) routes ─────────────────────────────────────────
_BYOK_REQUIRED_KEYS = [
    'GEMINI_API_KEY', 'GOOGLE_CSE_API_KEY', 'GOOGLE_API_KEY',
    'GOOGLE_CSE_CX', 'GOOGLE_CLIENT_ID', 'GOOGLE_CLIENT_SECRET',
]

def _byok_path(username: str) -> str:
    byok_dir = os.path.join(_PORTING_INPUT_DIR, 'byok')
    os.makedirs(byok_dir, exist_ok=True)
    return os.path.join(byok_dir, _porting_safe_name(username) + '.enc')


@app.post("/api/porting/byok/activate")
def byok_activate():
    username, err = _porting_login_required()
    if err:
        return err
    try:
        body = request.get_json(silent=True) or {}
        keys = {}
        missing = []
        for k in _BYOK_REQUIRED_KEYS:
            val = str(body.get(k, '')).strip()
            if not val:
                missing.append(k)
            else:
                keys[k] = val
        if missing:
            return jsonify({"error": f"Missing required keys: {', '.join(missing)}"}), 400
        raw = json.dumps({'username': username, 'keys': keys}).encode('utf-8')
        encrypted = _porting_encrypt(raw)
        dest = _byok_path(username)
        with open(dest, 'wb') as fh:
            fh.write(encrypted)
        log_infrastructure("byok_activated", username=username,
                           detail="BYOK keys activated", status="success")
        return jsonify({"ok": True, "byok_active": True})
    except Exception as exc:
        logger.exception("[porting/byok/activate]")
        log_error(source="byok_activate", message=str(exc), severity="error",
                  username=username, endpoint="/api/porting/byok/activate")
        return jsonify({"error": "BYOK activation failed", "detail": str(exc)}), 500


@app.get("/api/porting/byok/status")
def byok_status():
    username, err = _porting_login_required()
    if err:
        return err
    try:
        active = os.path.isfile(_byok_path(username))
        return jsonify({"byok_active": active})
    except Exception as exc:
        logger.exception("[porting/byok/status]")
        return jsonify({"error": "Could not check BYOK status", "detail": str(exc)}), 500


@app.get("/api/porting/credentials/status")
def porting_credentials_status():
    """Return whether the user has any uploaded credential files on file."""
    username, err = _porting_login_required()
    if err:
        return err
    try:
        safe_prefix = _porting_safe_name(username) + "_"
        has_creds = any(
            f.startswith(safe_prefix) and f.endswith(".enc")
            for f in os.listdir(_PORTING_INPUT_DIR)
        ) if os.path.isdir(_PORTING_INPUT_DIR) else False
        return jsonify({"credentials_on_file": has_creds})
    except Exception as exc:
        logger.exception("[porting/credentials/status]")
        return jsonify({"error": "Could not check credential status", "detail": str(exc)}), 500


@app.delete("/api/porting/byok/deactivate")
def byok_deactivate():
    username, err = _porting_login_required()
    if err:
        return err
    try:
        dest = _byok_path(username)
        if os.path.isfile(dest):
            os.remove(dest)
        log_infrastructure(
            "byok_deactivated",
            username=username,
            detail="BYOK keys file removed",
            status="success",
            key_type="ALL",
            deactivation_reason="manual",
        )
        return jsonify({"ok": True, "byok_active": False})
    except Exception as exc:
        logger.exception("[porting/byok/deactivate]")
        return jsonify({"error": "Could not deactivate BYOK", "detail": str(exc)}), 500


@app.post("/api/porting/byok/validate")
def byok_validate():
    """Validate BYOK keys by probing live Google Cloud APIs + checking credential formats.
    Steps:
      1. Gemini API  — list models (validates GEMINI_API_KEY + billing)
      2. Custom Search API — single query (validates GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX)
      3. GOOGLE_API_KEY format check
      4. OAuth client credential format check
    Returns a structured results array without storing anything."""
    username, err = _porting_login_required()
    if err:
        return err
    try:
        import re
        import urllib.request as _ureq
        import urllib.parse as _uparse

        body = request.get_json(silent=True) or {}
        keys = {}
        missing = []
        for k in _BYOK_REQUIRED_KEYS:
            raw = body.get(k)
            if not isinstance(raw, (str, int, float)):
                missing.append(k); continue
            val = str(raw).strip()
            if not val or len(val) > 512:
                missing.append(k)
            else:
                keys[k] = val
        if missing:
            return jsonify({"error": f"Missing required keys: {', '.join(missing)}"}), 400

        def _probe(url, timeout=8):
            """GET url; returns (http_status_or_None, body_text)."""
            try:
                with _ureq.urlopen(url, timeout=timeout) as resp:
                    return resp.status, resp.read().decode('utf-8', errors='replace')
            except Exception as exc:
                if hasattr(exc, 'code'):
                    try:
                        return exc.code, exc.read().decode('utf-8', errors='replace')
                    except Exception:
                        return exc.code, ''
                return None, str(exc)

        def _err_msg(body_text, fallback):
            try:
                return json.loads(body_text).get('error', {}).get('message', fallback)
            except Exception:
                return fallback

        results = []

        # ── Step 1: Gemini API (GEMINI_API_KEY + billing) ────────────────────────
        gemini_url = (
            "https://generativelanguage.googleapis.com/v1beta/models?key="
            + _uparse.quote(keys['GEMINI_API_KEY'], safe='')
        )
        status, body_text = _probe(gemini_url)
        if status == 200:
            results.append({'step': 'gemini', 'label': 'Gemini API', 'status': 'ok',
                            'detail': 'API key is valid and billing is active.'})
        elif status == 403:
            results.append({'step': 'gemini', 'label': 'Gemini API', 'status': 'error',
                            'detail': _err_msg(body_text, 'Gemini API is not enabled or billing is inactive on this project.'),
                            'consoleUrl': 'https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com'})
        elif status == 400:
            results.append({'step': 'gemini', 'label': 'Gemini API', 'status': 'error',
                            'detail': _err_msg(body_text, 'Invalid GEMINI_API_KEY.'),
                            'consoleUrl': 'https://console.cloud.google.com/apis/credentials'})
        else:
            detail = f'Unexpected HTTP {status}' if status else f'Could not reach Google APIs: {body_text}'
            results.append({'step': 'gemini', 'label': 'Gemini API', 'status': 'warn', 'detail': detail})

        # ── Step 2: Custom Search API (GOOGLE_CSE_API_KEY + GOOGLE_CSE_CX) ───────
        cse_url = (
            "https://customsearch.googleapis.com/customsearch/v1?key="
            + _uparse.quote(keys['GOOGLE_CSE_API_KEY'], safe='')
            + "&cx=" + _uparse.quote(keys['GOOGLE_CSE_CX'], safe='')
            + "&q=test&num=1"
        )
        status, body_text = _probe(cse_url)
        if status == 200:
            results.append({'step': 'cse', 'label': 'Custom Search API', 'status': 'ok',
                            'detail': 'CSE API key and Search Engine ID are valid.'})
        elif status == 403:
            results.append({'step': 'cse', 'label': 'Custom Search API', 'status': 'error',
                            'detail': _err_msg(body_text, 'Custom Search API is not enabled or billing is required.'),
                            'consoleUrl': 'https://console.cloud.google.com/apis/library/customsearch.googleapis.com'})
        elif status == 400:
            results.append({'step': 'cse', 'label': 'Custom Search API', 'status': 'error',
                            'detail': _err_msg(body_text, 'Invalid GOOGLE_CSE_API_KEY or GOOGLE_CSE_CX Search Engine ID.'),
                            'consoleUrl': 'https://console.cloud.google.com/apis/credentials'})
        else:
            detail = f'Unexpected HTTP {status}' if status else f'Could not reach Custom Search API: {body_text}'
            results.append({'step': 'cse', 'label': 'Custom Search API', 'status': 'warn', 'detail': detail})

        # ── Step 3: GOOGLE_API_KEY format ─────────────────────────────────────────
        google_api_key_ok = bool(re.fullmatch(r'AIza[0-9A-Za-z\-_]{35}', keys['GOOGLE_API_KEY']))
        results.append({
            'step': 'google_api_key', 'label': 'GOOGLE_API_KEY Format',
            'status': 'ok' if google_api_key_ok else 'warn',
            'detail': ('Key format is valid (AIza… 39-character format).' if google_api_key_ok
                       else 'Key format looks unusual — expected a 39-character key starting with "AIza".'),
            **({'consoleUrl': 'https://console.cloud.google.com/apis/credentials'} if not google_api_key_ok else {}),
        })

        # ── Step 4: OAuth client credentials ──────────────────────────────────────
        client_id_ok = bool(re.fullmatch(r'\d+-[a-zA-Z0-9]+\.apps\.googleusercontent\.com', keys['GOOGLE_CLIENT_ID']))
        client_secret_ok = bool(re.match(r'^(GOCSPX-[A-Za-z0-9_\-]{28,}|[A-Za-z0-9_\-]{24,})$', keys['GOOGLE_CLIENT_SECRET']))
        if not client_id_ok:
            results.append({'step': 'oauth', 'label': 'OAuth Client Credentials', 'status': 'error',
                            'detail': 'GOOGLE_CLIENT_ID must have the format <numbers>-<id>.apps.googleusercontent.com',
                            'consoleUrl': 'https://console.cloud.google.com/apis/credentials'})
        elif not client_secret_ok:
            results.append({'step': 'oauth', 'label': 'OAuth Client Credentials', 'status': 'warn',
                            'detail': 'GOOGLE_CLIENT_SECRET format looks unusual (expected "GOCSPX-…"). Verify it was copied from Google Cloud Console → Credentials → OAuth 2.0 Client.',
                            'consoleUrl': 'https://console.cloud.google.com/apis/credentials'})
        else:
            results.append({'step': 'oauth', 'label': 'OAuth Client Credentials', 'status': 'ok',
                            'detail': 'Client ID and Client Secret formats are valid.'})

        all_ok = all(r['status'] in ('ok', 'warn') for r in results)
        overall_status = "success" if all_ok else "fail"
        failed_steps = [r['label'] for r in results if r['status'] == 'error']
        log_infrastructure("byok_validation", username=username,
                           detail="; ".join(failed_steps) if failed_steps else "All checks passed",
                           status=overall_status)
        return jsonify({'ok': all_ok, 'results': results})
    except Exception as exc:
        logger.exception("[porting/byok/validate]")
        log_error(source="byok_validate", message=str(exc), severity="error",
                  username=username, endpoint="/api/porting/byok/validate")
        return jsonify({"error": "Validation failed", "detail": str(exc)}), 500


if __name__ == '__main__':
    port=int(os.getenv("PORT","8091"))
    logger.info(f"Starting AutoSourcing webbridge on :{port}")
    if not GOOGLE_CSE_API_KEY or not GOOGLE_CSE_CX:
        logger.warning("GOOGLE_CSE_API_KEY/CX not set. Search Results Only / Auto-expand may not produce rows.")
    
    # Using run_simple is implicitly handled by app.run when DispatcherMiddleware wraps the app
    # provided we monkeypatch app.wsgi_app correctly above.
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)