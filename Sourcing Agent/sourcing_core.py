"""
sourcing_core.py
Encapsulates sourcing + profile retrieval logic separated from chatbot_api.
Only pure functionality lives here; no FastAPI route code, no conversation state.
"""

import os
import json
import asyncio
import requests
from typing import Tuple, Optional, Dict, Any, List

# ---------------- Utility Helpers ----------------
def dedupe(items: List[str]) -> List[str]:
    out, seen = [], set()
    for x in items or []:
        if not isinstance(x, str):
            continue
        t = x.strip()
        if not t:
            continue
        k = t.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(t)
    return out

# ---------------- Country Code Helpers (now backed by countrycode.JSON) ---------------
_COUNTRYCODE_CACHE: Optional[Dict[str, str]] = None
_COUNTRYNAME_TO_CC: Optional[Dict[str, str]] = None

def _load_country_codes() -> Dict[str, str]:
    """
    Loads countrycode.JSON once and caches it.
    countrycode.JSON is expected to be a mapping of 'cc' -> 'Country Name' (ISO-2 keys).
    """
    global _COUNTRYCODE_CACHE, _COUNTRYNAME_TO_CC
    if _COUNTRYCODE_CACHE is not None and _COUNTRYNAME_TO_CC is not None:
        return _COUNTRYCODE_CACHE

    # Resolve JSON path (prefer alongside this file; fallback to CWD)
    candidates = []
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(here, "countrycode.JSON"))
    except Exception:
        pass
    candidates.append(os.path.join(os.getcwd(), "countrycode.JSON"))

    data = {}
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                break
            except Exception:
                data = {}

    # Fallback to empty if not found; callers will still use local aliases
    _COUNTRYCODE_CACHE = {str(k).strip().lower(): str(v).strip() for k, v in (data or {}).items() if str(k).strip()}
    # Build reverse map: country name (lower) -> code
    _COUNTRYNAME_TO_CC = {}
    for cc, name in _COUNTRYCODE_CACHE.items():
        nm = (name or "").strip().lower()
        if nm and cc:
            _COUNTRYNAME_TO_CC[nm] = cc
    return _COUNTRYCODE_CACHE

def country_to_cc(country: str):
    """
    Returns a 2-letter country code (lowercase) for the given country string.
    - Accepts direct ISO-2 codes.
    - Resolves full country names via countrycode.JSON (if available).
    - Includes common alias fallbacks for robustness.
    """
    if not country:
        return None
    c = country.strip().lower()

    # If user already passed a 2-letter alpha code, accept it
    if len(c) == 2 and c.isalpha():
        return c

    # Load JSON-backed maps
    codes = _load_country_codes()              # cc -> Country Name
    names_to_cc = _COUNTRYNAME_TO_CC or {}     # Country Name (lower) -> cc

    # Try exact country name match from JSON
    if c in names_to_cc:
        return names_to_cc[c]

    # Common aliases/synonyms
    alias = {
        "united states of america": "us", "united states": "us", "u.s.": "us",
        "u.s.a": "us", "usa": "us", "states": "us",
        "united kingdom": "gb", "uk": "gb", "england": "gb", "great britain": "gb",
        "south korea": "kr", "republic of korea": "kr", "korea republic": "kr", "s. korea": "kr", "korea": "kr",
        "north korea": "kp",
        "uae": "ae", "united arab emirates": "ae",
        # Frequent localized names can be added here if needed
    }
    if c in alias:
        return alias[c]

    # Last resort: minimal fallback map for common countries (kept for resilience)
    fallback = {
        "malaysia": "my", "singapore": "sg", "germany": "de", "france": "fr",
        "italy": "it", "spain": "es", "japan": "jp", "china": "cn", "india": "in",
        "brazil": "br", "canada": "ca", "australia": "au", "poland": "pl",
        "portugal": "pt", "netherlands": "nl", "sweden": "se", "norway": "no",
        "denmark": "dk", "finland": "fi", "switzerland": "ch", "mexico": "mx"
    }
    return fallback.get(c)

# ---------------- Query Builder (extracted) ----------------
def build_queries(job_title: str, country: str = "", companies=None):
    companies = companies or []
    title_quoted = f"\"{job_title.strip()}\"" if job_title else ""
    cc = country_to_cc(country)
    site = f"site:{cc}.linkedin.com/in" if cc else "site:linkedin.com/in"
    base_filters = ' -intitle:"jobs" -inurl:"dir/"'
    comp_block = ""
    valid_companies = [c.strip() for c in companies if c.strip()]
    if valid_companies:
        if len(valid_companies) == 1:
            comp_block = f"\"{valid_companies[0]}\""
        else:
            comp_block = "(" + " OR ".join(f"\"{c}\"" for c in valid_companies) + ")"
    blocks = [b for b in [title_quoted, comp_block] if b]
    primary = site + (" " + " AND ".join(blocks) if blocks else "") + base_filters
    fallback_blocks = [title_quoted] if title_quoted else []
    fallback = site + (" " + " AND ".join(fallback_blocks) if fallback_blocks else "") + base_filters
    return primary, (fallback if fallback != primary else "")

# ---------------- Robust HTTP helpers (return clean errors on non-JSON) ----------------
def _safe_parse_json(response) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        return response.json() or {}, None
    except Exception:
        # Return first 300 chars of text body to avoid flooding logs/clients
        body = (response.text or "")[:300]
        return None, f"Server returned invalid JSON (HTTP {response.status_code}): {body}"

# ---------------- Backend API Interaction ----------------
class SourcingEngine:
    """
    High-level sourcing API encapsulation for starting jobs, polling, and formatting status.
    """

    def __init__(self, base_url: Optional[str] = None):
        self.base = (base_url or os.getenv("AUTOSOURCING_BASE", "http://localhost:8091")).rstrip("/")

    def _post(self, path: str, payload: Dict[str, Any], timeout: int = 35) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            r = requests.post(f"{self.base}{path}", json=payload, timeout=timeout)
            if r.status_code != 200:
                data, err = _safe_parse_json(r)
                if data is not None:
                    return None, f"HTTP {r.status_code}: {data}"
                return None, f"HTTP {r.status_code}: {(r.text or '')[:300]}"
            data, err = _safe_parse_json(r)
            if err:
                return None, err
            return data, None
        except Exception as e:
            return None, f"Request error: {e}"

    def _get(self, path: str, timeout: int = 20) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
        try:
            r = requests.get(f"{self.base}{path}", timeout=timeout)
            if r.status_code != 200:
                data, err = _safe_parse_json(r)
                if data is not None:
                    return None, f"HTTP {r.status_code}: {data}"
                return None, f"HTTP {r.status_code}: {(r.text or '')[:300]}"
            data, err = _safe_parse_json(r)
            if err:
                return None, err
            return data, None
        except Exception as e:
            return None, f"Request error: {e}"

    def start_job(self,
                  job_title: str,
                  country: str,
                  companies: List[str],
                  seniority: Optional[str],
                  selected_sectors: List[str],
                  auto_suggested_companies: List[str]) -> Tuple[Optional[str], Optional[str]]:
        """
        Initiates a sourcing job and returns (job_id, error).
        """
        companies = companies or []
        auto_suggested_companies = auto_suggested_companies or []

        # AFFECTED CHANGE:
        # Combine explicit companies and auto-suggested companies when auto suggestions are present,
        # otherwise use the explicit companies. Previous logic tied this behavior to selected_sectors,
        # which could lead to unexpected merging; make merging explicit and predictable.
        if auto_suggested_companies:
            effective_companies = dedupe(list(companies) + list(auto_suggested_companies))
        else:
            effective_companies = dedupe(companies)

        primary_query, fallback_query = build_queries(job_title, country, effective_companies)

        # AFFECTED SECTION: Refine identity retrieval: prefer persisted file; if userid missing but username present, lookup DB.
        userid = ""
        username = ""
        try:
            ctx_path = os.path.join(os.getcwd(), ".chatbot_identity.json")
            if os.path.isfile(ctx_path):
                with open(ctx_path, "r", encoding="utf-8") as f:
                    ident = json.load(f) or {}
                userid = str(ident.get("userid") or "").strip()
                username = str(ident.get("username") or "").strip()
        except Exception:
            pass
        if not username:
            username = (os.getenv("CHATBOT_USERNAME") or os.getenv("DEFAULT_USER") or "").strip()
        if not userid and username:
            try:
                import psycopg2
                pg_host = os.getenv("PGHOST", "localhost")
                pg_port = int(os.getenv("PGPORT", "5432"))
                pg_user = os.getenv("PGUSER", "postgres")
                pg_password = os.getenv("PGPASSWORD", "") or "orlha"
                pg_db = os.getenv("PGDATABASE", "candidate_db")
                conn = psycopg2.connect(host=pg_host, port=pg_port, user=pg_user, password=pg_password, dbname=pg_db)
                cur = conn.cursor()
                cur.execute("SELECT userid FROM login WHERE username=%s", (username,))
                row = cur.fetchone()
                cur.close()
                conn.close()
                if row and row[0]:
                    userid = str(row[0]).strip()
            except Exception:
                pass
        # END AFFECTED SECTION

        payload = {
            "jobTitle": f"\"{job_title}\"" if job_title else "",
            "jobTitles": [job_title] if job_title else [],
            "companyExpression": " OR ".join(f"\"{c}\"" for c in effective_companies) if effective_companies else "",
            "companyNames": companies,
            "autoSuggestedCompanyNames": auto_suggested_companies,
            "languages": [],
            "languageQuery": "",
            "currentRole": False,
            "queries": [primary_query],
            "fallbackQueries": [fallback_query] if fallback_query else [],
            "country": country or "",
            "autoExpand": True,
            "manualUrls": [],
            "searchResultsOnly": True,
            "selectedSectors": selected_sectors or [],
            "channelGaming": False,
            "channelMedia": False,
            "channelTechnology": False,
            "xrayPlatformQueries": [],
            "seniority": seniority or "",
            "userid": userid,
            "username": username
        }
        data, err = self._post("/start_job", payload)
        if err:
            return None, f"Failed to start sourcing job: {err}"
        job_id = (data or {}).get("job_id")
        if not job_id:
            return None, "Sourcing job did not return a job_id."
        return job_id, None

    async def poll_status(self, job_id: str, attempts: int = 8, delay_sec: float = 1.5) -> Dict[str, Any]:
        """
        Polls job status until done or attempts exhausted.
        """
        last = {}
        for _ in range(max(1, attempts)):
            data, err = self._get(f"/job_status/{job_id}")
            if data is not None:
                last = data
                if bool(data.get("done")):
                    return data
            else:
                last = {"error": err}
            await asyncio.sleep(delay_sec)
        return last

    @staticmethod
    def extract_profile_count(status: Dict[str, Any]) -> Optional[int]:
        if not isinstance(status, dict):
            return None
        cnt = status.get("count")
        if isinstance(cnt, int):
            return cnt
        urls = status.get("urls")
        if isinstance(urls, list):
            return len(urls)
        return None

    def format_status(self, status: Dict[str, Any], job_id: str) -> str:
        if not isinstance(status, dict):
            return "No job status available."
        urls = status.get("urls") or []
        ocsv = status.get("output_csv")
        oxlsx = status.get("output_xlsx")
        done = status.get("done")
        if not done:
            msg = f"“Just a sec—I’m on it!”\nFound {len(urls)} profiles"
            if len(urls) > 0 and not (ocsv or oxlsx):
                msg += "\nOrganizing the results into an Excel sheet"
            return msg
        lines = ["“Pulled the data from Excel—ready to go”"]
        if ocsv:
            lines.append(f"CSV: {self.base}/download/{ocsv}")
        if oxlsx:
            # AFFECTED SECTION: Fix typo (oxxlsx -> oxlsx) already corrected previously
            lines.append(f"XLSX: {self.base}/download/{oxlsx}")
            # END AFFECTED SECTION
        if not ocsv and not oxlsx:
            lines.append("No outputs were generated.")
        return "\n".join(lines).strip()