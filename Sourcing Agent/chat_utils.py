import os, re, json
from datetime import datetime

SENIORITY_KEYWORDS = {
    "associate": "Associate", "junior": "Associate", "entry": "Associate",
    "manager": "Manager", "lead": "Manager",
    "director": "Director", "head": "Director",
    "vp": "Director", "vice president": "Director"
}

ROLE_LIKE_TERMS = {
    "technical art","technical artist","artist","game artist","environment artist",
    "character artist","concept artist","vfx","vfx artist","animator","animation",
    "rendering","shader","shading","graphics programmer","tools programmer",
    "programmer","engineer","developer","producer","designer","art director"
}

SECTOR_TERMS = [
    "industry","sector","domain","field","vertical","discipline",
    "profession","specialization","category","area"
]

SECTOR_STOPWORDS = {
    "what","other","sector","sectors","industry","industries","domain","domains",
    "field","fields","vertical","verticals","discipline","disciplines",
    "area","areas","category","categories","profession","professions",
    "specialization","specializations","the","and","or","of","to","in","a","an"
}

_SUB_TO_MAIN_CACHE = None
_SECTOR_TAXONOMY_CACHE = None

def dedupe(items):
    out, seen = [], set()
    for x in items or []:
        if not isinstance(x,str): continue
        t = x.strip()
        if not t: continue
        k = t.lower()
        if k in seen: continue
        seen.add(k); out.append(t)
    return out

def detect_seniority(text: str):
    tl = text.lower() if text else ""
    for k,v in SENIORITY_KEYWORDS.items():
        if re.search(rf"\b{re.escape(k)}\b", tl): return v
    return None

# --- Affected Section: expand sector normalization with medical/pharma synonyms ---
def normalize_sector_names(names):
    synonyms = {
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
        "pharmaceuticals": "pharma"
    }
    cleaned = []
    for n in names or []:
        if not isinstance(n,str): continue
        t = n.strip().strip(",.;:").lower()
        if not t or t in SECTOR_STOPWORDS: continue
        if len(re.sub(r"[^a-zA-Z]","",t))<3: continue
        if any(r in t for r in ROLE_LIKE_TERMS): continue
        if t in synonyms:
            t = synonyms[t]
        cleaned.append(t)
    return dedupe(cleaned)

def load_sector_taxonomy():
    global _SUB_TO_MAIN_CACHE, _SECTOR_TAXONOMY_CACHE
    if _SUB_TO_MAIN_CACHE is not None: return _SUB_TO_MAIN_CACHE
    mapping = {}
    def add_map(sub_name, main_name):
        if not sub_name or not main_name: return
        s=sub_name.strip().lower(); m=main_name.strip()
        if not s: return
        mapping.setdefault(s,[])
        if m not in mapping[s]: mapping[s].append(m)
    try:
        path=os.path.join(os.getcwd(),"sectors.json")
        if os.path.isfile(path):
            with open(path,"r",encoding="utf-8") as f:
                data=json.load(f)
            _SECTOR_TAXONOMY_CACHE=data
            if isinstance(data,dict):
                for main, subs in data.items():
                    if isinstance(subs,list):
                        for s in subs:
                            if isinstance(s,str): add_map(s,main)
                    add_map(main,main)
    except Exception:
        pass
    # Manual aliases
    manual = {
        "gaming":["Media, Gaming, Entertainment"],
        "video game":["Media, Gaming, Entertainment"],
        "game development":["Media, Gaming, Entertainment"],
        "clinical research":["Healthcare, Life Sciences"],
        "pharma":["Healthcare, Life Sciences"],
        "pharmaceuticals":["Healthcare, Life Sciences"],
        "biotech":["Healthcare, Life Sciences"],
        "finance":["Financial Services"],
        "banking":["Financial Services"],
        "fintech":["Financial Services"],
        "ecommerce":["Retail, E-commerce"],
        "retail":["Retail, E-commerce"],
        "semiconductor":["Semiconductors, Hardware"],
        "chip design":["Semiconductors, Hardware"],
        "automotive":["Automotive, Mobility"],
    }
    for k,v in manual.items():
        for m in v:
            add_map(k,m)
    _SUB_TO_MAIN_CACHE=mapping
    return mapping

def extract_sectors_regex(text: str):
    if not text: return []
    s = " " + text.strip() + " "
    terms="|".join([re.escape(t) for t in SECTOR_TERMS])
    candidates=[]
    pat_a=re.compile(rf"\b([A-Za-z0-9&/\- ,]+?)\s+(?:{terms})\b", re.I)
    for m in pat_a.finditer(s):
        frag=(m.group(1) or "").strip(" -,:;/")
        if frag: candidates.append(frag)
    pat_b=re.compile(
        rf"\b(?:{terms})\s+(?:of|in|for|within)?\s*(?:the\s+)?([A-Za-z0-9&/\- ,]+?)\b(?=[\.,;:]|\s(?:in|at|for|with|and|or|who|that|which)\b|$)",
        re.I
    )
    for m in pat_b.finditer(s):
        frag=(m.group(1) or "").strip(" -,:;/")
        if frag: candidates.append(frag)
    pat_c=re.compile(rf"\b(?:{terms})\s*[:\-]\s*([A-Za-z0-9&/\- ,]+?)\b(?=[\.,;:]|\s|$)", re.I)
    for m in pat_c.finditer(s):
        frag=(m.group(1) or "").strip(" -,:;/")
        if frag: candidates.append(frag)
    return dedupe(candidates)[:4]

def sector_keyword_match(text: str, extra_terms=None):
    tl=(text or "").lower()
    found=set()
    for s in extract_sectors_regex(text):
        found.add(s.lower())
    sub_map=load_sector_taxonomy()
    for sub in list(sub_map.keys())[:3000]:
        if sub and sub in tl: found.add(sub)
    for t in (extra_terms or []):
        if isinstance(t,str) and t.strip(): found.add(t.strip().lower())
    return [x.lower() for x in normalize_sector_names(list(found))]

def map_to_main_sectors(names):
    sub_map=load_sector_taxonomy()
    mains=[]
    for n in names or []:
        if not isinstance(n,str): continue
        key=n.strip().lower()
        if not key: continue
        if key in sub_map: mains.extend(sub_map[key])
        else: mains.append(n.strip())
    return normalize_sector_names(mains)

def country_to_cc(country: str):
    if not country: return None
    c=country.strip().lower()
    if len(c)==2 and c.isalpha(): return c
    alias={
        "united states of america":"us","united states":"us","u.s.":"us","u.s.a":"us","usa":"us",
        "united kingdom":"gb","uk":"gb","england":"gb","south korea":"kr","republic of korea":"kr",
        "north korea":"kp","uae":"ae"
    }
    if c in alias: return alias[c]
    fallback={
        "malaysia":"my","singapore":"sg","germany":"de","france":"fr","italy":"it","spain":"es",
        "japan":"jp","china":"cn","india":"in","brazil":"br","canada":"ca","australia":"au",
        "poland":"pl","portugal":"pt","netherlands":"nl","sweden":"se","norway":"no",
        "denmark":"dk","finland":"fi","switzerland":"ch"
    }
    return fallback.get(c)

def extract_country_hint(text: str):
    if not text: return ""
    m=re.search(r'\b(?:based in|located in|in|from)\s+([A-Za-z][A-Za-z\s&\.-]{1,50})\b', text, re.I)
    return (m.group(1) or "").strip(" ,.;:-") if m else ""