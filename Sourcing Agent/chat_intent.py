import re

def is_affirmative(t: str):
    return bool(re.search(r'\b(yes|yep|yeah|yup|sure|ok|okay|go ahead|do it|proceed|start|begin|review|excel)\b', (t or "").lower()))

def is_negative(t: str):
    return bool(re.search(r'\b(no|nope|not now|cancel|stop|abort|nah|reject)\b', (t or "").lower()))

def classify_intent(text: str):
    tl=(text or "").strip().lower()
    if re.search(r'\b(i\s+want\s+suggestion|i\s+am\s+asking\s+for\s+suggestion|suggest\s+companies|suggest\s+job\s+titles|company\s+ideas|job\s+title\s+ideas)\b', tl):
        return {"mode":"suggestion","kind":"both"}
    if re.search(r'\b(find\s+companies|company\s+recommendations)\b', tl):
        return {"mode":"suggestion","kind":"companies"}
    if re.search(r'\b(what|which|where)\b.*\bcompanies?\b', tl) and not re.search(r'\b(profile|candidate|linkedin|people)\b', tl):
        return {"mode":"suggestion","kind":"companies"}
    if any(p in tl for p in ["suggest","recommend","other job title","target companies"]):
        if "job" in tl: return {"mode":"suggestion","kind":"jobs"}
        if "compan" in tl: return {"mode":"suggestion","kind":"companies"}
        return {"mode":"suggestion","kind":"both"}
    if any(p in tl for p in ["proceed","start the search","run the search","find profiles","source ","fetch candidates","begin sourcing"]):
        return {"mode":"search","kind":""}
    if any(k in tl for k in ["job title","job titles","alternate job title"]):
        return {"mode":"suggestion","kind":"jobs"}
    if any(p in tl for p in ["how many profiles","profile count","result count"]):
        return {"mode":"count","kind":""}
    return {"mode":"uncertain","kind":""}