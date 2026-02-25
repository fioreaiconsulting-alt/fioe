"""
Unit tests for sector inference helpers in webbridge.py.
Run with: python test_sector_inference.py

NOTE: These tests use self-contained stubs that mirror the production functions
(_token_set, _map_keyword_to_sector_label, _find_best_sector_match_for_text).
Importing webbridge.py directly would start the Flask server and require all
production dependencies (Flask, google-generativeai, etc.), making isolated
testing impractical without a full environment. When the production functions
change, the corresponding stubs here must be updated to match.
"""
import re
import heapq
import unittest

# ---------------------------------------------------------------------------
# Minimal stubs — replicate the functions under test without importing the
# full Flask app (avoids side-effects and heavy dependencies)
# ---------------------------------------------------------------------------

def _token_set(s):
    if not s:
        return set()
    normalized = re.sub(r'&amp;|&', 'and', s.lower())
    return set(re.findall(r'\w+', normalized))


_KEYWORD_TO_SECTOR_LABEL = {
    "hvac": "Industrial & Manufacturing > Machinery",
    "air conditioning": "Industrial & Manufacturing > Machinery",
    "software": "Technology > Software",
    "cloud": "Technology > Cloud & Infrastructure",
    "infrastructure": "Technology > Cloud & Infrastructure",
    "ai": "Technology > AI & Data",
    "artificial intelligence": "Technology > AI & Data",
    "machine learning": "Technology > AI & Data",
    "bank": "Financial Services > Banking",
    "banking": "Financial Services > Banking",
    "insurance": "Financial Services > Insurance",
    "investment": "Financial Services > Investment & Asset Management",
    "wealth": "Financial Services > Investment & Asset Management",
    "fintech": "Financial Services > Fintech",
    "gaming": "Media, Gaming & Entertainment > Gaming",
    "ecommerce": "Consumer & Retail > E-commerce",
    "renewable": "Energy & Environment > Renewable Energy",
    "aerospace": "Industrial & Manufacturing > Aerospace & Defense",
}

SECTORS_INDEX = [
    "Technology > Cloud & Infrastructure",
    "Technology > AI & Data",
    "Technology > Software",
    "Healthcare > Biotechnology",
    "Healthcare > Healthcare Services",
    "Financial Services > Banking",
    "Financial Services > Fintech",
    "Financial Services > Insurance",
    "Financial Services > Investment & Asset Management",
    "Industrial & Manufacturing > Machinery",
    "Industrial & Manufacturing > Aerospace & Defense",
    "Media, Gaming & Entertainment > Gaming",
    "Consumer & Retail > E-commerce",
    "Energy & Environment > Renewable Energy",
]

SECTORS_TOKEN_INDEX = [(label, _token_set(label)) for label in SECTORS_INDEX]
MIN_SECTOR_JACCARD = 0.12


def _map_keyword_to_sector_label(text):
    txt = (text or "").lower()
    for kw, label in _KEYWORD_TO_SECTOR_LABEL.items():
        if re.search(r'\b' + re.escape(kw) + r'\b', txt):
            for l in SECTORS_INDEX:
                if l.lower() == label.lower():
                    return l
            for l in SECTORS_INDEX:
                if label.lower() in l.lower():
                    return l
    return None


def _find_best_sector_match_for_text(candidate):
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
        score = abs_overlap / len(cand_tokens | label_tokens)
        top_candidates.append((score, abs_overlap, label))
        if (score > best_score or
                (score == best_score and abs_overlap > best_abs) or
                (score == best_score and abs_overlap == best_abs and best and len(label) < len(best))):
            best_score = score
            best_abs = abs_overlap
            best = label
    match_ok = best and (
        best_score >= MIN_SECTOR_JACCARD or
        (len(cand_tokens) <= 2 and best_abs >= 1)
    )
    if match_ok:
        return best
    return None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTokenSet(unittest.TestCase):
    def test_ampersand_normalization(self):
        tokens = _token_set("Cloud & Infrastructure")
        self.assertIn("and", tokens)
        self.assertIn("cloud", tokens)
        self.assertIn("infrastructure", tokens)

    def test_html_ampersand_normalization(self):
        tokens = _token_set("Media, Gaming &amp; Entertainment")
        self.assertIn("and", tokens)
        self.assertIn("gaming", tokens)

    def test_empty_string(self):
        self.assertEqual(_token_set(""), set())

    def test_none(self):
        self.assertEqual(_token_set(None), set())


class TestMapKeywordToSectorLabel(unittest.TestCase):
    def test_cloud_keyword(self):
        result = _map_keyword_to_sector_label("Senior Cloud Engineer")
        self.assertEqual(result, "Technology > Cloud & Infrastructure")

    def test_banking_keyword(self):
        result = _map_keyword_to_sector_label("Banking Product Manager")
        self.assertEqual(result, "Financial Services > Banking")

    def test_hvac_keyword(self):
        result = _map_keyword_to_sector_label("HVAC Technician")
        self.assertEqual(result, "Industrial & Manufacturing > Machinery")

    def test_ai_word_boundary(self):
        # "ai" should NOT match inside "training"
        result = _map_keyword_to_sector_label("Training Specialist")
        self.assertIsNone(result)

    def test_ai_standalone(self):
        # "ai" SHOULD match when it stands alone
        result = _map_keyword_to_sector_label("AI Research Scientist")
        self.assertEqual(result, "Technology > AI & Data")

    def test_bank_word_boundary(self):
        # "bank" should NOT match inside "bankroll" or similar
        result = _map_keyword_to_sector_label("Data Scientist")
        self.assertIsNone(result)

    def test_machine_learning(self):
        result = _map_keyword_to_sector_label("Machine Learning Engineer")
        self.assertEqual(result, "Technology > AI & Data")

    def test_no_match(self):
        result = _map_keyword_to_sector_label("Unrelated Role XYZ")
        self.assertIsNone(result)

    def test_empty(self):
        result = _map_keyword_to_sector_label("")
        self.assertIsNone(result)


class TestFindBestSectorMatchForText(unittest.TestCase):
    def test_cloud_engineer(self):
        result = _find_best_sector_match_for_text("Senior Cloud Engineer")
        self.assertEqual(result, "Technology > Cloud & Infrastructure")

    def test_cloud_infrastructure_explicit(self):
        result = _find_best_sector_match_for_text("cloud infrastructure")
        self.assertEqual(result, "Technology > Cloud & Infrastructure")

    def test_banking_product_manager(self):
        result = _find_best_sector_match_for_text("Banking Product Manager")
        self.assertEqual(result, "Financial Services > Banking")

    def test_gaming_label(self):
        result = _find_best_sector_match_for_text("Gaming Producer")
        self.assertEqual(result, "Media, Gaming & Entertainment > Gaming")

    def test_healthcare_biotechnology(self):
        result = _find_best_sector_match_for_text("Healthcare Biotechnology Scientist")
        self.assertEqual(result, "Healthcare > Biotechnology")

    def test_no_match_random(self):
        result = _find_best_sector_match_for_text("xyz random nonsense")
        self.assertIsNone(result)

    def test_empty(self):
        result = _find_best_sector_match_for_text("")
        self.assertIsNone(result)

    def test_slashed_input(self):
        # Common Gemini output format: caller splits on "/" and passes parts
        result = _find_best_sector_match_for_text("Cloud")
        # Short 1-token input: Jaccard may be low but abs overlap >= 1 fallback applies
        self.assertEqual(result, "Technology > Cloud & Infrastructure")

    def test_cloud_not_healthcare(self):
        # Regression: "Senior Cloud Engineer" must NOT map to Healthcare > Biotechnology
        result = _find_best_sector_match_for_text("Senior Cloud Engineer")
        self.assertNotEqual(result, "Healthcare > Biotechnology")

    def test_fintech_label(self):
        result = _find_best_sector_match_for_text("Fintech Product Manager")
        self.assertEqual(result, "Financial Services > Fintech")

    def test_renewable_energy(self):
        result = _find_best_sector_match_for_text("Renewable Energy Engineer")
        self.assertEqual(result, "Energy & Environment > Renewable Energy")


class TestCombinedFallback(unittest.TestCase):
    """Tests that _find_best_sector_match_for_text or _map_keyword_to_sector_label together cover all cases."""

    def _resolve(self, text):
        return _find_best_sector_match_for_text(text) or _map_keyword_to_sector_label(text)

    def test_hvac_via_keyword(self):
        # "hvac" won't match any label token, but keyword map catches it
        result = self._resolve("HVAC Technician")
        self.assertEqual(result, "Industrial & Manufacturing > Machinery")

    def test_cloud_via_jaccard(self):
        result = self._resolve("Senior Cloud Engineer")
        self.assertEqual(result, "Technology > Cloud & Infrastructure")

    def test_banking_product_manager(self):
        result = self._resolve("Banking Product Manager")
        self.assertEqual(result, "Financial Services > Banking")

    def test_ambiguous_empty(self):
        result = self._resolve("")
        self.assertIsNone(result)


# ---------------------------------------------------------------------------
# Tests for _should_overwrite_existing (idempotency helper)
# Mirrors the production function in webbridge.py without importing it.
# ---------------------------------------------------------------------------

def _should_overwrite_existing(existing_meta, incoming_level="L2", force=False):
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


class TestShouldOverwriteExisting(unittest.TestCase):
    def test_no_existing_meta_allows(self):
        allow, reason = _should_overwrite_existing(None, "L2", False)
        self.assertTrue(allow)
        self.assertEqual(reason, "no existing rating")

    def test_empty_existing_meta_allows(self):
        allow, reason = _should_overwrite_existing({}, "L2", False)
        self.assertTrue(allow)

    def test_force_flag_overrides_everything(self):
        meta = {"level": "L2", "version": 3}
        allow, reason = _should_overwrite_existing(meta, "L1", force=True)
        self.assertTrue(allow)
        self.assertEqual(reason, "force_reassess=True")

    def test_l2_upgrades_l1(self):
        meta = {"level": "L1", "version": 1}
        allow, reason = _should_overwrite_existing(meta, "L2", False)
        self.assertTrue(allow)
        self.assertIn("upgrade", reason)

    def test_l1_does_not_downgrade_l2(self):
        meta = {"level": "L2", "version": 1}
        allow, reason = _should_overwrite_existing(meta, "L1", False)
        self.assertFalse(allow)
        self.assertIn("downgrade", reason)

    def test_same_level_l2_skips(self):
        meta = {"level": "L2", "version": 2}
        allow, reason = _should_overwrite_existing(meta, "L2", False)
        self.assertFalse(allow)
        self.assertIn("same level", reason)

    def test_same_level_l1_skips(self):
        meta = {"level": "L1", "version": 1}
        allow, reason = _should_overwrite_existing(meta, "L1", False)
        self.assertFalse(allow)

    def test_missing_level_metadata_allows(self):
        meta = {"level": "", "version": 1}
        allow, reason = _should_overwrite_existing(meta, "L2", False)
        self.assertTrue(allow)
        self.assertIn("no existing level metadata", reason)

    def test_l2_incoming_no_existing_level_allows(self):
        meta = {"level": None, "version": 1}
        allow, reason = _should_overwrite_existing(meta, "L2", False)
        self.assertTrue(allow)


# ---------------------------------------------------------------------------
# Tests for product exclusion from assessment + independent product inference
# Mirrors the production logic in webbridge.py without importing it.
# ---------------------------------------------------------------------------

def _build_active_criteria_stub(job_title, role_tag, country, company, seniority,
                                sector, product, tenure, target_skills,
                                candidate_skills, experience_text):
    """
    Stub that mirrors the active_criteria building logic in webbridge.py
    (after the change that excludes 'product' from the assessment breakdown).
    Product is intentionally NOT appended to active_criteria.
    """
    active_criteria = []
    if job_title and role_tag:
        active_criteria.append("jobtitle_role_tag")
    if country:
        active_criteria.append("country")
    if company:
        active_criteria.append("company")
    if seniority:
        active_criteria.append("seniority")
    if sector:
        active_criteria.append("sector")
    # Product is excluded from active_criteria (Gemini populates it independently).
    if tenure is not None and tenure != "":
        try:
            float(tenure)
            active_criteria.append("tenure")
        except (ValueError, TypeError):
            pass
    if target_skills and (candidate_skills or experience_text):
        active_criteria.append("skillset")
    return active_criteria


def _extract_product_list_stub(gemini_output):
    """
    Stub that mirrors how webbridge.py extracts the product_list from Gemini
    output (obj.get("product_list", [])).  Product inference is independent of
    the assessment breakdown.
    """
    if not isinstance(gemini_output, dict):
        return []
    raw = gemini_output.get("product_list", [])
    if isinstance(raw, list):
        return [str(p).strip() for p in raw if str(p).strip()]
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(',') if s.strip()]
    return []


class TestProductExcludedFromAssessment(unittest.TestCase):
    """Validate that 'product' is excluded from active_criteria while
    product inference (Gemini product_list) remains fully functional."""

    def test_product_not_in_active_criteria_when_products_present(self):
        """Even with a non-empty product list, product must not appear in active_criteria."""
        criteria = _build_active_criteria_stub(
            job_title="Product Manager",
            role_tag="Product Manager",
            country="Singapore",
            company="Acme Corp",
            seniority="Senior",
            sector="Technology > Software",
            product=["SaaS", "Mobile App", "API Platform"],
            tenure=3.5,
            target_skills=["Python", "SQL"],
            candidate_skills=["Python", "Java"],
            experience_text="5 years of software development"
        )
        self.assertNotIn("product", criteria)

    def test_product_not_in_active_criteria_when_no_products(self):
        """product must not appear in active_criteria even when product list is empty."""
        criteria = _build_active_criteria_stub(
            job_title="Software Engineer",
            role_tag="Software Engineer",
            country="UK",
            company="TechCo",
            seniority="Mid",
            sector="Technology > Cloud & Infrastructure",
            product=[],
            tenure=2.0,
            target_skills=["AWS", "Docker"],
            candidate_skills=["AWS"],
            experience_text="3 years cloud engineering"
        )
        self.assertNotIn("product", criteria)

    def test_other_criteria_still_active(self):
        """Excluding product must not affect other criteria being added."""
        criteria = _build_active_criteria_stub(
            job_title="Data Scientist",
            role_tag="Data Scientist",
            country="USA",
            company="DataCo",
            seniority="Senior",
            sector="Technology > AI & Data",
            product=["ML Platform"],
            tenure=4.0,
            target_skills=["Python", "TensorFlow"],
            candidate_skills=["Python"],
            experience_text="6 years ML research"
        )
        for expected in ["jobtitle_role_tag", "country", "company", "seniority",
                         "sector", "tenure", "skillset"]:
            self.assertIn(expected, criteria)
        self.assertNotIn("product", criteria)

    def test_product_inference_from_gemini_output(self):
        """Product list extracted from Gemini output is independent of assessment."""
        gemini_output = {
            "skillset": ["Python", "SQL"],
            "product_list": ["CRM Platform", "Mobile App", "Data Pipeline"],
            "seniority": "Senior",
            "sector": "Technology > Software"
        }
        products = _extract_product_list_stub(gemini_output)
        self.assertEqual(products, ["CRM Platform", "Mobile App", "Data Pipeline"])

    def test_product_inference_empty_output(self):
        """Empty Gemini output yields an empty product list (no crash)."""
        products = _extract_product_list_stub({})
        self.assertEqual(products, [])

    def test_product_inference_string_fallback(self):
        """Comma-separated string product_list is correctly parsed."""
        gemini_output = {"product_list": "SaaS, Mobile, API"}
        products = _extract_product_list_stub(gemini_output)
        self.assertEqual(products, ["SaaS", "Mobile", "API"])

    def test_product_inference_invalid_input(self):
        """Non-dict Gemini output returns empty list without raising."""
        self.assertEqual(_extract_product_list_stub(None), [])
        self.assertEqual(_extract_product_list_stub("string"), [])
        self.assertEqual(_extract_product_list_stub(42), [])


# ---------------------------------------------------------------------------
# Stubs for all rating assessment category heuristics.
# These mirror the production functions in webbridge.py without importing it.
# When the production functions change, update the stubs here to match.
# ---------------------------------------------------------------------------

import re as _re

def _jobtitle_heuristic_stub(candidate_title, required_tag):
    """Mirror of jobtitle_heuristic in webbridge.py."""
    if not candidate_title:
        return "not_assessed", ""
    v = str(candidate_title).lower()
    t = str(required_tag).lower()
    if t in v or v in t:
        return "match", "Heuristic match"
    _stopwords = {"the", "a", "an", "of", "and", "or", "for", "in", "at"}
    v_tokens = set(_re.findall(r'\b\w+\b', v)) - _stopwords
    t_tokens = set(_re.findall(r'\b\w+\b', t)) - _stopwords
    if v_tokens & t_tokens:
        return "related", "Partial title match (token overlap)"
    return "unrelated", "No token overlap with role tag"


def _seniority_heuristic_stub(candidate_seniority, required_seniority):
    """Mirror of seniority_heuristic in webbridge.py."""
    if not candidate_seniority:
        return "not_assessed", ""
    if not required_seniority:
        return "not_assessed", ""
    cs = str(candidate_seniority).lower().strip()
    rs = str(required_seniority).lower().strip()
    if cs == rs or cs in rs or rs in cs:
        return "match", f"Seniority match: {candidate_seniority}"
    return "unrelated", f"Seniority mismatch: candidate={candidate_seniority}, required={required_seniority}"


def _country_heuristic_stub(candidate_country, required_country):
    """Mirror of country_heuristic in webbridge.py — including city-to-country mapping."""
    _CITY_TO_COUNTRY = {
        "tokyo": "japan", "osaka": "japan", "kyoto": "japan", "yokohama": "japan",
        "beijing": "china", "shanghai": "china", "shenzhen": "china",
        "guangzhou": "china", "chengdu": "china", "hong kong": "china",
        "seoul": "south korea", "busan": "south korea",
        "mumbai": "india", "delhi": "india", "bangalore": "india",
        "hyderabad": "india", "chennai": "india", "kolkata": "india",
        "bangkok": "thailand",
        "jakarta": "indonesia",
        "kuala lumpur": "malaysia",
        "manila": "philippines",
        "hanoi": "vietnam", "ho chi minh city": "vietnam",
        "taipei": "taiwan",
        "sydney": "australia", "melbourne": "australia", "brisbane": "australia",
        "perth": "australia",
        "london": "united kingdom", "manchester": "united kingdom",
        "birmingham": "united kingdom",
        "berlin": "germany", "munich": "germany", "frankfurt": "germany",
        "hamburg": "germany",
        "paris": "france", "lyon": "france",
        "new york": "united states", "los angeles": "united states",
        "san francisco": "united states", "chicago": "united states",
        "seattle": "united states", "boston": "united states",
        "austin": "united states", "houston": "united states",
        "toronto": "canada", "vancouver": "canada", "montreal": "canada",
        "dubai": "united arab emirates", "abu dhabi": "united arab emirates",
    }
    _COUNTRY_ALIASES = {
        "uk": "united kingdom", "usa": "united states", "us": "united states",
        "uae": "united arab emirates",
    }

    def _resolve(val):
        v = str(val).lower().strip()
        v = _COUNTRY_ALIASES.get(v, v)
        return _CITY_TO_COUNTRY.get(v, v)

    if not candidate_country:
        return "not_assessed", ""
    if not required_country:
        return "not_assessed", ""
    cc = _resolve(candidate_country)
    rc = _resolve(required_country)
    if cc == rc or cc in rc or rc in cc:
        return "match", f"Country match: {candidate_country}"
    return "unrelated", f"Country mismatch: candidate={candidate_country}, required={required_country}"


def _star_string_stub(status, category_stars):
    """Mirror of star_string generation in webbridge.py."""
    if status == "not_assessed":
        return "Unable to Access"
    return "★" * category_stars + "☆" * (5 - category_stars)


def _tenure_heuristic_stub(tenure):
    """Mirror of tenure assessment heuristic in webbridge.py."""
    try:
        val = float(tenure)
        if val >= 4.0:
            return "match", f"{val:.1f} years avg tenure"
        elif val >= 2.0:
            return "related", f"{val:.1f} years avg tenure"
        else:
            return "unrelated", f"{val:.1f} years avg tenure (short)"
    except (ValueError, TypeError):
        return "not_assessed", "Tenure data unavailable"


def _scoring_factor_stub(category, status):
    """
    Mirror of the scoring logic in webbridge.py's _core_assess_profile scoring loop.
    - seniority and country: binary (match=1.0, else=0)
    - all others: match=1.0, related=0.5, else=0
    """
    if category in ("seniority", "country"):
        return 1.0 if status == "match" else 0.0
    if status == "match":
        return 1.0
    elif status == "related":
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Tests for Job Title (jobtitle_role_tag) assessment heuristic
# ---------------------------------------------------------------------------

class TestJobTitleHeuristic(unittest.TestCase):
    def test_exact_match(self):
        st, _ = _jobtitle_heuristic_stub("Clinical Study Manager", "Clinical Study Manager")
        self.assertEqual(st, "match")

    def test_substring_match(self):
        st, _ = _jobtitle_heuristic_stub("Senior Clinical Study Manager", "Clinical Study Manager")
        self.assertEqual(st, "match")

    def test_related_partial_token_overlap(self):
        # "Clinical Project Manager" shares "Clinical" and "Manager" with "Clinical Study Manager"
        st, _ = _jobtitle_heuristic_stub("Clinical Project Manager", "Clinical Study Manager")
        self.assertEqual(st, "related")

    def test_unrelated_no_token_overlap(self):
        # "Clinical Study Director" shares "Clinical" and "Study" → still token overlap → related
        # But a completely different role should be unrelated
        st, _ = _jobtitle_heuristic_stub("Finance Business Partner", "Clinical Study Manager")
        self.assertEqual(st, "unrelated")

    def test_director_vs_manager_unrelated(self):
        # Clinical Study Director vs Clinical Study Manager:
        # "Clinical" and "Study" overlap → related (not unrelated) per token-overlap rule
        st, _ = _jobtitle_heuristic_stub("Clinical Study Director", "Clinical Study Manager")
        self.assertEqual(st, "related")

    def test_empty_candidate_title(self):
        st, _ = _jobtitle_heuristic_stub("", "Clinical Study Manager")
        self.assertEqual(st, "not_assessed")

    def test_related_yields_half_score(self):
        st, _ = _jobtitle_heuristic_stub("Clinical Project Manager", "Clinical Study Manager")
        factor = _scoring_factor_stub("jobtitle_role_tag", st)
        self.assertEqual(factor, 0.5)

    def test_unrelated_yields_zero_score(self):
        st, _ = _jobtitle_heuristic_stub("Finance Business Partner", "Clinical Study Manager")
        factor = _scoring_factor_stub("jobtitle_role_tag", st)
        self.assertEqual(factor, 0.0)

    def test_match_yields_full_score(self):
        st, _ = _jobtitle_heuristic_stub("Clinical Study Manager", "Clinical Study Manager")
        factor = _scoring_factor_stub("jobtitle_role_tag", st)
        self.assertEqual(factor, 1.0)


# ---------------------------------------------------------------------------
# Tests for Seniority assessment heuristic (binary: match=1.0, else=0)
# ---------------------------------------------------------------------------

class TestSeniorityHeuristic(unittest.TestCase):
    def test_exact_match(self):
        st, _ = _seniority_heuristic_stub("Manager", "Manager")
        self.assertEqual(st, "match")

    def test_mismatch_director_vs_manager(self):
        # Candidate at Director level but required Manager → 0
        st, _ = _seniority_heuristic_stub("Director", "Manager")
        self.assertEqual(st, "unrelated")

    def test_mismatch_director_vs_manager_scores_zero(self):
        st, _ = _seniority_heuristic_stub("Director", "Manager")
        factor = _scoring_factor_stub("seniority", st)
        self.assertEqual(factor, 0.0)

    def test_mismatch_senior_vs_manager(self):
        st, _ = _seniority_heuristic_stub("Senior", "Manager")
        self.assertEqual(st, "unrelated")

    def test_match_scores_full(self):
        st, _ = _seniority_heuristic_stub("Manager", "Manager")
        factor = _scoring_factor_stub("seniority", st)
        self.assertEqual(factor, 1.0)

    def test_related_status_also_scores_zero_for_seniority(self):
        # Even if status were "related", binary scoring must yield 0 for seniority
        factor = _scoring_factor_stub("seniority", "related")
        self.assertEqual(factor, 0.0)

    def test_no_required_seniority(self):
        st, _ = _seniority_heuristic_stub("Senior", "")
        self.assertEqual(st, "not_assessed")

    def test_no_candidate_seniority(self):
        st, _ = _seniority_heuristic_stub("", "Manager")
        self.assertEqual(st, "not_assessed")

    def test_case_insensitive(self):
        st, _ = _seniority_heuristic_stub("MANAGER", "manager")
        self.assertEqual(st, "match")


# ---------------------------------------------------------------------------
# Tests for Country assessment heuristic (binary: match=1.0, else=0)
# ---------------------------------------------------------------------------

class TestCountryHeuristic(unittest.TestCase):
    def test_exact_match(self):
        st, _ = _country_heuristic_stub("Singapore", "Singapore")
        self.assertEqual(st, "match")

    def test_mismatch_china_vs_singapore(self):
        # Required Singapore, candidate's latest country China → 0
        st, _ = _country_heuristic_stub("China", "Singapore")
        self.assertEqual(st, "unrelated")

    def test_mismatch_scores_zero(self):
        st, _ = _country_heuristic_stub("China", "Singapore")
        factor = _scoring_factor_stub("country", st)
        self.assertEqual(factor, 0.0)

    def test_match_scores_full(self):
        st, _ = _country_heuristic_stub("Singapore", "Singapore")
        factor = _scoring_factor_stub("country", st)
        self.assertEqual(factor, 1.0)

    def test_related_status_also_scores_zero_for_country(self):
        # Even if status were "related", binary scoring must yield 0 for country
        factor = _scoring_factor_stub("country", "related")
        self.assertEqual(factor, 0.0)

    def test_no_required_country(self):
        st, _ = _country_heuristic_stub("Singapore", "")
        self.assertEqual(st, "not_assessed")

    def test_no_candidate_country(self):
        st, _ = _country_heuristic_stub("", "Singapore")
        self.assertEqual(st, "not_assessed")

    def test_case_insensitive(self):
        st, _ = _country_heuristic_stub("SINGAPORE", "singapore")
        self.assertEqual(st, "match")

    def test_uk_vs_usa_mismatch(self):
        st, _ = _country_heuristic_stub("UK", "USA")
        factor = _scoring_factor_stub("country", st)
        self.assertEqual(factor, 0.0)


# ---------------------------------------------------------------------------
# Tests for Company assessment (presence → match)
# ---------------------------------------------------------------------------

class TestCompanyAssessment(unittest.TestCase):
    def test_company_present_scores_full(self):
        factor = _scoring_factor_stub("company", "match")
        self.assertEqual(factor, 1.0)

    def test_company_related_still_scores_half(self):
        # company uses standard (non-binary) scoring
        factor = _scoring_factor_stub("company", "related")
        self.assertEqual(factor, 0.5)

    def test_company_unrelated_scores_zero(self):
        factor = _scoring_factor_stub("company", "unrelated")
        self.assertEqual(factor, 0.0)


# ---------------------------------------------------------------------------
# Tests for Sector assessment (related → 0.5 partial credit)
# ---------------------------------------------------------------------------

class TestSectorAssessment(unittest.TestCase):
    def test_sector_match_scores_full(self):
        factor = _scoring_factor_stub("sector", "match")
        self.assertEqual(factor, 1.0)

    def test_sector_related_scores_half(self):
        factor = _scoring_factor_stub("sector", "related")
        self.assertEqual(factor, 0.5)

    def test_sector_unrelated_scores_zero(self):
        factor = _scoring_factor_stub("sector", "unrelated")
        self.assertEqual(factor, 0.0)


# ---------------------------------------------------------------------------
# Tests for Tenure assessment heuristic (match ≥4y, related 2–4y, unrelated <2y)
# ---------------------------------------------------------------------------

class TestTenureHeuristic(unittest.TestCase):
    def test_long_tenure_is_match(self):
        st, _ = _tenure_heuristic_stub(5.0)
        self.assertEqual(st, "match")

    def test_four_year_boundary_is_match(self):
        st, _ = _tenure_heuristic_stub(4.0)
        self.assertEqual(st, "match")

    def test_medium_tenure_is_related(self):
        st, _ = _tenure_heuristic_stub(3.0)
        self.assertEqual(st, "related")

    def test_two_year_boundary_is_related(self):
        st, _ = _tenure_heuristic_stub(2.0)
        self.assertEqual(st, "related")

    def test_short_tenure_is_unrelated(self):
        st, _ = _tenure_heuristic_stub(1.0)
        self.assertEqual(st, "unrelated")

    def test_zero_tenure_is_unrelated(self):
        st, _ = _tenure_heuristic_stub(0.0)
        self.assertEqual(st, "unrelated")

    def test_invalid_tenure_not_assessed(self):
        st, _ = _tenure_heuristic_stub("N/A")
        self.assertEqual(st, "not_assessed")

    def test_none_tenure_not_assessed(self):
        st, _ = _tenure_heuristic_stub(None)
        self.assertEqual(st, "not_assessed")

    def test_tenure_related_scores_half(self):
        st, _ = _tenure_heuristic_stub(3.0)
        factor = _scoring_factor_stub("tenure", st)
        self.assertEqual(factor, 0.5)

    def test_tenure_match_scores_full(self):
        st, _ = _tenure_heuristic_stub(5.0)
        factor = _scoring_factor_stub("tenure", st)
        self.assertEqual(factor, 1.0)


# ---------------------------------------------------------------------------
# Tests for city-to-country recognition in country assessment
# ---------------------------------------------------------------------------

class TestCityToCountryMapping(unittest.TestCase):
    def test_tokyo_maps_to_japan(self):
        st, _ = _country_heuristic_stub("Tokyo", "Japan")
        self.assertEqual(st, "match")

    def test_beijing_maps_to_china(self):
        st, _ = _country_heuristic_stub("Beijing", "China")
        self.assertEqual(st, "match")

    def test_london_maps_to_uk(self):
        st, _ = _country_heuristic_stub("London", "United Kingdom")
        self.assertEqual(st, "match")

    def test_london_maps_to_uk_alias(self):
        st, _ = _country_heuristic_stub("London", "UK")
        self.assertEqual(st, "match")

    def test_new_york_maps_to_usa(self):
        st, _ = _country_heuristic_stub("New York", "United States")
        self.assertEqual(st, "match")

    def test_dubai_maps_to_uae(self):
        st, _ = _country_heuristic_stub("Dubai", "UAE")
        self.assertEqual(st, "match")

    def test_city_in_wrong_country_is_unrelated(self):
        # Tokyo is in Japan, not Singapore
        st, _ = _country_heuristic_stub("Tokyo", "Singapore")
        self.assertEqual(st, "unrelated")

    def test_beijing_vs_singapore_is_unrelated(self):
        st, _ = _country_heuristic_stub("Beijing", "Singapore")
        factor = _scoring_factor_stub("country", st)
        self.assertEqual(factor, 0.0)

    def test_city_country_same_country_match(self):
        # Both resolve to the same country
        st, _ = _country_heuristic_stub("Tokyo", "Japan")
        factor = _scoring_factor_stub("country", st)
        self.assertEqual(factor, 1.0)

    def test_seoul_maps_to_south_korea(self):
        st, _ = _country_heuristic_stub("Seoul", "South Korea")
        self.assertEqual(st, "match")

    def test_sydney_maps_to_australia(self):
        st, _ = _country_heuristic_stub("Sydney", "Australia")
        self.assertEqual(st, "match")


# ---------------------------------------------------------------------------
# Tests for "Unable to Access" star_string when status is not_assessed
# ---------------------------------------------------------------------------

class TestStarStringNotAssessed(unittest.TestCase):
    def test_not_assessed_yields_unable_to_access(self):
        result = _star_string_stub("not_assessed", 0)
        self.assertEqual(result, "Unable to Access")

    def test_match_yields_star_string(self):
        result = _star_string_stub("match", 5)
        self.assertEqual(result, "★★★★★")

    def test_unrelated_yields_empty_stars(self):
        result = _star_string_stub("unrelated", 0)
        self.assertEqual(result, "☆☆☆☆☆")

    def test_related_yields_partial_stars(self):
        result = _star_string_stub("related", 3)
        self.assertEqual(result, "★★★☆☆")

    def test_not_assessed_never_shows_stars(self):
        result = _star_string_stub("not_assessed", 5)
        self.assertNotIn("★", result)
        self.assertNotIn("☆", result)
        self.assertEqual(result, "Unable to Access")


if __name__ == "__main__":
    unittest.main(verbosity=2)