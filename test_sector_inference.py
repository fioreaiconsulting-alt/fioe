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


if __name__ == "__main__":
    unittest.main(verbosity=2)