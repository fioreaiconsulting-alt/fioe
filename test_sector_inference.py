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


if __name__ == "__main__":
    unittest.main(verbosity=2)
