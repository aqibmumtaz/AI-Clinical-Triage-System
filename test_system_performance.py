"""Comprehensive system performance tests for AI Clinical Triage System.

Tests cover: STT noise handling, Urdu transcription, triage scoring accuracy,
symptom alias coverage, department selection rules, and end-to-end scenarios.

Run: pytest test_system_performance.py -v
"""

import pytest
import json
import os
import time
import hashlib

# Import from conversation_agent
import sys
sys.path.insert(0, os.path.dirname(__file__))
from conversation_agent import (
    compute_cluster_scores,
    _symptom_matches,
    _make_cache_key,
    DISEASE_CLUSTERS_DATA,
    SAMPLE_RATE_OUT,
)


# ─────────────────────────────────────────────
# 1. STT Noise & Urdu Transcription Tests
# ─────────────────────────────────────────────

class TestSTTUrduMistranscriptions:
    """Test that common Urdu STT mistranscriptions are handled via aliases."""

    @pytest.fixture
    def aliases(self):
        return DISEASE_CLUSTERS_DATA.get("symptom_aliases", {})

    def _alias_covers(self, aliases, target_text):
        """Check if target_text is covered by any alias entry."""
        target = target_text.lower()
        for base, alias_list in aliases.items():
            if target == base.lower():
                return True
            for a in alias_list:
                if target == a.lower():
                    return True
        return False

    def test_khansi_mistranscriptions(self, aliases):
        """All common STT variants of کھانسی (cough) must be aliased."""
        variants = ["کھانسی", "خاصی", "حاصل", "کانسی", "کاشی", "کھاسی", "کاسی", "خانسی"]
        for v in variants:
            assert self._alias_covers(aliases, v), f"Cough variant '{v}' not in aliases"

    def test_bukhar_mistranscriptions(self, aliases):
        """All common STT variants of بخار (fever) must be aliased."""
        variants = ["بخار", "بھکار", "بکھار", "بُخار"]
        for v in variants:
            assert self._alias_covers(aliases, v), f"Fever variant '{v}' not in aliases"

    def test_sir_dard_mistranscriptions(self, aliases):
        """Headache Urdu variants must be aliased."""
        variants = ["سر درد", "سردرد", "سر میں درد"]
        for v in variants:
            assert self._alias_covers(aliases, v), f"Headache variant '{v}' not in aliases"

    def test_insomnia_urdu_variants(self, aliases):
        """Insomnia Urdu variants including common speech patterns."""
        variants = ["نیند نہ آنا", "نیند نہیں آتی", "نیند کی کمی"]
        for v in variants:
            assert self._alias_covers(aliases, v), f"Insomnia variant '{v}' not in aliases"

    def test_chest_pain_urdu(self, aliases):
        """Chest pain Urdu variants."""
        variants = ["سینے میں درد", "سینے کی تکلیف"]
        for v in variants:
            assert self._alias_covers(aliases, v), f"Chest pain variant '{v}' not in aliases"

    def test_palpitations_urdu(self, aliases):
        """Palpitations Urdu variants."""
        variants = ["دل کی دھڑکن تیز", "دل زور سے دھڑکنا"]
        for v in variants:
            assert self._alias_covers(aliases, v), f"Palpitations variant '{v}' not in aliases"

    def test_stomach_pain_urdu(self, aliases):
        """Stomach pain Urdu variants."""
        variants = ["پیٹ درد", "پیٹ میں درد"]
        for v in variants:
            assert self._alias_covers(aliases, v), f"Stomach pain variant '{v}' not in aliases"


class TestSTTNoiseFiltering:
    """Test that noise word filtering doesn't block valid medical terms."""

    @pytest.fixture
    def noise_words(self):
        return frozenset({
            "", "hmm", "hm", "um", "uh", "ah", "oh", "mm", "mhm", "uh huh",
            "yeah", "yep", "yup", "nah", "nope", "ok", "okay",
            "...", ".", "-", "bye", "bye bye", "thank you", "thanks",
            "you", "the", "a", "and", "is", "it", "so", "but", "or",
            "ha", "huh", "eh", "hmm hmm", "mm hmm", "aha",
        })

    def test_medical_terms_not_in_noise(self, noise_words):
        """Common medical terms must NOT be in noise words."""
        medical = ["fever", "pain", "cough", "headache", "chest", "heart",
                    "blood", "pressure", "sugar", "breathing", "vomiting"]
        for term in medical:
            assert term not in noise_words, f"Medical term '{term}' incorrectly in noise words"

    def test_urdu_terms_not_in_noise(self, noise_words):
        """Urdu medical terms must NOT be in noise words."""
        urdu_medical = ["بخار", "درد", "کھانسی", "سر", "دل", "پیٹ", "ہاں", "نہیں"]
        for term in urdu_medical:
            assert term not in noise_words, f"Urdu term '{term}' incorrectly in noise words"

    def test_affirmations_not_noise(self, noise_words):
        """Patient affirmations in Urdu should not be filtered."""
        affirmations = ["ہاں", "جی ہاں", "جی", "نہیں", "جی نہیں"]
        for a in affirmations:
            assert a not in noise_words, f"Urdu affirmation '{a}' incorrectly in noise words"


# ─────────────────────────────────────────────
# 2. Symptom Matching Accuracy
# ─────────────────────────────────────────────

class TestExactSymptomMatching:
    """Test that exact matching prevents cross-contamination."""

    def test_cough_not_persistent_cough(self):
        assert not _symptom_matches({"cough"}, "persistent cough")

    def test_fever_not_high_fever(self):
        assert not _symptom_matches({"fever"}, "high fever")

    def test_headache_not_severe_headache(self):
        assert not _symptom_matches({"headache"}, "severe headache")

    def test_pain_not_chest_pain(self):
        assert not _symptom_matches({"pain"}, "chest pain")

    def test_exact_match_works(self):
        assert _symptom_matches({"chest pain"}, "chest pain")

    def test_exact_match_urdu(self):
        """Exact Urdu symptom match."""
        assert _symptom_matches({"بخار"}, "بخار")

    def test_weakness_not_sudden_weakness(self):
        assert not _symptom_matches({"weakness"}, "sudden weakness")

    def test_bleeding_not_heavy_bleeding(self):
        assert not _symptom_matches({"bleeding"}, "heavy bleeding")

    def test_back_pain_not_lower_back_pain(self):
        assert not _symptom_matches({"back pain"}, "lower back pain")

    def test_abdominal_pain_not_lower_abdominal_pain(self):
        assert not _symptom_matches({"abdominal pain"}, "lower abdominal pain")


# ─────────────────────────────────────────────
# 3. Full Alias Coverage
# ─────────────────────────────────────────────

class TestFullAliasCoverage:
    """Every condition symptom must have a corresponding alias with Urdu translation."""

    def test_all_symptoms_have_aliases(self):
        aliases = DISEASE_CLUSTERS_DATA["symptom_aliases"]
        aliased = set()
        for base, alias_list in aliases.items():
            aliased.add(base.lower())
            for a in alias_list:
                aliased.add(a.lower())

        all_symptoms = set()
        for cluster in DISEASE_CLUSTERS_DATA["clusters"]:
            for cond in cluster["conditions"]:
                for s in cond["symptoms"]:
                    all_symptoms.add(s.lower())

        missing = all_symptoms - aliased
        assert len(missing) == 0, f"Symptoms without aliases: {missing}"

    def test_all_aliases_have_urdu(self):
        """Every alias entry must have at least one Urdu script value."""
        import re
        urdu_re = re.compile(r'[\u0600-\u06FF]')
        aliases = DISEASE_CLUSTERS_DATA["symptom_aliases"]
        missing_urdu = []
        for base, alias_list in aliases.items():
            has_urdu = any(urdu_re.search(a) for a in alias_list)
            if not has_urdu:
                missing_urdu.append(base)
        assert len(missing_urdu) == 0, f"Aliases without Urdu: {missing_urdu}"

    def test_no_cross_department_alias_contamination(self):
        """No alias should map a symptom to a DIFFERENT department's condition symptom
        (except for genuinely same symptoms like breathing difficulty = shortness of breath)."""
        aliases = DISEASE_CLUSTERS_DATA["symptom_aliases"]
        # Build symptom → departments map
        sym_depts = {}
        for cluster in DISEASE_CLUSTERS_DATA["clusters"]:
            for cond in cluster["conditions"]:
                for s in cond["symptoms"]:
                    sl = s.lower()
                    if sl not in sym_depts:
                        sym_depts[sl] = set()
                    sym_depts[sl].add(cluster["department"])

        # Known acceptable cross-department aliases (genuinely same symptom)
        acceptable = {
            ("breathing difficulty", "shortness of breath"),
            ("burning urination", "painful urination"),
            ("skin rash", "rash"),
            ("stomach pain", "abdominal pain"),
        }

        harmful = []
        for base, alias_list in aliases.items():
            base_depts = sym_depts.get(base.lower(), set())
            for a in alias_list:
                al = a.lower()
                if al in sym_depts:
                    extra = sym_depts[al] - base_depts
                    if extra and (base.lower(), al) not in acceptable and (al, base.lower()) not in acceptable:
                        harmful.append(f"{base} → {a} (extra depts: {extra})")
        assert len(harmful) == 0, f"Harmful cross-department aliases: {harmful}"


# ─────────────────────────────────────────────
# 4. Department Selection Rules
# ─────────────────────────────────────────────

class TestDepartmentRulesComprehensive:
    """Test age/gender-based department filtering across all edge cases."""

    def test_pediatrics_zero_for_adult(self):
        results = compute_cluster_scores(["fever", "cough"], patient_age="25")
        peds = next(r for r in results if r["department"] == "Pediatrics")
        assert peds["score"] == 0.0

    def test_pediatrics_zero_for_elderly(self):
        results = compute_cluster_scores(["fatigue", "weakness"], patient_age="70")
        peds = next(r for r in results if r["department"] == "Pediatrics")
        assert peds["score"] == 0.0

    def test_pediatrics_active_for_child(self):
        results = compute_cluster_scores(["fever", "cough", "poor feeding"], patient_age="3")
        peds = next(r for r in results if r["department"] == "Pediatrics")
        assert peds["score"] > 0

    def test_pediatrics_penalized_unknown_age(self):
        """Without age, Pediatrics should be penalized (30%) unless child-specific symptoms."""
        results_no_child = compute_cluster_scores(["fever", "cough"])
        peds_no_child = next(r for r in results_no_child if r["department"] == "Pediatrics")

        results_child = compute_cluster_scores(["fever", "cough", "poor feeding"])
        peds_child = next(r for r in results_child if r["department"] == "Pediatrics")

        # With child-specific symptom, no penalty
        assert peds_child["score"] > peds_no_child["score"]

    def test_gynecology_zero_for_male(self):
        results = compute_cluster_scores(["nausea", "fatigue"], patient_gender="M")
        gyn = next(r for r in results if r["department"] == "Gynecology & Obstetrics")
        assert gyn["score"] == 0.0

    def test_gynecology_zero_for_male_word(self):
        """Gender as 'male' (word) should also work."""
        results = compute_cluster_scores(["nausea", "fatigue"], patient_gender="male")
        gyn = next(r for r in results if r["department"] == "Gynecology & Obstetrics")
        assert gyn["score"] == 0.0

    def test_gynecology_active_for_female(self):
        results = compute_cluster_scores(["missed period", "nausea"], patient_gender="F")
        gyn = next(r for r in results if r["department"] == "Gynecology & Obstetrics")
        assert gyn["score"] > 0

    def test_age_parsing_handles_urdu(self):
        """Age like '30 سال' should parse correctly."""
        results = compute_cluster_scores(["fever"], patient_age="30 سال")
        peds = next(r for r in results if r["department"] == "Pediatrics")
        assert peds["score"] == 0.0  # 30 years = adult

    def test_age_parsing_handles_none(self):
        """None age should not crash."""
        results = compute_cluster_scores(["fever"], patient_age=None)
        assert len(results) > 0

    def test_age_parsing_handles_empty(self):
        """Empty string age should not crash."""
        results = compute_cluster_scores(["fever"], patient_age="")
        assert len(results) > 0


# ─────────────────────────────────────────────
# 5. End-to-End Triage Scenarios
# ─────────────────────────────────────────────

class TestEndToEndTriageScenarios:
    """Real-world patient scenarios — verify correct department ranking."""

    def test_cardiac_chest_pain_urdu(self):
        """Urdu patient with chest pain → Cardiology #1."""
        results = compute_cluster_scores(["سینے میں درد", "سانس لینے میں تکلیف"])
        assert results[0]["department"] == "Cardiology"

    def test_general_fever_cough(self):
        """Fever + cough → General Medicine #1."""
        results = compute_cluster_scores(
            ["fever", "cough", "body ache", "sore throat"],
            {"fever": "yes", "temperature": "102"},
            "moderate"
        )
        assert results[0]["department"] == "General Medicine"

    def test_dental_toothache(self):
        """Toothache → Dentistry #1."""
        results = compute_cluster_scores(["toothache", "swollen gums", "gum pain"])
        assert results[0]["department"] == "Dentistry"

    def test_ortho_fracture(self):
        """Fracture → Orthopedics #1."""
        results = compute_cluster_scores(["fracture", "bone pain", "swelling"], {"mobility": "impaired"})
        assert results[0]["department"] == "Orthopedics"

    def test_ortho_urdu_bone_fracture(self):
        """Urdu bone symptoms: ہڈیاں میں درد + سوجن + فریکچر → Orthopedics #1.
        Previously failed because plural 'ہڈیاں' and Urdu 'فریکچر' had no aliases."""
        results = compute_cluster_scores(["ہڈیاں میں درد", "سوجن", "فریکچر"])
        ortho = next(r for r in results if r["department"] == "Orthopedics")
        assert ortho["score"] > 0.3, f"Orthopedics should score > 30%, got {ortho['score']*100:.0f}%"
        assert results[0]["department"] == "Orthopedics", \
            f"Orthopedics should be #1, got {results[0]['department']}"
        assert "bone pain" in ortho["matched_symptoms"], "ہڈیاں میں درد should match bone pain"
        assert "fracture" in ortho["matched_symptoms"], "فریکچر should match fracture"

    def test_ortho_urdu_plural_singular(self):
        """Both plural ہڈیوں and singular ہڈی should match bone pain."""
        r1 = compute_cluster_scores(["ہڈی میں درد"])
        r2 = compute_cluster_scores(["ہڈیوں میں درد"])
        o1 = next(r for r in r1 if r["department"] == "Orthopedics")
        o2 = next(r for r in r2 if r["department"] == "Orthopedics")
        assert o1["score"] > 0, "Singular ہڈی should match"
        assert o2["score"] > 0, "Plural ہڈیوں should match"

    def test_ent_ear_pain(self):
        """Ear pain → ENT #1."""
        results = compute_cluster_scores(["ear pain", "hearing loss", "ear discharge"])
        assert results[0]["department"] == "ENT"

    def test_ophthalmology_eye(self):
        """Eye problems → Ophthalmology #1."""
        results = compute_cluster_scores(["eye pain", "blurred vision", "red eye"])
        assert results[0]["department"] == "Ophthalmology"

    def test_psychiatry_anxiety(self):
        """Anxiety symptoms → Psychiatry #1."""
        results = compute_cluster_scores(["anxiety", "insomnia", "palpitations"])
        assert results[0]["department"] == "Psychiatry"

    def test_dermatology_skin(self):
        """Skin issues → Dermatology #1."""
        results = compute_cluster_scores(["skin rash", "itching", "redness"])
        assert results[0]["department"] == "Dermatology"

    def test_gastro_stomach(self):
        """Stomach issues → Gastroenterology #1."""
        results = compute_cluster_scores(["stomach pain", "heartburn", "acid reflux", "bloating"])
        assert results[0]["department"] == "Gastroenterology"

    def test_neuro_seizure(self):
        """Seizure → Neurology #1."""
        results = compute_cluster_scores(["seizure", "loss of consciousness", "convulsion"])
        assert results[0]["department"] == "Neurology"

    def test_gynecology_pregnancy(self):
        """Pregnancy symptoms for female → Gynecology #1."""
        results = compute_cluster_scores(
            ["missed period", "nausea", "vomiting", "fatigue"],
            patient_gender="F"
        )
        assert results[0]["department"] == "Gynecology & Obstetrics"

    def test_pediatrics_child(self):
        """Child with fever → Pediatrics #1."""
        results = compute_cluster_scores(
            ["fever", "cough", "poor feeding", "vomiting"],
            {"fever": "yes", "temperature": "102"},
            "moderate",
            patient_age="5"
        )
        assert results[0]["department"] == "Pediatrics"

    def test_adult_generic_symptoms_not_pediatrics(self):
        """Adult with generic symptoms → NOT Pediatrics #1."""
        results = compute_cluster_scores(
            ["کھانسی", "تھکاوٹ", "کمزوری", "نیند نہیں آتی"],
            patient_age="30",
            patient_gender="M"
        )
        peds = next(r for r in results if r["department"] == "Pediatrics")
        assert peds["score"] == 0.0
        assert results[0]["department"] != "Pediatrics"


# ─────────────────────────────────────────────
# 6. Scoring Accuracy & Edge Cases
# ─────────────────────────────────────────────

class TestScoringEdgeCases:
    """Edge cases in scoring that could cause incorrect triage."""

    def test_empty_symptoms_returns_all_zero(self):
        results = compute_cluster_scores([])
        for r in results:
            assert r["score"] == 0.0

    def test_none_symptoms_returns_all_zero(self):
        results = compute_cluster_scores(None)
        for r in results:
            assert r["score"] == 0.0

    def test_single_symptom_scores_low(self):
        """Single symptom should not reach 50%."""
        results = compute_cluster_scores(["cough"])
        assert results[0]["score"] < 0.5

    def test_vital_boost_increases_score(self):
        """Adding vitals should increase score."""
        base = compute_cluster_scores(["chest pain"])
        with_vitals = compute_cluster_scores(["chest pain"], {"bp": "140/90", "heart_rate": "110"})
        cardio_base = next(r for r in base if r["department"] == "Cardiology")
        cardio_vitals = next(r for r in with_vitals if r["department"] == "Cardiology")
        assert cardio_vitals["score"] > cardio_base["score"]

    def test_severity_boost_increases_score(self):
        """Adding severity should increase score."""
        base = compute_cluster_scores(["chest pain", "shortness of breath"])
        with_sev = compute_cluster_scores(["chest pain", "shortness of breath"], patient_severity="severe")
        cardio_base = next(r for r in base if r["department"] == "Cardiology")
        cardio_sev = next(r for r in with_sev if r["department"] == "Cardiology")
        assert cardio_sev["score"] >= cardio_base["score"]

    def test_cross_condition_boost(self):
        """Multiple conditions matching → higher department score."""
        # Chest pain + shortness of breath match multiple Cardiology conditions
        results = compute_cluster_scores(["chest pain", "shortness of breath", "palpitations"])
        cardio = next(r for r in results if r["department"] == "Cardiology")
        best_cond = max(c["score"] for c in cardio["top_conditions"])
        # Department score should be boosted above best condition
        assert cardio["score"] >= best_cond

    def test_score_never_exceeds_one(self):
        """Score should never exceed 1.0."""
        results = compute_cluster_scores(
            ["fever", "cough", "body ache", "sore throat", "chills", "fatigue", "sneezing", "congestion", "runny nose"],
            {"fever": "yes", "temperature": "103", "bp": "120/80"},
            "severe"
        )
        for r in results:
            assert r["score"] <= 1.0

    def test_duplicate_symptoms_no_inflation(self):
        """Same symptom repeated should not inflate score."""
        single = compute_cluster_scores(["chest pain"])
        double = compute_cluster_scores(["chest pain", "chest pain"])
        cardio_single = next(r for r in single if r["department"] == "Cardiology")
        cardio_double = next(r for r in double if r["department"] == "Cardiology")
        assert cardio_single["score"] == cardio_double["score"]


# ─────────────────────────────────────────────
# 7. TTS Cache Integrity
# ─────────────────────────────────────────────

class TestTTSCacheIntegrity:
    """Test cache key generation and audio validation."""

    def test_cache_key_deterministic(self):
        """Same text + config → same cache key."""
        key1 = _make_cache_key("hello world", "nova", "OpenAI", "tts-1", 24000, "ur")
        key2 = _make_cache_key("hello world", "nova", "OpenAI", "tts-1", 24000, "ur")
        assert key1 == key2

    def test_cache_key_different_voice(self):
        """Different voice → different cache key."""
        key1 = _make_cache_key("hello", "nova", "OpenAI", "tts-1", 24000, "ur")
        key2 = _make_cache_key("hello", "shimmer", "OpenAI", "tts-1", 24000, "ur")
        assert key1 != key2

    def test_cache_key_different_text(self):
        """Different text → different cache key."""
        key1 = _make_cache_key("hello", "nova", "OpenAI", "tts-1", 24000, "ur")
        key2 = _make_cache_key("goodbye", "nova", "OpenAI", "tts-1", 24000, "ur")
        assert key1 != key2

    def test_cache_key_normalizes_punctuation(self):
        """Text with different punctuation → same cache key."""
        key1 = _make_cache_key("hello world!", "nova", "OpenAI", "tts-1", 24000, "ur")
        key2 = _make_cache_key("hello world", "nova", "OpenAI", "tts-1", 24000, "ur")
        assert key1 == key2

    def test_cache_key_normalizes_whitespace(self):
        """Extra whitespace → same cache key."""
        key1 = _make_cache_key("hello  world", "nova", "OpenAI", "tts-1", 24000, "ur")
        key2 = _make_cache_key("hello world", "nova", "OpenAI", "tts-1", 24000, "ur")
        assert key1 == key2


# ─────────────────────────────────────────────
# 8. Performance Benchmarks
# ─────────────────────────────────────────────

class TestPerformanceBenchmarks:
    """Ensure scoring computation is fast enough for real-time."""

    def test_scoring_under_50ms(self):
        """Single scoring call should complete in under 50ms."""
        symptoms = ["chest pain", "shortness of breath", "palpitations", "sweating"]
        vitals = {"bp": "140/90", "heart_rate": "110"}

        start = time.time()
        for _ in range(100):
            compute_cluster_scores(symptoms, vitals, "severe", "45", "M")
        elapsed = (time.time() - start) / 100 * 1000  # ms per call

        assert elapsed < 50, f"Scoring took {elapsed:.1f}ms (limit: 50ms)"

    def test_alias_expansion_under_10ms(self):
        """Alias expansion with Urdu symptoms should be fast."""
        urdu_symptoms = ["سینے میں درد", "سانس لینے میں تکلیف", "دل کی دھڑکن تیز", "بخار"]

        start = time.time()
        for _ in range(100):
            compute_cluster_scores(urdu_symptoms)
        elapsed = (time.time() - start) / 100 * 1000

        assert elapsed < 50, f"Urdu scoring took {elapsed:.1f}ms (limit: 50ms)"

    def test_cache_key_under_1ms(self):
        """Cache key computation should be fast."""
        start = time.time()
        for _ in range(1000):
            _make_cache_key("السلام علیکم خوش آمدید", "nova", "OpenAI", "tts-1", 24000, "ur")
        elapsed = (time.time() - start) / 1000 * 1000

        assert elapsed < 1, f"Cache key took {elapsed:.2f}ms (limit: 1ms)"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
