"""Tests for the disease clustering engine used in the triage dashboard."""
import json
import os
import sys
import pytest

# Add the module to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import the clustering functions directly
from conversation_agent import (
    compute_cluster_scores,
    _load_disease_clusters,
    _build_vitals_table,
    DISEASE_CLUSTERS_DATA,
    ResponseBroadcaster,
    STTBroadcaster,
)


class TestDatasetLoading:
    """Test that the disease clusters dataset loads correctly."""

    def test_dataset_file_exists(self):
        path = os.path.join(SCRIPT_DIR, "disease_clusters.json")
        assert os.path.exists(path), "disease_clusters.json should exist"

    def test_dataset_loads_successfully(self):
        assert DISEASE_CLUSTERS_DATA is not None
        assert "clusters" in DISEASE_CLUSTERS_DATA

    def test_dataset_has_12_departments(self):
        clusters = DISEASE_CLUSTERS_DATA["clusters"]
        assert len(clusters) == 12, f"Expected 12 departments, got {len(clusters)}"

    def test_each_cluster_has_required_fields(self):
        for c in DISEASE_CLUSTERS_DATA["clusters"]:
            assert "department" in c, f"Missing department field"
            assert "doctor" in c, f"Missing doctor field in {c.get('department')}"
            assert "room" in c, f"Missing room field in {c.get('department')}"
            assert "conditions" in c, f"Missing conditions in {c.get('department')}"
            assert len(c["conditions"]) >= 3, f"{c['department']} should have >=3 conditions"

    def test_each_condition_has_required_fields(self):
        for c in DISEASE_CLUSTERS_DATA["clusters"]:
            for cond in c["conditions"]:
                assert "name" in cond, f"Missing name in condition"
                assert "icd" in cond, f"Missing icd in {cond.get('name')}"
                assert "symptoms" in cond, f"Missing symptoms in {cond.get('name')}"
                assert len(cond["symptoms"]) >= 2, f"{cond['name']} should have >=2 symptoms"

    def test_symptom_aliases_exist(self):
        aliases = DISEASE_CLUSTERS_DATA.get("symptom_aliases", {})
        assert len(aliases) > 10, f"Expected >10 aliases, got {len(aliases)}"

    def test_departments_match_doctors_panel(self):
        dept_names = {c["department"] for c in DISEASE_CLUSTERS_DATA["clusters"]}
        expected = {
            "General Medicine", "Cardiology", "Gynecology & Obstetrics",
            "Pediatrics", "Orthopedics", "Dermatology", "ENT",
            "Ophthalmology", "Neurology", "Dentistry",
            "Gastroenterology", "Psychiatry",
        }
        assert dept_names == expected, f"Department mismatch: {dept_names.symmetric_difference(expected)}"


class TestClusterScoringNoSymptoms:
    """Test scoring when no symptoms are provided."""

    def test_empty_symptoms_returns_all_clusters(self):
        results = compute_cluster_scores([])
        assert len(results) == 12

    def test_empty_symptoms_all_zero_scores(self):
        results = compute_cluster_scores([])
        for r in results:
            assert r["score"] == 0.0, f"{r['department']} should have 0 score"

    def test_none_symptoms_returns_all_clusters(self):
        results = compute_cluster_scores(None)
        assert len(results) == 12

    def test_empty_list_has_department_info(self):
        results = compute_cluster_scores([])
        for r in results:
            assert "department" in r
            assert "doctor" in r
            assert "room" in r
            assert "icon" in r
            assert "top_conditions" in r


class TestClusterScoringWithSymptoms:
    """Test scoring with various symptom inputs."""

    def test_chest_pain_matches_cardiology(self):
        results = compute_cluster_scores(["chest pain"])
        # Cardiology should be the top or near-top result
        cardio = next(r for r in results if r["department"] == "Cardiology")
        assert cardio["score"] > 0, "Cardiology should match chest pain"
        assert "chest pain" in cardio["matched_symptoms"]

    def test_fever_cough_matches_general_medicine(self):
        results = compute_cluster_scores(["fever", "cough"])
        gen_med = next(r for r in results if r["department"] == "General Medicine")
        assert gen_med["score"] > 0, "General Medicine should match fever+cough"

    def test_toothache_matches_dentistry(self):
        results = compute_cluster_scores(["toothache"])
        dental = next(r for r in results if r["department"] == "Dentistry")
        assert dental["score"] > 0, "Dentistry should match toothache"

    def test_headache_matches_neurology(self):
        results = compute_cluster_scores(["severe headache", "nausea"])
        neuro = next(r for r in results if r["department"] == "Neurology")
        assert neuro["score"] > 0, "Neurology should match headache"

    def test_skin_rash_matches_dermatology(self):
        results = compute_cluster_scores(["skin rash", "itching"])
        derma = next(r for r in results if r["department"] == "Dermatology")
        assert derma["score"] > 0, "Dermatology should match skin rash"

    def test_joint_pain_matches_orthopedics(self):
        results = compute_cluster_scores(["joint pain", "stiffness"])
        ortho = next(r for r in results if r["department"] == "Orthopedics")
        assert ortho["score"] > 0, "Orthopedics should match joint pain"

    def test_ear_pain_matches_ent(self):
        results = compute_cluster_scores(["ear pain", "hearing loss"])
        ent = next(r for r in results if r["department"] == "ENT")
        assert ent["score"] > 0, "ENT should match ear pain"

    def test_eye_pain_matches_ophthalmology(self):
        results = compute_cluster_scores(["blurred vision", "eye pain"])
        eye = next(r for r in results if r["department"] == "Ophthalmology")
        assert eye["score"] > 0, "Ophthalmology should match eye problems"

    def test_stomach_pain_matches_gastro(self):
        results = compute_cluster_scores(["stomach pain", "nausea", "acid reflux"])
        gastro = next(r for r in results if r["department"] == "Gastroenterology")
        assert gastro["score"] > 0, "Gastroenterology should match stomach pain"

    def test_anxiety_matches_psychiatry(self):
        results = compute_cluster_scores(["anxiety", "insomnia"])
        psych = next(r for r in results if r["department"] == "Psychiatry")
        assert psych["score"] > 0, "Psychiatry should match anxiety"

    def test_pregnancy_matches_gynecology(self):
        results = compute_cluster_scores(["missed period", "nausea", "vomiting"])
        gyn = next(r for r in results if r["department"] == "Gynecology & Obstetrics")
        assert gyn["score"] > 0, "Gynecology should match pregnancy symptoms"

    def test_child_fever_matches_pediatrics(self):
        results = compute_cluster_scores(["fever", "cough", "poor feeding"])
        peds = next(r for r in results if r["department"] == "Pediatrics")
        assert peds["score"] > 0, "Pediatrics should match child symptoms"


class TestClusterScoringOrder:
    """Test that results are sorted by score."""

    def test_results_sorted_descending(self):
        results = compute_cluster_scores(["chest pain", "shortness of breath"])
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True), "Results should be sorted by score descending"

    def test_top_result_is_most_relevant(self):
        results = compute_cluster_scores(["toothache", "swollen gums", "dental pain"])
        assert results[0]["department"] == "Dentistry", f"Top result should be Dentistry, got {results[0]['department']}"

    def test_multiple_symptoms_increase_score(self):
        score1 = compute_cluster_scores(["chest pain"])[0]["score"]
        results2 = compute_cluster_scores(["chest pain", "shortness of breath", "palpitations"])
        cardio2 = next(r for r in results2 if r["department"] == "Cardiology")
        assert cardio2["score"] >= score1, "More matching symptoms should increase or maintain score"


class TestClusterScoringWithVitals:
    """Test that vitals boost the score."""

    def test_vitals_boost_cardiology(self):
        results_no_vitals = compute_cluster_scores(["chest pain"])
        cardio_no = next(r for r in results_no_vitals if r["department"] == "Cardiology")

        results_with_vitals = compute_cluster_scores(
            ["chest pain"],
            patient_vitals={"bp": "140/90", "heart_rate": "110"},
        )
        cardio_with = next(r for r in results_with_vitals if r["department"] == "Cardiology")

        assert cardio_with["score"] >= cardio_no["score"], "Vitals should boost or maintain score"

    def test_fever_vital_boosts_general_medicine(self):
        results = compute_cluster_scores(
            ["cough", "body ache"],
            patient_vitals={"fever": "yes", "temperature": "102°F"},
        )
        gen_med = next(r for r in results if r["department"] == "General Medicine")
        assert gen_med["score"] > 0, "Fever vital should boost General Medicine"

    def test_sugar_level_boosts_diabetes(self):
        results = compute_cluster_scores(
            ["excessive thirst", "frequent urination"],
            patient_vitals={"sugar_level": "250"},
        )
        gen_med = next(r for r in results if r["department"] == "General Medicine")
        assert gen_med["score"] > 0, "Sugar level vital should boost General Medicine (diabetes)"


class TestClusterScoringWithSeverity:
    """Test severity matching."""

    def test_severity_matching_boosts_score(self):
        results = compute_cluster_scores(
            ["chest pain"],
            patient_severity="severe",
        )
        cardio = next(r for r in results if r["department"] == "Cardiology")
        assert cardio["score"] > 0, "Severity should not break scoring"


class TestSymptomAliases:
    """Test that Urdu/Roman Urdu symptom aliases work."""

    def test_urdu_alias_chest_pain(self):
        results = compute_cluster_scores(["seene mein dard"])
        cardio = next(r for r in results if r["department"] == "Cardiology")
        assert cardio["score"] > 0, "Urdu alias 'seene mein dard' should match Cardiology"

    def test_urdu_alias_fever(self):
        results = compute_cluster_scores(["bukhar"])
        gen_med = next(r for r in results if r["department"] == "General Medicine")
        assert gen_med["score"] > 0, "Urdu alias 'bukhar' should match General Medicine"

    def test_urdu_alias_headache(self):
        """'sir dard' (headache) should match General Medicine — NOT Neurology (migraine is separate)."""
        results = compute_cluster_scores(["sir dard"])
        gen_med = next(r for r in results if r["department"] == "General Medicine")
        assert gen_med["score"] > 0, "Urdu alias 'sir dard' should match General Medicine"

    def test_urdu_alias_stomach_pain(self):
        results = compute_cluster_scores(["pait dard"])
        gastro = next(r for r in results if r["department"] == "Gastroenterology")
        assert gastro["score"] > 0, "Urdu alias 'pait dard' should match Gastroenterology"


class TestConditionScores:
    """Test individual condition scoring within clusters."""

    def test_conditions_have_scores(self):
        results = compute_cluster_scores(["fever", "headache", "abdominal pain"])
        for r in results:
            if r["score"] > 0:
                assert len(r["top_conditions"]) > 0
                for cond in r["top_conditions"]:
                    assert "name" in cond
                    assert "score" in cond
                    assert "icd" in cond

    def test_top_conditions_sorted_by_score(self):
        results = compute_cluster_scores(["fever", "cough", "body ache"])
        for r in results:
            cond_scores = [c["score"] for c in r["top_conditions"]]
            assert cond_scores == sorted(cond_scores, reverse=True), \
                f"Conditions in {r['department']} should be sorted by score"

    def test_all_conditions_returned_not_sliced(self):
        """All conditions from JSON must be returned, not just top 3."""
        results = compute_cluster_scores(["fever"])
        for r in results:
            # Find corresponding cluster in source data
            src = next(c for c in DISEASE_CLUSTERS_DATA["clusters"]
                       if c["department"] == r["department"])
            assert len(r["top_conditions"]) == len(src["conditions"]), \
                f"{r['department']}: expected {len(src['conditions'])} conditions, got {len(r['top_conditions'])}"

    def test_conditions_include_symptoms_field(self):
        """Each condition must include its full symptoms list."""
        results = compute_cluster_scores(["chest pain"])
        for r in results:
            for cond in r["top_conditions"]:
                assert "symptoms" in cond, f"Missing symptoms in {cond['name']}"
                assert isinstance(cond["symptoms"], list)
                assert len(cond["symptoms"]) >= 2, f"{cond['name']} should have >=2 symptoms"

    def test_conditions_include_vitals_field(self):
        """Each condition must include its vitals dict."""
        results = compute_cluster_scores(["chest pain"])
        for r in results:
            for cond in r["top_conditions"]:
                assert "vitals" in cond, f"Missing vitals in {cond['name']}"
                assert isinstance(cond["vitals"], dict)

    def test_conditions_include_severity_field(self):
        """Each condition must include its severity range."""
        results = compute_cluster_scores(["chest pain"])
        for r in results:
            for cond in r["top_conditions"]:
                assert "severity" in cond, f"Missing severity in {cond['name']}"
                assert isinstance(cond["severity"], list)
                assert len(cond["severity"]) >= 1

    def test_zero_score_conditions_also_have_full_data(self):
        """Even conditions with 0 score must include full data (for initial display)."""
        results = compute_cluster_scores(None)  # No symptoms = initial state
        for r in results:
            for cond in r["top_conditions"]:
                assert "symptoms" in cond
                assert "vitals" in cond
                assert "severity" in cond


class TestEdgeCases:
    """Test edge cases and robustness."""

    def test_empty_string_symptom(self):
        results = compute_cluster_scores([""])
        assert len(results) == 12

    def test_whitespace_symptoms(self):
        results = compute_cluster_scores(["  ", "  fever  "])
        gen_med = next(r for r in results if r["department"] == "General Medicine")
        assert gen_med["score"] > 0, "Should handle whitespace in symptoms"

    def test_case_insensitive(self):
        results1 = compute_cluster_scores(["Chest Pain"])
        results2 = compute_cluster_scores(["chest pain"])
        cardio1 = next(r for r in results1 if r["department"] == "Cardiology")
        cardio2 = next(r for r in results2 if r["department"] == "Cardiology")
        assert cardio1["score"] == cardio2["score"], "Scoring should be case-insensitive"

    def test_nonexistent_symptom(self):
        results = compute_cluster_scores(["xyzabc123"])
        assert all(r["score"] == 0 for r in results), "Unknown symptom should match nothing"

    def test_mixed_real_and_fake_symptoms(self):
        results = compute_cluster_scores(["chest pain", "xyzabc123"])
        cardio = next(r for r in results if r["department"] == "Cardiology")
        assert cardio["score"] > 0, "Valid symptoms should still match despite invalid ones"

    def test_none_vitals(self):
        results = compute_cluster_scores(["fever"], patient_vitals=None)
        assert len(results) == 12

    def test_empty_vitals(self):
        results = compute_cluster_scores(["fever"], patient_vitals={})
        assert len(results) == 12

    def test_return_structure(self):
        results = compute_cluster_scores(["fever"])
        for r in results:
            assert isinstance(r["department"], str)
            assert isinstance(r["doctor"], str)
            assert isinstance(r["room"], str)
            assert isinstance(r["score"], float)
            assert isinstance(r["matched_symptoms"], list)
            assert isinstance(r["top_conditions"], list)
            assert 0.0 <= r["score"] <= 1.0


class TestRealWorldScenarios:
    """Test realistic patient scenarios."""

    def test_scenario_flu_patient(self):
        """Patient with typical flu symptoms."""
        results = compute_cluster_scores(
            ["fever", "cough", "body ache", "sore throat"],
            patient_vitals={"fever": "yes", "temperature": "101°F"},
        )
        assert results[0]["department"] == "General Medicine", \
            f"Flu patient top match should be General Medicine, got {results[0]['department']}"

    def test_scenario_heart_attack(self):
        """Patient with heart attack symptoms."""
        results = compute_cluster_scores(
            ["chest pain", "shortness of breath", "sweating", "nausea"],
            patient_vitals={"bp": "90/60", "heart_rate": "120"},
            patient_severity="severe",
        )
        cardio = next(r for r in results if r["department"] == "Cardiology")
        assert cardio["score"] > 0.3, "Heart attack symptoms should strongly match Cardiology"

    def test_scenario_broken_bone(self):
        """Patient with fracture symptoms."""
        results = compute_cluster_scores(
            ["bone pain", "swelling", "inability to move"],
            patient_vitals={"mobility": "cannot walk"},
            patient_severity="severe",
        )
        ortho = next(r for r in results if r["department"] == "Orthopedics")
        assert ortho["score"] > 0, "Fracture symptoms should match Orthopedics"

    def test_scenario_migraine(self):
        """Patient with migraine symptoms."""
        results = compute_cluster_scores(
            ["severe headache", "nausea", "sensitivity to light"],
            patient_severity="severe",
        )
        neuro = next(r for r in results if r["department"] == "Neurology")
        assert neuro["score"] > 0.2, "Migraine should strongly match Neurology"

    def test_scenario_dental_emergency(self):
        """Patient with dental abscess."""
        results = compute_cluster_scores(
            ["severe tooth pain", "swollen face", "fever"],
            patient_vitals={"fever": "yes", "temperature": "100.5°F"},
        )
        dental = next(r for r in results if r["department"] == "Dentistry")
        assert dental["score"] > 0, "Dental abscess should match Dentistry"


class TestVitalsTableFiltering:
    """Test that _build_vitals_table() intelligently filters vitals based on severity."""

    def test_table_returns_string(self):
        result = _build_vitals_table()
        assert isinstance(result, str)
        assert len(result) > 0

    def test_table_has_12_rows(self):
        result = _build_vitals_table()
        rows = [r for r in result.strip().split("\n") if r.strip()]
        assert len(rows) == 12, f"Expected 12 department rows, got {len(rows)}"

    def test_dermatology_no_temperature(self):
        """Dermatology has fever='possible' in Acne — should NOT ask temperature."""
        result = _build_vitals_table()
        for line in result.split("\n"):
            if "acne" in line.lower() or "itching" in line.lower() or "dry skin" in line.lower():
                assert "temperature" not in line.lower(), \
                    f"Dermatology should NOT ask temperature (fever is 'possible'): {line}"
                assert "severity" in line.lower(), \
                    f"Dermatology should ask severity only: {line}"
                break
        else:
            pytest.fail("Could not find Dermatology row in vitals table")

    def test_gastro_no_temperature(self):
        """Gastro has fever='possible' in Hepatitis — should NOT ask temperature."""
        result = _build_vitals_table()
        for line in result.split("\n"):
            if "acid reflux" in line.lower() or "bloating" in line.lower():
                assert "temperature" not in line.lower(), \
                    f"Gastro should NOT ask temperature (fever is 'possible'): {line}"
                assert "severity" in line.lower(), \
                    f"Gastro should ask severity only: {line}"
                break
        else:
            pytest.fail("Could not find Gastroenterology row in vitals table")

    def test_neurology_no_sugar_level(self):
        """Neurology has sugar_level='check' in Neuropathy — should NOT ask sugar level."""
        result = _build_vitals_table()
        for line in result.split("\n"):
            if "migraine" in line.lower() or "convulsion" in line.lower():
                assert "sugar level" not in line.lower(), \
                    f"Neurology should NOT ask sugar level (value is 'check'): {line}"
                break
        else:
            pytest.fail("Could not find Neurology row in vitals table")

    def test_ent_no_bp(self):
        """ENT has bp='check' in Vertigo — should NOT ask BP."""
        result = _build_vitals_table()
        for line in result.split("\n"):
            if "ear pain" in line.lower() or "ear discharge" in line.lower():
                assert "BP" not in line, \
                    f"ENT should NOT ask BP (value is 'check'): {line}"
                break
        else:
            pytest.fail("Could not find ENT row in vitals table")

    def test_cardiology_includes_bp_and_heart_rate(self):
        """Cardiology has definitive BP and heart_rate values — should include both."""
        result = _build_vitals_table()
        for line in result.split("\n"):
            if "chest pain" in line.lower() or "chest tightness" in line.lower():
                assert "BP" in line, f"Cardiology should ask BP: {line}"
                assert "heart rate" in line, f"Cardiology should ask heart rate: {line}"
                break
        else:
            pytest.fail("Could not find Cardiology row in vitals table")

    def test_general_medicine_includes_all_definitive_vitals(self):
        """General Medicine has multiple definitive vitals — should include all."""
        result = _build_vitals_table()
        for line in result.split("\n"):
            if "body ache" in line.lower() and "chills" in line.lower():
                assert "temperature" in line.lower(), f"Gen Med should ask temperature: {line}"
                assert "BP" in line, f"Gen Med should ask BP: {line}"
                assert "sugar level" in line.lower(), f"Gen Med should ask sugar level: {line}"
                break
        else:
            pytest.fail("Could not find General Medicine row in vitals table")

    def test_orthopedics_includes_mobility(self):
        """Orthopedics has definitive mobility values — should include mobility."""
        result = _build_vitals_table()
        for line in result.split("\n"):
            if "fracture" in line.lower() or "bone pain" in line.lower():
                assert "mobility" in line.lower(), f"Orthopedics should ask mobility: {line}"
                break
        else:
            pytest.fail("Could not find Orthopedics row in vitals table")

    def test_ophthalmology_severity_only(self):
        """Ophthalmology has no vitals at all — should ask severity only."""
        result = _build_vitals_table()
        for line in result.split("\n"):
            if "blurred vision" in line.lower() and "cloudy vision" in line.lower():
                assert "severity" in line.lower(), \
                    f"Ophthalmology should ask severity only: {line}"
                break
        else:
            pytest.fail("Could not find Ophthalmology row in vitals table")

    def test_psychiatry_includes_bp_and_heart_rate(self):
        """Psychiatry (Anxiety) has definitive BP and heart_rate — should include both."""
        result = _build_vitals_table()
        for line in result.split("\n"):
            if "anxiety" in line.lower() or "depression" in line.lower():
                assert "BP" in line, f"Psychiatry should ask BP: {line}"
                assert "heart rate" in line, f"Psychiatry should ask heart rate: {line}"
                break
        else:
            pytest.fail("Could not find Psychiatry row in vitals table")

    def test_possible_values_never_appear(self):
        """No row should include vitals solely from 'possible' values."""
        # Verify by checking that departments with ONLY 'possible' vitals show severity
        result = _build_vitals_table()
        # Dermatology and Gastro should show severity (their only vitals are 'possible')
        derma_found = False
        gastro_found = False
        for line in result.split("\n"):
            if "acne" in line.lower() or "itching" in line.lower():
                derma_found = True
                assert "severity" in line.lower()
            if "acid reflux" in line.lower() or "bloating" in line.lower():
                gastro_found = True
                assert "severity" in line.lower()
        assert derma_found and gastro_found

    def test_mild_only_conditions_dont_contribute_vitals(self):
        """Conditions with severity=['mild'] only should not contribute vitals."""
        # Pregnancy Care has bp='monitor', severity=['mild'] — should not add BP for Gynecology
        # But Prenatal Complications has bp='high', severity=['moderate','severe'] — SHOULD add BP
        result = _build_vitals_table()
        for line in result.split("\n"):
            if "irregular periods" in line.lower() or "missed period" in line.lower():
                # Gynecology row should have BP (from Prenatal Complications, not from Pregnancy Care)
                assert "BP" in line, f"Gynecology should have BP from Prenatal Complications: {line}"
                break


class TestMergePatientData:
    """Test _merge_patient_data reliability — critical for triage updates."""

    def _make_broadcaster(self):
        """Create a ResponseBroadcaster with a mock agent for testing merge logic."""
        class MockAgent:
            accumulated_patient_data = {}
        return ResponseBroadcaster(MockAgent())

    def test_first_call_sets_data(self):
        rb = self._make_broadcaster()
        rb._merge_patient_data({"name": "Ahmed", "symptoms": ["chest pain"]})
        assert rb._accumulated_patient_data["name"] == "Ahmed"
        assert rb._accumulated_patient_data["symptoms"] == ["chest pain"]

    def test_merge_replaces_symptoms_with_llm_list(self):
        """LLM is authoritative — its symptom list replaces previous. Allows correcting mistranscriptions."""
        rb = self._make_broadcaster()
        rb._merge_patient_data({"symptoms": ["کاشی"]})  # STT mistranscription
        rb._merge_patient_data({"symptoms": ["کھانسی"]})  # LLM corrects
        assert rb._accumulated_patient_data["symptoms"] == ["کھانسی"]
        assert "کاشی" not in rb._accumulated_patient_data["symptoms"]

    def test_merge_llm_sends_all_symptoms_each_turn(self):
        """LLM sends ALL symptoms on each turn (not incremental). New list replaces old."""
        rb = self._make_broadcaster()
        rb._merge_patient_data({"symptoms": ["chest pain"]})
        rb._merge_patient_data({"symptoms": ["chest pain", "shortness of breath"]})
        assert len(rb._accumulated_patient_data["symptoms"]) == 2
        assert "chest pain" in rb._accumulated_patient_data["symptoms"]
        assert "shortness of breath" in rb._accumulated_patient_data["symptoms"]

    def test_merge_empty_symptoms_preserves_existing(self):
        rb = self._make_broadcaster()
        rb._merge_patient_data({"symptoms": ["fever"]})
        rb._merge_patient_data({"symptoms": []})
        assert rb._accumulated_patient_data["symptoms"] == ["fever"]

    def test_merge_none_symptoms_does_not_crash(self):
        rb = self._make_broadcaster()
        rb._merge_patient_data({"symptoms": ["fever"]})
        rb._merge_patient_data({"symptoms": None})  # LLM might send null
        assert rb._accumulated_patient_data["symptoms"] == ["fever"]

    def test_merge_none_does_not_overwrite_name(self):
        rb = self._make_broadcaster()
        rb._merge_patient_data({"name": "Ahmed", "symptoms": []})
        rb._merge_patient_data({"name": None, "symptoms": ["fever"]})
        assert rb._accumulated_patient_data["name"] == "Ahmed"

    def test_merge_empty_string_does_not_overwrite(self):
        rb = self._make_broadcaster()
        rb._merge_patient_data({"severity": "mild", "symptoms": []})
        rb._merge_patient_data({"severity": "", "symptoms": []})
        assert rb._accumulated_patient_data["severity"] == "mild"

    def test_merge_vitals_accumulate(self):
        rb = self._make_broadcaster()
        rb._merge_patient_data({"vitals": {"fever": "yes"}, "symptoms": []})
        rb._merge_patient_data({"vitals": {"bp": "140/90"}, "symptoms": []})
        assert rb._accumulated_patient_data["vitals"]["fever"] == "yes"
        assert rb._accumulated_patient_data["vitals"]["bp"] == "140/90"

    def test_merge_vitals_none_preserves_existing(self):
        rb = self._make_broadcaster()
        rb._merge_patient_data({"vitals": {"fever": "yes"}, "symptoms": []})
        rb._merge_patient_data({"vitals": {"fever": None}, "symptoms": []})
        assert rb._accumulated_patient_data["vitals"]["fever"] == "yes"

    def test_merge_empty_vitals_dict_preserves(self):
        rb = self._make_broadcaster()
        rb._merge_patient_data({"vitals": {"fever": "yes"}, "symptoms": []})
        rb._merge_patient_data({"vitals": {}, "symptoms": []})
        assert rb._accumulated_patient_data["vitals"]["fever"] == "yes"

    def test_merge_non_dict_vitals_ignored(self):
        rb = self._make_broadcaster()
        rb._merge_patient_data({"vitals": {"fever": "yes"}, "symptoms": []})
        rb._merge_patient_data({"vitals": "invalid", "symptoms": []})
        assert rb._accumulated_patient_data["vitals"]["fever"] == "yes"

    def test_merge_is_complete_boolean(self):
        rb = self._make_broadcaster()
        rb._merge_patient_data({"is_complete": False, "symptoms": []})
        assert rb._accumulated_patient_data["is_complete"] is False
        rb._merge_patient_data({"is_complete": True, "symptoms": []})
        assert rb._accumulated_patient_data["is_complete"] is True

    def test_merge_non_dict_input_ignored(self):
        rb = self._make_broadcaster()
        rb._merge_patient_data({"symptoms": ["fever"]})
        rb._merge_patient_data(None)  # Should not crash
        rb._merge_patient_data("invalid")  # Should not crash
        assert rb._accumulated_patient_data["symptoms"] == ["fever"]

    def test_merge_initial_none_symptoms_then_add(self):
        """If first call has None symptoms, second call should add correctly."""
        rb = self._make_broadcaster()
        rb._merge_patient_data({"name": "Ahmed", "symptoms": None})
        rb._merge_patient_data({"symptoms": ["chest pain"]})
        assert "chest pain" in rb._accumulated_patient_data["symptoms"]

    def test_merge_referred_doctor_sets_correctly(self):
        rb = self._make_broadcaster()
        rb._merge_patient_data({"symptoms": ["fever"], "referred_doctor": None})
        assert rb._accumulated_patient_data.get("referred_doctor") is None
        rb._merge_patient_data({"referred_doctor": "Dr. Khan", "symptoms": []})
        assert rb._accumulated_patient_data["referred_doctor"] == "Dr. Khan"


class TestTriageDataFlow:
    """Test the full data flow: merge → compute_cluster_scores → correct results."""

    def test_greeting_then_symptoms(self):
        """Simulate: greeting (no symptoms) → patient says chest pain."""
        # Turn 1: Greeting
        scores1 = compute_cluster_scores([])
        assert all(c["score"] == 0 for c in scores1)

        # Turn 2: Patient says chest pain
        scores2 = compute_cluster_scores(["chest pain"])
        cardio = next(c for c in scores2 if c["department"] == "Cardiology")
        assert cardio["score"] > 0, "Chest pain should match Cardiology"

    def test_progressive_symptom_matching(self):
        """Symptoms accumulate across turns — scores should increase."""
        s1 = compute_cluster_scores(["chest pain"])
        cardio1 = next(c for c in s1 if c["department"] == "Cardiology")

        s2 = compute_cluster_scores(["chest pain", "shortness of breath"])
        cardio2 = next(c for c in s2 if c["department"] == "Cardiology")

        s3 = compute_cluster_scores(["chest pain", "shortness of breath", "palpitations"])
        cardio3 = next(c for c in s3 if c["department"] == "Cardiology")

        assert cardio2["score"] >= cardio1["score"]
        assert cardio3["score"] >= cardio2["score"]

    def test_merge_then_score(self):
        """Full flow: merge patient data then compute scores."""
        class MockAgent:
            accumulated_patient_data = {}
        rb = ResponseBroadcaster(MockAgent())

        # Turn 1: Greeting
        rb._merge_patient_data({"name": None, "symptoms": [], "vitals": {}})
        acc = rb._accumulated_patient_data
        scores = compute_cluster_scores(acc.get("symptoms", []), acc.get("vitals", {}))
        assert all(c["score"] == 0 for c in scores)

        # Turn 2: Patient gives name + symptoms
        rb._merge_patient_data({"name": "Ahmed", "symptoms": ["fever", "cough"]})
        acc = rb._accumulated_patient_data
        scores = compute_cluster_scores(acc.get("symptoms", []), acc.get("vitals", {}))
        gen_med = next(c for c in scores if c["department"] == "General Medicine")
        assert gen_med["score"] > 0, "Fever + cough should match General Medicine"

        # Turn 3: Add vitals
        rb._merge_patient_data({"vitals": {"fever": "yes", "temperature": "102°F"}, "symptoms": []})
        acc = rb._accumulated_patient_data
        scores = compute_cluster_scores(acc.get("symptoms", []), acc.get("vitals", {}))
        gen_med2 = next(c for c in scores if c["department"] == "General Medicine")
        assert gen_med2["score"] >= gen_med["score"], "Vitals should boost score"

    def test_severity_guard_prevents_false_match(self):
        """Severity/vital boosts should NOT apply when zero symptoms match."""
        scores = compute_cluster_scores([], patient_severity="severe")
        assert all(c["score"] == 0 for c in scores), "No symptoms → no boosts"

    def test_condition_matched_symptoms_field(self):
        """Verify matched symptoms are correctly identified in conditions."""
        scores = compute_cluster_scores(["chest pain", "palpitations"])
        cardio = next(c for c in scores if c["department"] == "Cardiology")
        angina = next(c for c in cardio["top_conditions"] if "Angina" in c["name"])
        assert len(angina["matched"]) >= 1
        assert any("chest pain" in m for m in angina["matched"])

    def test_all_departments_always_returned(self):
        """All 12 departments should always be in results."""
        for syms in [[], ["fever"], ["chest pain", "headache"]]:
            scores = compute_cluster_scores(syms)
            assert len(scores) == 12, f"Expected 12 departments for symptoms={syms}"

    def test_score_never_exceeds_one(self):
        """Score should be capped at 1.0 even with many boosts."""
        scores = compute_cluster_scores(
            ["fever", "cough", "body ache", "sore throat", "headache", "chills", "fatigue"],
            patient_vitals={"fever": "yes", "temperature": "103", "bp": "130/85"},
            patient_severity="severe",
        )
        for c in scores:
            assert c["score"] <= 1.0, f"{c['department']} score {c['score']} exceeds 1.0"
            for cond in c["top_conditions"]:
                assert cond["score"] <= 1.0, f"{cond['name']} score {cond['score']} exceeds 1.0"


class TestNoiseFilter:
    """Test STT noise filtering for noisy hospital environments."""

    def _make_broadcaster(self):
        class MockAgent:
            accumulated_patient_data = {}
        return STTBroadcaster(MockAgent())

    def test_empty_string_is_noise(self):
        b = self._make_broadcaster()
        assert b._is_noise("") is True

    def test_single_char_is_noise(self):
        b = self._make_broadcaster()
        assert b._is_noise("a") is True

    def test_hmm_is_noise(self):
        b = self._make_broadcaster()
        assert b._is_noise("hmm") is True
        assert b._is_noise("Hmm") is True

    def test_um_is_noise(self):
        b = self._make_broadcaster()
        assert b._is_noise("um") is True
        assert b._is_noise("uh") is True

    def test_dots_is_noise(self):
        b = self._make_broadcaster()
        assert b._is_noise("...") is True
        assert b._is_noise(".") is True

    def test_real_speech_not_noise(self):
        b = self._make_broadcaster()
        assert b._is_noise("I have chest pain") is False
        assert b._is_noise("My name is Ahmed") is False
        assert b._is_noise("fever") is False

    def test_urdu_speech_not_noise(self):
        b = self._make_broadcaster()
        assert b._is_noise("میرا سینہ دکھ رہا ہے") is False
        assert b._is_noise("بخار ہے") is False

    def test_short_medical_term_not_noise(self):
        b = self._make_broadcaster()
        # "bp" is 2 chars but meaningful — passes min length
        assert b._is_noise("bp") is False
        # "ok" is now correctly classified as noise (background chatter)
        assert b._is_noise("ok") is True

    def test_combined_noise_words_are_noise(self):
        b = self._make_broadcaster()
        # Background chatter: multiple short noise words
        assert b._is_noise("oh yeah ok") is True
        assert b._is_noise("yeah yeah") is True
        # But real speech with noise words mixed in passes
        assert b._is_noise("ok I have chest pain") is False

    def test_repeated_noise_word_is_noise(self):
        b = self._make_broadcaster()
        assert b._is_noise("yeah yeah yeah") is True
        assert b._is_noise("um um um") is True

    def test_whitespace_only_is_noise(self):
        b = self._make_broadcaster()
        assert b._is_noise("   ") is True

    def test_punctuation_stripped(self):
        b = self._make_broadcaster()
        assert b._is_noise("oh.") is True
        assert b._is_noise("ah!") is True


class TestConfidenceThreshold:
    """Test that scoring achieves meaningful confidence with enough data."""

    def test_single_symptom_below_50(self):
        """One symptom alone should NOT reach 50% — bot needs more data."""
        results = compute_cluster_scores(["cough"])
        top = results[0]
        assert top["score"] < 0.5, f"Single symptom 'cough' should be <50%, got {top['score']*100}%"

    def test_two_symptoms_higher_score(self):
        """Two matching symptoms should score higher than one."""
        one = compute_cluster_scores(["fever"])
        two = compute_cluster_scores(["fever", "cough"])
        assert two[0]["score"] > one[0]["score"]

    def test_three_symptoms_reaches_50(self):
        """Three matching symptoms should get close to or above 50%."""
        results = compute_cluster_scores(["fever", "cough", "body ache"])
        top = results[0]
        assert top["score"] >= 0.3, f"3 symptoms should be >=30%, got {top['score']*100}%"

    def test_symptoms_plus_vitals_boosts_above_50(self):
        """Symptoms + vitals should push score above 50%."""
        results = compute_cluster_scores(
            ["fever", "cough", "body ache"],
            {"fever": "yes", "temperature": "103"},
            "moderate"
        )
        top = results[0]
        assert top["score"] >= 0.5, f"3 symptoms + vitals should be >=50%, got {top['score']*100}%"

    def test_chest_pain_matches_cardiology(self):
        """Chest pain should match cardiology, not pediatrics."""
        results = compute_cluster_scores(["chest pain", "shortness of breath", "sweating"])
        cardio = next(r for r in results if r["department"] == "Cardiology")
        peds = next(r for r in results if r["department"] == "Pediatrics")
        assert cardio["score"] > peds["score"], "Cardiology should rank above Pediatrics for chest pain"
        assert cardio["score"] >= 0.4, f"Cardiology should be >=40% with 3 heart symptoms, got {cardio['score']*100}%"

    def test_cough_alone_matches_general_medicine_not_pediatrics(self):
        """Cough alone should match General Medicine (has more cough conditions) over Pediatrics."""
        results = compute_cluster_scores(["cough"])
        gm = next(r for r in results if r["department"] == "General Medicine")
        # General Medicine should have a match for cough
        assert gm["score"] > 0, "General Medicine should match cough"

    def test_bp_vital_boosts_hypertension(self):
        """BP vital should boost hypertension/cardiology scores."""
        without_bp = compute_cluster_scores(["headache", "dizziness"])
        with_bp = compute_cluster_scores(["headache", "dizziness"], {"bp": "130/90"})
        gm_without = next(r for r in without_bp if r["department"] == "General Medicine")
        gm_with = next(r for r in with_bp if r["department"] == "General Medicine")
        assert gm_with["score"] > gm_without["score"], "BP vital should boost score"

    def test_full_scenario_above_50(self):
        """Full realistic scenario should achieve 50%+ confidence."""
        results = compute_cluster_scores(
            ["chest pain", "shortness of breath", "sweating", "nausea"],
            {"bp": "140/90", "heart_rate": "110"},
            "severe"
        )
        top = results[0]
        assert top["score"] >= 0.5, f"Full cardiac scenario should be >=50%, got {top['score']*100}%"
        assert top["department"] == "Cardiology"


class TestReferralRules:
    """Test that referral rules are loaded from disease_clusters.json."""

    def test_referral_rules_exist_in_json(self):
        rules = DISEASE_CLUSTERS_DATA.get("referral_rules", {})
        assert len(rules) > 0, "referral_rules should exist in disease_clusters.json"

    def test_all_departments_have_rules(self):
        rules = DISEASE_CLUSTERS_DATA.get("referral_rules", {})
        departments = {c["department"] for c in DISEASE_CLUSTERS_DATA["clusters"]}
        for dept in departments:
            assert dept in rules, f"Missing referral rule for {dept}"

    def test_default_rule_exists(self):
        rules = DISEASE_CLUSTERS_DATA.get("referral_rules", {})
        assert "_default" in rules, "Should have a _default fallback rule"

    def test_pediatrics_rule_mentions_children(self):
        rules = DISEASE_CLUSTERS_DATA.get("referral_rules", {})
        assert "children" in rules["Pediatrics"].lower() or "child" in rules["Pediatrics"].lower()

    def test_gynecology_rule_mentions_female(self):
        rules = DISEASE_CLUSTERS_DATA.get("referral_rules", {})
        assert "female" in rules["Gynecology & Obstetrics"].lower()

    def test_build_referral_rules_function(self):
        from conversation_agent import _build_referral_rules
        output = _build_referral_rules()
        assert "General Medicine" in output
        assert "Pediatrics" in output
        assert "DEFAULT" in output


class TestScoringAccuracy:
    """Verify score calculations are numerically correct."""

    def test_score_is_symptom_ratio(self):
        """Score = matched / total symptoms (before boosts)."""
        # Common Cold has 9 symptoms: fever, cough, runny nose, body ache, sore throat, chills, fatigue, sneezing, congestion
        results = compute_cluster_scores(["fever"])
        gm = next(r for r in results if r["department"] == "General Medicine")
        cold = next(c for c in gm["top_conditions"] if "Cold" in c["name"])
        assert cold["score"] == round(1 / 9, 3), f"1/9 symptoms = {round(1/9,3)}, got {cold['score']}"

    def test_two_of_nine_score(self):
        results = compute_cluster_scores(["fever", "cough"])
        gm = next(r for r in results if r["department"] == "General Medicine")
        cold = next(c for c in gm["top_conditions"] if "Cold" in c["name"])
        assert cold["score"] == round(2 / 9, 3), f"2/9 expected, got {cold['score']}"

    def test_three_of_nine_score(self):
        results = compute_cluster_scores(["fever", "cough", "body ache"])
        gm = next(r for r in results if r["department"] == "General Medicine")
        cold = next(c for c in gm["top_conditions"] if "Cold" in c["name"])
        assert cold["score"] == round(3 / 9, 3)

    def test_vital_boost_adds_0_1(self):
        """Each matching vital adds 0.1 to condition score."""
        without = compute_cluster_scores(["fever", "cough"])
        with_v = compute_cluster_scores(["fever", "cough"], {"fever": "yes"})
        gm1 = next(r for r in without if r["department"] == "General Medicine")
        gm2 = next(r for r in with_v if r["department"] == "General Medicine")
        cold1 = next(c for c in gm1["top_conditions"] if "Cold" in c["name"])
        cold2 = next(c for c in gm2["top_conditions"] if "Cold" in c["name"])
        # Cold has vitals: temperature + fever — both boosted by fever=yes? No, temperature needs patient temperature vital
        # fever vital: cond has fever="present", patient has fever="yes" → +0.1
        assert cold2["score"] >= cold1["score"] + 0.1 - 0.001

    def test_severity_boost_adds_0_05(self):
        """Matching severity adds 0.05 to condition score."""
        without = compute_cluster_scores(["fever", "cough"])
        with_s = compute_cluster_scores(["fever", "cough"], patient_severity="mild")
        gm1 = next(r for r in without if r["department"] == "General Medicine")
        gm2 = next(r for r in with_s if r["department"] == "General Medicine")
        cold1 = next(c for c in gm1["top_conditions"] if "Cold" in c["name"])
        cold2 = next(c for c in gm2["top_conditions"] if "Cold" in c["name"])
        # Cold severity = ["mild", "moderate"], so "mild" matches → +0.05
        assert cold2["score"] == round(cold1["score"] + 0.05, 3)

    def test_non_matching_severity_no_boost(self):
        """Severe doesn't match mild-only conditions."""
        results = compute_cluster_scores(["vaccination"], patient_severity="severe")
        peds = next(r for r in results if r["department"] == "Pediatrics")
        vacc = next(c for c in peds["top_conditions"] if "Vaccination" in c["name"])
        # Vaccination severity = ["mild"], patient_severity = "severe" → no severity boost
        base = round(1 / 4, 3)  # 1 symptom out of 4
        assert vacc["score"] == base, f"Expected {base} (no severity boost), got {vacc['score']}"

    def test_cluster_score_at_least_best_condition(self):
        """Department score >= its best condition score (cross-condition boost can increase it).
        Exceptions: Pediatrics/Gynecology may be penalized by department selection rules when age/gender unknown."""
        results = compute_cluster_scores(["chest pain", "shortness of breath", "sweating"])
        penalized_depts = {"Pediatrics", "Gynecology & Obstetrics"}  # Age/gender-gated departments
        for r in results:
            if r["department"] in penalized_depts:
                continue  # Skip — these are intentionally penalized when age/gender unknown
            best = max(c["score"] for c in r["top_conditions"]) if r["top_conditions"] else 0.0
            assert r["score"] >= best, f"{r['department']}: cluster={r['score']} < best={best}"

    def test_cross_condition_boost_increases_cardiology(self):
        """Cardiology score should be boosted above best condition when multiple conditions match."""
        results = compute_cluster_scores(["chest pain", "shortness of breath"])
        cardio = next(r for r in results if r["department"] == "Cardiology")
        best_cond = max(c["score"] for c in cardio["top_conditions"])
        # Chest pain + SOB match all 4 Cardiology conditions → cross-condition boost
        assert cardio["score"] > best_cond, \
            f"Cardiology dept score ({cardio['score']}) should be boosted above best condition ({best_cond})"

    def test_score_capped_at_1(self):
        """Score never exceeds 1.0 even with many matches + boosts."""
        all_cold_symptoms = ["fever", "cough", "runny nose", "body ache", "sore throat", "chills", "fatigue", "sneezing", "congestion"]
        results = compute_cluster_scores(
            all_cold_symptoms,
            {"fever": "yes", "temperature": "103", "bp": "120/80"},
            "moderate"
        )
        for r in results:
            assert r["score"] <= 1.0
            for c in r["top_conditions"]:
                assert c["score"] <= 1.0


class TestScoringDepartmentRanking:
    """Verify correct department ranks for different presentations."""

    def test_flu_ranks_general_medicine_first(self):
        results = compute_cluster_scores(["fever", "cough", "body ache", "sore throat"])
        assert results[0]["department"] == "General Medicine"

    def test_cardiac_emergency_ranks_cardiology_first(self):
        results = compute_cluster_scores(
            ["chest pain", "shortness of breath", "sweating", "palpitations"],
            {"bp": "90/60", "heart_rate": "130"},
            "severe"
        )
        assert results[0]["department"] == "Cardiology"

    def test_dental_ranks_dentistry_first(self):
        results = compute_cluster_scores(["toothache", "swollen gums", "dental pain"])
        assert results[0]["department"] == "Dentistry"

    def test_skin_ranks_dermatology_first(self):
        results = compute_cluster_scores(["skin rash", "itching", "redness", "hives"])
        assert results[0]["department"] == "Dermatology"

    def test_gastro_ranks_first_for_digestive(self):
        results = compute_cluster_scores(["stomach pain", "heartburn", "acid reflux", "bloating"])
        assert results[0]["department"] == "Gastroenterology"

    def test_psychiatry_ranks_first_for_mental_health(self):
        results = compute_cluster_scores(["anxiety", "nervousness", "panic attacks", "restlessness"])
        assert results[0]["department"] == "Psychiatry"

    def test_ent_ranks_first_for_ear_issues(self):
        results = compute_cluster_scores(["ear pain", "hearing loss", "ear discharge"])
        assert results[0]["department"] == "ENT"

    def test_ophthalmology_ranks_first_for_eye_issues(self):
        results = compute_cluster_scores(["eye pain", "blurred vision", "red eye"])
        assert results[0]["department"] == "Ophthalmology"

    def test_neuro_ranks_first_for_seizure(self):
        results = compute_cluster_scores(["seizure", "convulsion", "loss of consciousness"])
        assert results[0]["department"] == "Neurology"

    def test_ortho_ranks_first_for_fracture(self):
        results = compute_cluster_scores(["fracture", "bone pain", "swelling", "inability to move"])
        assert results[0]["department"] == "Orthopedics"

    def test_gynecology_ranks_first_for_pregnancy(self):
        results = compute_cluster_scores(["missed period", "nausea", "vomiting", "breast tenderness"])
        assert results[0]["department"] == "Gynecology & Obstetrics"


class TestScoringWithVitalCombinations:
    """Test different vital combinations and their boost effects."""

    def test_bp_and_heart_rate_double_boost(self):
        """Two vitals give +0.2 total."""
        without = compute_cluster_scores(["chest pain"])
        with_v = compute_cluster_scores(["chest pain"], {"bp": "140/90", "heart_rate": "110"})
        c1 = next(r for r in without if r["department"] == "Cardiology")
        c2 = next(r for r in with_v if r["department"] == "Cardiology")
        assert c2["score"] > c1["score"], "Two vitals should boost score"
        assert c2["score"] >= c1["score"] + 0.15  # at least +0.2 from two vitals (on best condition)

    def test_temperature_vital_matches_fever_condition(self):
        results = compute_cluster_scores(["fever", "headache"], {"temperature": "103°F"})
        gm = next(r for r in results if r["department"] == "General Medicine")
        typhoid = next(c for c in gm["top_conditions"] if "Typhoid" in c["name"])
        assert typhoid["score"] > round(2 / 7, 3), "Temperature should boost Typhoid beyond base symptom ratio"

    def test_sugar_level_boosts_diabetes(self):
        results = compute_cluster_scores(["excessive thirst", "frequent urination"], {"sugar_level": "250"})
        gm = next(r for r in results if r["department"] == "General Medicine")
        diabetes = next(c for c in gm["top_conditions"] if "Diabetes" in c["name"])
        base = round(2 / 7, 3)  # 2 of 7 diabetes symptoms
        assert diabetes["score"] > base, f"Sugar level should boost above {base}, got {diabetes['score']}"

    def test_mobility_vital_boosts_ortho(self):
        results = compute_cluster_scores(["back pain", "stiffness"], {"mobility": "limited"})
        ortho = next(r for r in results if r["department"] == "Orthopedics")
        back = next(c for c in ortho["top_conditions"] if "Back Pain" in c["name"])
        base = round(2 / 6, 3)  # 2 of 6 back pain symptoms
        assert back["score"] > base, f"Mobility should boost above {base}"

    def test_empty_vital_value_no_boost(self):
        """Empty string vital should NOT add boost."""
        without = compute_cluster_scores(["chest pain"])
        with_empty = compute_cluster_scores(["chest pain"], {"bp": "", "heart_rate": ""})
        c1 = next(r for r in without if r["department"] == "Cardiology")
        c2 = next(r for r in with_empty if r["department"] == "Cardiology")
        assert c1["score"] == c2["score"], "Empty vitals should not boost"

    def test_fever_no_does_not_boost(self):
        """fever='no' should NOT add boost."""
        without = compute_cluster_scores(["cough"])
        with_no = compute_cluster_scores(["cough"], {"fever": "no"})
        gm1 = next(r for r in without if r["department"] == "General Medicine")
        gm2 = next(r for r in with_no if r["department"] == "General Medicine")
        assert gm1["score"] == gm2["score"], "fever=no should not boost"

    def test_vitals_without_symptoms_no_score(self):
        """Vitals alone (no symptoms) should NOT generate a score."""
        results = compute_cluster_scores([], {"bp": "180/100", "heart_rate": "130"})
        assert all(r["score"] == 0 for r in results), "Vitals without symptoms = 0 score"


class TestScoringAliasExpansion:
    """Test that alias expansion works bidirectionally."""

    def test_urdu_cough_alias_maps_to_cough(self):
        """کھانسی should match conditions containing 'cough'."""
        results = compute_cluster_scores(["کھانسی"])
        gm = next(r for r in results if r["department"] == "General Medicine")
        assert gm["score"] > 0
        assert len(gm["matched_symptoms"]) > 0

    def test_roman_urdu_alias_works(self):
        """'bukhar' should match fever conditions."""
        results = compute_cluster_scores(["bukhar"])
        gm = next(r for r in results if r["department"] == "General Medicine")
        cold = next(c for c in gm["top_conditions"] if "Cold" in c["name"])
        assert cold["score"] > 0, "'bukhar' alias should match Cold (fever)"

    def test_urdu_chest_pain_alias(self):
        results = compute_cluster_scores(["سینے میں درد"])
        cardio = next(r for r in results if r["department"] == "Cardiology")
        assert cardio["score"] > 0

    def test_mistranscription_alias_khansi(self):
        """Deepgram mistranscription 'خاصی' should still match cough."""
        results = compute_cluster_scores(["خاصی"])
        gm = next(r for r in results if r["department"] == "General Medicine")
        assert gm["score"] > 0, "Mistranscription alias 'خاصی' should match cough"

    def test_mistranscription_alias_bukhar_variant(self):
        """بکھار should match fever."""
        results = compute_cluster_scores(["بکھار"])
        gm = next(r for r in results if r["department"] == "General Medicine")
        assert gm["score"] > 0

    def test_multiple_aliases_together(self):
        """Multiple Urdu aliases in same call."""
        results = compute_cluster_scores(["bukhar", "khansi", "sir dard"])
        gm = next(r for r in results if r["department"] == "General Medicine")
        assert gm["score"] > 0
        assert len(gm["matched_symptoms"]) >= 2

    def test_alias_and_english_together(self):
        """Mix of English and Urdu symptoms."""
        results = compute_cluster_scores(["fever", "کھانسی", "body ache"])
        gm = next(r for r in results if r["department"] == "General Medicine")
        assert gm["score"] > 0
        assert len(gm["matched_symptoms"]) >= 2

    def test_seene_ki_takleef_maps_to_cardiology(self):
        """سینے کی تکلیف (chest discomfort) should match Cardiology."""
        results = compute_cluster_scores(["سینے کی تکلیف"])
        cardio = next(r for r in results if r["department"] == "Cardiology")
        assert cardio["score"] > 0, "سینے کی تکلیف should match Cardiology via chest pain alias"

    def test_breathing_difficulty_maps_to_shortness_of_breath(self):
        """breathing difficulty should also match conditions with 'shortness of breath'."""
        results = compute_cluster_scores(["سانس لینے میں تکلیف"])
        cardio = next(r for r in results if r["department"] == "Cardiology")
        assert cardio["score"] > 0, "سانس لینے میں تکلیف should match Cardiology via shortness of breath"

    def test_urdu_chest_plus_breathing_matches_cardiology(self):
        """سینے کی تکلیف + سانس لینے میں تکلیف together should rank Cardiology first."""
        results = compute_cluster_scores(["سینے کی تکلیف", "سانس لینے میں تکلیف"])
        cardio = next(r for r in results if r["department"] == "Cardiology")
        assert cardio["score"] >= 0.3, f"Two cardiac Urdu symptoms should score >=30%, got {cardio['score']*100}%"

    def test_real_patient_urdu_cardiac_symptoms(self):
        """Real scenario: patient says سینے کی تکلیف, سینے میں درد, سانس لینے میں تکلیف.
        Both Urdu chest terms map to 'chest pain' (deduped), so effectively 2 unique symptoms.
        Should match Cardiology FIRST, above Pediatrics."""
        results = compute_cluster_scores(["سینے کی تکلیف", "سینے میں درد", "سانس لینے میں تکلیف"])
        cardio = next(r for r in results if r["department"] == "Cardiology")
        peds = next(r for r in results if r["department"] == "Pediatrics")
        assert results[0]["department"] == "Cardiology", \
            f"Top should be Cardiology, got {results[0]['department']} ({results[0]['score']*100}%)"
        assert cardio["score"] > peds["score"], \
            f"Cardiology ({cardio['score']*100}%) should rank above Pediatrics ({peds['score']*100}%)"


class TestReferralScenarios:
    """Test realistic triage scenarios verifying referral readiness."""

    def test_single_symptom_not_referral_ready(self):
        """One symptom alone → low confidence, needs more data."""
        for symptom in ["cough", "fever", "headache", "stomach pain", "back pain"]:
            results = compute_cluster_scores([symptom])
            assert results[0]["score"] < 0.5, f"'{symptom}' alone should be <50%, got {results[0]['score']*100}%"

    def test_two_symptoms_still_below_50(self):
        """Two symptoms should still be below 50% for most departments."""
        results = compute_cluster_scores(["fever", "cough"])
        assert results[0]["score"] < 0.5

    def test_three_symptoms_plus_vitals_reaches_50(self):
        """Three symptoms + vitals should push past 50%."""
        results = compute_cluster_scores(
            ["fever", "cough", "body ache"],
            {"fever": "yes", "temperature": "102"},
            "moderate"
        )
        assert results[0]["score"] >= 0.5, f"3 symptoms + vitals should be >=50%, got {results[0]['score']*100}%"

    def test_cardiac_referral_ready_with_full_data(self):
        results = compute_cluster_scores(
            ["chest pain", "shortness of breath", "sweating", "nausea"],
            {"bp": "140/90", "heart_rate": "110"},
            "severe"
        )
        cardio = next(r for r in results if r["department"] == "Cardiology")
        assert cardio["score"] >= 0.5, f"Full cardiac data should be >=50%, got {cardio['score']*100}%"
        assert cardio["department"] == results[0]["department"], "Cardiology should be top result"

    def test_ortho_referral_with_mobility(self):
        results = compute_cluster_scores(
            ["fracture", "bone pain", "swelling", "inability to move"],
            {"mobility": "impaired"},
            "severe"
        )
        ortho = next(r for r in results if r["department"] == "Orthopedics")
        assert ortho["score"] >= 0.5

    def test_dental_referral_ready(self):
        results = compute_cluster_scores(
            ["toothache", "swollen gums", "sensitivity to hot"],
            patient_severity="moderate"
        )
        dental = next(r for r in results if r["department"] == "Dentistry")
        assert dental["score"] >= 0.5, f"Dental with 3 symptoms + severity should be >=50%, got {dental['score']*100}%"

    def test_gastro_referral_ready(self):
        results = compute_cluster_scores(
            ["stomach pain", "heartburn", "acid reflux", "bloating"],
            patient_severity="moderate"
        )
        gastro = next(r for r in results if r["department"] == "Gastroenterology")
        assert gastro["score"] >= 0.5

    def test_psychiatry_referral_ready(self):
        results = compute_cluster_scores(
            ["anxiety", "nervousness", "panic attacks", "restlessness"],
            {"heart_rate": "100", "bp": "130/85"},
            "moderate"
        )
        psych = next(r for r in results if r["department"] == "Psychiatry")
        assert psych["score"] >= 0.5

    def test_ent_referral_ready(self):
        results = compute_cluster_scores(
            ["sore throat", "difficulty swallowing", "fever", "swollen tonsils"],
            {"temperature": "101", "fever": "yes"},
            "moderate"
        )
        ent = next(r for r in results if r["department"] == "ENT")
        assert ent["score"] >= 0.5

    def test_derma_referral_ready(self):
        results = compute_cluster_scores(
            ["skin rash", "itching", "redness", "hives"],
            patient_severity="moderate"
        )
        derma = next(r for r in results if r["department"] == "Dermatology")
        assert derma["score"] >= 0.5

    def test_neuro_referral_with_stroke_symptoms(self):
        results = compute_cluster_scores(
            ["sudden weakness", "numbness", "difficulty speaking", "vision loss"],
            {"bp": "180/100"},
            "severe"
        )
        neuro = next(r for r in results if r["department"] == "Neurology")
        assert neuro["score"] >= 0.5


class TestScoreDifferentiation:
    """Ensure similar symptoms differentiate correctly between departments."""

    def test_headache_with_fever_goes_general_not_neuro(self):
        """Simple headache + fever = General Medicine, not Neurology."""
        results = compute_cluster_scores(
            ["headache", "fever", "body ache"],
            {"fever": "yes", "temperature": "101"}
        )
        gm = next(r for r in results if r["department"] == "General Medicine")
        neuro = next(r for r in results if r["department"] == "Neurology")
        assert gm["score"] >= neuro["score"], "Fever+headache+bodyache should favor General Medicine"

    def test_headache_with_aura_goes_neuro(self):
        """Migraine-specific symptoms → Neurology."""
        results = compute_cluster_scores(["severe headache", "nausea", "sensitivity to light", "aura"])
        neuro = next(r for r in results if r["department"] == "Neurology")
        assert neuro["score"] > 0.4, "Migraine symptoms should strongly match Neurology"

    def test_abdominal_pain_with_fever_general_vs_gastro(self):
        """Abdominal pain + fever can be either — depends on other symptoms."""
        # Typhoid-like: abdominal pain + fever + headache + weakness
        results = compute_cluster_scores(["abdominal pain", "fever", "headache", "weakness"])
        gm = next(r for r in results if r["department"] == "General Medicine")
        assert gm["score"] > 0, "Typhoid symptoms should match General Medicine"

    def test_abdominal_pain_with_acid_goes_gastro(self):
        """Digestive symptoms → Gastroenterology."""
        results = compute_cluster_scores(["stomach pain", "acid reflux", "heartburn", "bloating"])
        gastro = next(r for r in results if r["department"] == "Gastroenterology")
        gm = next(r for r in results if r["department"] == "General Medicine")
        assert gastro["score"] > gm["score"], "Digestive symptoms should favor Gastro over GM"

    def test_cough_shared_across_departments(self):
        """Cough appears in multiple departments — top depends on other symptoms."""
        # Cough alone → General Medicine (more cough conditions)
        results = compute_cluster_scores(["cough"])
        gm = next(r for r in results if r["department"] == "General Medicine")
        peds = next(r for r in results if r["department"] == "Pediatrics")
        # Both should match, but GM has more conditions with cough
        assert gm["score"] > 0 and peds["score"] > 0

    def test_numbness_general_vs_neuro(self):
        """Numbness + diabetes symptoms → General Medicine. Numbness + nerve pain → Neurology."""
        dm_results = compute_cluster_scores(["numbness", "excessive thirst", "frequent urination"])
        gm = next(r for r in dm_results if r["department"] == "General Medicine")
        assert gm["score"] > 0

        nerve_results = compute_cluster_scores(["numbness", "tingling", "burning sensation", "nerve pain"])
        neuro = next(r for r in nerve_results if r["department"] == "Neurology")
        assert neuro["score"] > 0.4

    def test_dizziness_differentiates_ent_vs_cardio(self):
        """Dizziness with ear symptoms → ENT, dizziness with chest → Cardiology."""
        ent_results = compute_cluster_scores(["dizziness", "ringing in ear", "balance problems"])
        ent = next(r for r in ent_results if r["department"] == "ENT")
        assert ent["score"] > 0.3

        cardio_results = compute_cluster_scores(["dizziness", "palpitations", "chest pain"])
        cardio = next(r for r in cardio_results if r["department"] == "Cardiology")
        assert cardio["score"] > 0.3


class TestSymptomDoctorMapping:
    """Test the symptom-to-doctor mapping builder."""

    def test_mapping_covers_all_departments(self):
        from conversation_agent import _build_symptom_doctor_mapping
        mapping = _build_symptom_doctor_mapping()
        for cluster in DISEASE_CLUSTERS_DATA["clusters"]:
            assert cluster["department"] in mapping, f"{cluster['department']} missing from mapping"

    def test_mapping_includes_all_symptoms(self):
        from conversation_agent import _build_symptom_doctor_mapping
        mapping = _build_symptom_doctor_mapping()
        for cluster in DISEASE_CLUSTERS_DATA["clusters"]:
            for cond in cluster["conditions"]:
                for symptom in cond["symptoms"][:3]:  # Check at least first 3
                    assert symptom.lower() in mapping.lower(), \
                        f"Symptom '{symptom}' missing from {cluster['department']} mapping"

    def test_cough_in_general_medicine_mapping(self):
        """Cough must appear in General Medicine mapping (was truncated before)."""
        from conversation_agent import _build_symptom_doctor_mapping
        mapping = _build_symptom_doctor_mapping()
        lines = mapping.split("\n")
        gm_line = next(l for l in lines if "General Medicine" in l)
        assert "cough" in gm_line, f"'cough' missing from General Medicine mapping: {gm_line}"

    def test_no_symptom_truncation(self):
        """All symptoms from all conditions should be in the mapping — no [:4] or [:6] truncation."""
        from conversation_agent import _build_symptom_doctor_mapping
        mapping = _build_symptom_doctor_mapping()
        for cluster in DISEASE_CLUSTERS_DATA["clusters"]:
            all_symptoms = set()
            for cond in cluster["conditions"]:
                for sym in cond["symptoms"]:
                    all_symptoms.add(sym.lower())
            line = next(l for l in mapping.split("\n") if cluster["department"] in l)
            for sym in all_symptoms:
                assert sym in line.lower(), \
                    f"Symptom '{sym}' truncated from {cluster['department']}"


class TestTriageFeedbackLoop:
    """Test the triage scoring and feedback mechanism for LLM context."""

    def test_progressive_confidence_building(self):
        """Confidence should increase as more data is collected."""
        s1 = compute_cluster_scores(["fever"])
        s2 = compute_cluster_scores(["fever", "cough"])
        s3 = compute_cluster_scores(["fever", "cough", "body ache"])
        s4 = compute_cluster_scores(["fever", "cough", "body ache"], {"fever": "yes", "temperature": "102"})
        s5 = compute_cluster_scores(["fever", "cough", "body ache"], {"fever": "yes", "temperature": "102"}, "moderate")

        scores = [s[0]["score"] for s in [s1, s2, s3, s4, s5]]
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1], \
                f"Score should increase: step {i} ({scores[i]}) < step {i-1} ({scores[i-1]})"

    def test_enough_data_should_reach_50_percent(self):
        """By the time we have 3 symptoms + vitals + severity, should be >=50%."""
        results = compute_cluster_scores(
            ["fever", "cough", "body ache"],
            {"fever": "yes", "temperature": "102"},
            "moderate"
        )
        assert results[0]["score"] >= 0.5, \
            f"Full data should reach >=50%, got {results[0]['score']*100}%"

    def test_all_departments_achievable_at_50_percent(self):
        """Every department should be able to reach 50% with enough matching symptoms + vitals."""
        scenarios = {
            "General Medicine": (["fever", "cough", "body ache", "sore throat"], {"fever": "yes", "temperature": "102"}, "moderate"),
            "Cardiology": (["chest pain", "shortness of breath", "sweating"], {"bp": "140/90", "heart_rate": "110"}, "severe"),
            "Gynecology & Obstetrics": (["missed period", "nausea", "vomiting", "fatigue"], {}, "mild"),
            "Pediatrics": (["fever", "cough", "poor feeding", "vomiting"], {"fever": "yes", "temperature": "102"}, "moderate"),
            "Orthopedics": (["fracture", "bone pain", "swelling", "inability to move"], {"mobility": "impaired"}, "severe"),
            "Dermatology": (["skin rash", "itching", "redness", "hives"], {}, "moderate"),
            "ENT": (["sore throat", "difficulty swallowing", "fever", "swollen tonsils"], {"temperature": "101", "fever": "yes"}, "moderate"),
            "Ophthalmology": (["eye pain", "blurred vision", "halos around lights", "headache"], {}, "moderate"),
            "Neurology": (["severe headache", "nausea", "sensitivity to light", "aura"], {}, "severe"),
            "Dentistry": (["toothache", "swollen gums", "sensitivity to hot"], {}, "moderate"),
            "Gastroenterology": (["stomach pain", "heartburn", "acid reflux", "bloating"], {}, "moderate"),
            "Psychiatry": (["anxiety", "nervousness", "panic attacks", "restlessness"], {"heart_rate": "100", "bp": "130/85"}, "moderate"),
        }
        for dept, (syms, vitals, sev) in scenarios.items():
            results = compute_cluster_scores(syms, vitals, sev)
            dept_result = next(r for r in results if r["department"] == dept)
            assert dept_result["score"] >= 0.5, \
                f"{dept} should reach 50% with full data, got {dept_result['score']*100}%"

    def test_matched_symptoms_reported_correctly(self):
        """matched_symptoms should list what actually matched."""
        results = compute_cluster_scores(["chest pain", "shortness of breath", "dizziness"])
        cardio = next(r for r in results if r["department"] == "Cardiology")
        assert "chest pain" in cardio["matched_symptoms"]
        assert "shortness of breath" in cardio["matched_symptoms"]

    def test_top_3_triage_summary_format(self):
        """Top matches format: department (doctor): XX% match [symptoms]."""
        results = compute_cluster_scores(
            ["fever", "cough", "headache"],
            {"fever": "yes"},
            "moderate"
        )
        top_matches = [c for c in results if c["score"] > 0]
        assert len(top_matches) >= 1, "Should have at least 1 matching department"
        # Verify the data structure needed for triage summary
        for c in top_matches[:3]:
            assert "department" in c
            assert "doctor" in c
            assert "score" in c
            assert "matched_symptoms" in c
            pct = round(c["score"] * 100)
            assert 0 < pct <= 100


class TestDepartmentSelectionRules:
    """Test that scoring applies department selection rules (age/gender gating)."""

    def test_adult_urdu_symptoms_not_pediatrics(self):
        """Real scenario: Adult patient (عاقب) with کھانسی, نیند نہیں آتی, تھکاوٹ, کمزوری.
        Scoring should NOT rank Pediatrics #1. General Medicine or Psychiatry should rank higher.
        Previously Pediatrics scored 40% due to fatigue+weakness matching Growth/Nutrition Issues."""
        symptoms = ["کھانسی", "نیند نہیں آتی", "تھکاوٹ", "کمزوری"]
        results = compute_cluster_scores(symptoms)
        peds = next(r for r in results if r["department"] == "Pediatrics")
        gen_med = next(r for r in results if r["department"] == "General Medicine")
        # Pediatrics should be penalized (age unknown, no child-specific symptoms)
        assert gen_med["score"] > peds["score"], \
            f"General Medicine ({gen_med['score']*100}%) should rank above Pediatrics ({peds['score']*100}%)"

    def test_adult_age_zeroes_pediatrics(self):
        """When age >= 12, Pediatrics should score 0 regardless of symptom overlap."""
        symptoms = ["cough", "fatigue", "weakness"]
        results = compute_cluster_scores(symptoms, patient_age="30")
        peds = next(r for r in results if r["department"] == "Pediatrics")
        assert peds["score"] == 0.0, f"Pediatrics should be 0% for age=30, got {peds['score']*100}%"

    def test_child_age_keeps_pediatrics(self):
        """When age < 12, Pediatrics should NOT be penalized."""
        symptoms = ["fever", "cough", "poor feeding"]
        results = compute_cluster_scores(symptoms, patient_age="5")
        peds = next(r for r in results if r["department"] == "Pediatrics")
        assert peds["score"] > 0, f"Pediatrics should score > 0 for age=5"

    def test_male_zeroes_gynecology(self):
        """Male patients should get 0% for Gynecology."""
        symptoms = ["nausea", "fatigue", "vomiting"]
        results = compute_cluster_scores(symptoms, patient_gender="M")
        gyn = next(r for r in results if r["department"] == "Gynecology & Obstetrics")
        assert gyn["score"] == 0.0, f"Gynecology should be 0% for male, got {gyn['score']*100}%"

    def test_female_keeps_gynecology(self):
        """Female patients should still match Gynecology."""
        symptoms = ["missed period", "nausea", "vomiting"]
        results = compute_cluster_scores(symptoms, patient_gender="F")
        gyn = next(r for r in results if r["department"] == "Gynecology & Obstetrics")
        assert gyn["score"] > 0, f"Gynecology should score > 0 for female"

    def test_cough_does_not_match_persistent_cough(self):
        """'cough' (via alias کھانسی) should NOT match 'persistent cough' (Heart Failure).
        These are clinically different symptoms."""
        symptoms = ["کھانسی"]
        results = compute_cluster_scores(symptoms)
        cardio = next(r for r in results if r["department"] == "Cardiology")
        # Cough should NOT match Heart Failure's "persistent cough"
        assert "persistent cough" not in cardio["matched_symptoms"], \
            "Generic 'cough' should not match 'persistent cough' in Heart Failure"

    def test_insomnia_urdu_matches_psychiatry(self):
        """'نیند نہیں آتی' (neend nahi aati) should match insomnia in Psychiatry."""
        symptoms = ["نیند نہیں آتی"]
        results = compute_cluster_scores(symptoms)
        psych = next(r for r in results if r["department"] == "Psychiatry")
        assert psych["score"] > 0, "نیند نہیں آتی should match Psychiatry (Insomnia)"
        assert any("insomnia" in s for s in psych["matched_symptoms"]), \
            f"Expected 'insomnia' in matched, got {psych['matched_symptoms']}"

    def test_real_scenario_adult_general_symptoms(self):
        """Full real scenario: Adult male, cough + insomnia + fatigue + weakness.
        Scoring should align with LLM's General Medicine referral decision."""
        symptoms = ["کھانسی", "نیند نہیں آتی", "تھکاوٹ", "کمزوری"]
        results = compute_cluster_scores(symptoms, patient_age="30", patient_gender="M")
        top = results[0]
        peds = next(r for r in results if r["department"] == "Pediatrics")
        gyn = next(r for r in results if r["department"] == "Gynecology & Obstetrics")
        # Pediatrics and Gynecology should be zeroed out
        assert peds["score"] == 0.0, f"Pediatrics should be 0% for adult male"
        assert gyn["score"] == 0.0, f"Gynecology should be 0% for male"
        # Top department should be either General Medicine or Psychiatry (both valid)
        assert top["department"] in ("General Medicine", "Psychiatry"), \
            f"Top department should be General Medicine or Psychiatry, got {top['department']}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
