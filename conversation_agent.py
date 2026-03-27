"""
Hospital Support Desk Agent — Pipecat Pipeline (STT → LLM → TTS)
FastAPI + WebSocket (UI) + WebRTC (audio) backend with custom HTML/CSS/JS frontend.
Voice-based patient intake agent for hospital reception.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OMP conflict for noisereduce + torch
import sys
import json
import asyncio
import time
import logging
import traceback
import base64
import hashlib
import re as re_module
from pathlib import Path
from typing import Dict
from contextlib import asynccontextmanager

from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, "agent_terminal.log")
with open(LOG_FILE, "w") as f:
    f.write(f"=== Hospital Support Desk Agent Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

class FlushFileHandler(logging.FileHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[FlushFileHandler(LOG_FILE, mode='a'), logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("agent")

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("websockets").setLevel(logging.WARNING)
logging.getLogger("pipecat").setLevel(logging.WARNING)
logging.getLogger("aioice").setLevel(logging.WARNING)
logging.getLogger("aiortc").setLevel(logging.WARNING)

class TeeWriter:
    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file
    def write(self, msg):
        self.original.write(msg)
        with open(self.log_file, "a") as f:
            f.write(msg)
            f.flush()
    def flush(self):
        self.original.flush()
    def isatty(self):
        return False
    def fileno(self):
        return self.original.fileno()

sys.stdout = TeeWriter(sys.__stdout__, LOG_FILE)
sys.stderr = TeeWriter(sys.__stderr__, LOG_FILE)

SAMPLE_RATE_IN = 16000   # STT input (Deepgram expects 16kHz)
SAMPLE_RATE_OUT = 24000  # TTS output (smoother audio)
SESSION_FILE = os.path.join(OUTPUT_DIR, "session_state.json")


def _load_session():
    """Load persisted session state from disk."""
    try:
        if os.path.exists(SESSION_FILE):
            with open(SESSION_FILE, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"Session: Failed to load: {e}")
    return {}


def _save_session(data: dict):
    """Save session state to disk."""
    try:
        with open(SESSION_FILE, "w") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Session: Failed to save: {e}")


# ─────────────────────────────────────────────
# Disease Clustering Engine
# ─────────────────────────────────────────────

DISEASE_CLUSTERS_FILE = os.path.join(SCRIPT_DIR, "disease_clusters.json")

def _load_disease_clusters():
    """Load disease clusters dataset from JSON file."""
    try:
        with open(DISEASE_CLUSTERS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        logger.info(f"Disease clusters: Loaded {len(data.get('clusters', []))} departments")
        return data
    except Exception as e:
        logger.warning(f"Disease clusters: Failed to load: {e}")
        return {"clusters": [], "symptom_aliases": {}}

DISEASE_CLUSTERS_DATA = _load_disease_clusters()


def _symptom_matches(variants: set, condition_symptom: str) -> bool:
    """Check if any patient symptom variant matches a condition symptom.

    Uses exact match OR whole-word containment — NOT raw substring.
    "cough" matches "cough" but NOT "persistent cough".
    "chest pain" matches "chest pain" exactly.
    Urdu variants use exact match only (no word splitting needed).
    """
    cs = condition_symptom
    for v in variants:
        # Exact match
        if v == cs:
            return True
        # Whole-word containment: "v" must be a full word/phrase within "cs" or vice versa
        # Split both into word sets and check if one is a subset of the other
        v_words = set(v.split())
        cs_words = set(cs.split())
        if v_words == cs_words:
            return True
        # For Urdu: exact match only (already handled above)
    return False


def compute_cluster_scores(patient_symptoms, patient_vitals=None, patient_severity=None, patient_age=None, patient_gender=None):
    """Compute match scores for each department/disease cluster based on patient symptoms and vitals.

    Applies department selection rules from disease_clusters.json:
    - Pediatrics: ONLY for children (under 12). Adults → General Medicine.
    - Gynecology: ONLY for female patients.
    - General Medicine: DEFAULT boost for common adult symptoms.

    Returns list of clusters sorted by score descending, each with:
    - department, doctor, room, icon
    - score (0.0-1.0) — overall match percentage
    - matched_symptoms — which patient symptoms matched
    - top_conditions — conditions sorted by relevance with their own scores
    """
    # Parse age for department filtering
    _age_num = None
    if patient_age:
        import re as _re
        age_match = _re.search(r'(\d+)', str(patient_age))
        if age_match:
            _age_num = int(age_match.group(1))
    if not patient_symptoms:
        # Return all clusters with zero score (initial state)
        return [
            {
                "department": c["department"],
                "doctor": c["doctor"],
                "room": c["room"],
                "icon": c.get("icon", "🏥"),
                "score": 0.0,
                "matched_symptoms": [],
                "top_conditions": [
                    {
                        "name": cond["name"], "icd": cond["icd"], "score": 0.0, "matched": [],
                        "symptoms": cond.get("symptoms", []),
                        "vitals": cond.get("vitals", {}),
                        "severity": cond.get("severity", []),
                    }
                    for cond in c.get("conditions", [])
                ],
            }
            for c in DISEASE_CLUSTERS_DATA.get("clusters", [])
        ]

    # Normalize patient symptoms to lowercase
    sym_lower = [s.lower().strip() for s in patient_symptoms if s]

    # Pre-compute alias variants for each original patient symptom.
    # Each original symptom expands to its alias group (Urdu translations + English synonyms).
    # Crucially, each original symptom can match at most ONE condition symptom,
    # preventing alias chains from inflating scores.
    aliases = DISEASE_CLUSTERS_DATA.get("symptom_aliases", {})
    sym_variants = {}
    for orig in sym_lower:
        variants = {orig}
        for canonical, alias_list in aliases.items():
            c_l = canonical.lower()
            aliases_l = {a.lower() for a in alias_list}
            if orig in aliases_l or orig == c_l:
                variants.add(c_l)
                variants.update(aliases_l)
        sym_variants[orig] = variants

    results = []
    for cluster in DISEASE_CLUSTERS_DATA.get("clusters", []):
        cluster_matched = set()
        condition_scores = []

        for cond in cluster.get("conditions", []):
            cond_symptoms = [s.lower() for s in cond.get("symptoms", [])]
            matched = []
            matched_cond = set()  # condition symptoms already matched
            for orig in sym_lower:
                for cs in cond_symptoms:
                    if cs not in matched_cond and _symptom_matches(sym_variants[orig], cs):
                        matched.append(cs)
                        matched_cond.add(cs)
                        break

            # Vital boost: if patient vitals match condition vital indicators
            vital_boost = 0.0
            if patient_vitals and cond.get("vitals"):
                cond_vitals = cond["vitals"]
                v = patient_vitals
                if cond_vitals.get("fever") and v.get("fever") and v["fever"].lower() not in ("", "no", "none"):
                    vital_boost += 0.1
                if cond_vitals.get("bp") and v.get("bp") and v["bp"].strip():
                    vital_boost += 0.1
                if cond_vitals.get("heart_rate") and v.get("heart_rate") and v["heart_rate"].strip():
                    vital_boost += 0.1
                if cond_vitals.get("temperature") and v.get("temperature") and v["temperature"].strip():
                    vital_boost += 0.1
                if cond_vitals.get("sugar_level") and v.get("sugar_level") and v["sugar_level"].strip():
                    vital_boost += 0.1
                if cond_vitals.get("mobility") and v.get("mobility") and v["mobility"].strip():
                    vital_boost += 0.1

            # Severity boost
            severity_boost = 0.0
            if patient_severity and cond.get("severity"):
                if patient_severity.lower() in [s.lower() for s in cond["severity"]]:
                    severity_boost = 0.05

            cond_score = 0.0
            if cond_symptoms and matched:
                cond_score = min(1.0, (len(matched) / len(cond_symptoms)) + vital_boost + severity_boost)

            cluster_matched.update(matched)
            condition_scores.append({
                "name": cond["name"],
                "icd": cond["icd"],
                "score": round(cond_score, 3),
                "matched": list(set(matched)),
                "symptoms": cond.get("symptoms", []),
                "vitals": cond.get("vitals", {}),
                "severity": cond.get("severity", []),
            })

        # Cluster score = best condition score + cross-condition confidence boost.
        # When symptoms match MULTIPLE conditions in a department (e.g., chest pain matches
        # Angina, Hypertensive HD, Arrhythmia, Heart Failure in Cardiology), the diagnostic
        # confidence is much higher than matching just 1 condition in Pediatrics.
        # This aligns scoring with LLM referral decisions.
        condition_scores.sort(key=lambda x: x["score"], reverse=True)
        best_score = condition_scores[0]["score"] if condition_scores else 0.0

        strong_conditions = sum(1 for c in condition_scores if c["score"] >= 0.15)
        if strong_conditions > 1:
            cross_boost = best_score * 0.15 * (strong_conditions - 1)
            best_score = min(1.0, best_score + cross_boost)

        # --- Department selection rules (align scoring with LLM judgement) ---
        dept = cluster["department"]

        # Pediatrics: ONLY for children under 12. If age unknown or >= 12, suppress.
        # Exception: child-specific symptoms like "vaccination", "growth monitoring", "well child"
        if dept == "Pediatrics":
            child_specific = {"vaccination", "checkup", "growth monitoring", "well child", "delayed development", "poor feeding", "poor weight gain"}
            has_child_specific = bool(cluster_matched & child_specific)
            if _age_num is not None and _age_num >= 12:
                best_score = 0.0  # Adult — zero out Pediatrics
            elif _age_num is None and not has_child_specific:
                best_score *= 0.3  # Age unknown + no child-specific symptoms → heavy penalty

        # Gynecology: ONLY for female patients
        if dept == "Gynecology & Obstetrics":
            if patient_gender and patient_gender.strip().upper() in ("M", "MALE"):
                best_score = 0.0  # Male — zero out Gynecology

        results.append({
            "department": cluster["department"],
            "doctor": cluster["doctor"],
            "room": cluster["room"],
            "icon": cluster.get("icon", "🏥"),
            "score": round(best_score, 3),
            "matched_symptoms": list(cluster_matched),
            "top_conditions": condition_scores,
        })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def _build_vitals_table(severity_labels="mild/moderate/severe"):
    """Build the vitals follow-up guide from disease_clusters.json for the system prompt.

    Single source of truth — same JSON drives the triage dashboard AND the bot's questions.

    Filtering logic:
    - INCLUDE vitals with definitive values: "present", "elevated", "high", "low",
      "irregular", "variable", "impaired", "limited", "monitor"
    - EXCLUDE vitals with non-definitive values: "possible", "check"
      (these are just indicators, not things the bot should actively ask about)
    - A vital is included for a cluster only if ANY condition in that cluster has
      a definitive value for it (even if other conditions have "possible"/"check").
    - Severity-aware: conditions with only "mild" severity don't contribute vitals
      to the "ask" list — mild complaints only need severity confirmation.
    """
    VITAL_DISPLAY = {
        "temperature": "temperature",
        "fever": "temperature",
        "bp": "BP",
        "heart_rate": "heart rate",
        "sugar_level": "sugar level",
        "mobility": "mobility",
    }
    # Values that mean "don't actively ask — just an indicator"
    EXCLUDE_VALUES = {"possible", "check"}

    rows = []
    for cluster in DISEASE_CLUSTERS_DATA.get("clusters", []):
        symptoms_sample = set()
        valid_vitals = set()
        severity_set = set()
        for cond in cluster.get("conditions", []):
            cond_severities = set(cond.get("severity", []))
            severity_set.update(cond_severities)
            for sym in cond.get("symptoms", [])[:3]:
                symptoms_sample.add(sym)
            # Only include vitals from conditions that can be moderate/severe
            if not cond_severities.issubset({"mild"}):
                for vk, vval in cond.get("vitals", {}).items():
                    if vk in VITAL_DISPLAY and vval not in EXCLUDE_VALUES:
                        valid_vitals.add(VITAL_DISPLAY[vk])

        symptom_text = ", ".join(sorted(symptoms_sample)[:5])
        if valid_vitals:
            vital_text = ", ".join(sorted(valid_vitals))
        else:
            vital_text = f"severity ({severity_labels})"

        rows.append(f"   | {symptom_text} | {vital_text} |")

    return "\n".join(rows)


def _build_symptom_doctor_mapping():
    """Auto-generate Symptom-to-Doctor mapping from disease_clusters.json.
    Single source of truth — same JSON drives triage dashboard AND bot referrals.
    """
    lines = []
    for cluster in DISEASE_CLUSTERS_DATA.get("clusters", []):
        # Collect ALL unique symptoms across conditions in this cluster
        all_symptoms = set()
        for cond in cluster.get("conditions", []):
            for sym in cond.get("symptoms", []):
                all_symptoms.add(sym.lower())
        symptom_text = ", ".join(sorted(all_symptoms))
        dept = cluster["department"]
        doctor = cluster["doctor"]
        lines.append(f"- {symptom_text} -> {dept} ({doctor})")
    return "\n".join(lines)


def _build_referral_rules():
    """Auto-generate department referral rules from disease_clusters.json."""
    rules = DISEASE_CLUSTERS_DATA.get("referral_rules", {})
    lines = []
    for dept, rule in rules.items():
        if dept == "_default":
            lines.append(f"- **DEFAULT**: {rule}")
        else:
            lines.append(f"- **{dept}**: {rule}")
    return "\n".join(lines)


DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
ELEVENLABS_KEY = os.getenv("ELEVEN_LABS_API_KEY", "") or os.getenv("ELEVENLABS_API_KEY", "")
ANTHROPIC_KEY = os.getenv("ANTHROPIC_API_KEY", "")

logger.info(f"DEEPGRAM_KEY set: {bool(DEEPGRAM_KEY)} (len={len(DEEPGRAM_KEY)})")
logger.info(f"OPENAI_KEY set: {bool(OPENAI_KEY)} (len={len(OPENAI_KEY)})")
logger.info(f"ELEVENLABS_KEY set: {bool(ELEVENLABS_KEY)} (len={len(ELEVENLABS_KEY)})")
logger.info(f"ANTHROPIC_KEY set: {bool(ANTHROPIC_KEY)} (len={len(ANTHROPIC_KEY)})")

# ─────────────────────────────────────────────
# Hospital Configuration
# ─────────────────────────────────────────────

HOSPITAL_NAME = "City General Hospital"

DOCTORS_PANEL = [
    {"name": "Dr. Ahmed Khan", "specialty": "General Medicine", "room": "101", "available": True, "gender": "M",
     "qualification": "MBBS, FCPS (Medicine)", "experience": "15 years", "timings": "9:00 AM - 2:00 PM",
     "phone": "Ext. 1101", "languages": ["Urdu", "English"],
     "expertise": "Fever, Flu, Infections, Diabetes Management, Hypertension, General Checkups"},
    {"name": "Dr. Fatima Zahra", "specialty": "Gynecology & Obstetrics", "room": "205", "available": True, "gender": "F",
     "qualification": "MBBS, FCPS (Gynecology)", "experience": "12 years", "timings": "10:00 AM - 3:00 PM",
     "phone": "Ext. 2205", "languages": ["Urdu", "English"],
     "expertise": "Pregnancy Care, Prenatal Checkups, Women's Health, Fertility Consultation"},
    {"name": "Dr. Ali Raza", "specialty": "Cardiology", "room": "302", "available": True, "gender": "M",
     "qualification": "MBBS, MRCP, Fellowship Cardiology (UK)", "experience": "18 years", "timings": "9:00 AM - 1:00 PM",
     "phone": "Ext. 3302", "languages": ["Urdu", "English", "Punjabi"],
     "expertise": "Heart Disease, ECG, Echo, Angiography, Blood Pressure, Chest Pain"},
    {"name": "Dr. Sarah Malik", "specialty": "Pediatrics", "room": "108", "available": True, "gender": "F",
     "qualification": "MBBS, FCPS (Pediatrics)", "experience": "10 years", "timings": "9:00 AM - 4:00 PM",
     "phone": "Ext. 1108", "languages": ["Urdu", "English"],
     "expertise": "Child Health, Vaccination, Growth Monitoring, Newborn Care, Childhood Infections"},
    {"name": "Dr. Usman Tariq", "specialty": "Orthopedics", "room": "210", "available": False, "gender": "M",
     "qualification": "MBBS, MS (Orthopedics)", "experience": "14 years", "timings": "On Leave",
     "phone": "Ext. 2210", "languages": ["Urdu", "English"],
     "expertise": "Fractures, Joint Replacement, Sports Injuries, Back Pain, Arthritis"},
    {"name": "Dr. Ayesha Siddiqui", "specialty": "Dermatology", "room": "115", "available": True, "gender": "F",
     "qualification": "MBBS, FCPS (Dermatology)", "experience": "8 years", "timings": "11:00 AM - 5:00 PM",
     "phone": "Ext. 1115", "languages": ["Urdu", "English"],
     "expertise": "Skin Allergies, Acne, Eczema, Psoriasis, Hair Loss, Cosmetic Dermatology"},
    {"name": "Dr. Hassan Javed", "specialty": "ENT", "room": "118", "available": True, "gender": "M",
     "qualification": "MBBS, FCPS (ENT)", "experience": "11 years", "timings": "10:00 AM - 3:00 PM",
     "phone": "Ext. 1118", "languages": ["Urdu", "English", "Punjabi"],
     "expertise": "Ear Infections, Hearing Loss, Sinus, Tonsils, Voice Disorders, Vertigo"},
    {"name": "Dr. Nadia Hussain", "specialty": "Ophthalmology", "room": "122", "available": True, "gender": "F",
     "qualification": "MBBS, FCPS (Ophthalmology), Fellowship (Retina)", "experience": "13 years", "timings": "9:00 AM - 2:00 PM",
     "phone": "Ext. 1122", "languages": ["Urdu", "English"],
     "expertise": "Vision Problems, Cataract Surgery, Glaucoma, Retina Disorders, Eye Infections"},
    {"name": "Dr. Bilal Qureshi", "specialty": "Neurology", "room": "310", "available": True, "gender": "M",
     "qualification": "MBBS, FCPS (Neurology), Fellowship (Epilepsy)", "experience": "16 years", "timings": "10:00 AM - 2:00 PM",
     "phone": "Ext. 3310", "languages": ["Urdu", "English"],
     "expertise": "Headaches, Migraines, Epilepsy, Stroke, Numbness, Nerve Disorders"},
    {"name": "Dr. Zainab Akhtar", "specialty": "Dentistry", "room": "G-05", "available": True, "gender": "F",
     "qualification": "BDS, FCPS (Orthodontics)", "experience": "9 years", "timings": "9:00 AM - 5:00 PM",
     "phone": "Ext. 0005", "languages": ["Urdu", "English"],
     "expertise": "Dental Fillings, Root Canal, Braces, Tooth Extraction, Teeth Whitening"},
    {"name": "Dr. Imran Shah", "specialty": "Gastroenterology", "room": "215", "available": False, "gender": "M",
     "qualification": "MBBS, FCPS (Gastroenterology)", "experience": "20 years", "timings": "On Leave",
     "phone": "Ext. 2215", "languages": ["Urdu", "English"],
     "expertise": "Stomach Ulcers, Liver Disease, Endoscopy, IBS, Acid Reflux, Hepatitis"},
    {"name": "Dr. Maryam Riaz", "specialty": "Psychiatry", "room": "320", "available": True, "gender": "F",
     "qualification": "MBBS, FCPS (Psychiatry)", "experience": "7 years", "timings": "11:00 AM - 4:00 PM",
     "phone": "Ext. 3320", "languages": ["Urdu"],
     "expertise": "Anxiety, Depression, Sleep Disorders, PTSD, Counseling, Stress Management"},
]

def get_doctors_panel_text():
    lines = []
    for d in DOCTORS_PANEL:
        status = "Available" if d["available"] else "Not Available Today"
        lines.append(f"- {d['name']} | {d['specialty']} | Room {d['room']} | {status} | Languages: {', '.join(d['languages'])}")
    return "\n".join(lines)

_DOCTORS_PANEL_TEXT = get_doctors_panel_text()

# Language-specific conversation flow examples
LANG_FLOW = {
    "ur": {
        "script_rule": "IMPORTANT: Always respond in PURE Urdu script (اردو رسم الخط). NEVER use Roman Urdu transliteration. NEVER mix English words into Urdu sentences — use the Urdu equivalent. For example: say 'بے ترتیب' not 'irregular', say 'بلڈ پریشر' not 'blood pressure', say 'شوگر' not 'sugar'. Only use English for proper nouns (doctor names, hospital name).",
        "phrasing": "Use polite Pakistani Urdu phrasing — NEVER blunt direct questions.",
        "greeting_response": 'If the patient greets you first (e.g., "Assalam o Alaikum", "السلام علیکم"), ALWAYS respond with "وعلیکم السلام!" before continuing.',
        "greeting_example": f"\"السلام علیکم! {HOSPITAL_NAME} میں خوش آمدید۔ کیا آپ اپنا نام بتا سکتے ہیں؟\"",
        "symptoms_example": "\"کیا آپ بتا سکتے ہیں آپ کو کیا مسئلہ درپیش ہے؟\"",
        "duration_example": "\"کیا آپ بتا سکتے ہیں یہ مسئلہ کب سے ہے؟\"",
        "severity_labels": "ہلکا/درمیانہ/شدید — mild/moderate/severe",
        "refer_example": "\"میں آپ کو [Doctor name] صاحب/صاحبہ کے پاس بھیج رہا/رہی ہوں، [department]، [room number]۔\"",
        "thanks": "\"آپ کا شکریہ\"",
        "well_wish": "\"اللہ آپ کو جلدی شفا دے\" or \"ان شاء اللہ جلدی ٹھیک ہو جائیں گے\"",
        "spoken_text_note": "Use Urdu script.",
    },
    "en": {
        "script_rule": "",
        "phrasing": "Use polite, professional English — warm but efficient.",
        "greeting_response": 'If the patient greets you, respond warmly before continuing.',
        "greeting_example": f"\"Welcome to {HOSPITAL_NAME}! I'm here to help. May I have your name please?\"",
        "symptoms_example": "\"Could you tell me what seems to be the problem?\"",
        "duration_example": "\"How long have you been experiencing this?\"",
        "severity_labels": "mild/moderate/severe",
        "refer_example": "\"I'm referring you to [Doctor name], [department], [room number].\"",
        "thanks": "\"Thank you, take care!\"",
        "well_wish": "\"Wishing you a speedy recovery\" or \"Get well soon\"",
        "spoken_text_note": "Use English.",
    },
    "hi": {
        "script_rule": "IMPORTANT: Always respond in Hindi Devanagari script (हिंदी). NEVER use Roman Hindi transliteration.",
        "phrasing": "Use polite Hindi phrasing — respectful and warm.",
        "greeting_response": 'If the patient greets you (e.g., "Namaste", "नमस्ते"), respond with "नमस्ते!" before continuing.',
        "greeting_example": f"\"नमस्ते! {HOSPITAL_NAME} में आपका स्वागत है। क्या आप अपना नाम बता सकते हैं?\"",
        "symptoms_example": "\"आप बताइए, आपको क्या तकलीफ़ हो रही है?\"",
        "duration_example": "\"यह समस्या कब से है?\"",
        "severity_labels": "हल्का/मध्यम/गंभीर — mild/moderate/severe",
        "refer_example": "\"मैं आपको [Doctor name] जी के पास भेज रहा/रही हूँ, [department], [room number]।\"",
        "thanks": "\"धन्यवाद, अपना ख़्याल रखिए!\"",
        "well_wish": "\"जल्दी ठीक हो जाइए\" or \"भगवान आपको जल्दी स्वस्थ करे\"",
        "spoken_text_note": "Use Hindi Devanagari script.",
    },
    "ar": {
        "script_rule": "IMPORTANT: Always respond in Arabic script. NEVER use transliteration.",
        "phrasing": "Use polite formal Arabic — respectful and professional.",
        "greeting_response": 'If the patient greets you (e.g., "السلام عليكم"), respond with "وعليكم السلام!" before continuing.',
        "greeting_example": f"\"السلام عليكم! أهلاً بكم في {HOSPITAL_NAME}. هل يمكنني معرفة اسمكم؟\"",
        "symptoms_example": "\"ما هي المشكلة الصحية التي تعاني منها؟\"",
        "duration_example": "\"منذ متى تعاني من هذه المشكلة؟\"",
        "severity_labels": "خفيف/متوسط/شديد — mild/moderate/severe",
        "refer_example": "\"سأحيلك إلى [Doctor name]، [department]، [room number].\"",
        "thanks": "\"شكراً لك!\"",
        "well_wish": "\"أسأل الله لك الشفاء العاجل\"",
        "spoken_text_note": "Use Arabic script.",
    },
    "pa": {
        "script_rule": "IMPORTANT: Always respond in Punjabi Shahmukhi or Gurmukhi script as appropriate.",
        "phrasing": "Use polite Punjabi phrasing — warm and friendly.",
        "greeting_response": 'If the patient greets you (e.g., "ਸਤ ਸ੍ਰੀ ਅਕਾਲ"), respond warmly before continuing.',
        "greeting_example": f"\"ਸਤ ਸ੍ਰੀ ਅਕਾਲ! {HOSPITAL_NAME} ਵਿੱਚ ਤੁਹਾਡਾ ਸਵਾਗਤ ਹੈ। ਤੁਹਾਡਾ ਨਾਮ ਕੀ ਹੈ ਜੀ?\"",
        "symptoms_example": "\"ਦੱਸੋ ਜੀ, ਤੁਹਾਨੂੰ ਕੀ ਤਕਲੀਫ਼ ਹੈ?\"",
        "duration_example": "\"ਇਹ ਤਕਲੀਫ਼ ਕਦੋਂ ਤੋਂ ਹੈ?\"",
        "severity_labels": "ਹਲਕਾ/ਦਰਮਿਆਨਾ/ਤੇਜ਼ — mild/moderate/severe",
        "refer_example": "\"ਮੈਂ ਤੁਹਾਨੂੰ [Doctor name] ਜੀ ਕੋਲ ਭੇਜ ਰਿਹਾ/ਰਹੀ ਹਾਂ, [department], [room number]।\"",
        "thanks": "\"ਧੰਨਵਾਦ ਜੀ!\"",
        "well_wish": "\"ਰੱਬ ਤੁਹਾਨੂੰ ਜਲਦੀ ਠੀਕ ਕਰੇ\"",
        "spoken_text_note": "Use Punjabi.",
    },
    "ms": {
        "script_rule": "",
        "phrasing": "Use polite formal Malay — respectful and professional.",
        "greeting_response": 'If the patient greets you (e.g., "Assalamualaikum"), respond with "Waalaikumsalam!" before continuing.',
        "greeting_example": f"\"Selamat datang ke {HOSPITAL_NAME}! Boleh saya tahu nama anda?\"",
        "symptoms_example": "\"Boleh ceritakan apa masalah kesihatan anda?\"",
        "duration_example": "\"Sudah berapa lama anda mengalami masalah ini?\"",
        "severity_labels": "ringan/sederhana/teruk — mild/moderate/severe",
        "refer_example": "\"Saya akan rujuk anda kepada [Doctor name], [department], [room number].\"",
        "thanks": "\"Terima kasih!\"",
        "well_wish": "\"Semoga cepat sembuh\"",
        "spoken_text_note": "Use Malay.",
    },
}


def _build_hospital_prompt(lang_code="ur"):
    """Build language-aware system prompt."""
    flow = LANG_FLOW.get(lang_code, LANG_FLOW["en"])
    lang_name = LANGUAGE_NAMES.get(lang_code, "English")

    script_rule = f"\n- {flow['script_rule']}" if flow["script_rule"] else ""

    return f"""You are the AI Reception Assistant at {HOSPITAL_NAME}. Your name is "Medi". You help patients get to the right doctor fast.

## Communication Style
- Keep every response SHORT — 1-2 sentences max. This is a voice conversation, not text.
- Be warm but efficient. No lengthy greetings or speeches.
- Ask ONE question at a time. Never ask multiple questions in one turn.
- You MUST respond in {lang_name} throughout the ENTIRE conversation. NEVER switch languages unless the patient explicitly speaks a different language.{script_rule}

## Conversation Flow (follow this order strictly)
{flow['phrasing']}
- {flow['greeting_response']}
1. **Quick greeting + name**: {flow['greeting_example']} — Greet ONLY ONCE on your very first response. NEVER repeat the greeting or hospital name after the first turn.
2. **Symptoms**: {flow['symptoms_example']} — Once you have the name, move directly to asking symptoms. Do NOT greet again.
3. **Duration**: {flow['duration_example']} — When acknowledging symptoms, say something natural like "آپ کی تکلیف سمجھ آئی" (I understand your problem) or "ٹھیک ہے" (okay). NEVER say "شکریہ" (thank you) for symptoms — it sounds unnatural.
4. **Relevant vitals/follow-up** — Use this EXACT lookup table (auto-generated from hospital disease database). Ask ONLY the listed follow-up for the matching symptom category. NEVER ask a vital not listed for that category.
   | Symptom Category | Valid Follow-up to Ask |
   |---|---|
{_build_vitals_table(flow['severity_labels'])}

   RULES:
   - Do NOT ask about fever/temperature unless the table says "temperature" for that category.
   - Do NOT ask about BP/heart rate unless the table says "BP" or "heart rate" for that category.
   - If no specific vital is listed, ask severity ({flow['severity_labels']}) only.
   - Severity-based intelligence: If the patient's complaint sounds MILD (e.g., mild rash, routine checkup), ask severity confirmation only. If MODERATE or SEVERE (e.g., chest pain, high fever, difficulty breathing), ask the relevant vitals from the table.
   - Ask only ONE follow-up per turn.
   - You MUST ask at least ONE follow-up from the table before referring.
   - When patient gives vital values (e.g., "BP 90 130", "sugar 200", "temperature 103"), ALWAYS record the exact numbers in vitals. Format BP as "130/90", temperature as "103°F", sugar as "200 mg/dL".
5. **Age** (optional): Only ask if medically relevant for triage. Skip for obvious referrals.
6. **Smart follow-up — ask TARGETED questions to narrow diagnosis**: After the patient tells their first symptom, think about which conditions in the disease database match. Then ask about OTHER symptoms specific to those conditions to confirm or rule out.
   Examples:
   - Patient says "chest pain" → top matches are Angina (chest tightness, shortness of breath, sweating) and GERD (acid reflux, burning). Ask: "کیا سانس لینے میں تکلیف ہے؟" (shortness of breath?) to differentiate.
   - Patient says "headache" → could be Migraine (severe, nausea, light sensitivity) or Hypertension (dizziness, blurred vision). Ask: "کیا چکر بھی آ رہے ہیں؟" (dizziness?) to narrow down.
   - Patient says "cough" → could be Cold (fever, runny nose) or Asthma (wheezing, breathing difficulty). Ask: "کیا بخار بھی ہے؟" (fever?) to differentiate.
   - If only ONE vague symptom, ask: "کیا اور کوئی تکلیف ہے؟" (any other symptoms?)
   - Ask relevant vitals from the table for the top matching conditions.
   - Only refer when you have: name + at least 2 symptoms OR 1 symptom + severity + vital

   **CRITICAL — Handle off-topic answers**: If you asked about a SPECIFIC symptom (e.g., "کیا سانس لینے میں تکلیف ہے؟") and the patient answers a DIFFERENT question instead (e.g., gives duration "ایک مہینے سے" or gives vitals), you MUST:
   1. Accept and record the information they gave (e.g., save the duration)
   2. STILL ask your unanswered symptom question again on the NEXT turn. Do NOT skip it.
   Example: You asked "کیا سانس لینے میں تکلیف ہے؟" → Patient says "ایک مہینے سے" → You say "ٹھیک ہے، ایک مہینے سے۔ لیکن کیا سانس لینے میں بھی تکلیف ہے؟"
   The patient may not have understood your question, so gently re-ask. Only move on if they clearly answer yes or no to your symptom question.
7. **Refer**: Your tool results contain TRIAGE SCORES with a "REFER TO →" directive. You MUST follow it.
   MINIMUM requirements before referring:
   - At least 2 different symptoms collected, OR 1 symptom + severity + at least one vital value
   - Duration asked
   - At least ONE vital or severity collected
   Then refer when EITHER condition is met:
   a) A department reaches >= 50% confidence → refer immediately to the "REFER TO →" doctor
   b) You have collected symptoms + duration + severity + vitals → refer to the "REFER TO →" doctor regardless of score
   MANDATORY: You MUST refer to the exact doctor/department/room shown in "REFER TO →". Do NOT pick a different doctor. Do NOT default to General Medicine. The triage system has already applied department selection rules (age, gender, symptom specificity). Trust it.
   NEVER ask more than 5-6 questions total. NEVER refer after just 1 symptom without severity and vitals. Say: {flow['refer_example']}
8. **End session**: After giving the referral and well-wish, say {flow['thanks']} to signal conversation is complete.

## MANDATORY: Patient Screening Tool
You MUST call the update_patient_screening tool on EVERY response — no exceptions, even your first greeting.
- Put your spoken response in the "spoken_text" field (1-2 sentences). {flow['spoken_text_note']}
- Fill patient data fields with ALL previously collected data plus any new info from the current turn.
- IMPORTANT: Always keep numbers as digits (103, 120/80, 98.6) — NEVER transliterate numbers to words.
- Use empty string "" for unknown fields, empty array [] for no symptoms yet.
- Set is_complete=true ONLY after final referral and farewell.
- Set emergency=true for emergency cases.
- **CRITICAL — symptoms array must include ALL patient complaints**: Every health issue the patient mentions is a symptom. If they say "دل کی دھڑکن تیز ہے" (fast heartbeat), add "دل کی دھڑکن تیز" to symptoms AND set heart_rate in vitals. If they say "بخار ہے" (fever), add "بخار" to symptoms AND set fever vitals. NEVER drop a complaint from the symptoms array — it drives the triage scoring. On each turn, include ALL previously collected symptoms PLUS any new ones from this turn.
Do NOT generate any text outside of the tool call — all your output goes through the tool.

## Key Rules
- Goal is to get patient to the right doctor in UNDER 2 MINUTES
- NEVER diagnose — only triage and refer
- IMPORTANT: The patient's speech comes through voice recognition which may have errors. Use context to understand what they mean. For example: "حاصل"/"خاصی"/"کاشی"/"کھاسی"/"کاسی" all mean "کھانسی" (cough), "بھکار"/"بکھار" means "بخار" (fever). Always interpret the patient's intent, not their exact words. In the symptoms array, ALWAYS write the CORRECT standard Urdu term (e.g., "کھانسی" not "کاشی") — normalize mistranscriptions before saving to symptoms.
- NEVER skip an unanswered question. If you asked about a symptom and the patient answered something else (duration, vitals, etc.), capture what they said BUT re-ask your symptom question on the next turn. Do not move to the next step until the symptom question is answered with yes/no.
- If symptoms clearly point to a specialty, skip extra questions and refer immediately
- For emergencies (severe chest pain, breathing difficulty, heavy bleeding, loss of consciousness) — IMMEDIATELY direct to Emergency, set "emergency": true. No questions.
- If a referred doctor is unavailable, suggest the nearest alternative
- If the patient does NOT respond for a while (you receive no new input), gently repeat your last question in a polite way. For example: "کیا آپ نے سنا؟" (did you hear?) or rephrase your question shorter and simpler. Do NOT just repeat the exact same sentence.
- End with a brief well-wish: {flow['well_wish']}

## Doctors Panel (Current)
{_DOCTORS_PANEL_TEXT}

## Symptom to Doctor Mapping (for smart follow-up questions ONLY — do NOT use for referral decisions)
{_build_symptom_doctor_mapping()}
NOTE: This mapping is for asking targeted follow-up questions only. For actual referral, ALWAYS use the "REFER TO →" directive from TRIAGE SCORES.

## Other Inquiries
- Fees -> Billing counter, Ground Floor, Main Lobby
- Lab tests -> Laboratory, 1st Floor, Room 130
- X-ray/scans -> Radiology, 1st Floor, Room 140
- Medicine -> Pharmacy, Ground Floor, Wing B

REMINDER: You MUST respond in {lang_name} and call update_patient_screening on EVERY response. No exceptions.
"""


# ─────────────────────────────────────────────
# Config Maps
# ─────────────────────────────────────────────

LLM_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo", "claude-haiku-4-5-20251001", "claude-sonnet-4-6-20250514"]

CLAUDE_MODELS = {"claude-haiku-4-5-20251001", "claude-sonnet-4-6-20250514"}
STT_MODELS = ["Deepgram Nova-3", "OpenAI Realtime"]
STT_LANGUAGES = {"Urdu": "ur", "English": "en", "Hindi": "hi", "Arabic": "ar", "Punjabi": "pa", "Malay": "ms"}

PERSONALITY_MAP = {
    "warm": "Extremely warm, compassionate, and reassuring. Speak like a caring elder who genuinely cares about the patient's well-being.",
    "professional": "Professional, efficient, and courteous. Be direct but polite — like a well-trained hospital receptionist.",
    "friendly": "Friendly, casual, and approachable. Use a light conversational tone to put patients at ease.",
}

LANG_STYLE_MAP = {
    "formal": "Use formal Pakistani Urdu in Urdu script. Use آپ، جناب، صاحب/صاحبہ۔ Maintain ادب and تمیز۔ In English use polite professional tone.",
    "casual": "Use casual Pakistani Urdu in Urdu script like everyday conversation. Simple words, approachable tone. Like talking to someone at home — 'آپ فکر مت کیجیے'، 'کوئی بات نہیں'۔",
    "native": "Use rich Pakistani Urdu in Urdu script with local idioms and expressions. Speak like a Pakistani — 'ان شاء اللہ ٹھیک ہو جائیں گے'، 'اللہ کرے جلدی صحت ہو'، 'آپ نے بالکل صحیح کیا جو آئے'۔ Warm, authentic desi tone.",
}

VOICE_GENDER = {
    "alloy": "F", "echo": "M", "fable": "M", "onyx": "M", "nova": "F", "shimmer": "F",
    "Rachel": "F", "Drew": "M", "Clyde": "M", "Paul": "M", "Domi": "F",
    "Dave": "M", "Fin": "M", "Sarah": "F", "Antoni": "M", "Thomas": "M",
}

TTS_PROVIDERS = {
    "OpenAI": {"voices": [
        ("alloy", "F"), ("echo", "M"), ("fable", "M"), ("onyx", "M"), ("nova", "F"), ("shimmer", "F"),
    ]},
    "ElevenLabs": {"voices": [
        ("Rachel", "F"), ("Drew", "M"), ("Clyde", "M"), ("Paul", "M"), ("Domi", "F"),
        ("Dave", "M"), ("Fin", "M"), ("Sarah", "F"), ("Antoni", "M"), ("Thomas", "M"),
    ]},
}

ELEVENLABS_VOICE_MAP = {
    "Rachel": "21m00Tcm4TlvDq8ikWAM", "Drew": "29vD33N1CtxCmqQRPOHJ",
    "Clyde": "2EiwWnXFnvU5JabPnv8n", "Paul": "5Q0t7uMcjvnagumLfvZi",
    "Domi": "AZnzlk1XvdvUeBnXmlld", "Dave": "CYw3kZ02Hs0563khs1Fj",
    "Fin": "D38z5RcWu1voky8WS1ja", "Sarah": "EXAVITQu4vr4xnSDxMaL",
    "Antoni": "ErXwobaYiN019PkySvjV", "Thomas": "GBv7mTt0atIp3Br8iCZE",
}


# ─────────────────────────────────────────────
# System Prompt Builder
# ─────────────────────────────────────────────

LANGUAGE_NAMES = {"ur": "Urdu", "en": "English", "hi": "Hindi", "ar": "Arabic", "pa": "Punjabi", "ms": "Malay"}

GENDER_EXAMPLES = {
    "ur": {
        "M": ("میں آپ کو بھیج رہا ہوں", "میں آپ کی مدد کروں گا", "آپ فکر نہ کریں"),
        "F": ("میں آپ کو بھیج رہی ہوں", "میں آپ کی مدد کروں گی", "آپ فکر نہ کریں"),
    },
    "hi": {
        "M": ("मैं आपको भेज रहा हूँ", "मैं आपकी मदद करूँगा", "आप चिंता न करें"),
        "F": ("मैं आपको भेज रही हूँ", "मैं आपकी मदद करूँगी", "आप चिंता न करें"),
    },
    "en": {
        "M": ("I'm referring you", "I'll help you", "Don't worry"),
        "F": ("I'm referring you", "I'll help you", "Don't worry"),
    },
}

def build_system_prompt(agent_name="Amina", voice_gender="F", personality="warm", lang_style="formal", stt_language="ur"):
    lang_name = LANGUAGE_NAMES.get(stt_language, "Urdu")
    examples = GENDER_EXAMPLES.get(stt_language, GENDER_EXAMPLES.get("en"))
    gender_ex = examples.get(voice_gender, examples["F"])
    gender_label = "MALE" if voice_gender == "M" else "FEMALE"
    gender_instruction = (
        f"\n\n## Your Gender Identity\n"
        f"You are a {gender_label} assistant named {agent_name}. "
        f"Refer to yourself in {'masculine' if voice_gender == 'M' else 'feminine'} context "
        f"(e.g., '{gender_ex[0]}', '{gender_ex[1]}', '{gender_ex[2]}')."
    )
    personality_text = PERSONALITY_MAP.get(personality, PERSONALITY_MAP["warm"])
    lang_text = LANG_STYLE_MAP.get(lang_style, LANG_STYLE_MAP["formal"])
    base_prompt = _build_hospital_prompt(stt_language)
    base_prompt = base_prompt.replace('Your name is "Medi"', f'Your name is "{agent_name}"')
    return base_prompt + gender_instruction + f"\n\n## Personality\n{personality_text}" + f"\n\n## Language Style\n{lang_text}"


# ─────────────────────────────────────────────
# Pipecat Imports
# ─────────────────────────────────────────────

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.frames.frames import (
    Frame,
    TranscriptionFrame,
    InterimTranscriptionFrame,
    TextFrame,
    AudioRawFrame,
    OutputAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    LLMFullResponseStartFrame,
    LLMFullResponseEndFrame,
    LLMMessagesFrame,
    EndFrame,
    FunctionCallResultProperties,
)
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.services.llm_service import FunctionCallParams
from pipecat.transports.base_transport import TransportParams
from pipecat.services.deepgram.stt import DeepgramSTTService, LiveOptions
from pipecat.services.openai.stt import OpenAISTTService
from pipecat.transcriptions.language import Language
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.services.anthropic.llm import AnthropicLLMService
from pipecat.services.openai.tts import OpenAITTSService
from pipecat.services.elevenlabs.tts import ElevenLabsTTSService
from pipecat.transports.smallwebrtc.connection import IceServer, SmallWebRTCConnection
from pipecat.transports.smallwebrtc.transport import SmallWebRTCTransport
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams

# --- Confidence-filtered Deepgram STT ---
# Pipecat 0.0.85 doesn't expose Deepgram confidence scores in TranscriptionFrame.
# This subclass intercepts the raw Deepgram response and drops low-confidence
# transcriptions (background noise, distant speakers) before they enter the pipeline.
STT_CONFIDENCE_THRESHOLD = 0.45  # Urdu-friendly threshold — Urdu scores 0.5-0.7 typically

class ConfidenceFilteredDeepgramSTT(DeepgramSTTService):
    """DeepgramSTTService that filters out low-confidence transcriptions."""

    async def _on_message(self, *args, **kwargs):
        result = kwargs["result"]
        if len(result.channel.alternatives) == 0:
            return
        alt = result.channel.alternatives[0]
        confidence = alt.confidence
        transcript = alt.transcript or ""
        is_final = result.is_final

        # Only filter final transcriptions — let interims through for live UI
        if is_final and len(transcript) > 0 and confidence is not None and confidence < STT_CONFIDENCE_THRESHOLD:
            logger.info(f"STT filtered (confidence={confidence:.2f} < {STT_CONFIDENCE_THRESHOLD}): '{transcript[:60]}'")
            return

        # Pass through to parent
        await super()._on_message(*args, **kwargs)


# ICE servers for WebRTC NAT traversal
ICE_SERVERS = [
    IceServer(urls="stun:stun.l.google.com:19302"),
    IceServer(urls="stun:stun1.l.google.com:19302"),
    IceServer(urls="stun:stun2.l.google.com:19302"),
    IceServer(
        urls=[
            "turn:a.relay.metered.ca:80",
            "turn:a.relay.metered.ca:80?transport=tcp",
            "turn:a.relay.metered.ca:443",
            "turns:a.relay.metered.ca:443",
        ],
        username="83eebabf8b4cce9d5dbcb649",
        credential="2D7JvfkOQtBdYW3R",
    ),
]


# ─────────────────────────────────────────────
# Patient Screening Tool (Function Calling)
# ─────────────────────────────────────────────

PATIENT_SCREENING_TOOL = FunctionSchema(
    name="update_patient_screening",
    description="Send your spoken response AND update patient screening. You MUST call this tool on EVERY turn.",
    properties={
        "spoken_text": {"type": "string", "description": "Your spoken response to the patient (1-2 sentences). This will be spoken aloud via TTS. Use Urdu script by default."},
        "name": {"type": "string", "description": "Patient name (empty string if not yet known)"},
        "age": {"type": "string", "description": "Patient age (empty string if unknown)"},
        "gender": {"type": "string", "enum": ["M", "F", ""], "description": "Patient gender"},
        "symptoms": {"type": "array", "items": {"type": "string"}, "description": "Only actual medical symptoms (e.g. chest pain, fever, headache). Never include partial words, filler words, or non-medical text."},
        "duration": {"type": "string", "description": "Symptom duration (empty string if unknown)"},
        "severity": {"type": "string", "enum": ["mild", "moderate", "severe", ""], "description": "Severity level. ONLY use: mild, moderate, severe, or empty string. NEVER use numbers or ranges."},
        "vitals": {
            "type": "object",
            "properties": {
                "fever": {"type": "string", "description": "Fever status (e.g. 'yes', 'no', 'high')"},
                "bp": {"type": "string", "description": "Blood pressure as systolic/diastolic (e.g. '130/90'). When patient says '90 130' or 'BP 130 90', format as '130/90'"},
                "heart_rate": {"type": "string", "description": "Heart rate in bpm (e.g. '90 bpm')"},
                "temperature": {"type": "string", "description": "Temperature reading (e.g. '103°F', '39°C')"},
                "sugar_level": {"type": "string", "description": "Sugar/glucose level (e.g. '200 mg/dL', 'high')"},
                "mobility": {"type": "string", "description": "Mobility status (e.g. 'limited', 'normal')"},
            },
            "description": "Relevant vital signs collected",
        },
        "referred_doctor": {"type": "string", "description": "Doctor name if referred (empty if not yet)"},
        "referred_dept": {"type": "string", "description": "Department if referred"},
        "referred_room": {"type": "string", "description": "Room number if referred"},
        "is_complete": {"type": "boolean", "description": "True ONLY after final referral and farewell"},
        "emergency": {"type": "boolean", "description": "True for emergency cases"},
    },
    required=["spoken_text", "name", "symptoms", "is_complete"],
)


# ─────────────────────────────────────────────
# TTS Proxy — enables live provider/model/voice switching
# ─────────────────────────────────────────────

class TTSProxy(FrameProcessor):
    """Transparent proxy that delegates all frames to the active TTS service.
    Supports hot-swapping the underlying TTS without rebuilding the pipeline.

    The TTS service is spliced into the pipeline chain:
      prev → TTSProxy._tts (actual TTS) → next
    TTSProxy itself is invisible — it just manages the TTS service's links.
    """

    def __init__(self, tts_service):
        super().__init__()
        self._tts = tts_service

    def link(self, processor):
        """When pipeline links TTSProxy → next, actually link TTS → next."""
        super().link(processor)
        # The actual TTS service should output to our next processor
        self._tts.link(processor)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        # Do NOT call super() — that would bypass TTS and push frame directly downstream
        # Instead, delegate entirely to the TTS service which handles TextFrame→AudioFrame conversion
        await self._tts.process_frame(frame, direction)

    async def swap_tts(self, new_tts):
        """Hot-swap the TTS service. Safe to call between utterances."""
        old_tts = self._tts
        self._tts = new_tts
        # Wire new TTS to output to the same downstream processor
        if self._next:
            self._tts.link(self._next)
        self._tts._prev = self._prev
        # Cleanup old service
        try:
            await old_tts.cleanup()
        except Exception:
            pass
        logger.info(f"TTSProxy: Swapped to {new_tts.__class__.__name__}")


# ─────────────────────────────────────────────
# TTS Audio Cache — disk-backed, normalized text keys
# ─────────────────────────────────────────────

TTS_CACHE_DIR = os.path.join(OUTPUT_DIR, "tts_cache")
TTS_CACHE_INDEX = os.path.join(TTS_CACHE_DIR, "cache_index.json")
TTS_CACHE_MAX_ENTRIES = 500

def _normalize_for_cache(text):
    """Normalize text for cache key: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re_module.sub(r'[^\w\s]', '', text)
    text = re_module.sub(r'\s+', ' ', text)
    return text

def _make_cache_key(text, voice, provider, model, sample_rate, language):
    """Generate cache key from normalized text + voice config."""
    normalized = _normalize_for_cache(text)
    raw = f"{normalized}|{voice}|{provider}|{model}|{sample_rate}|{language}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class TTSAudioCache:
    """Disk-backed TTS audio cache with normalized text keys."""

    def __init__(self):
        os.makedirs(TTS_CACHE_DIR, exist_ok=True)
        self._index = {}
        self._load_index()
        self._stats = {"hits": 0, "misses": 0}

    def _load_index(self):
        try:
            if os.path.exists(TTS_CACHE_INDEX):
                with open(TTS_CACHE_INDEX, "r") as f:
                    self._index = json.load(f)
                # Migrate flat files into language subdirectories
                migrated = 0
                for key, entry in list(self._index.items()):
                    fname = entry.get("file", "")
                    if "/" not in fname:
                        # Flat file — detect language from text (Urdu chars → ur, else en)
                        text = entry.get("text", "")
                        lang = "ur" if any('\u0600' <= c <= '\u06FF' for c in text) else "en"
                        lang_dir = os.path.join(TTS_CACHE_DIR, lang)
                        os.makedirs(lang_dir, exist_ok=True)
                        old_path = os.path.join(TTS_CACHE_DIR, fname)
                        new_fname = f"{lang}/{fname}"
                        new_path = os.path.join(TTS_CACHE_DIR, new_fname)
                        if os.path.exists(old_path):
                            os.rename(old_path, new_path)
                            entry["file"] = new_fname
                            migrated += 1
                if migrated:
                    logger.info(f"TTSCache: Migrated {migrated} files into language dirs")
                    self._save_index()
                # Validate: remove entries with missing/corrupt files
                orphaned = []
                for key, entry in list(self._index.items()):
                    filepath = os.path.join(TTS_CACHE_DIR, entry.get("file", ""))
                    if not os.path.exists(filepath):
                        orphaned.append(key)
                    elif os.path.getsize(filepath) < 1000:  # < 1KB = corrupt
                        orphaned.append(key)
                        try:
                            os.remove(filepath)
                        except OSError:
                            pass
                for key in orphaned:
                    del self._index[key]
                if orphaned:
                    logger.info(f"TTSCache: Cleaned {len(orphaned)} orphaned/corrupt entries")
                    self._save_index()
                logger.info(f"TTSCache: Loaded {len(self._index)} cached entries")
        except Exception as e:
            logger.warning(f"TTSCache: Failed to load index: {e}")
            self._index = {}

    def _save_index(self):
        try:
            with open(TTS_CACHE_INDEX, "w") as f:
                json.dump(self._index, f, indent=2)
        except Exception as e:
            logger.warning(f"TTSCache: Failed to save index: {e}")

    def get(self, key):
        """Get cached audio bytes. Returns (audio_bytes, sample_rate, channels) or None."""
        if key not in self._index:
            self._stats["misses"] += 1
            return None
        entry = self._index[key]
        filepath = os.path.join(TTS_CACHE_DIR, entry["file"])
        if not os.path.exists(filepath):
            del self._index[key]
            self._stats["misses"] += 1
            return None
        try:
            with open(filepath, "rb") as f:
                audio = f.read()
            self._stats["hits"] += 1
            total = self._stats["hits"] + self._stats["misses"]
            if total % 10 == 0:
                rate = self._stats["hits"] / total * 100
                logger.info(f"TTSCache: {rate:.0f}% hit rate ({self._stats['hits']}/{total})")
            return (audio, entry["sample_rate"], entry["channels"])
        except Exception:
            self._stats["misses"] += 1
            return None

    def put(self, key, audio_bytes, sample_rate, channels, text, voice, provider, model, language="en"):
        """Store audio in cache. Rejects audio smaller than 5KB (likely corrupt/empty)."""
        if len(audio_bytes) < 5000:
            logger.warning(f"TTSCache: Rejecting tiny audio ({len(audio_bytes)} bytes) for '{text[:40]}'")
            return
        # LRU eviction if at capacity
        if len(self._index) >= TTS_CACHE_MAX_ENTRIES:
            oldest_key = min(self._index, key=lambda k: self._index[k].get("created", 0))
            old_file = os.path.join(TTS_CACHE_DIR, self._index[oldest_key]["file"])
            try:
                os.remove(old_file)
            except Exception:
                pass
            del self._index[oldest_key]

        lang_dir = os.path.join(TTS_CACHE_DIR, language)
        filename = f"{language}/{key}.pcm"
        filepath = os.path.join(TTS_CACHE_DIR, filename)
        try:
            os.makedirs(lang_dir, exist_ok=True)
            with open(filepath, "wb") as f:
                f.write(audio_bytes)
            self._index[key] = {
                "file": filename,
                "text": text[:100],
                "voice": voice,
                "provider": provider,
                "model": model,
                "sample_rate": sample_rate,
                "channels": channels,
                "size": len(audio_bytes),
                "created": time.time(),
            }
            self._save_index()
            logger.debug(f"TTSCache: Stored '{text[:50]}...' ({len(audio_bytes)} bytes)")
        except Exception as e:
            logger.warning(f"TTSCache: Failed to store: {e}")

    @property
    def stats(self):
        return self._stats


# Global cache instance
_tts_cache = TTSAudioCache()


# ─────────────────────────────────────────────
# Number-to-Urdu-words converter (for TTS pronunciation)
# Text display keeps digits "302", but TTS says "تین سو دو"
# ─────────────────────────────────────────────

_URDU_ONES = {
    0: "صفر", 1: "ایک", 2: "دو", 3: "تین", 4: "چار",
    5: "پانچ", 6: "چھ", 7: "سات", 8: "آٹھ", 9: "نو",
    10: "دس", 11: "گیارہ", 12: "بارہ", 13: "تیرہ", 14: "چودہ",
    15: "پندرہ", 16: "سولہ", 17: "سترہ", 18: "اٹھارہ", 19: "انیس",
    20: "بیس", 21: "اکیس", 22: "بائیس", 23: "تئیس", 24: "چوبیس",
    25: "پچیس", 26: "چھبیس", 27: "ستائیس", 28: "اٹھائیس", 29: "انتیس",
    30: "تیس", 31: "اکتیس", 32: "بتیس", 33: "تینتیس", 34: "چونتیس",
    35: "پینتیس", 36: "چھتیس", 37: "سینتیس", 38: "اڑتیس", 39: "انتالیس",
    40: "چالیس", 50: "پچاس", 60: "ساٹھ", 70: "ستر", 80: "اسی", 90: "نوے",
}


def _number_to_urdu(n):
    """Convert integer (0-999) to Urdu words."""
    if n in _URDU_ONES:
        return _URDU_ONES[n]
    if n < 100:
        tens = (n // 10) * 10
        ones = n % 10
        if tens in _URDU_ONES and ones in _URDU_ONES:
            return f"{_URDU_ONES[tens]} {_URDU_ONES[ones]}"
        return str(n)
    if n < 1000:
        hundreds = n // 100
        remainder = n % 100
        h_word = _URDU_ONES.get(hundreds, str(hundreds))
        result = f"{h_word} سو"
        if remainder > 0:
            result += f" {_number_to_urdu(remainder)}"
        return result
    return str(n)


def _numbers_to_urdu_words(text):
    """Replace digit-based numbers with Urdu words for TTS pronunciation.
    Only converts 1-3 digit numbers. Preserves surrounding text."""
    return re_module.sub(r'\b(\d{1,3})\b', lambda m: _number_to_urdu(int(m.group())), text)


class TTSCacheGate(FrameProcessor):
    """Sits BEFORE TTS in pipeline. Buffers TextFrames, checks cache on response end.
    On cache hit: emits cached audio directly, swallows TextFrames (TTS never sees them).
    On cache miss: forwards buffered TextFrames to TTS for processing."""

    def __init__(self, agent):
        super().__init__()
        self._agent = agent
        self._text_buffer = []  # buffered TextFrames
        self._accumulated_text = ""
        self._buffering = False
        self._cache_hit = False

    def _get_cache_key(self, text):
        return _make_cache_key(
            text,
            self._agent.tts_voice,
            self._agent.tts_provider,
            self._agent.tts_model,
            SAMPLE_RATE_OUT,
            self._agent.stt_language,
        )

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._text_buffer = []
            self._accumulated_text = ""
            self._buffering = True
            self._cache_hit = False
            await self.push_frame(frame, direction)

        elif isinstance(frame, TextFrame) and self._buffering:
            self._text_buffer.append(frame)
            self._accumulated_text += frame.text
            # Don't forward yet — buffer until response ends

        elif isinstance(frame, LLMFullResponseEndFrame) and self._buffering:
            self._buffering = False
            # Strip [PATIENT_DATA] from text for cache key
            clean = re_module.sub(r'\[PATIENT_DATA\].*?\[/PATIENT_DATA\]', '', self._accumulated_text, flags=re_module.DOTALL).strip()
            clean = re_module.sub(r'\[PATIENT_DATA\].*', '', clean, flags=re_module.DOTALL).strip()

            if clean:
                key = self._get_cache_key(clean)
                cached = _tts_cache.get(key)

                if cached:
                    # CACHE HIT — emit cached audio directly, skip TTS
                    audio_bytes, sample_rate, channels = cached
                    self._cache_hit = True
                    logger.info(f"TTSCache HIT: '{clean[:50]}...'")
                    await self.push_frame(TTSStartedFrame(), direction)
                    # Emit audio in chunks (8KB each for smooth streaming)
                    chunk_size = 8192
                    for i in range(0, len(audio_bytes), chunk_size):
                        chunk = audio_bytes[i:i+chunk_size]
                        await self.push_frame(
                            OutputAudioRawFrame(audio=chunk, sample_rate=sample_rate, num_channels=channels),
                            direction
                        )
                    await self.push_frame(TTSStoppedFrame(), direction)
                    # Forward end frame but NOT text frames (TTS skipped)
                    await self.push_frame(frame, direction)
                    return
                else:
                    logger.debug(f"TTSCache MISS: '{clean[:50]}...'")

            # CACHE MISS — forward all buffered TextFrames to TTS
            # Convert numbers to Urdu words for proper TTS pronunciation
            # (text display keeps digits, only audio gets Urdu words)
            for tf in self._text_buffer:
                if self._agent.stt_language == "ur":
                    converted = _numbers_to_urdu_words(tf.text)
                    await self.push_frame(TextFrame(text=converted), direction)
                else:
                    await self.push_frame(tf, direction)
            self._text_buffer = []
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)


class TTSCacheCapture(FrameProcessor):
    """Sits AFTER TTS in pipeline. On cache miss, captures audio output and stores in cache.

    TTS processes sentences individually, emitting separate TTSStarted/TTSStopped pairs
    per sentence. We accumulate audio across ALL sentences and save to cache only when
    the full LLM response ends (LLMFullResponseEndFrame).
    """

    def __init__(self, agent, cache_gate):
        super().__init__()
        self._agent = agent
        self._gate = cache_gate
        self._audio_chunks = []
        self._capturing = False
        self._sample_rate = SAMPLE_RATE_OUT
        self._channels = 1

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            # Reset for new response — start fresh accumulation
            if not self._gate._cache_hit:
                self._capturing = True
                self._audio_chunks = []
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSStartedFrame):
            # Don't reset chunks here — just forward the frame
            await self.push_frame(frame, direction)

        elif isinstance(frame, AudioRawFrame) and self._capturing:
            self._audio_chunks.append(frame.audio)
            self._sample_rate = frame.sample_rate
            self._channels = frame.num_channels
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSStoppedFrame):
            # Don't save yet — more sentences may follow
            await self.push_frame(frame, direction)

        elif isinstance(frame, LLMFullResponseEndFrame):
            # Full response done — save ALL accumulated audio to cache
            if self._capturing and self._audio_chunks:
                clean = self._gate._accumulated_text
                clean = re_module.sub(r'\[PATIENT_DATA\].*?\[/PATIENT_DATA\]', '', clean, flags=re_module.DOTALL).strip()
                clean = re_module.sub(r'\[PATIENT_DATA\].*', '', clean, flags=re_module.DOTALL).strip()
                if clean:
                    key = _make_cache_key(
                        clean, self._agent.tts_voice, self._agent.tts_provider,
                        self._agent.tts_model, self._sample_rate, self._agent.stt_language
                    )
                    audio_bytes = b''.join(self._audio_chunks)
                    _tts_cache.put(
                        key, audio_bytes, self._sample_rate, self._channels,
                        clean, self._agent.tts_voice, self._agent.tts_provider, self._agent.tts_model,
                        self._agent.stt_language
                    )
                    logger.info(f"TTSCache SAVED: '{clean[:50]}...' ({len(audio_bytes)} bytes, {len(self._audio_chunks)} chunks)")
            self._capturing = False
            self._audio_chunks = []
            self._gate._cache_hit = False
            await self.push_frame(frame, direction)

        else:
            await self.push_frame(frame, direction)


# ─────────────────────────────────────────────
# Custom Pipecat Processors
# ─────────────────────────────────────────────

from pipecat.audio.filters.base_audio_filter import BaseAudioFilter as _BaseAudioFilter


class FacebookDenoiserFilter(_BaseAudioFilter):
    """Audio filter using Facebook's Denoiser (DNS64) for real-time speech enhancement.
    Removes background noise AND background human voices using deep learning.
    Runs on CPU in real-time at 50ms+ chunks. Supports 16kHz natively."""

    def __init__(self):
        self._model = None
        self._sample_rate = 16000
        self._filtering = True

    async def start(self, sample_rate: int):
        self._sample_rate = sample_rate
        try:
            import torch
            from denoiser import pretrained
            self._model = pretrained.dns64()
            self._model.eval()
            logger.info(f"FacebookDenoiser: Model loaded (SR={self._model.sample_rate}Hz)")
        except Exception as e:
            logger.warning(f"FacebookDenoiser: Failed to load model: {e}")
            self._model = None

    async def stop(self):
        self._model = None

    async def process_frame(self, frame):
        from pipecat.frames.frames import FilterEnableFrame
        if isinstance(frame, FilterEnableFrame):
            self._filtering = frame.enable

    async def filter(self, audio: bytes) -> bytes:
        if not self._filtering or self._model is None:
            return audio
        try:
            import torch
            import numpy as np
            # Convert bytes → tensor
            samples = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
            tensor = torch.from_numpy(samples).unsqueeze(0).unsqueeze(0)  # [1, 1, N]
            # Denoise
            with torch.no_grad():
                enhanced = self._model(tensor)
            # Convert back to bytes
            out = (enhanced.squeeze().numpy() * 32768.0).clip(-32768, 32767).astype(np.int16)
            return out.tobytes()
        except Exception:
            return audio  # Fallback: return original on any error


def _fire_and_forget(coro):
    """Schedule a coroutine without blocking the caller. Used for WebSocket broadcasts
    in the audio pipeline to prevent UI updates from adding latency to STT/TTS flow."""
    import asyncio
    asyncio.ensure_future(coro)

class AudioLevelMonitor(FrameProcessor):
    """Computes audio RMS from incoming AudioRawFrames and broadcasts to UI mic meter.
    Lightweight: uses int math, broadcasts every 10th frame (~200ms), non-blocking."""

    def __init__(self, agent):
        super().__init__()
        self._agent = agent
        self._frame_count = 0

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, AudioRawFrame):
            self._frame_count += 1
            # Broadcast every 10th frame (~200ms) — minimal overhead
            if self._frame_count % 10 == 0:
                try:
                    # Fast int-based RMS — no numpy, no float conversion
                    data = frame.audio
                    n = len(data) // 2
                    if n > 0:
                        import struct
                        samples = struct.unpack(f'<{n}h', data)
                        sum_sq = sum(s * s for s in samples[::8])  # sample every 8th for speed
                        rms = (sum_sq / (n // 8)) ** 0.5 / 32768.0 if n >= 8 else 0.0
                    else:
                        rms = 0.0
                    _fire_and_forget(self._agent.broadcast({"type": "mic_level", "rms": round(rms, 4)}))
                except Exception:
                    pass
        await self.push_frame(frame, direction)


class STTBroadcaster(FrameProcessor):
    """Broadcasts STT events (interim + final transcription) to WebSocket UI.
    Filters out noise: very short transcripts and common noise artifacts.
    """

    # Common noise artifacts that STT produces from background sounds / chatter
    NOISE_WORDS = frozenset({
        "", "hmm", "hm", "um", "uh", "ah", "oh", "mm", "mhm", "uh huh",
        "yeah", "yep", "yup", "nah", "nope", "ok", "okay",
        "...", ".", "-", "bye", "bye bye", "thank you", "thanks",
        "you", "the", "a", "and", "is", "it", "so", "but", "or",
        "ha", "huh", "eh", "hmm hmm", "mm hmm", "aha",
    })
    MIN_TRANSCRIPT_LEN = 2  # Reject transcripts shorter than 2 chars

    def __init__(self, agent):
        super().__init__()
        self._agent = agent

    def _is_noise(self, text: str) -> bool:
        """Check if transcript is likely background noise, not real speech.
        Filters: very short text, known noise words, repeated single words from chatter.
        """
        cleaned = text.strip().lower().rstrip(".,!?;:")
        if len(cleaned) < self.MIN_TRANSCRIPT_LEN:
            return True
        if cleaned in self.NOISE_WORDS:
            return True
        # Repeated single word (e.g., background chatter: "go go go", "yeah yeah")
        words = cleaned.split()
        if len(words) >= 2 and len(set(words)) == 1 and words[0] in self.NOISE_WORDS:
            return True
        # All words are noise words (e.g., "oh yeah ok" from background)
        if len(words) <= 3 and all(w in self.NOISE_WORDS for w in words):
            return True
        return False

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterimTranscriptionFrame):
            if frame.text:
                self._agent.stt_live = frame.text + "..."
                _fire_and_forget(self._agent.broadcast({"type": "stt_live", "text": self._agent.stt_live}))

        elif isinstance(frame, TranscriptionFrame):
            if frame.text:
                self._agent.stt_live = ""
                _fire_and_forget(self._agent.broadcast({"type": "stt_live", "text": ""}))

                # Filter noise transcriptions
                if self._is_noise(frame.text):
                    logger.info(f"STT noise filtered: '{frame.text}'")
                    _fire_and_forget(self._agent.broadcast({"type": "status", "status": "listening"}))
                    return  # Don't push to LLM or log as conversation

                self._agent.conversation_log.append({"role": "patient", "text": frame.text})
                self._agent._persist()
                _fire_and_forget(self._agent.broadcast({"type": "conversation", "role": "patient", "text": frame.text}))
                _fire_and_forget(self._agent.broadcast({"type": "status", "status": "thinking"}))
                logger.info(f"Patient said: '{frame.text}'")

        await self.push_frame(frame, direction)


class ResponseBroadcaster(FrameProcessor):
    """Broadcasts LLM response + TTS status events to WebSocket UI."""

    def __init__(self, agent):
        super().__init__()
        self._agent = agent
        self._llm_response_text = ""
        self._in_patient_data = False
        self._buffered_frames = []  # Buffer frames while checking for [PATIENT_DATA]
        self._accumulated_patient_data = {}  # Persists across responses — never loses data
        self._tool_called = False  # Set by function handler when tool is called
        self._spoken_text_from_tool = ""  # Spoken text extracted from tool call

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, LLMFullResponseStartFrame):
            self._llm_response_text = ""
            self._in_patient_data = False
            self._buffered_frames = []
            self._tool_called = False
            self._spoken_text_from_tool = ""
            _fire_and_forget(self._agent.broadcast({"type": "status", "status": "speaking"}))

        elif isinstance(frame, TextFrame):
            self._llm_response_text += frame.text

            # Already in patient data block — suppress
            if self._in_patient_data:
                if "[/PATIENT_DATA]" in self._llm_response_text:
                    self._in_patient_data = False
                return

            # Check if full tag appeared
            if "[PATIENT_DATA]" in self._llm_response_text:
                self._in_patient_data = True
                # Discard buffered frames (they contain partial tag text like "[", "PATIENT", etc.)
                self._buffered_frames = []
                return

            # Check if we might be in the middle of the tag — look for partial "[PATIENT_DATA]"
            # Buffer frames if the accumulated text ends with a prefix of "[PATIENT_DATA]"
            tag = "[PATIENT_DATA]"
            tail = self._llm_response_text[-len(tag):]  # last N chars
            might_be_tag = False
            for i in range(1, len(tag)):
                if tail.endswith(tag[:i]):
                    might_be_tag = True
                    break

            if might_be_tag:
                # Buffer this frame — don't send to TTS yet
                self._buffered_frames.append((frame, direction))
                return
            else:
                # Flush any buffered frames — they weren't part of a tag
                for bf, bd in self._buffered_frames:
                    await self.push_frame(bf, bd)
                self._buffered_frames = []
                await self.push_frame(frame, direction)
                return

        elif isinstance(frame, LLMFullResponseEndFrame):
            import re, json

            # PATH A: Tool-based flow (spoken_text came via function call)
            if self._tool_called and self._spoken_text_from_tool:
                display_text = self._spoken_text_from_tool.strip()
                if display_text:
                    # Push spoken text as TextFrames → flows to cache gate → TTS
                    await self.push_frame(TextFrame(text=display_text), FrameDirection.DOWNSTREAM)
                    # Log conversation
                    self._agent.conversation_log.append({"role": "medi", "text": display_text})
                    self._agent._persist()
                    _fire_and_forget(self._agent.broadcast({"type": "conversation", "role": "medi", "text": display_text}))
                    logger.info(f"Agent replied (tool): '{display_text}'")
                self._llm_response_text = ""
                _fire_and_forget(self._agent.broadcast({"type": "status", "status": "listening"}))
                await self.push_frame(frame, direction)
                return

            # PATH B: Text-based flow (fallback for models without tool support)
            if self._llm_response_text:
                full = self._llm_response_text
                display_text = full

                # Try multiple patterns — LLM may use markdown, backticks, etc.
                pd_match = re.search(r'\[PATIENT_DATA\]\s*(.*?)\s*\[/PATIENT_DATA\]', full, re.DOTALL)
                if not pd_match:
                    pd_match = re.search(r'\[PATIENT_DATA\]\s*(\{.*)', full, re.DOTALL)
                if not pd_match:
                    pd_match = re.search(r'(\{[^{}]*"symptoms"[^{}]*\})\s*$', full, re.DOTALL)

                if pd_match:
                    json_str = pd_match.group(1).strip()
                    json_str = re.sub(r'^```\w*\n?|```$', '', json_str).strip()
                    json_str = re.sub(r'\[/PATIENT_DATA\]\s*$', '', json_str).strip()
                    display_text = full[:pd_match.start()].strip()
                    display_text = re.sub(r'\s*\[PATIENT_DATA\].*', '', display_text, flags=re.DOTALL).strip()
                    try:
                        patient_data = json.loads(json_str)
                        self._merge_patient_data(patient_data)
                        acc = self._accumulated_patient_data
                        self._agent.accumulated_patient_data = acc
                        _fire_and_forget(self._agent.broadcast({"type": "patient_data", "data": acc}))
                        cluster_scores = compute_cluster_scores(acc.get("symptoms", []), acc.get("vitals", {}), acc.get("severity"), acc.get("age"), acc.get("gender"))
                        _fire_and_forget(self._agent.broadcast({"type": "cluster_data", "clusters": cluster_scores}))
                        logger.info(f"Patient data extracted: {acc}")
                    except Exception as e:
                        logger.warning(f"Failed to parse patient data JSON: {e}\nRaw: {json_str[:200]}")
                else:
                    display_text = re.sub(r'\[PATIENT_DATA\].*', '', display_text, flags=re.DOTALL).strip()
                    if self._tool_called:
                        logger.debug("No inline [PATIENT_DATA] but tool was called — data already broadcast")
                    else:
                        await self._extract_patient_data_async()
                        logger.debug(f"No patient data — triggered async extraction: {full[-100:]}")

                if display_text:
                    self._agent.conversation_log.append({"role": "medi", "text": display_text})
                    self._agent._persist()
                    _fire_and_forget(self._agent.broadcast({"type": "conversation", "role": "medi", "text": display_text}))
                    logger.info(f"Agent replied: '{display_text}'")
                self._llm_response_text = ""
            _fire_and_forget(self._agent.broadcast({"type": "status", "status": "listening"}))
            await self.push_frame(frame, direction)
            return

        await self.push_frame(frame, direction)

    def _merge_patient_data(self, new_data):
        """Merge new patient data into accumulated data — never overwrite with null/empty."""
        if not isinstance(new_data, dict):
            return
        if not self._accumulated_patient_data:
            self._accumulated_patient_data = new_data
            return
        acc = self._accumulated_patient_data
        for key, val in new_data.items():
            if key == "vitals":
                if not isinstance(val, dict):
                    continue
                if "vitals" not in acc or not isinstance(acc.get("vitals"), dict):
                    acc["vitals"] = {}
                for vk, vv in val.items():
                    if vv is not None:
                        acc["vitals"][vk] = vv
            elif key == "symptoms":
                if not isinstance(val, list):
                    continue
                # LLM is authoritative — use its symptom list directly.
                # The LLM is instructed to include ALL previously collected symptoms
                # plus new ones on each turn, so its list is the source of truth.
                # This allows the LLM to correct mistranscriptions (e.g., "کاشی" → "کھانسی").
                new_symptoms = [s for s in val if isinstance(s, str) and s]
                if new_symptoms:
                    acc["symptoms"] = new_symptoms
            elif val is not None and val != "" and val != "null":
                acc[key] = val

    async def _extract_patient_data_async(self):
        """When LLM skips [PATIENT_DATA], make a quick extraction call from conversation log."""
        import json as json_mod
        log = self._agent.conversation_log
        if not log or len(log) < 2:
            return

        # Build conversation summary for extraction
        conv_text = "\n".join(f"{'Patient' if m['role']=='patient' else 'Agent'}: {m['text']}" for m in log[-6:])

        extraction_prompt = (
            "Extract patient data from this hospital reception conversation. "
            "Return ONLY a JSON object with these fields: "
            '{"name": str|null, "age": str|null, "gender": "M"|"F"|null, '
            '"symptoms": [str], "duration": str|null, "severity": str|null, '
            '"vitals": {"fever": str|null, "bp": str|null, "heart_rate": str|null, '
            '"temperature": str|null, "sugar_level": str|null, "mobility": str|null}, '
            '"referred_doctor": str|null, "referred_dept": str|null, "referred_room": str|null}\n\n'
            f"Conversation:\n{conv_text}\n\nJSON:"
        )

        try:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=OPENAI_KEY)
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": extraction_prompt}],
                max_tokens=300,
                temperature=0,
            )
            raw = resp.choices[0].message.content.strip()
            # Clean markdown wrapper if present
            import re
            raw = re.sub(r'^```\w*\n?|```$', '', raw).strip()
            patient_data = json_mod.loads(raw)
            self._merge_patient_data(patient_data)
            acc = self._accumulated_patient_data
            self._agent.accumulated_patient_data = acc
            _fire_and_forget(self._agent.broadcast({"type": "patient_data", "data": acc}))
            cluster_scores = compute_cluster_scores(acc.get("symptoms", []), acc.get("vitals", {}), acc.get("severity"), acc.get("age"), acc.get("gender"))
            _fire_and_forget(self._agent.broadcast({"type": "cluster_data", "clusters": cluster_scores}))
            logger.info(f"Async extraction: {acc}")
        except Exception as e:
            logger.warning(f"Async patient data extraction failed: {e}")
            # Still re-send accumulated if we have any
            if self._accumulated_patient_data:
                acc = self._accumulated_patient_data
                self._agent.accumulated_patient_data = acc
                _fire_and_forget(self._agent.broadcast({"type": "patient_data", "data": acc}))
                cluster_scores = compute_cluster_scores(acc.get("symptoms", []), acc.get("vitals", {}), acc.get("severity"), acc.get("age"), acc.get("gender"))
                _fire_and_forget(self._agent.broadcast({"type": "cluster_data", "clusters": cluster_scores}))

    def reset_patient_data(self):
        """Reset accumulated data for new patient session."""
        self._accumulated_patient_data = {}


# ─────────────────────────────────────────────
# Conversation Agent (Pipecat-based)
# ─────────────────────────────────────────────

class ConversationAgent:
    _SETTINGS_KEYS = [
        "agent_name", "voice_gender", "personality", "lang_style",
        "llm_model", "stt_model", "stt_language", "tts_provider", "tts_model", "tts_voice",
    ]

    def __init__(self):
        logger.info("Agent: Initializing Hospital Support Desk Agent...")
        self.status = "ready"
        self.stt_live = ""
        self.conversation_log = []
        self._patient_count = 0
        self._ws_clients = set()
        self._app_loop = None
        self._pipeline_task = None
        self._pipeline_runner = None

        # Agent persona settings (defaults)
        self.agent_name = "Amina"
        self.voice_gender = "F"
        self.personality = "warm"
        self.lang_style = "formal"
        self.llm_model = "gpt-4o-mini"
        self.stt_model = "Deepgram Nova-3"
        self.stt_language = "ur"
        self.tts_provider = "OpenAI"
        self.tts_model = "tts-1"
        self.tts_voice = "nova"

        # Restore saved settings (but NOT conversation) on startup
        saved = _load_session()
        if saved:
            for key in self._SETTINGS_KEYS:
                if key in saved:
                    setattr(self, key, saved[key])
            logger.info(f"Session: Restored settings — agent={self.agent_name}, voice={self.tts_voice}, provider={self.tts_provider}")
        # Clear conversation on startup (fresh start)
        self._persist()

        # Patient data — persists across WS reconnects within a session
        self.accumulated_patient_data = {}

        # Active pipecat services (updated per session)
        self._llm_service = None
        self._tts_service = None
        self._tts_proxy = None
        self._llm_context = None

        # WebRTC connection management
        self._pcs_map: Dict[str, SmallWebRTCConnection] = {}

        logger.info("Agent: Initialized")

    def _reset_session(self):
        """Clear conversation log and reset transient state. Settings are preserved."""
        self.conversation_log = []
        self.stt_live = ""
        self.status = "ready"
        self.accumulated_patient_data = {}
        self._persist()

    def _persist(self):
        """Save current settings + conversation to disk (non-blocking via thread pool)."""
        data = {k: getattr(self, k) for k in self._SETTINGS_KEYS}
        data["conversation_log"] = self.conversation_log
        import asyncio
        try:
            loop = asyncio.get_running_loop()
            loop.run_in_executor(None, _save_session, data)
        except RuntimeError:
            _save_session(data)  # No event loop — save synchronously

    async def broadcast(self, msg: dict):
        dead = set()
        for ws in list(self._ws_clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.add(ws)
        self._ws_clients -= dead

    def _build_system_prompt(self):
        return build_system_prompt(self.agent_name, self.voice_gender, self.personality, self.lang_style, self.stt_language)

    def _create_tts_service(self):
        """Create TTS service based on current provider setting."""
        if self.tts_provider == "ElevenLabs" and ELEVENLABS_KEY:
            voice_id = ELEVENLABS_VOICE_MAP.get(self.tts_voice, "21m00Tcm4TlvDq8ikWAM")
            # Use flash_v2_5 by default — pipecat 0.0.85 WebSocket API doesn't support eleven_v3 yet.
            # flash_v2_5 handles Urdu text even though not officially listed.
            # eleven_v3 works via REST API but fails on WebSocket (HTTP 403).
            el_model = self.tts_model if self.tts_model.startswith("eleven_") else "eleven_flash_v2_5"
            if el_model == "eleven_v3":
                el_model = "eleven_flash_v2_5"  # Fallback — v3 not supported by pipecat WebSocket
            logger.info(f"TTS: Using ElevenLabs model={el_model} voice={self.tts_voice} id={voice_id}")
            return ElevenLabsTTSService(
                api_key=ELEVENLABS_KEY,
                voice_id=voice_id,
                model=el_model,
                sample_rate=SAMPLE_RATE_OUT,
            )
        else:
            oai_model = self.tts_model if self.tts_model.startswith("tts-") else "tts-1"
            logger.info(f"TTS: Using OpenAI model={oai_model} voice={self.tts_voice}")
            return OpenAITTSService(
                api_key=OPENAI_KEY,
                voice=self.tts_voice,
                model=oai_model,
                sample_rate=SAMPLE_RATE_OUT,
            )

    # Map language codes to pipecat Language enum for OpenAI STT
    _LANG_TO_PIPECAT = {
        "ur": Language.UR_PK, "en": Language.EN, "hi": Language.HI,
        "ar": Language.AR, "pa": Language.PA, "ms": Language.MS,
    }

    def _create_stt_service(self):
        """Create STT service based on current stt_model setting."""
        if self.stt_model == "OpenAI Realtime":
            lang = self._LANG_TO_PIPECAT.get(self.stt_language, Language.EN)
            logger.info(f"STT: Using OpenAI gpt-4o-transcribe language={lang}")
            return OpenAISTTService(
                api_key=OPENAI_KEY,
                model="gpt-4o-transcribe",
                language=lang,
            )
        else:
            logger.info(f"STT: Using Deepgram Nova-3 language={self.stt_language} (confidence filter={STT_CONFIDENCE_THRESHOLD})")
            return ConfidenceFilteredDeepgramSTT(
                api_key=DEEPGRAM_KEY,
                live_options=LiveOptions(
                    model="nova-3",
                    language=self.stt_language,
                    encoding="linear16",
                    sample_rate=SAMPLE_RATE_IN,
                    channels=1,
                    punctuate=True,
                    smart_format=True,
                    interim_results=True,
                    endpointing=600,       # 600ms — balanced for Urdu speech patterns
                    utterance_end_ms="1500",  # 1.5s utterance timeout
                    no_delay=True,         # Reduce STT latency — send results immediately
                    vad_events=True,       # Enable Deepgram's own VAD events
                    filler_words=False,    # Exclude um/uh for cleaner transcription
                ),
            )

    def get_state(self):
        return {
            "conversation": self.conversation_log,
            "status": self.status,
            "stt_live": self.stt_live,
            "recording": self._pipeline_task is not None,
            "doctors": DOCTORS_PANEL,
            "disease_clusters": DISEASE_CLUSTERS_DATA,
            "patient_data": self.accumulated_patient_data if self.accumulated_patient_data else None,
            "config": {
                "stt_model": self.stt_model,
                "stt_language": next((k for k, v in STT_LANGUAGES.items() if v == self.stt_language), "Urdu"),
                "llm_model": self.llm_model,
                "tts_provider": self.tts_provider,
                "tts_model": self.tts_model,
                "tts_voice": self.tts_voice,
                "voice_gender": self.voice_gender,
                "agent_name": self.agent_name,
                "personality": self.personality,
                "lang_style": self.lang_style,
            },
        }

    async def stop(self):
        if self._pipeline_task:
            await self._pipeline_task.queue_frame(EndFrame())
            self._pipeline_task = None
        # Close all WebRTC connections
        for pc_id, pc in list(self._pcs_map.items()):
            try:
                await pc.disconnect()
            except Exception:
                pass
        self._pcs_map.clear()
        self._reset_session()
        self.status = "ended"
        _fire_and_forget(self.broadcast({"type": "stt_live", "text": ""}))
        _fire_and_forget(self.broadcast({"type": "status", "status": "ended"}))
        logger.info("Agent: Session ended — session reset")

    async def run_bot(self, webrtc_connection: SmallWebRTCConnection):
        """Run a pipecat pipeline for one patient session."""
        self._patient_count += 1
        self._reset_session()
        self.status = "connecting"
        _fire_and_forget(self.broadcast({"type": "status", "status": "connecting"}))
        _fire_and_forget(self.broadcast({"type": "clear"}))

        # Noise filter — cleans audio BEFORE VAD for better speech detection + cleaner STT
        # If Krisp available: use it + allow interruptions (best experience)
        # If not: disable interruptions so background noise can't cut off bot speech
        noise_filter = None
        has_ai_noise_filter = False
        try:
            from pipecat.audio.filters.krisp_filter import KrispFilter
            noise_filter = KrispFilter()
            has_ai_noise_filter = True
            logger.info("Audio filter: Krisp (AI noise cancellation) — interruptions ENABLED")
        except Exception:
            logger.info("Audio filter: None (Krisp not available) — interruptions DISABLED to handle background noise")

        # Transport — Silero VAD tuned for Urdu speech in hospital environments
        # Note: Do NOT use vad_enabled=True with vad_analyzer — it's deprecated and causes double processing
        transport = SmallWebRTCTransport(
            webrtc_connection=webrtc_connection,
            params=TransportParams(
                audio_in_enabled=True,
                audio_out_enabled=True,
                audio_in_sample_rate=SAMPLE_RATE_IN,
                audio_out_sample_rate=SAMPLE_RATE_OUT,
                audio_in_filter=noise_filter,
                vad_analyzer=SileroVADAnalyzer(
                    params=VADParams(
                        confidence=0.6,    # Balanced for Urdu speech
                        start_secs=0.2,    # 200ms — reliable speech onset detection
                        stop_secs=0.6,     # 600ms — synced with Deepgram endpointing
                        min_volume=0.5,    # Moderate threshold
                    ),
                ),
            ),
        )

        # STT — use selected model (Deepgram or OpenAI)
        stt = self._create_stt_service()

        # LLM — select provider based on model name
        system_prompt = self._build_system_prompt()
        if self.llm_model in CLAUDE_MODELS:
            self._llm_service = AnthropicLLMService(
                api_key=ANTHROPIC_KEY,
                model=self.llm_model,
            )
        else:
            self._llm_service = OpenAILLMService(
                api_key=OPENAI_KEY,
                model=self.llm_model,
            )

        # LLM Context with system prompt + patient screening tool
        messages = [{"role": "system", "content": system_prompt}]
        tools = [{"type": "function", "function": PATIENT_SCREENING_TOOL.to_default_dict()}]
        self._llm_context = OpenAILLMContext(messages=messages, tools=tools, tool_choice="required")
        context_aggregator = self._llm_service.create_context_aggregator(self._llm_context)

        # TTS — choose provider based on current setting
        self._tts_service = self._create_tts_service()

        # Broadcasters: STT events before aggregator, LLM/TTS events after TTS
        audio_level_monitor = AudioLevelMonitor(self)
        stt_broadcaster = STTBroadcaster(self)
        response_broadcaster = ResponseBroadcaster(self)

        # Register patient screening tool handler — broadcasts data to WebSocket UI
        agent_ref = self  # closure reference

        async def handle_patient_screening(params: FunctionCallParams):
            data = dict(params.arguments)
            # Extract spoken text (will be injected into pipeline by ResponseBroadcaster)
            spoken_text = data.pop("spoken_text", "")
            response_broadcaster._spoken_text_from_tool = spoken_text
            response_broadcaster._tool_called = True

            try:
                vitals = data.get("vitals", {})
                if not isinstance(vitals, dict):
                    vitals = {}
                # Ensure symptoms is always a list
                symptoms = data.get("symptoms", [])
                if not isinstance(symptoms, list):
                    symptoms = [symptoms] if symptoms else []
                patient_data = {
                    "name": data.get("name") or None,
                    "age": data.get("age") or None,
                    "gender": data.get("gender") or None,
                    "symptoms": symptoms,
                    "duration": data.get("duration") or None,
                    "severity": data.get("severity") or None,
                    "vitals": vitals,
                    "referred_doctor": data.get("referred_doctor") or None,
                    "referred_dept": data.get("referred_dept") or None,
                    "referred_room": data.get("referred_room") or None,
                    "is_complete": data.get("is_complete", False),
                    "emergency": data.get("emergency", False),
                }
                response_broadcaster._merge_patient_data(patient_data)
                acc = response_broadcaster._accumulated_patient_data
                # Sync to agent-level state (survives WS reconnects)
                agent_ref.accumulated_patient_data = acc
                # UI update — non-blocking
                _fire_and_forget(agent_ref.broadcast({"type": "patient_data", "data": acc}))

                # Compute triage scores when symptoms OR vitals exist (skip on greeting/name turns)
                symptoms_list = acc.get("symptoms", [])
                vitals_data = acc.get("vitals", {})
                has_clinical_data = bool(symptoms_list) or bool(vitals_data and any(v for v in vitals_data.values() if v))
                if has_clinical_data:
                    cluster_scores = compute_cluster_scores(
                        symptoms_list,
                        acc.get("vitals", {}),
                        acc.get("severity"),
                        acc.get("age"),
                        acc.get("gender"),
                    )
                    # UI triage dashboard — non-blocking
                    _fire_and_forget(agent_ref.broadcast({"type": "cluster_data", "clusters": cluster_scores}))
                else:
                    cluster_scores = []

                # Build triage summary for LLM context — directive referral guidance
                top_matches = [c for c in cluster_scores if c["score"] > 0]
                if top_matches:
                    # Build availability lookup from DOCTORS_PANEL
                    _avail = {d["name"]: d["available"] for d in DOCTORS_PANEL}

                    # Always refer to the HIGHEST scoring department — correct specialty matters most.
                    # If doctor is unavailable, still refer to the department with a note.
                    refer_to = top_matches[0]
                    if not _avail.get(refer_to["doctor"], True):
                        refer_to = dict(refer_to)  # copy to modify
                        refer_to["_unavailable"] = True

                    top = top_matches[0]
                    top_score = top["score"]
                    top_pct = round(top_score * 100)
                    triage_lines = []
                    for c in top_matches[:3]:
                        pct = round(c["score"] * 100)
                        matched = ", ".join(c.get("matched_symptoms", []))
                        avail_tag = "" if _avail.get(c["doctor"], True) else " [NOT AVAILABLE]"
                        triage_lines.append(f"{c['department']} ({c['doctor']}, Room {c['room']}): {pct}% [{matched}]{avail_tag}")
                    triage_summary = "[TRIAGE SCORES: " + " | ".join(triage_lines) + "]"
                    if top_score >= 0.5:
                        unavail_note = " (DOCTOR NOT AVAILABLE TODAY — tell patient to check at reception for alternative)" if refer_to.get("_unavailable") else ""
                        triage_summary += f" [READY TO REFER → {refer_to['doctor']}, {refer_to['department']}, Room {refer_to['room']}{unavail_note}]"
                    else:
                        unavail_note = " (DOCTOR NOT AVAILABLE TODAY)" if refer_to.get("_unavailable") else ""
                        triage_summary += f" [CONFIDENCE LOW ({top_pct}%) — ask more symptoms or vitals. When ready, REFER TO → {refer_to['doctor']}, {refer_to['department']}, Room {refer_to['room']}{unavail_note}]"
                else:
                    triage_summary = "[TRIAGE: No matches yet — gather symptoms]"

                logger.info(f"Tool: spoken='{spoken_text[:60]}' symptoms={acc.get('symptoms', [])} severity={acc.get('severity')} triage_top={top_matches[0]['department'] if top_matches else 'none'}:{round(top_matches[0]['score']*100) if top_matches else 0}%")
            except Exception as e:
                logger.error(f"Tool handler error: {e}\n{traceback.format_exc()}")
                triage_summary = ""
                # Still try to broadcast whatever we have
                try:
                    acc = response_broadcaster._accumulated_patient_data
                    if acc:
                        _fire_and_forget(agent_ref.broadcast({"type": "patient_data", "data": acc}))
                except Exception:
                    pass

            # Return triage summary with spoken text — LLM sees scores on next turn
            result_text = spoken_text
            if triage_summary:
                result_text = f"{spoken_text}\n{triage_summary}"
            await params.result_callback(
                result_text,
                properties=FunctionCallResultProperties(run_llm=False),
            )

        self._llm_service.register_function("update_patient_screening", handle_patient_screening)

        # TTS Cache: gate (before TTS) checks cache, capture (after TTS) stores misses
        cache_gate = TTSCacheGate(self)
        cache_capture = TTSCacheCapture(self, cache_gate)

        # User idle detection — gently re-prompts after 15s of silence
        from pipecat.processors.user_idle_processor import UserIdleProcessor
        async def _handle_user_idle(processor, retry_count):
            if retry_count >= 3:
                return False  # Give up after 3 retries
            prompts = {
                "ur": "(مریض خاموش ہے — شائستگی سے اپنا آخری سوال دوبارہ پوچھیں، مختصر اور آسان الفاظ میں)",
                "en": "(Patient is silent — politely rephrase your last question, shorter and simpler)",
            }
            prompt = prompts.get(self.stt_language, prompts["en"])
            await task.queue_frames([LLMMessagesFrame(
                messages=[{"role": "user", "content": prompt}]
            )])
            return True  # Keep retrying

        user_idle = UserIdleProcessor(callback=_handle_user_idle, timeout=15.0)

        # Pipeline: audio in → STT → stt_broadcast → idle → user context → LLM → response_broadcast → cache_gate → TTS → cache_capture → assistant context → audio out
        pipeline = Pipeline([
            transport.input(),
            audio_level_monitor,
            stt,
            stt_broadcaster,
            user_idle,
            context_aggregator.user(),
            self._llm_service,
            response_broadcaster,
            cache_gate,
            self._tts_service,
            cache_capture,
            context_aggregator.assistant(),
            transport.output(),
        ])

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                audio_in_sample_rate=SAMPLE_RATE_IN,
                audio_out_sample_rate=SAMPLE_RATE_OUT,
                allow_interruptions=True,  # Always allow — STT runs continuously, never blocked by TTS
                enable_metrics=True,
            ),
        )
        self._pipeline_task = task

        @transport.event_handler("on_client_connected")
        async def on_connected(transport, client):
            self.status = "listening"
            _fire_and_forget(self.broadcast({"type": "status", "status": "listening"}))
            logger.info(f"WebRTC: Client connected, patient #{self._patient_count}")
            # Auto-greeting: inject a user message to trigger the LLM immediately
            hn = HOSPITAL_NAME
            greeting_prompts = {
                "ur": f"(نیا مریض آیا ہے۔ آپ پہلے بول رہے ہیں — السلام علیکم کہیں (وعلیکم السلام نہیں)، {hn} میں خوش آمدید کہیں، اور نام پوچھیں۔)",
                "en": f"(A new patient has arrived. Say welcome to {hn}, mention the hospital name, and ask their name.)",
                "hi": f"(नया मरीज़ आया है। नमस्ते कहें, {hn} का नाम लें, स्वागत करें, और नाम पूछें।)",
                "ar": f"(مريض جديد وصل. قل السلام عليكم، اذكر اسم {hn}، رحب به، واسأله عن اسمه.)",
                "pa": f"(ਨਵਾਂ ਮਰੀਜ਼ ਆਇਆ ਹੈ। ਸਤ ਸ੍ਰੀ ਅਕਾਲ ਕਹੋ, {hn} ਦਾ ਨਾਮ ਲਓ, ਜੀ ਆਇਆਂ ਨੂੰ ਕਹੋ, ਅਤੇ ਨਾਮ ਪੁੱਛੋ।)",
                "ms": f"(Pesakit baru tiba. Ucap salam, sebut nama {hn}, alu-alukan, dan tanya nama mereka.)",
            }
            prompt = greeting_prompts.get(self.stt_language, greeting_prompts["en"])
            await task.queue_frames([LLMMessagesFrame(
                messages=[{"role": "user", "content": prompt}]
            )])

        @transport.event_handler("on_client_disconnected")
        async def on_disconnected(transport, client):
            self.status = "ended"
            _fire_and_forget(self.broadcast({"type": "status", "status": "ended"}))
            await task.cancel()
            self._pipeline_task = None
            logger.info("WebRTC: Client disconnected")

        runner = PipelineRunner(handle_sigint=False)
        logger.info(f"Agent: Patient #{self._patient_count} pipeline started")
        await runner.run(task)
        self._pipeline_task = None
        logger.info(f"Agent: Patient #{self._patient_count} pipeline ended")


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

agent = ConversationAgent()

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Shutdown: close all WebRTC connections
    for pc_id, pc in list(agent._pcs_map.items()):
        try:
            await pc.disconnect()
        except Exception:
            pass
    agent._pcs_map.clear()

app = FastAPI(title="Hospital Support Desk", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# HTML Frontend
_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "templates", "conversation_agent.html")
with open(_TEMPLATE_PATH, "r", encoding="utf-8") as _f:
    HTML_PAGE = _f.read().replace("{{HOSPITAL_NAME}}", HOSPITAL_NAME)
logger.info(f"Loaded HTML template from {_TEMPLATE_PATH} ({len(HTML_PAGE)} chars)")


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.post("/api/offer")
async def offer(request: dict, background_tasks: BackgroundTasks):
    """Handle WebRTC offer — creates or reuses a connection and starts the pipeline."""
    pc_id = request.get("pc_id")

    if pc_id and pc_id in agent._pcs_map:
        connection = agent._pcs_map[pc_id]
        logger.info(f"WebRTC: Reusing connection {pc_id}")
        await connection.renegotiate(
            sdp=request["sdp"],
            type=request["type"],
            restart_pc=request.get("restart_pc", False),
        )
    else:
        connection = SmallWebRTCConnection(ice_servers=ICE_SERVERS)
        await connection.initialize(sdp=request["sdp"], type=request["type"])

        @connection.event_handler("closed")
        async def handle_closed(conn: SmallWebRTCConnection):
            cid = conn.pc_id
            logger.info(f"WebRTC: Connection closed {cid}")
            if cid in agent._pcs_map:
                del agent._pcs_map[cid]

        background_tasks.add_task(agent.run_bot, connection)

    answer = connection.get_answer()
    agent._pcs_map[answer["pc_id"]] = connection
    logger.info("WebRTC: Answer generated")
    return answer


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    agent._ws_clients.add(websocket)
    agent._app_loop = asyncio.get_event_loop()
    logger.info(f"WS: Client connected (total: {len(agent._ws_clients)})")

    try:
        await websocket.send_json({"type": "state", **agent.get_state()})
    except Exception:
        pass

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "stop":
                await agent.stop()
            elif action == "settings_batch":
                # Batch update all settings at once (faster than individual messages)
                settings = data.get("settings", {})
                prompt_changed = False
                for key, value in settings.items():
                    if key == "stt_language":
                        agent.stt_language = STT_LANGUAGES.get(value, "ur")
                    elif key == "llm_model":
                        agent.llm_model = value
                    elif key == "stt_model":
                        agent.stt_model = value
                    elif key == "tts_provider":
                        agent.tts_provider = value
                    elif key == "tts_model":
                        agent.tts_model = value
                    elif key == "tts_voice":
                        agent.tts_voice = value
                    elif key in ("voice_gender", "agent_gender"):
                        agent.voice_gender = value
                        prompt_changed = True
                    elif key == "agent_name":
                        agent.agent_name = value
                        prompt_changed = True
                    elif key == "agent_personality":
                        agent.personality = value
                        prompt_changed = True
                    elif key == "agent_lang_style":
                        agent.lang_style = value
                        prompt_changed = True
                if prompt_changed:
                    _update_system_prompt()
                agent._persist()
                logger.info(f"WS: Batch settings updated ({len(settings)} keys)")
            elif action == "setting":
                key, value = data.get("key"), data.get("value")
                if key == "stt_language":
                    agent.stt_language = STT_LANGUAGES.get(value, "ur")
                elif key == "llm_model":
                    old_is_claude = agent.llm_model in CLAUDE_MODELS
                    new_is_claude = value in CLAUDE_MODELS
                    agent.llm_model = value
                    if old_is_claude != new_is_claude:
                        # Provider changed — needs new session
                        _fire_and_forget(agent.broadcast({"type": "info", "text": f"LLM switched to {value} — takes effect on next session"}))
                    elif agent._llm_service:
                        agent._llm_service.set_model_name(value)
                elif key == "stt_model":
                    agent.stt_model = value
                    _fire_and_forget(agent.broadcast({"type": "info", "text": f"STT model set to {value} — takes effect on next session"}))
                elif key == "tts_provider":
                    agent.tts_provider = value
                    # Takes effect on next session — no live swap to avoid pipeline breakage
                elif key == "tts_model":
                    agent.tts_model = value
                    # Takes effect on next session
                elif key == "tts_voice":
                    agent.tts_voice = value
                    # Voice change is safe live — just updates a property, no service swap
                    if agent._tts_service:
                        if agent.tts_provider == "OpenAI":
                            agent._tts_service._voice = value
                        elif agent.tts_provider == "ElevenLabs":
                            voice_id = ELEVENLABS_VOICE_MAP.get(value, "21m00Tcm4TlvDq8ikWAM")
                            agent._tts_service._voice_id = voice_id
                            logger.info(f"TTS: ElevenLabs voice updated to {value} ({voice_id})")
                elif key == "voice_gender":
                    agent.voice_gender = value
                    _update_system_prompt()
                elif key == "agent_name":
                    agent.agent_name = value
                    _update_system_prompt()
                elif key == "agent_gender":
                    agent.voice_gender = value
                    _update_system_prompt()
                elif key == "agent_personality":
                    agent.personality = value
                    _update_system_prompt()
                elif key == "agent_lang_style":
                    agent.lang_style = value
                    _update_system_prompt()
                logger.info(f"WS: Setting {key}={value}")
                agent._persist()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WS: Error: {e}")
    finally:
        agent._ws_clients.discard(websocket)
        logger.info(f"WS: Client disconnected (total: {len(agent._ws_clients)})")


def _update_system_prompt():
    """Update system prompt in active LLM context if a session is running."""
    if agent._llm_context and hasattr(agent._llm_context, 'messages') and agent._llm_context.messages:
        new_prompt = agent._build_system_prompt()
        agent._llm_context.messages[0] = {"role": "system", "content": new_prompt}
        logger.info("LLM: System prompt updated mid-session")


if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {HOSPITAL_NAME} Support Desk on port 7871")
    uvicorn.run(app, host="0.0.0.0", port=7871, log_level="warning")
