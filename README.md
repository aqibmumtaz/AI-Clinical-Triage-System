# AI Clinical Triage System

## Overview

The **AI Clinical Triage System** is a real-time, voice-based clinical triage platform that serves as the first point of contact for patients arriving at a hospital. The system combines Speech-to-Text (STT), Large Language Models (LLM), and Text-to-Speech (TTS) in a seamless pipeline to enable natural voice conversations in **6 languages** (Urdu, English, Hindi, Arabic, Punjabi, Malay).

It collects patient symptoms through natural conversation, scores them against a disease cluster database (70+ conditions across 12 departments), and routes patients to the correct specialist — all in under 2 minutes.

## Architecture

```
Browser (WebRTC Audio + WebSocket UI)
    |                    |
    v                    v
WebRTC Transport    WebSocket (/ws)
    |                    |
    v                    v
Pipecat Pipeline    UI State Manager
    |
    v
[Mic 16kHz] → [Silero VAD 0.6] → [AudioLevelMonitor] → [ConfidenceFilteredSTT 0.45]
→ [STTBroadcaster + NoiseFilter] → [UserContext] → [GPT-4o-mini + TriageScoring]
→ [ResponseBroadcaster] → [TTSCacheGate] → [TTS 24kHz] → [TTSCacheCapture] → [WebRTC Out]

All UI broadcasts: non-blocking (fire-and-forget) | allow_interruptions=true | STT always running
```

### Key Components

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **STT** | Deepgram Nova-3 / OpenAI gpt-4o-transcribe | Dual STT providers with confidence-based noise filtering |
| **VAD** | Silero VAD (neural) | Voice activity detection — filters background speakers |
| **LLM** | OpenAI GPT-4o-mini / GPT-4o / Claude | Conversational AI for patient triage |
| **Triage Engine** | Custom scoring algorithm | Disease cluster matching with alias expansion, cross-condition boost, department rules |
| **TTS** | OpenAI TTS-1 / ElevenLabs Flash v2.5 | Natural voice synthesis with multilingual support |
| **Transport** | WebRTC (SmallWebRTC) | Browser-to-server audio streaming with AEC |
| **Pipeline** | Pipecat-AI 0.0.85 | Async frame-based pipeline orchestration |
| **Frontend** | HTML/CSS/JS (single file) | Real-time triage dashboard, patient screening card, doctors panel |
| **Backend** | FastAPI + Uvicorn | HTTP, WebSocket, and WebRTC endpoints |

## Triage Scoring Engine

The scoring engine is the core of the system — it matches patient symptoms against 70+ medical conditions organized in 12 departments.

### How It Works

1. **Symptom Alias Expansion** — Patient symptoms (in any language) are expanded via alias lookup. "کھانسی" → "cough", "سینے میں درد" → "chest pain"
2. **Per-Symptom Matching** — Each patient symptom matches at most ONE condition symptom (prevents double-counting)
3. **Exact Match Only** — "cough" matches "cough" but NOT "persistent cough" (clinically different)
4. **Vital Boost** — +10% per matching vital sign (BP, heart rate, temperature, sugar, mobility)
5. **Severity Boost** — +5% when severity matches condition severity levels
6. **Cross-Condition Boost** — When symptoms match multiple conditions in a department, confidence increases (+15% per additional strong condition)
7. **Department Selection Rules** — Age-gated (Pediatrics: children only), gender-gated (Gynecology: females only)
8. **Doctor Availability Check** — Unavailable doctors are skipped in referral directives

### Scoring Formula

```
condition_score = (matched_symptoms / total_condition_symptoms) + vital_boost + severity_boost
department_score = best_condition_score + cross_condition_boost
final_score = apply_department_rules(department_score, age, gender)
```

### Disease Clusters

| Department | Doctor | Conditions | Key Symptoms |
|-----------|--------|-----------|--------------|
| General Medicine | Dr. Ahmed Khan | 7 | Fever, cough, cold, body ache, UTI, diabetes, hypertension |
| Cardiology | Dr. Ali Raza | 4 | Chest pain, palpitations, shortness of breath, heart failure |
| Gynecology | Dr. Fatima Zahra | 4 | Pregnancy, menstrual disorders, PID, prenatal |
| Pediatrics | Dr. Sarah Malik | 4 | Childhood fever, asthma, nutrition, vaccination |
| Orthopedics | Dr. Usman Tariq | 3 | Fractures, arthritis, back pain |
| Dermatology | Dr. Ayesha Siddiqui | 3 | Skin allergy, psoriasis, acne |
| ENT | Dr. Hassan Javed | 3 | Ear infection, sinusitis, vertigo |
| Ophthalmology | Dr. Nadia Hussain | 3 | Vision problems, glaucoma, cataract |
| Neurology | Dr. Bilal Qureshi | 3 | Migraine, epilepsy, stroke, neuropathy |
| Dentistry | Dr. Zainab Akhtar | 2 | Toothache, dental abscess |
| Gastroenterology | Dr. Imran Shah | 4 | Gastritis, IBS, hepatitis, ulcer |
| Psychiatry | Dr. Maryam Riaz | 4 | Anxiety, depression, insomnia, PTSD |

## Noise Filtering (Three-Layer Defense)

### Layer 1: Silero VAD (Audio Level)
- Neural voice activity detection (ONNX model)
- `confidence=0.6` — balanced for Urdu speech
- `min_volume=0.5` — moderate threshold
- `start_secs=0.2` — 200ms speech onset detection
- `stop_secs=0.6` — 600ms silence to end utterance (synced with Deepgram)

### Layer 2: Confidence Filtering (STT Level)
- Custom `ConfidenceFilteredDeepgramSTT` subclass
- Drops final transcriptions with confidence < 0.45 (Urdu-friendly)
- Lets interim results through for live UI display
- Null-safe (handles missing confidence gracefully)

### Layer 3: Noise Word Filter (Text Level)
- Drops known noise words (hmm, ok, yeah, etc.)
- Drops repeated single words and short noise phrases
- Preserves valid Urdu medical terms

### Current Working Pipeline Parameters
```
Transport:     allow_interruptions=true, audio_in_filter=None
Silero VAD:    confidence=0.6, start_secs=0.2, stop_secs=0.6, min_volume=0.5
Deepgram STT:  endpointing=600ms, utterance_end_ms=1500, no_delay=true, vad_events=true
STT Filter:    confidence_threshold=0.45
LLM:           gpt-4o-mini, tool_choice=required
TTS:           OpenAI tts-1, voice=nova, 24kHz
Performance:   All UI broadcasts non-blocking, _persist() in thread pool
Conda env:     speech-ai-rd
```

## Quick Start

### Prerequisites
- Python 3.10+
- API keys: OpenAI, Deepgram, ElevenLabs (optional)

### Setup

```bash
# Use the speech-ai-rd conda environment
conda activate speech-ai-rd
cd R&D/AI-Clinical-Triage-System
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

### Run

```bash
python conversation_agent.py
# Open http://localhost:7871 in browser
```

### Test

```bash
# Triage scoring tests (188 tests)
pytest test_disease_clusters.py -v

# Automated agent tests (77 tests)
cd automated_testing
pytest test_conversation_agent.py -v
```

## Directory Structure

```
R&D/AI-Clinical-Triage-System/
├── conversation_agent.py          # Main backend (FastAPI + Pipecat + Triage Engine)
├── disease_clusters.json          # Disease database (12 depts, 70+ conditions, 173 aliases)
├── test_disease_clusters.py       # 188 triage scoring tests
├── requirements.txt               # Python dependencies
├── templates/
│   └── conversation_agent.html    # Frontend UI (triage dashboard, patient card, settings)
├── automated_testing/
│   └── test_conversation_agent.py # 77 automated tests
└── output/
    ├── session_state.json         # Persisted session settings
    └── agent_terminal.log         # Runtime logs
```

## Features

### Clinical Triage
- **Real-time disease cluster scoring** — live percentage-based triage as symptoms are collected
- **173 symptom aliases** — covers Urdu, Roman Urdu, English, and common mistranscriptions
- **Smart follow-up questions** — asks targeted symptoms to differentiate between conditions
- **Cross-condition confidence boost** — higher confidence when multiple conditions in a department match
- **Age/gender-aware scoring** — Pediatrics suppressed for adults, Gynecology for males
- **Doctor availability check** — skips unavailable doctors in referral
- **Urdu tooltips** — hover over any English symptom tag to see Urdu translation

### Conversation Agent
- **10 configurable personas** (5 male, 5 female) with distinct voices
- **Dual STT providers** — Deepgram Nova-3 and OpenAI gpt-4o-transcribe
- **Background noise filtering** — two-layer defense (VAD + confidence filtering)
- **TTS audio caching** — disk-backed LRU cache, ~85% cost reduction
- **Number-to-Urdu TTS** — room numbers spoken in Urdu words while display keeps digits
- **Off-topic answer handling** — re-asks unanswered symptom questions
- **Pure Urdu output** — no English word mixing in Urdu responses

### Supported Languages
| Language | Direction | STT | LLM | TTS |
|----------|-----------|-----|-----|-----|
| Urdu | RTL | Deepgram Nova-3 / OpenAI | GPT-4o-mini | OpenAI / ElevenLabs |
| English | LTR | Deepgram Nova-3 / OpenAI | GPT-4o-mini | OpenAI / ElevenLabs |
| Hindi | LTR | Deepgram Nova-3 / OpenAI | GPT-4o-mini | OpenAI / ElevenLabs |
| Arabic | RTL | Deepgram Nova-3 / OpenAI | GPT-4o-mini | OpenAI / ElevenLabs |
| Punjabi | RTL | Deepgram Nova-3 / OpenAI | GPT-4o-mini | OpenAI / ElevenLabs |
| Malay | LTR | Deepgram Nova-3 / OpenAI | GPT-4o-mini | OpenAI / ElevenLabs |

## Test Results

| Category | Tests | Status |
|----------|-------|--------|
| Scoring Accuracy | 8 | PASS |
| Department Ranking | 11 | PASS |
| Vital Combinations | 7 | PASS |
| Alias Expansion | 11 | PASS |
| Referral Scenarios | 11 | PASS |
| Score Differentiation | 7 | PASS |
| Symptom-Doctor Mapping | 4 | PASS |
| Triage Feedback Loop | 5 | PASS |
| Department Selection Rules | 8 | PASS |
| API Endpoints | 4 | PASS |
| WebSocket Protocol | 5 | PASS |
| Session Management | 4 | PASS |
| Settings Configuration | 16 | PASS |
| System Prompt | 15 | PASS |
| Performance Benchmarks | 6 | PASS |
| Frontend Integrity | 16 | PASS |
| Model Configuration | 11 | PASS |
| **Total** | **252** | **ALL PASS** |

## API Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Serve frontend HTML |
| POST | `/api/offer` | WebRTC SDP offer/answer exchange |
| WS | `/ws` | UI state, settings, conversation, triage scores |

## License

Proprietary. All rights reserved. Aqib Mumtaz.
