# AI Clinical Triage System
## Project Overview Document

---

**Document Version:** 3.0
**Date:** March 22, 2026
**Author:** Dr. Aqib Mumtaz, Ph.D. — Head of AI
**Classification:** Confidential — For Internal & Stakeholder Review

---

## 1. Executive Summary

Our AI Clinical Triage System is a **voice-first, multilingual clinical triage platform** designed to automate patient intake at hospitals and clinics. The system conducts natural voice conversations with patients in their preferred language, collects symptoms, scores them against a disease cluster database using an intelligent triage scoring engine, and routes patients to the appropriate specialist — all in real-time, without human intervention.

The platform currently supports **6 languages** (Urdu, English, Hindi, Arabic, Punjabi, Malay), **12 medical specialties**, **70+ medical conditions**, **199 symptoms with 173 Urdu/English aliases**, and delivers end-to-end voice response times under **5 seconds**.

### Key Metrics

| Metric | Value |
|--------|-------|
| Languages Supported | 6 (expandable to 30+) |
| Medical Specialties | 12 departments |
| Medical Conditions | 70+ (disease clusters with ICD-10 codes) |
| Symptom Aliases | 173 entries covering 199 condition symptoms |
| Triage Scoring Tests | 188 tests, 100% pass rate |
| Voice Response Time | 3-5 seconds end-to-end |
| STT Options | Deepgram Nova-3, OpenAI gpt-4o-transcribe |
| STT Noise Filtering | Two-layer: Silero VAD (0.80) + Deepgram confidence (0.70) |
| Concurrent Sessions | Scalable via cloud deployment |
| Availability | 24/7/365 |
| Total Automated Tests | 265 tests, 100% pass rate |

---

## 2. Problem Statement

### The Hospital Reception Bottleneck

Hospitals in South Asia, the Middle East, and Southeast Asia face critical challenges at their front desk:

1. **Long wait times** — Patients wait 15-45 minutes for basic registration and triage, especially during peak hours.
2. **Language barriers** — Multilingual populations (Pakistan: Urdu/Punjabi/English; Malaysia: Malay/English/Mandarin; UAE: Arabic/Urdu/Hindi/English) create communication gaps with reception staff.
3. **Inconsistent triage** — Junior receptionists lack medical knowledge to properly route patients, leading to wrong department visits and wasted time.
4. **Staff shortages** — Qualified medical receptionists are expensive and hard to retain, especially for 24/7 coverage.
5. **No documentation** — Verbal interactions at reception are lost — no record of initial complaints for the doctor.
6. **Peak hour collapse** — Emergency situations overwhelm reception, delaying care for non-emergency patients.

### Market Size

| Market | 2024 Value | Projected | CAGR |
|--------|-----------|-----------|------|
| Global Healthcare AI | $14.9B | $110.6B by 2030 | 38.6% |
| Conversational AI in Healthcare | $13.7B | $48.9B by 2030 | 23.8% |
| Middle East Healthcare AI | $436M | $8.4B by 2033 | 37.0% |
| Middle East Hospital Services | $114B | $172.8B by 2033 | 4.7% |

*Sources: Grand View Research, MarketsandMarkets, BCC Research (2024-2025)*

**Target Market Infrastructure:**

| Country | Hospitals | Healthcare Market | Key Driver |
|---------|-----------|-------------------|------------|
| Pakistan | 1,830 (1,147 public + 683 private) | $7B | 240M population, 4+ languages |
| Saudi Arabia | 2,590 (290 hospitals + 2,300 centers) | $67B | Vision 2030 — $65B investment |
| UAE | 972 | $22B → $50B by 2029 | 8% CAGR, premium spend |
| Qatar | 100+ | $12B | #1 Arab world healthcare ranking |
| Malaysia | 355 (148 public + 207 private) | $28B → $48B by 2028 | MyDigital, 13.5% CAGR |

- **<5% of hospitals** in these regions have any form of AI automation at reception
- **Average hospital spends $180K-$350K/year** on front desk staffing (3-shift coverage)

*Sources: WHO EMRO, Statista, US-Saudi Business Council, UAE Open Data, Malaysia MOH (2024)*

---

## 3. Solution Architecture

### 3.1 System Overview

```
                    ┌──────────────────────────────────────┐
                    │         BROWSER (Patient)            │
                    │                                      │
                    │  ┌──────────┐     ┌───────────────┐  │
                    │  │ WebRTC   │     │  WebSocket    │  │
                    │  │ Audio    │     │  UI Messages  │  │
                    │  └────┬─────┘     └──────┬────────┘  │
                    └───────┼──────────────────┼───────────┘
                            │                  │
                    ┌───────┼──────────────────┼───────────┐
                    │       │    CLOUD SERVER   │           │
                    │       v                  v           │
                    │  ┌─────────┐      ┌──────────┐      │
                    │  │ WebRTC  │      │ FastAPI  │      │
                    │  │Transport│      │ WS Hub   │      │
                    │  └────┬────┘      └──────────┘      │
                    │       │                              │
                    │       v                              │
                    │  ┌──────────────────────────────┐    │
                    │  │      PIPECAT PIPELINE        │    │
                    │  │                              │    │
                    │  │  Audio → VAD → STT → LLM → TTS    │    │
                    │  │    (Silero) (Confidence   ↓       │    │
                    │  │             Filtered) Triage      │    │
                    │  │                      Scoring      │    │
                    │  │                      Engine       │    │
                    │  └──────────────────────────────┘    │
                    └──────────────────────────────────────┘
```

### 3.2 Technology Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Voice Activity Detection** | Silero VAD (neural ONNX) | Filters background noise — confidence=0.6, min_volume=0.5, stop_secs=0.6 |
| **Speech-to-Text** | ConfidenceFilteredDeepgramSTT / OpenAI gpt-4o-transcribe | Dual STT with confidence-based noise filtering (drops < 0.70). Switchable from settings |
| **Language Model** | OpenAI GPT-4o-mini / GPT-4o / Claude | Follows triage directive — MUST refer to highest-scoring department |
| **Triage Scoring Engine** | Custom Python (disease_clusters.json) | 70+ conditions, 173 aliases, exact matching, cross-condition boost, age/gender rules |
| **Text-to-Speech** | OpenAI TTS-1 / ElevenLabs Flash v2.5 | Multi-sentence audio caching, number-to-Urdu conversion for Urdu TTS |
| **Audio Transport** | WebRTC (STUN/TURN) | Real-time bidirectional audio with built-in echo cancellation. Works through NAT/firewalls |
| **Pipeline** | Pipecat-AI 0.0.85 | Async frame-based pipeline: VAD → STT → Broadcaster → LLM → Response → TTS Cache → TTS → Audio |
| **Backend** | FastAPI (Python) | HTTP, WebSocket, WebRTC endpoints. Tool-based LLM integration with patient screening |
| **Frontend** | Single-file HTML/CSS/JS | Real-time triage dashboard with Urdu tooltips, patient screening card, cost analysis |

### 3.3 Triage Scoring Engine

The core intelligence of the system — a real-time scoring algorithm that matches patient symptoms against 70+ medical conditions organized in 12 departments.

**Scoring Pipeline:**

1. **Symptom Alias Expansion** — Patient symptoms (in any language/script) are expanded via 173-entry alias lookup. "کھانسی" → "cough", "سینے میں درد" → "chest pain", "نیند نہیں آتی" → "insomnia"
2. **Per-Original-Symptom Matching** — Each patient symptom matches at most ONE condition symptom, preventing alias chain inflation
3. **Exact Word-Set Matching** — "cough" matches "cough" but NOT "persistent cough" (clinically different symptoms)
4. **Vital Boost** — +10% per matching vital (BP, heart rate, temperature, sugar, mobility, fever)
5. **Severity Boost** — +5% when patient severity matches condition severity levels
6. **Cross-Condition Boost** — When symptoms match multiple conditions in a department, confidence increases: `best_score + best_score × 0.15 × (strong_conditions - 1)`
7. **Department Selection Rules** — Age-gated: Pediatrics zeroed for adults (≥12). Gender-gated: Gynecology zeroed for males. Unknown age: 70% penalty on Pediatrics unless child-specific symptoms present
8. **Doctor Availability Check** — Unavailable doctors skipped in referral directive; [NOT AVAILABLE] tag shown
9. **Directive Referral** — Triage returns explicit "REFER TO → Dr. Name, Department, Room" — the LLM MUST follow this directive

**Scoring Formula:**

```
condition_score = (matched / total_symptoms) + vital_boost + severity_boost
department_score = best_condition_score + cross_condition_boost
final_score = apply_department_rules(department_score, age, gender)
```

**Noise Filtering (Three-Layer Defense, no audio processing filters):**

| Layer | Technology | Threshold | Purpose |
|-------|-----------|-----------|---------|
| Audio Level | Silero VAD (neural) | confidence=0.6, min_volume=0.5, stop_secs=0.6 | Filters non-speech audio |
| STT Level | ConfidenceFilteredDeepgramSTT | confidence < 0.45 dropped | Filters low-confidence transcriptions (Urdu-friendly) |
| Text Level | Noise word filter | frozenset lookup | Drops "hmm", "ok", "yeah" etc. |

**Pipeline Performance:**
- All UI broadcasts: non-blocking (fire-and-forget)
- File I/O (_persist): thread pool
- Scoring: skipped on greeting/name turns
- allow_interruptions=true (STT always running)
- Pipeline overhead: < 1ms per turn

### 3.4 Conversation Flow

```
1.  Patient opens browser → WebRTC audio connected
2.  Agent greets patient in selected language (pure Urdu script, no English mixing)
3.  Patient speaks → Silero VAD filters background noise → Confidence-filtered STT
4.  LLM processes speech → Calls update_patient_screening tool on EVERY response
5.  Triage scoring engine scores symptoms against 70+ conditions in real-time
6.  Triage dashboard updates live — department confidence percentages shown
7.  Agent collects: Name → Symptoms → Duration → Vitals (symptom-relevant only)
8.  Smart follow-up: asks targeted symptoms to differentiate conditions (e.g., chest pain → "کیا سانس لینے میں تکلیف ہے؟")
9.  Off-topic answer handling: if patient answers a different question, agent re-asks
10. TTS converts response → Number-to-Urdu conversion → Audio caching → WebRTC output
11. When triage score ≥ 50%: agent follows "REFER TO →" directive from scoring engine
12. Agent provides: Doctor name, department, room number (availability checked)
13. Patient screening card + triage slip available for printing
```

---

## 4. Core Features

### 4.1 Multilingual Voice Interaction

The system natively handles 6 languages with automatic text direction (RTL/LTR) support:

| Language | Region | Direction | Script | Cultural Adaptation |
|----------|--------|-----------|--------|-------------------|
| Urdu | Pakistan, India | RTL | Nastaliq | آپ/جناب forms, Islamic greetings |
| English | Global | LTR | Latin | Professional medical terminology |
| Hindi | India | LTR | Devanagari | Respectful "aap" forms |
| Arabic | Middle East | RTL | Arabic | Islamic honorifics, Gulf dialect awareness |
| Punjabi | Pakistan, India | RTL | Shahmukhi | Regional expressions, cultural warmth |
| Malay | Malaysia, Indonesia | LTR | Latin | Formal "Encik/Puan" honorifics |

### 4.2 Configurable Agent Personas

10 pre-built personas with gender-appropriate voice, grammar, and cultural behavior:

| Persona | Gender | Voice Provider | Voice | Use Case |
|---------|--------|---------------|-------|----------|
| Amina | Female | OpenAI | Nova | Default — warm, caring receptionist |
| Ayesha | Female | OpenAI | Shimmer | Professional, efficient |
| Fatima | Female | OpenAI | Alloy | Friendly, approachable |
| Hira | Female | ElevenLabs | Rachel | Premium natural voice |
| Sarah | Female | ElevenLabs | Sarah | Gentle, reassuring |
| Ali | Male | OpenAI | Echo | Professional male agent |
| Hassan | Male | OpenAI | Onyx | Deep, authoritative voice |
| Ahmed | Male | OpenAI | Fable | Warm, storyteller quality |
| Bilal | Male | ElevenLabs | Drew | Natural conversational |
| Usman | Male | ElevenLabs | Paul | Calm, measured delivery |

Each persona adapts:
- **Grammatical gender** — Urdu masculine/feminine verb endings
- **Cultural honorifics** — Gender-appropriate terms of address
- **Voice characteristics** — Matching TTS voice and tone
- **Personality traits** — Warm, Professional, or Friendly

### 4.3 Real-Time Disease Cluster Triage

The triage scoring engine scores patient symptoms in real-time against a disease cluster database. Each department shows a live confidence percentage on the triage dashboard. The system maps patient symptoms (including Urdu aliases) to 12 specialist departments and 70+ conditions:

| Symptom Category | Specialist | Room |
|-----------------|-----------|------|
| Fever, flu, general illness | General Medicine | 101 |
| Chest pain, blood pressure | Cardiology | 302 |
| Pregnancy, women's health | Gynecology | 205 |
| Children's illness, vaccination | Pediatrics | 108 |
| Bone fractures, joint pain | Orthopedics | 210 |
| Skin rash, allergies | Dermatology | 115 |
| Ear, nose, throat | ENT | 118 |
| Eye problems, vision | Ophthalmology | 122 |
| Headaches, seizures | Neurology | 310 |
| Toothache, dental | Dentistry | G-05 |
| Stomach pain, digestion | Gastroenterology | 215 |
| Anxiety, depression | Psychiatry | 320 |
| **Severe/Life-threatening** | **Emergency** | **Ground Floor** |

The agent checks doctor availability in real-time and suggests alternatives when a specialist is unavailable.

**Triage Dashboard Features:**
- **Live confidence percentages** per department — updates with each new symptom
- **Expandable condition cards** — click to see individual condition scores, matched/unmatched symptom pills, vital requirements, severity levels
- **Urdu tooltips** — hover over any English symptom tag to see Urdu translation (all 199 symptoms covered)
- **Matched symptom highlighting** — green pills for matched symptoms, gray for unmatched
- **ICD-10 codes** — each condition shows its ICD-10 classification code
- **Cross-condition boost indicator** — shows when multiple conditions in a department are matched

**Scoring-LLM Alignment:**
The triage scoring engine and LLM referral decisions are fully aligned. The scoring engine applies department selection rules (age, gender, symptom specificity) and returns a directive "REFER TO → Dr. Name, Department, Room" which the LLM must follow. This eliminates conflicts between the scoring dashboard and the bot's verbal referral.

### 4.4 Session Management

- **Settings persistence** — All configuration survives page refresh
- **Conversation logging** — Full transcript saved for doctor review
- **Session reset** — Clean state on new patient, end session, or app restart
- **Settings lock** — Prevents accidental config changes during active calls
- **Multi-client support** — Multiple browser tabs can view the same session

### 4.5 Real-Time UI

- **Live STT display** — Shows patient's speech as it's being recognized
- **Conversation bubbles** — Chat-style display with timestamps
- **Status indicators** — Listening/Thinking/Speaking with animated indicators
- **Doctors panel** — Interactive doctor directory with full profiles
- **Model profile tags** — Visual indicators of active model characteristics
- **RTL/LTR auto-switching** — UI direction adapts to selected language

### 4.6 Patient Screening Card

Real-time patient data extraction during conversation. The LLM appends structured `[PATIENT_DATA]` JSON after every response, and the frontend renders a live screening card displaying:

- **Demographics** — Name, age, gender
- **Clinical data** — Symptoms, duration, severity
- **Vitals** — Fever, blood pressure, heart rate, temperature, sugar level, mobility
- **Referral** — Referred doctor, department, room number

The card updates progressively as the conversation unfolds. A **print slip** function allows the screening data to be printed as a triage slip for the patient.

### 4.7 Symptom-to-Vitals Mapping

A 15-category mapping embedded in the system prompt ensures the agent only asks vitals relevant to the patient's presenting symptoms. For example, fever triggers temperature questions; chest pain triggers blood pressure and heart rate inquiries. This prevents unnecessary questioning and keeps conversations focused.

### 4.8 Conversation Flow Stepper

A visual progress indicator in the UI shows which step of the conversation the patient is at: **Name → Symptoms → Duration → Vitals → Refer**. Each step is color-coded with animated status dots, giving both the patient and staff a clear view of conversation progress.

### 4.9 TTS Audio Caching

A disk-backed LRU cache (500 entries) stores TTS audio keyed by normalized text + voice configuration (SHA-256 hash). Two pipeline processors work in tandem:

- **TTSCacheGate** — Intercepts text before TTS; serves cached audio on hit. On cache miss, applies number-to-Urdu conversion (e.g., "302" → "تین سو دو") before forwarding to TTS
- **TTSCacheCapture** — Accumulates audio across ALL sentences in a multi-sentence LLM response (using LLMFullResponseStartFrame/EndFrame lifecycle), then saves the complete audio to cache

**Multi-sentence handling:** TTS processes each sentence individually (separate TTSStarted/TTSStopped frames). The capture processor accumulates audio across all sentences and saves once at response end — ensuring the full response audio is cached, not just the last sentence.

This reduces TTS API costs by approximately **85% at steady state** after ~10 patients with similar conversation patterns. Cached audio is served with zero TTS latency.

### 4.10 Live Model Switching

LLM model and TTS voice can be changed live during an active call without restarting the session. STT model and TTS provider changes apply on the next session to prevent pipeline breakage. All model options are accessible from the settings panel.

### 4.11 Patient Data TTS Suppression

A buffering mechanism prevents `[PATIENT_DATA]` JSON blocks from being spoken aloud by TTS. The system handles streaming word-by-word TextFrames and suppresses the structured data portion while allowing the conversational response to flow through to speech synthesis.

### 4.12 Cost Analysis Dashboard

The frontend includes a detailed cost comparison table showing per-minute costs for all **24 STT+LLM+TTS combinations**, with columns for cached pricing, percentage savings, and monthly projections for both 8h×5d and 24×7 operation scenarios.

### 4.13 Urdu Script Output

The system prompt instructs the LLM to respond in proper Urdu script (اردو رسم الخط) instead of Roman Urdu transliteration. All prompt examples have been converted to Urdu script for consistency and natural readability.

---

## 5. Technical Performance

### 5.1 Latency Breakdown

| Stage | Typical Latency | Component |
|-------|----------------|-----------|
| Speech Detection (VAD) | ~300ms | Pipecat built-in |
| STT Processing | ~500ms | Deepgram Nova-3 / OpenAI gpt-4o-transcribe |
| LLM First Token | ~800ms | GPT-4o-mini |
| LLM Full Response | ~1500ms | GPT-4o-mini (2-4 sentences) |
| TTS First Audio | ~500ms (~0ms cached) | OpenAI TTS-1 streaming (cache hit = instant) |
| **Total End-to-End** | **~3-5 seconds** | - |

### 5.2 Automated Test Results

265 automated tests covering 17 categories — **100% pass rate**:

**Triage Scoring Engine Tests (188 tests):**

| Category | Tests | Status |
|----------|-------|--------|
| Scoring Accuracy | 8 | PASS |
| Department Ranking | 11 | PASS |
| Vital Combinations | 7 | PASS |
| Alias Expansion (Urdu/English) | 11 | PASS |
| Referral Scenarios | 11 | PASS |
| Score Differentiation | 7 | PASS |
| Symptom-Doctor Mapping | 4 | PASS |
| Triage Feedback Loop | 5 | PASS |
| Department Selection Rules | 8 | PASS |

**Platform Tests (77 tests):**

| Category | Tests | Status |
|----------|-------|--------|
| API Endpoints | 4 | PASS |
| WebSocket Protocol | 5 | PASS |
| Session Management | 4 | PASS |
| Settings Configuration | 16 | PASS |
| System Prompt Generation | 15 | PASS |
| Performance Benchmarks | 6 | PASS |
| Frontend Integrity | 16 | PASS |
| Model Configuration Matrix | 11 | PASS |

### 5.3 Performance Benchmarks

| Metric | Threshold | Actual | Status |
|--------|-----------|--------|--------|
| Page Load | < 200ms | < 50ms | PASS |
| WebSocket Connect | < 500ms | < 100ms | PASS |
| Setting Roundtrip | < 300ms | < 150ms | PASS |
| Stop Action | < 500ms | < 200ms | PASS |
| Session Persistence | < 500ms | < 200ms | PASS |
| 5 Concurrent Connections | < 2000ms | < 500ms | PASS |

---

## 6. Deployment Options

### 6.1 Cloud Deployment (Recommended)

```
                    Internet
                       │
                ┌──────┴──────┐
                │  CDN / LB   │
                └──────┬──────┘
                       │
            ┌──────────┼──────────┐
            │          │          │
        ┌───┴───┐  ┌───┴───┐  ┌──┴────┐
        │ Pod 1 │  │ Pod 2 │  │ Pod 3 │  (Kubernetes / ECS)
        └───┬───┘  └───┬───┘  └───┬───┘
            │          │          │
        ┌───┴──────────┴──────────┴───┐
        │      TURN Server Pool       │
        │   (WebRTC Media Relay)      │
        └─────────────────────────────┘
```

- **Container**: Docker image (~500MB)
- **Scaling**: Horizontal — each pod handles ~50 concurrent sessions
- **TURN servers**: Required for WebRTC in corporate/hospital networks
- **SSL**: Required for WebRTC in production (browser security policy)

### 6.2 On-Premise Kiosk

A tablet or touchscreen kiosk at the hospital entrance:
- Patient walks up, taps "Start"
- Speaks into built-in microphone
- Agent responds through speakers
- Prints a triage slip with doctor assignment

### 6.3 Phone/IVR Integration

Future: Connect to hospital phone system via SIP/Twilio:
- Patient calls hospital number
- AI agent answers, conducts triage
- Transfers to appropriate department

---

## 7. Competitive Analysis

| Feature | Our Solution | Ada Health | Babylon Health | Generic Chatbot |
|---------|----------|-----------|---------------|----------------|
| **Voice-first interaction** | Yes | No (text only) | Limited | No |
| **Multilingual (6+ languages)** | Yes | 10+ (text) | English/German | Varies |
| **RTL language support** | Yes (Urdu, Arabic, Punjabi) | Partial | No | No |
| **Real-time streaming** | Yes (WebRTC) | N/A | N/A | N/A |
| **Cultural adaptation** | Yes (gender, honorifics) | No | No | No |
| **Disease cluster scoring** | Yes (70+ conditions, real-time %) | No | No | No |
| **Symptom alias matching** | 173 aliases (Urdu/English/Roman) | Limited | English only | No |
| **Hospital-specific triage** | Yes (12 specialties, ICD-10) | General health | General health | No |
| **Doctor availability** | Real-time with auto-fallback | No | No | No |
| **Background noise filtering** | Two-layer (VAD + STT confidence) | N/A | N/A | N/A |
| **On-premise deployment** | Yes | Cloud only | Cloud only | Varies |
| **Cost per interaction** | ~$0.05-0.15 | $0.50+ | $1.00+ | $0.01-0.05 |
| **Setup time** | Hours | Weeks | Months | Hours |

---

## 8. Business Model

### 8.1 Revenue Streams

| Model | Pricing | Target |
|-------|---------|--------|
| **SaaS Subscription** | $500-2,000/month per hospital | Small-medium hospitals |
| **Enterprise License** | $20,000-50,000/year | Hospital chains |
| **Per-Interaction** | $0.10-0.25 per patient interaction | High-volume clinics |
| **Kiosk Hardware + Software** | $5,000 one-time + $200/month | Reception replacement |
| **White-Label** | Custom pricing | Healthcare IT vendors |

### 8.2 Cost Structure (Per Interaction)

| Component | Cost (Uncached) | Cost (Cached) | Notes |
|-----------|----------------|---------------|-------|
| Deepgram STT | ~$0.0059/min | ~$0.0059/min | Nova-3 pay-as-you-go |
| OpenAI STT | ~$0.006/min | ~$0.006/min | gpt-4o-transcribe (alternative) |
| OpenAI LLM | ~$0.01-0.03 | ~$0.01-0.03 | GPT-4o-mini (~500 tokens) |
| OpenAI TTS | ~$0.015/1K chars | ~$0.002/1K chars | TTS-1 — ~85% cache hit rate at steady state |
| Infrastructure | ~$0.005 | ~$0.005 | Cloud compute + bandwidth |
| **Total** | **~$0.04-0.06** | **~$0.02-0.04** | Per 3-minute interaction |

With TTS caching enabled, steady-state cost drops by ~30-40%. At $0.15/interaction pricing, **gross margin is ~70-85%**.

### 8.3 ROI for Hospitals

| Metric | Without AI | With AI |
|--------|-----------|---------|
| Reception staff (3 shifts) | 6-9 staff | 2-3 staff |
| Annual staff cost | $180K-350K | $60K-120K |
| AI system cost | $0 | $12K-24K/year |
| **Annual savings** | - | **$100K-200K+** |
| Patient wait time | 15-45 min | 2-5 min |
| 24/7 coverage | No (costly) | Yes (included) |
| Triage accuracy | 70-80% | 90%+ |
| Documentation | None | Full transcript |

---

## 9. Roadmap

### Phase 1 — Current (Q1 2026)
- [x] Core voice pipeline (STT → LLM → TTS)
- [x] 6 language support with RTL/LTR
- [x] 10 configurable personas
- [x] 12 specialty triage
- [x] WebRTC browser audio
- [x] 77 automated tests, 100% pass rate
- [x] Session persistence and settings management
- [x] Dual STT providers (Deepgram Nova-3 + OpenAI gpt-4o-transcribe)
- [x] TTS audio caching (~85% cost reduction at steady state)
- [x] Real-time patient screening card with live data extraction
- [x] Symptom-to-vitals mapping (15 categories)
- [x] Conversation flow stepper (Name → Symptoms → Duration → Vitals → Refer)
- [x] Live model switching (LLM, TTS voice during call; STT on next session)
- [x] Patient data TTS suppression
- [x] Urdu script output (proper رسم الخط instead of Roman Urdu)
- [x] Cost analysis dashboard (24 STT+LLM+TTS combinations)
- [x] Print triage slip
- [x] Real-time triage scoring engine (70+ conditions, 12 departments)
- [x] Disease cluster database with 173 symptom aliases (Urdu + English)
- [x] Two-layer noise filtering (Silero VAD + Deepgram confidence)
- [x] Age/gender-aware department selection rules
- [x] Cross-condition confidence boost for multi-condition departments
- [x] Exact symptom matching (prevents cross-contamination)
- [x] Doctor availability check in referral directives
- [x] Urdu tooltips on all triage symptom tags (199 symptoms)
- [x] Off-topic answer handling (re-asks unanswered symptom questions)
- [x] Number-to-Urdu TTS conversion (room numbers spoken in Urdu)
- [x] Pure Urdu output (no English word mixing)
- [x] 188 triage scoring tests, 100% pass rate

### Phase 2 — Q2 2026
- [ ] Patient registration form auto-fill
- [ ] Electronic Medical Record (EMR) integration (HL7/FHIR)
- [ ] Appointment scheduling with calendar integration
- [ ] SMS/WhatsApp follow-up messages
- [ ] Queue management system integration
- [ ] Analytics dashboard (interaction volume, wait times, triage accuracy)

### Phase 3 — Q3-Q4 2026
- [ ] On-premise kiosk hardware design
- [ ] Phone/IVR integration (Twilio/SIP)
- [ ] Multi-hospital deployment (cloud-native)
- [ ] Custom medical knowledge base per hospital
- [ ] Voice biometric patient identification
- [ ] Wheelchair/escort auto-dispatch

### Phase 4 — 2027
- [ ] FDA/CE medical device pre-certification (Class I)
- [ ] Insurance verification integration
- [ ] Multi-tenant SaaS platform
- [ ] Mobile app (iOS/Android)
- [ ] Regional language expansion (Bengali, Tamil, Turkish, Bahasa Indonesia)

---

## 10. Team & Leadership

### Head of AI

**Dr. Aqib Mumtaz, Ph.D.** — Head of AI
- Ph.D. in Computer Science (AI specialization), COMSATS University, Lahore (2019-2024)
- Research collaboration with University of Bergen, Norway
- 19+ years experience across AI/ML, Deep Learning, Computer Vision, NLP, and ASR
- Published researcher — 5 peer-reviewed papers (anomaly detection, video surveillance, deep learning)
- Core expertise: Transformer architectures, OpenAI Whisper, attention mechanisms, real-time AI
- State-of-the-art AI technologies: Large Language Models (GPT-4o, Qwen3-ASR), WebRTC real-time streaming, pipecat-ai pipeline orchestration, Deepgram Nova-3 STT, ElevenLabs neural TTS
- Led AI research teams of 22+ engineers; reviewed Pakistan's National AI Policy (ITU)

### Team Growth Plan (with funding)

| Role | Count | Priority |
|------|-------|----------|
| AI/ML Engineer | 2 | High |
| Full-Stack Developer | 1 | High |
| DevOps / Cloud Engineer | 1 | Medium |
| Healthcare Domain Expert | 1 | Medium |
| UI/UX Designer | 1 | Medium |
| Business Development | 1 | High |
| QA Engineer | 1 | Medium |

---

## 11. Risk Analysis

| Risk | Impact | Likelihood | Mitigation |
|------|--------|-----------|-----------|
| STT accuracy in noisy hospital | High | Medium | Two-layer noise filtering: Silero VAD (0.80 confidence) + ConfidenceFilteredDeepgramSTT (drops < 0.70). Endpointing synced at 900ms |
| LLM refers wrong doctor | High | Low | Triage scoring engine returns directive "REFER TO →" — LLM must follow. Scoring applies age/gender rules, doctor availability. 188 tests validate scoring accuracy |
| Scoring-LLM misalignment | Medium | Low | Department selection rules removed from system prompt (scoring handles them). LLM follows triage directive, not its own judgment |
| Alias cross-contamination | Medium | Low | Exact word-set matching prevents "cough" matching "persistent cough". All 173 aliases audited — zero harmful cross-department aliases |
| Internet outage | High | Low | On-premise fallback mode, offline triage rules |
| Data privacy (patient info) | Critical | Medium | End-to-end encryption, no PII storage, HIPAA/local compliance |
| API cost escalation | Medium | Low | TTS audio caching (~85% hit rate), dual STT providers for cost optimization, model switching |
| Patient reluctance (elderly) | Medium | High | Kiosk with large buttons, human fallback, gradual adoption |

---

## 12. Contact

**AI-Powered Healthcare Solutions**

For demo, or inquiries:
- Email: aqib.mumtaz@gmail.com

---

*This document is confidential and intended for authorized personnel and stakeholders only. All financial projections are estimates based on current market data and are subject to change.*
