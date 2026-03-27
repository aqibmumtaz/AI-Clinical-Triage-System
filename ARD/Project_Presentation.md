# AI Clinical Triage System
## Presentation

---

*AI-Powered Clinical Triage for Hospital Reception*
*March 2026 — v3.0*

---

# SLIDE 1: The Vision

## Every Patient Deserves Instant, Compassionate Care — In Their Own Language

Imagine walking into a hospital — anxious, in pain, possibly in a foreign city — and being greeted instantly by a warm, intelligent assistant who speaks your language, understands your symptoms, and guides you to the right doctor in under 2 minutes.

**No waiting. No language barriers. No paperwork. Just care.**

That's what our AI Clinical Triage System delivers — today. With real-time disease cluster scoring, 173 multilingual symptom aliases, and intelligent triage that adapts by age, gender, and clinical context.

---

# SLIDE 2: The Problem

## Hospital Reception is Broken

### The Patient Experience Today

```
Patient arrives (anxious, in pain)
        ↓
Wait in queue: 15-45 minutes
        ↓
Struggle to communicate: language barriers
        ↓
Misrouted to wrong department: 30% of the time
        ↓
Wait again at wrong department
        ↓
Finally see the right doctor: 1-2 hours later
```

### The Numbers Tell the Story

| Problem | Impact |
|---------|--------|
| **15-45 min** average wait at reception | Patient dissatisfaction, delayed care |
| **30%** of patients misrouted | Wasted doctor time, hospital inefficiency |
| **6-9 reception staff** required for 24/7 coverage | $180K-350K/year labor cost |
| **Zero documentation** of initial complaints | Doctor has no context before consultation |
| **Peak hours** overwhelm reception | Emergency patients delayed by routine cases |
| **Multilingual populations** underserved | Pakistan alone has 4+ major languages |

### Who Suffers?

- **Patients** — Anxiety, delayed treatment, communication frustration
- **Hospitals** — High staffing costs, inefficient triage, liability risk
- **Doctors** — No patient context, repeated history-taking, wasted time
- **Healthcare System** — Bottleneck at the very first touchpoint

---

# SLIDE 3: Our Solution

## AI That Talks, Listens, and Cares — In 6 Languages

### What We Built

A **real-time, voice-based AI clinical triage platform** that:

1. **Greets** the patient warmly in their preferred language (pure Urdu script, no English mixing)
2. **Listens** with two-layer noise filtering — Silero VAD filters background speakers, confidence threshold drops noise artifacts
3. **Scores** symptoms in real-time against 70+ conditions using disease cluster scoring engine with 173 multilingual aliases
4. **Asks** targeted follow-up questions based on differential diagnosis — "chest pain" → asks about shortness of breath to differentiate Angina vs GERD
5. **Triages** using intelligent scoring — cross-condition boost, age/gender rules, doctor availability check
6. **Directs** the patient to the exact doctor, department, and room shown in triage scores
7. **Documents** the entire conversation + live patient screening card with printable triage slip

### In Action

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│   Patient: "مجھے سینے میں درد ہو رہا ہے"              │
│   (I'm having chest pain)                           │
│                                                     │
│   Agent (Amina): "آپ کی تکلیف سن کر افسوس ہوا۔        │
│   کب سے ہے یہ درد؟"                                  │
│   (I'm sorry to hear that. How long have            │
│   you had this pain?)                               │
│                                                     │
│   Patient: "کل رات سے"                               │
│   (Since last night)                                │
│                                                     │
│   Agent: "میں آپ کو ابھی ڈاکٹر علی رضا،               │
│   کارڈیالوجسٹ کے پاس بھیج رہی ہوں۔                    │
│   کمرہ نمبر 302، تیسری منزل۔"                        │
│   (I'm sending you to Dr. Ali Raza,                 │
│   Cardiologist. Room 302, 3rd floor.)               │
│                                                     │
│   Total time: 90 seconds                            │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

# SLIDE 4: How It Works

## The Technology Behind the Voice

### The Pipeline (under 5 seconds end-to-end)

```
   Patient Speaks          AI Processes                AI Responds
   ┌──────────┐         ┌────────────────────┐        ┌──────────┐
   │ Browser  │ WebRTC  │  Cloud AI Pipeline │ WebRTC │ Browser  │
   │ Mic      │───────→ │                    │───────→│ Speaker  │
   └──────────┘  ~50ms  │ VAD: ~300ms        │  ~50ms └──────────┘
                        │ STT: 500ms (filtered)       │
                        │ LLM: 800ms         │ ┌──────────────┐
                        │ Triage: <10ms      │ │ Triage       │
                        │ TTS: 500ms (cached)│ │ Dashboard    │
                        └────────────────────┘ │ (live scores)│
                         Total: ~3-5s          └──────────────┘
```

### Key Technologies

| Component | What It Does | Why It's Special |
|-----------|-------------|-----------------|
| **Deepgram Nova-3 / OpenAI gpt-4o-transcribe** | Converts speech to text in real-time | Confidence-filtered — drops background noise (< 0.70). 95%+ accuracy |
| **Silero VAD** | Neural voice activity detection | Filters distant speakers (confidence=0.80, min_volume=0.65) |
| **Triage Scoring Engine** | Matches symptoms to 70+ conditions | 173 aliases, exact matching, cross-condition boost, age/gender rules |
| **GPT-4o-mini** | Understands context, generates responses | Follows triage directive — MUST refer to highest-scoring department |
| **OpenAI TTS / ElevenLabs** | Converts text to natural speech | Human-like voices, number-to-Urdu conversion, audio caching |
| **WebRTC** | Streams audio browser ↔ server | Zero latency, built-in echo cancellation, works through firewalls |
| **Pipecat-AI** | Orchestrates the full pipeline | Production-grade, handles interruptions, built-in metrics |

### What Makes It Different

- **Clinical triage, not chatbot** — Real-time disease cluster scoring against 70+ conditions. Live confidence percentages per department.
- **173 multilingual symptom aliases** — Patient says "کھانسی" → scores as "cough" across all conditions. Urdu, Roman Urdu, English, and common mistranscriptions covered.
- **Intelligent scoring** — Cross-condition boost, vital boost, severity boost, age/gender department rules. Scoring aligns with LLM referral decisions.
- **Two-layer noise filtering** — Silero VAD (neural, 0.80 confidence) + Deepgram STT confidence threshold (0.70). Background speakers filtered without blocking single-word patient responses.
- **Culturally intelligent** — Gender grammar, cultural honorifics, pure Urdu script output (no English mixing). Number-to-Urdu TTS conversion.
- **Hospital-specific** — Configured for specific doctors, rooms, departments. Doctor availability checked before referral.
- **Live triage dashboard** — Urdu tooltips on hover for all 199 symptoms. Real-time scoring updates with each new symptom.
- **265 automated tests** — 188 triage scoring + 77 platform tests, 100% pass rate.

---

# SLIDE 5: The Market

## Healthcare AI: Fastest Growing AI Segment

### Global Market Opportunity

| Market | 2024 | Projected | CAGR |
|--------|------|-----------|------|
| Global Healthcare AI | $14.9B | $110.6B by 2030 | 38.6% |
| Conversational AI in Healthcare | $13.7B | $48.9B by 2030 | 23.8% |
| Middle East Healthcare AI | $436M | $8.4B by 2033 | 37.0% |
| Middle East Hospital Services | $114B | $172.8B by 2033 | 4.7% |

*Sources: Grand View Research, MarketsandMarkets, BCC Research (2024-2025)*

### Our Target Markets

**Phase 1 — Pakistan (Year 1):**
- **1,830 hospitals** (1,147 public + 683 private), 123,000+ beds
- $7B hospital market (2024), growing at 4.3% CAGR
- 240M population, 4+ major languages, <5% AI adoption at reception

**Phase 2 — Saudi Arabia & Gulf (Year 1-2):**
- **Saudi Arabia**: 290+ hospitals + 2,300 primary care centers under privatization. $67B healthcare market. Vision 2030 investing $65B in healthcare infrastructure. Recruiting 175K healthcare professionals by 2030
- **UAE**: 972 hospitals and healthcare facilities. $22B healthcare market, growing to $50B by 2029 (8% CAGR)
- **Qatar**: $12B healthcare sector, ranked #1 in Arab world for healthcare quality
- **Kuwait, Bahrain, Oman**: Mandatory health insurance driving expansion. Active digital health initiatives

**Phase 3 — Malaysia & Expansion (Year 2-3):**
- **Malaysia**: 355 hospitals (148 public + 207 private). $28B healthcare market, growing to $48B by 2028 (13.5% CAGR). Government MyDigital push
- **India**: 70,000+ hospitals. World's largest healthcare market by volume
- **Indonesia**: 3,000+ hospitals, 280M population

### Total Addressable Market (TAM)

| Segment | Hospitals | Healthcare Market |
|---------|-----------|-------------------|
| Pakistan | ~1,830 | $7B |
| Saudi Arabia | ~2,590 | $67B |
| UAE | ~972 | $22B |
| Other GCC (Qatar, Kuwait, Bahrain, Oman) | ~500 | $30B |
| Malaysia | ~355 | $28B |
| **Phase 1-3 Total** | **~6,250** | **$154B** |
| India + Indonesia (expansion) | ~73,000 | $200B+ |

*Sources: WHO EMRO, Statista, US-Saudi Business Council, UAE Open Data, Malaysia MOH (2024)*

---

# SLIDE 6: Competitive Advantage

## Why We Win

### 1. Real-Time Clinical Triage with Multilingual Voice — Nobody Else Does This

Most healthcare AI is **text-based**, **English-only**, and **diagnosis-focused**. We are the only solution that:
- Performs **real-time disease cluster scoring** against 70+ conditions with live confidence percentages
- Handles **voice conversations** in Urdu, Arabic, and Punjabi (RTL languages) with 173 symptom aliases
- Uses **intelligent scoring** with cross-condition boost, age/gender rules, and exact symptom matching
- Filters **background noise** with two-layer VAD + STT confidence filtering
- Applies **cultural communication norms** (Islamic greetings, gender grammar, pure Urdu script)

### 2. Hospital-Specific, Not Generic

| Generic Health AI | Our AI Clinical Triage System |
|-------------------|----------|
| "You might have condition X" | "REFER TO → Dr. Ali Raza, Cardiology, Room 302 (48% match)" |
| No real-time scoring | Live disease cluster scoring — 70+ conditions, ICD-10 codes |
| No context of hospital | Knows every doctor, room, schedule, and availability |
| Diagnosis-focused (risky) | Triage-focused (safe, referral-based, scoring-driven) |
| English only | 173 symptom aliases across Urdu, English, Roman Urdu |
| Requires medical certification | Reception function — no medical claims |

### 3. Deployment Speed

```
Generic EHR Integration:  6-12 months, $200K+ implementation
Our Setup:            2-4 hours, $500/month
```

We don't replace the EHR — we sit **in front** of it. The agent handles the reception interaction, then passes the documented conversation to the existing hospital system.

### 4. Cost Structure

Our per-interaction cost of **$0.04-0.06** (dropping to **$0.02-0.04** with TTS caching at steady state) allows us to price at $0.10-0.25/interaction and maintain **70-85% gross margins** — comparable to SaaS economics, not healthcare services economics. A detailed cost analysis dashboard covers all 24 STT+LLM+TTS combinations with monthly projections.

---

# SLIDE 7: Business Model

## Multiple Revenue Streams, SaaS Economics

### Pricing Tiers

| Tier | Price | Includes | Target |
|------|-------|----------|--------|
| **Starter** | $500/month | 2,000 interactions, 1 language, basic personas | Small clinics |
| **Professional** | $1,500/month | 10,000 interactions, 3 languages, all personas, analytics | Medium hospitals |
| **Enterprise** | $3,000/month | Unlimited interactions, all languages, custom personas, EMR integration, SLA | Hospital chains |
| **Kiosk** | $5,000 setup + $200/month | Hardware + software, self-service terminal | Reception replacement |

### Unit Economics

| Metric | Value |
|--------|-------|
| Average Revenue Per Hospital | $1,500/month ($18K/year) |
| Cost per Interaction | $0.04-0.06 |
| Average Interactions per Hospital | 3,000/month |
| Cost per Hospital | $150-180/month |
| **Gross Margin** | **88-90%** |
| Customer Acquisition Cost (CAC) | $2,000-5,000 |
| Lifetime Value (LTV) | $54K (3-year retention) |
| **LTV/CAC Ratio** | **11-27x** |

### Path to Revenue

| Quarter | Milestone | Revenue |
|---------|-----------|---------|
| Q2 2026 | 5 pilot hospitals (Pakistan) | $7.5K/month |
| Q4 2026 | 25 hospitals (Pakistan + UAE) | $37.5K/month |
| Q2 2027 | 100 hospitals (3 countries) | $150K/month |
| Q4 2027 | 300 hospitals (5 countries) | $450K/month |
| **Q4 2028** | **1,000 hospitals** | **$1.5M/month ($18M ARR)** |

---

# SLIDE 8: Traction & Proof Points

## What We've Built and Proven

### Technical Milestones (Completed)

| Milestone | Status | Details |
|-----------|--------|---------|
| Core voice pipeline | Done | STT → LLM → TTS with <5s latency |
| 6 language support | Done | Urdu, English, Hindi, Arabic, Punjabi, Malay |
| RTL/LTR auto-switching | Done | UI adapts direction per language |
| 10 configurable personas | Done | 5M/5F with gender-appropriate grammar |
| 12 specialty triage | Done | Full doctor panel with real-time availability |
| WebRTC audio | Done | Browser-based, zero installation |
| Session persistence | Done | Settings survive refresh, conversations logged |
| Settings lock during calls | Done | Prevents config changes during active sessions |
| Automated test suite | Done | 77 tests, 100% pass rate |
| Performance benchmarks | Done | All metrics under threshold |
| Dual STT providers | Done | Deepgram Nova-3 + OpenAI gpt-4o-transcribe, switchable from UI |
| TTS audio caching | Done | Disk-backed LRU (500 entries), ~85% cost savings at steady state |
| Patient screening card | Done | Real-time data extraction with printable triage slip |
| Symptom-to-vitals mapping | Done | 15-category intelligent vitals collection |
| Conversation flow stepper | Done | Visual progress: Name → Symptoms → Duration → Vitals → Refer |
| Live model switching | Done | Change LLM/TTS voice mid-call; STT changes on next session |
| Urdu script output | Done | Proper اردو رسم الخط instead of Roman Urdu |
| Cost analysis dashboard | Done | 24 combinations with monthly projections |
| Triage scoring engine | Done | 70+ conditions, 12 departments, real-time scoring |
| 173 symptom aliases | Done | Urdu + English + Roman Urdu + mistranscriptions |
| Two-layer noise filtering | Done | Silero VAD (0.80) + Deepgram confidence (0.70) |
| Age/gender department rules | Done | Pediatrics age-gated, Gynecology gender-gated |
| Cross-condition boost | Done | +15% per additional strong condition in department |
| Exact symptom matching | Done | "cough" ≠ "persistent cough" (prevents cross-contamination) |
| Doctor availability check | Done | Unavailable doctors skipped in referral |
| Urdu triage tooltips | Done | All 199 symptoms with Urdu hover translations |
| Off-topic answer handling | Done | Re-asks unanswered symptom questions |
| Number-to-Urdu TTS | Done | Room "302" spoken as "تین سو دو" |
| Pure Urdu output | Done | No English word mixing in responses |
| 188 triage scoring tests | Done | Scoring accuracy, alias expansion, referral scenarios |

### Key Performance Indicators

| KPI | Target | Actual |
|-----|--------|--------|
| End-to-end voice latency | < 5 seconds | 3-5 seconds |
| STT accuracy | > 90% | 95%+ |
| Triage test pass rate | 100% | 100% (188/188) |
| Platform test pass rate | 100% | 100% (77/77) |
| Symptom alias coverage | 100% | 100% (199/199 symptoms covered) |
| Page load time | < 200ms | < 50ms |
| Languages supported | 4+ | 6 |

---

# SLIDE 9: The Team

## Experienced AI + Healthcare Engineering

### Leadership

**Dr. Aqib Mumtaz, Ph.D.** — Head of AI
- Ph.D. in Computer Science (AI specialization), COMSATS University, Lahore (2019-2024)
- Research collaboration with University of Bergen, Norway — Anomaly Detection
- 19+ years experience in AI/ML, Deep Learning, Computer Vision, NLP
- Published researcher — 5 peer-reviewed papers in anomaly detection, video surveillance, deep learning
- Expertise: ASR (OpenAI Whisper, Wave2Vec2), Transformer architectures, attention mechanisms, real-time AI systems
- State-of-the-art AI stack: GPT-4o, Qwen3-ASR, Deepgram Nova-3, ElevenLabs neural TTS, WebRTC streaming, pipecat-ai orchestration
- Previously led AI research teams of 22+ engineers (SlashNext — phishing detection AI)
- Reviewed Pakistan's National AI Policy (ITU collaboration)
- Award and Honour — University of Kagoshima, Japan

### Core Competencies

| Capability | Evidence |
|-----------|---------|
| **ASR/NLP** | Production ASR system with 70,000+ ICD-10 codes, morphological indexing, fuzzy matching |
| **Deep Learning** | Transformer architectures, attention mechanisms, Whisper fine-tuning, generative models |
| **Real-time Systems** | WebRTC-based transcription bots running in production |
| **Medical AI** | Symptom extraction, negation detection, drug/lab entity recognition |
| **Multilingual** | Urdu, English, Hindi, Arabic, Punjabi, Malay — with cultural adaptation |
| **Research** | 5 published papers, international collaborations (Norway, Japan) |
| **Full-Stack** | Frontend, backend, infrastructure, cloud deployment — end-to-end delivery |

### Hiring Plan (with funding)

| Role | Count | Timeline |
|------|-------|----------|
| AI/ML Engineer | 2 | Immediate |
| Full-Stack Developer | 1 | Month 1-2 |
| Healthcare Domain Expert | 1 | Month 2-3 |
| Business Development (MENA) | 1 | Month 1 |
| DevOps / SRE | 1 | Month 3 |

---

# SLIDE 10: What We're Asking For

## Seed Round: $500K - $1M

### Use of Funds

| Category | Allocation | Purpose |
|----------|-----------|---------|
| **Engineering** (50%) | $250-500K | 4-5 engineers for 12 months. EMR integration, analytics dashboard, kiosk hardware, phone integration |
| **Go-to-Market** (25%) | $125-250K | 5 pilot deployments, sales team for Pakistan/UAE, marketing materials, conference presence |
| **Infrastructure** (15%) | $75-150K | Cloud hosting, TURN servers, API costs for pilot hospitals |
| **Operations** (10%) | $50-100K | Legal (healthcare compliance), accounting, office |

### Milestones This Round Achieves

| Timeline | Milestone | Metric |
|----------|-----------|--------|
| Month 3 | 5 pilot hospitals live | 5 paying customers |
| Month 6 | EMR integration (HL7/FHIR) | Enterprise feature |
| Month 9 | 25 hospitals across 2 countries | $37.5K MRR |
| Month 12 | Series A ready | $50K+ MRR, 50+ hospitals |

### What Investors Get

- **Equity**: 15-20% (seed stage)
- **Market**: First-mover in RTL voice AI for healthcare
- **Technology**: Production-proven, 77-test-validated platform
- **Revenue path**: Clear unit economics with 88-90% gross margins
- **Expansion**: Each new language/country multiplies TAM without rebuilding core tech

---

# SLIDE 11: Why Now?

## The Convergence of Five Trends

### 1. AI Quality Just Crossed the Threshold
GPT-4o-mini delivers human-level conversation quality at $0.01/interaction. Two years ago, this cost $1.00+ and was too slow for real-time voice.

### 2. Voice AI Infrastructure Matured
Deepgram, ElevenLabs, and Pipecat have made production-grade voice pipelines possible in weeks, not years. WebRTC handles browser audio natively.

### 3. Post-COVID Healthcare Digitalization
Hospitals are actively seeking automation. COVID proved that human-only reception is a single point of failure. Government healthcare digitalization programs are funded and active (Saudi Vision 2030, Malaysia MyDigital, Pakistan Digital Health).

### 4. Multilingual AI Just Became Viable
Until 2024, AI voice quality in Urdu, Arabic, and Hindi was poor. Nova-3 and multilingual TTS models now deliver production-quality in these languages for the first time.

### 5. Healthcare Staffing Crisis
Global healthcare worker shortage of 14.7 million (WHO, 2023), projected 11M by 2030. Reception staff turnover is 30-40% annually. AI doesn't call in sick, doesn't need training, and works 24/7.

---

# SLIDE 12: The Ask

## Let's Transform Healthcare Together

### What We Offer

- **Working product** — Not a concept. A production-ready AI clinical triage system you can demo today.
- **Real-time scoring** — 70+ conditions, 12 departments, live confidence percentages. 188 triage tests, 100% pass rate.
- **173 multilingual aliases** — Urdu, Roman Urdu, English, and common STT mistranscriptions. All 199 symptoms with Urdu tooltips.
- **Intelligent noise filtering** — Two-layer: Silero VAD (neural) + Deepgram confidence. Works in noisy hospital environments.
- **Scoring-LLM alignment** — Triage engine decides the doctor, LLM follows the directive. No conflicting recommendations.
- **Cultural intelligence** — Pure Urdu script, gender grammar, Islamic greetings, number-to-Urdu TTS.
- **Hospital-specific** — Configured for real doctors, rooms, schedules, and availability.
- **Cost-optimized** — TTS caching ~85%, dual STT providers, 265 automated tests.
- **Massive market** — 6,250+ hospitals across Pakistan, Saudi Arabia, UAE, Gulf, and Malaysia.

### What We Need

- **$500K-1M seed funding** to deploy in 25 hospitals and build enterprise features
- **Strategic partners** in healthcare (hospital chains, health ministries, healthcare IT vendors)
- **Advisors** with healthcare operations and MENA/South Asia business experience

### The Opportunity

The first company to own **voice AI at hospital reception** in South Asia and MENA will have an unassailable position. Every hospital is a potential customer. Every patient interaction builds our data moat. Every language we add multiplies our market.

**We've built the technology. Now we need to deploy it.**

---

## Contact

**AI-Powered Healthcare Solutions**

For demo, or inquiries:
- Email: aqib.mumtaz@gmail.com

---

*Confidential. For authorized stakeholders only. Financial projections are estimates based on current market research and are subject to change. This presentation does not constitute an offer of securities.*
