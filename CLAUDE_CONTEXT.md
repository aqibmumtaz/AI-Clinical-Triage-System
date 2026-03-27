---
name: ai_clinical_triage_system
description: Full project context for AI Clinical Triage System — architecture, pipeline, scoring, fixes, and current state
type: project
---

## Project: AI Clinical Triage System

**Directory**: `R&D/AI-Clinical-Triage-System/`
**Conda env**: `ai-clinical-triage` (Python 3.10) — create if missing, install from requirements.txt
**Run**: `python conversation_agent.py` → serves on port 7871
**Port**: 7871

## Quick Start (when user says "run project")

```bash
# 1. Create conda env if not exists
conda create -n ai-clinical-triage python=3.10 -y 2>/dev/null || true

# 2. Activate
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate ai-clinical-triage

# 3. Install dependencies
pip install -r requirements.txt

# 4. Kill any existing process on port 7871
lsof -i TCP:7871 | awk '/LISTEN/{print $2}' | xargs kill -9 2>/dev/null

# 5. Run
cd R&D/AI-Clinical-Triage-System
python conversation_agent.py
# Open http://localhost:7871
```

**IMPORTANT**: Always use conda env named after the project. If env doesn't exist, create it first with Python 3.10 and install requirements.txt.
**Main file**: `conversation_agent.py` (~115KB)
**Frontend**: `templates/conversation_agent.html` (~118KB)
**Disease DB**: `disease_clusters.json` (12 departments, 70+ conditions, 199 condition symptoms, 173+ aliases)
**Tests**: 252 total (188 in test_disease_clusters.py + 62 in test_system_performance.py + 2 ortho)

## Architecture — Pipecat 0.0.85 Pipeline

```
WebRTC Input → Silero VAD → AudioLevelMonitor → ConfidenceFilteredDeepgramSTT
→ STTBroadcaster → UserContextAggregator → LLM (GPT-4o-mini, tool_choice=required)
→ ResponseBroadcaster → TTSCacheGate → TTS (OpenAI/ElevenLabs) → TTSCacheCapture
→ AssistantContextAggregator → WebRTC Output
```

## Current Working Parameters (VERIFIED — don't change without testing)

```
Transport:     allow_interruptions=true, audio_in_filter=None
Silero VAD:    confidence=0.6, start_secs=0.2, stop_secs=0.6, min_volume=0.5
Deepgram STT:  endpointing=600ms, utterance_end_ms=1500, no_delay=true, vad_events=true
STT Filter:    confidence_threshold=0.45 (Urdu-friendly)
LLM:           gpt-4o-mini, tool_choice=required
TTS:           OpenAI tts-1, voice=nova, 24kHz
Performance:   All UI broadcasts non-blocking, _persist() in thread pool, scoring skipped on greeting turns
```

## Key Systems Implemented

### 1. Triage Scoring Engine (`compute_cluster_scores()`)
- Per-original-symptom matching (prevents alias double-counting)
- Exact word-set matching (`_symptom_matches()`) — "cough" ≠ "persistent cough"
- Cross-condition boost: `best_score + best_score × 0.15 × (strong_conditions - 1)`
- Department rules: Pediatrics zeroed for age≥12, Gynecology zeroed for males
- Doctor availability check — unavailable doctors skipped in REFER TO directive
- Triage summary returned to LLM: "CONFIDENCE LOW (38%) — ask more" or "READY TO REFER → Dr. X"

### 2. ConfidenceFilteredDeepgramSTT (custom subclass)
- Drops final transcriptions with confidence < 0.45
- Passes interim results through for live UI
- Null-safe confidence handling

### 3. TTSCacheGate + TTSCacheCapture
- Disk-backed LRU cache for TTS audio
- Accumulates audio across ALL sentences (LLMFullResponseStart→End lifecycle)
- Number-to-Urdu converter in cache-miss path (302 → تین سو دو)
- Minimum 5KB audio threshold to reject corrupt entries

### 4. STTBroadcaster — Noise Word Filter
- Drops: hmm, ok, yeah, repeated words, < 2 chars
- Non-blocking UI broadcasts (stt_live, conversation, status)
- _persist() runs in thread pool

### 5. ResponseBroadcaster
- Extracts spoken_text from tool call
- Non-blocking broadcasts for patient_data, cluster_data
- Strips [PATIENT_DATA] blocks from TTS path

### 6. AudioLevelMonitor
- Lightweight int-based RMS (no numpy), every 10th frame
- Broadcasts mic_level to UI for meter animation

## Alias System
- 173+ aliases covering Urdu, Roman Urdu, English, STT mistranscriptions
- Cross-contamination cleaned: removed headache→migraine, weakness→fatigue, congestion→nasal congestion, cough→persistent cough, fever→high fever, headache→severe headache, stomach pain→lower abdominal pain, back pain→lower back pain
- STT mistranscription aliases added: کاشی/کھاسی/کاسی→cough, نیند نہیں آتی→insomnia, ہڈیاں/فریکچر→bone pain/fracture

## Noise Filtering (3 layers, no audio processing)
1. **Silero VAD** (audio) — blocks non-speech
2. **STT confidence filter** (transcription) — drops < 0.45
3. **Noise word filter** (text) — drops known noise words

**Tried and removed:**
- NoisereduceFilter — OMP library conflict, caused audio distortion
- FacebookDenoiserFilter — crashes in pipecat pipeline (BaseAudioFilter incompatibility)
- KrispFilter — requires paid SDK (sales approval needed at sdk.krisp.ai)
- ai-coustics (AICFilter) — requires paid license

## ElevenLabs Status
- Free tier: blocked ("unusual activity detected")
- Creator key (sk_4...): WORKS — tier=creator, 286K chars remaining
- eleven_v3: NOT supported by pipecat WebSocket API (HTTP 403)
- eleven_flash_v2_5: works with Creator key
- Code auto-falls back from eleven_v3 to eleven_flash_v2_5

## UI Features
- Triage dashboard with live scoring
- Urdu tooltips on symptom tags (hover)
- Pricing on all dropdowns (STT, LLM, TTS) — verified against official sources
- LLM models grouped by provider (OpenAI → Anthropic)
- Mic level meter bar
- Cost analysis table
- Title: "City General Hospital" / subtitle: "AI Clinical Triage System"

## System Prompt Key Rules
- Pure Urdu output (no English mixing)
- 1-2 sentences max per turn
- ONE question at a time
- Conversation flow: Greeting → Name → Symptoms → Duration → Vitals → Smart follow-up → Refer
- Minimum requirements before referral: 2 symptoms OR 1 symptom + severity + vital
- MUST follow REFER TO directive from triage scores
- Re-ask unanswered questions (off-topic answer handling)

## Documentation
- `README.md` — Full project documentation
- `ARD/Project_Overview.md` — Architecture document (v3.0)
- `ARD/Project_Presentation.md` — Presentation slides (v3.0)
- `ARD/generate_docs.py` — PDF/PPTX generator

## Performance
- Pipeline overhead: < 1ms (all broadcasts non-blocking)
- Scoring: ~5-10ms per call
- Total end-to-end: ~2-4 seconds
- 252 tests: ALL PASS

## Conversation Context (What Was Discussed & Built)

### Session 1 (2026-03-21): Core Triage + TTS Fixes
- Built 188 test cases for scoring and referral
- Fixed TTS cache only playing last sentence → accumulate across all sentences
- Fixed alias double-counting → per-original-symptom matching
- Added cross-condition confidence boost for better department differentiation
- Tightened referral rules → minimum 2 symptoms before referring
- Added number-to-Urdu TTS converter (302 → تین سو دو)
- Expanded symptom_aliases from 26 → 173+ entries with full Urdu coverage
- Separated condition symptoms from alias groups (chest tightness ≠ chest pain alias)
- Fixed second symptom being missed → off-topic answer handling in system prompt

### Session 2 (2026-03-22): Noise, Performance, Alignment
- Built ConfidenceFilteredDeepgramSTT with 0.45 threshold
- Tried and removed: NoisereduceFilter (OMP crash), FacebookDenoiser (pipeline crash), Krisp (paid SDK), ai-coustics (paid license)
- Made all UI broadcasts non-blocking (fire-and-forget)
- Made _persist() non-blocking (thread pool)
- Skip scoring on greeting/name turns
- Cleaned 8 cross-contamination alias pairs
- Added department selection rules (age/gender gating)
- Added exact word-set matching (_symptom_matches)
- Added doctor availability check
- Added triage directive alignment (REFER TO →)
- Added Urdu tooltips on triage tags
- Added pricing labels on all dropdowns (verified against official sources)
- Grouped LLM models by provider
- Added AudioLevelMonitor for mic meter
- Tuned VAD: confidence=0.6, stop_secs=0.6 (Urdu-friendly)
- Reverted allow_interruptions to true (False caused audio flooding)
- Tested ElevenLabs: free tier blocked, Creator key works, eleven_v3 not supported by pipecat WS

### Session 3 (2026-03-23): Ortho Fix + Repeating Questions
- Fixed bone pain/fracture routing to Gynecology → added missing Urdu aliases (ہڈیاں, فریکچر)
- Fixed unavailable doctor handling → falls back with note
- Fixed LLM repeating questions → added conversation history awareness in system prompt
- Verified ElevenLabs Creator key working (REST API 200, WS needs flash_v2_5 not v3)
- Removed eleven_v3 from UI dropdown

### Session 4 (2026-03-27): Refactoring & Documentation
- Renamed directory: multilingual → AI-Clinical-Triage-System
- Updated all path references in README, ARD docs
- Updated requirements.txt with actual dependencies
- Created CLAUDE.md (rules), CLAUDE_CONTEXT.md (project context), CLAUDE_TRACKER.md (execution log)

### User Preferences Observed
- Wants short, direct answers — no lengthy explanations
- Says "restart" or "restart app" to restart the server
- Says "run project" or "run kar do" to start everything from scratch
- Prefers to test live after each change
- Expects all Urdu output — no English mixing
- Checks logs and scoring alignment carefully
- Wants pricing verified against official sources
- Values system reliability over new features
