# CLAUDE.md — AI Clinical Triage System

> Entry point for Claude. Read this first, then CLAUDE_CONTEXT.md for project details, then CLAUDE_TRACKER.md for execution history.

## How to Run

```bash
# Kill existing → activate env → install deps → start server
lsof -i TCP:7871 | awk '/LISTEN/{print $2}' | xargs kill -9 2>/dev/null
eval "$(conda shell.bash hook 2>/dev/null)" && conda activate ai-clinical-triage
pip install -r requirements.txt  # only if env is new
python conversation_agent.py     # serves on http://localhost:7871
```

If conda env `ai-clinical-triage` doesn't exist:
```bash
conda create -n ai-clinical-triage python=3.10 -y
```

## Rules (MUST follow)

### API Keys
- **NEVER read API key values** from .env files — ask user to copy manually if needed
- **NEVER print, log, or display** key values in output
- Keys are in `.env` — check key existence with `len()` only

### Code Changes
- **NEVER change pipeline parameters** (VAD, endpointing, confidence thresholds) without running all 252 tests
- **NEVER add audio processing filters** (NoisereduceFilter, FacebookDenoiser) — they cause crashes/distortion. See CLAUDE_TRACKER.md for history
- **NEVER change `allow_interruptions=true`** — setting it to False causes buffered audio flooding
- **NEVER add `ElevenLabsTTSParams`** — doesn't exist in pipecat 0.0.85

### Environment
- Always use conda env named after the project: `ai-clinical-triage`
- Always `conda activate` before running any python command
- Only run benchmarks when explicitly asked

### Testing
- Run tests after any scoring/alias changes: `pytest test_disease_clusters.py test_system_performance.py -v`
- Current passing: 252 tests

## Experiences & Lessons Learned

### What WORKS (don't change)
- Clean pipeline: VAD + STT confidence + noise word filter (no audio processing)
- VAD confidence=0.6, min_volume=0.5, stop_secs=0.6 — Urdu-friendly
- STT confidence threshold 0.45 — lower catches valid Urdu, higher drops it
- All UI broadcasts fire-and-forget (non-blocking)
- _persist() in thread pool
- Per-original-symptom matching in scoring
- Exact word-set matching (_symptom_matches)

### What FAILED (don't repeat)
- NoisereduceFilter → OMP library conflict, audio distortion even with KMP_DUPLICATE_LIB_OK
- FacebookDenoiserFilter (denoiser library) → crashes pipecat pipeline, BaseAudioFilter incompatibility
- KrispFilter → requires paid SDK, not installable without license
- AICFilter (ai-coustics) → requires paid license key
- allow_interruptions=False → buffers audio during bot speech, causes double-transcription flooding
- VAD confidence=0.80 → too strict for Urdu, drops valid speech
- STT confidence=0.70 → drops valid Urdu transcriptions (0.5-0.7 is normal for Urdu)
- eleven_v3 model → NOT supported by pipecat WebSocket API, HTTP 403
- ElevenLabsTTSParams → doesn't exist in pipecat 0.0.85, causes ImportError
- Adding "persistent cough" as alias of "cough" → cross-contamination, inflates Cardiology scores
- Adding "fatigue" as alias of "weakness" → inflates 4 extra departments

### Common Issues & Quick Fixes
- **No voice output**: Check TTS provider in session_state.json — switch to OpenAI if ElevenLabs fails
- **ElevenLabs 403**: Free tier blocked — need Creator key (sk_4...) or use OpenAI TTS
- **"Connecting..." stuck**: Check logs for ImportError or ValidationError — usually a bad import
- **Wrong department referral**: Check alias cross-contamination, run scoring test for that symptom set
- **STT not transcribing**: Check Deepgram key, confidence threshold, VAD min_volume
- **Bot repeating questions**: LLM context issue — check system prompt conversation flow rules
