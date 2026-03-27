# CLAUDE_TRACKER.md — Execution & Development Log

> Long-term tracker for bugs, fixes, decisions, and things to watch during ongoing development.

## Active Watch List

| Item | Status | Notes |
|------|--------|-------|
| ElevenLabs free tier | BLOCKED | "Unusual activity detected" — need Creator key or paid plan |
| Krisp SDK | NOT INSTALLED | Requires sales approval at sdk.krisp.ai, 14-day trial after approval |
| Background human voices | UNSOLVED | No audio filter can separate voices without Krisp. Current defense: VAD + confidence only |
| Deepgram Urdu accuracy | MONITORING | Confidence scores 0.5-0.7 typical for Urdu. Threshold at 0.45 to avoid false drops |
| pipecat version | 0.0.85 | Check for updates — newer versions may add Krisp/AIC support or fix BaseAudioFilter |

## Bugs Found & Fixed (Chronological)

### 2026-03-21: TTS Cache Only Playing Last Sentence
- **Root cause**: TTSCacheCapture reset audio on each TTSStartedFrame, saved on TTSStoppedFrame. TTS processes sentences individually → only last sentence cached.
- **Fix**: Accumulate audio from LLMFullResponseStartFrame, save once at LLMFullResponseEndFrame.

### 2026-03-21: Alias Double-Counting Inflating Scores
- **Root cause**: Global alias expansion created one big set. "fever" expanded to include "high fever", both matched same condition → inflated score.
- **Fix**: Per-original-symptom matching — each original symptom gets its own variant set, matches at most ONE condition symptom.

### 2026-03-21: Scoring Not Syncing with LLM Referral
- **Root cause**: Cardiology 33% vs Pediatrics 20% — correct ranking but low scores. LLM made correct call but dashboard showed confusing numbers.
- **Fix**: Added cross-condition confidence boost: `best_score + best_score × 0.15 × (strong_conditions - 1)`.

### 2026-03-21: Bot Referring After Only 1 Symptom
- **Root cause**: System prompt rule (b) was too loose — LLM interpreted 1 symptom + duration = enough.
- **Fix**: Added explicit minimum: "At least 2 symptoms OR 1 symptom + severity + vital."

### 2026-03-21: Room Number "302" Not Pronounced in Urdu
- **Fix**: Built `_numbers_to_urdu_words()` converter, applied only in TTSCacheGate cache-miss path for Urdu. Display keeps digits.

### 2026-03-22: Cross-Contamination — 8 Alias Pairs Removed
- headache→migraine, weakness→fatigue, congestion→nasal congestion
- cough→persistent cough, fever→high fever, headache→severe headache
- stomach pain→lower abdominal pain, back pain→lower back pain
- **Impact**: Fixed wrong department routing for common symptoms.

### 2026-03-22: VAD/Noise Filters Broke STT
- **Root cause**: Aggressive settings (VAD 0.80, STT confidence 0.70) dropped valid Urdu speech.
- **Fix**: Reverted to 0.6/0.45. Removed all audio processing filters (NoisereduceFilter, FacebookDenoiser).

### 2026-03-22: NoisereduceFilter Audio Distortion
- **Root cause**: OMP library conflict between torch and noisereduce. Even KMP_DUPLICATE_LIB_OK didn't fully fix it.
- **Fix**: Removed entirely. Clean pipeline without audio processing works best.

### 2026-03-22: FacebookDenoiserFilter Pipeline Crash
- **Root cause**: Custom FacebookDenoiserFilter didn't extend pipecat's BaseAudioFilter properly. Pydantic ValidationError on TransportParams.
- **Fix**: Removed. Krisp is the only viable AI noise filter for pipecat.

### 2026-03-22: allow_interruptions=False Caused Double Transcriptions
- **Root cause**: With interruptions disabled, VAD still detected speech during bot TTS. Audio buffered and flooded STT when bot finished → multiple transcriptions of same utterance.
- **Fix**: Reverted to allow_interruptions=true. STT always running is the correct approach.

### 2026-03-22: ElevenLabsTTSParams ImportError
- **Root cause**: `ElevenLabsTTSParams` doesn't exist in pipecat 0.0.85. Added for auto_mode fix but class doesn't exist.
- **Fix**: Removed the import. Use default ElevenLabsTTSService constructor.

### 2026-03-22: eleven_v3 Model HTTP 403
- **Root cause**: pipecat's WebSocket TTS API doesn't support eleven_v3 model. REST API works but WebSocket gets 403.
- **Fix**: Auto-fallback from eleven_v3 to eleven_flash_v2_5 in code. Removed eleven_v3 from UI dropdown.

### 2026-03-23: Bone Pain/Fracture Going to Gynecology
- **Root cause**: (1) Missing Urdu aliases — ہڈیاں (plural) vs ہڈی (singular), فریکچر not in aliases. (2) Dr. Usman Tariq unavailable.
- **Fix**: Added plural/STT aliases. Added unavailable doctor handling — falls back to General Medicine with note.

## Decisions Made

| Date | Decision | Reasoning |
|------|----------|-----------|
| 2026-03-21 | Per-symptom matching over global expansion | Prevents alias chain inflation |
| 2026-03-22 | No audio processing filters | All tested options cause crashes or distortion |
| 2026-03-22 | allow_interruptions=true always | False causes worse problems than noise |
| 2026-03-22 | STT confidence 0.45 for Urdu | Urdu scores 0.5-0.7, lower threshold catches all valid speech |
| 2026-03-22 | Non-blocking all UI broadcasts | Reduced pipeline overhead from ~200-500ms to < 1ms |
| 2026-03-22 | Scoring skipped on greeting turns | No symptoms → no scoring needed → faster response |
| 2026-03-22 | OpenAI TTS as default | More reliable than ElevenLabs, works without paid plan |
| 2026-03-23 | Removed eleven_v3 from dropdown | Not supported by pipecat WebSocket API |
| 2026-03-27 | Renamed directory to AI-Clinical-Triage-System | Matches project name |

## Performance Benchmarks (2026-03-22)

| Component | Target | Actual | Status |
|-----------|--------|--------|--------|
| Scoring (English) | < 50ms | 4.2ms | PASS |
| Scoring (Urdu + aliases) | < 50ms | 5.2ms | PASS |
| Scoring (empty/greeting) | < 1ms | 0.03ms | PASS |
| Symptom matching | < 0.1ms | 0.001ms | PASS |
| Cache key (SHA-256) | < 1ms | 0.005ms | PASS |
| Noise word lookup | < 0.01ms | 0.0001ms | PASS |
| Async fire-and-forget | < 1ms | 0.072ms | PASS |
| Total pipeline overhead | < 5ms | < 1ms | PASS |

## Test Coverage

| Suite | File | Tests | Status |
|-------|------|-------|--------|
| Triage Scoring | test_disease_clusters.py | 188 | ALL PASS |
| System Performance | test_system_performance.py | 62 | ALL PASS |
| Ortho Fix | test_system_performance.py | 2 | ALL PASS |
| **Total** | | **252** | **ALL PASS** |
