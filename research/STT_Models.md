# Open-Source STT (Speech-to-Text) Models for Urdu

> Research for replacing Deepgram Nova-3 in AI Clinical Triage System
> Date: 2026-04-09

## Current Setup
- **Provider:** Deepgram Nova-3 (paid API)
- **Secondary:** OpenAI gpt-4o-transcribe
- **Config:** 16kHz, linear16, mono, confidence threshold 0.45, endpointing 600ms
- **Framework:** pipecat 0.0.85

---

## All STT Models Evaluated

| # | Model | Urdu Support | WER (Urdu) | VRAM/RAM | Real-time Streaming | Pipecat Integration | Self-Hosted | Notes |
|---|-------|-------------|-----------|----------|-------------------|-------------------|-------------|-------|
| 1 | **Faster-Whisper large-v3-turbo-urdu** | Yes (fine-tuned) | ~26% | 6 GB GPU | Yes (segmented) | Built-in | Yes | **Top pick.** Zero custom code |
| 2 | **Faster-Whisper large-v3-urdu** | Yes (fine-tuned) | ~22% | 10 GB GPU | Yes (segmented) | Built-in | Yes | Best accuracy, slower |
| 3 | **Whisper large-v3-turbo** (base) | Yes | ~35-40% | 6 GB GPU | Yes (segmented) | Built-in | Yes | No Urdu fine-tuning |
| 4 | **Whisper large-v3** (base) | Yes | ~30-35% | 10 GB GPU | Yes (segmented) | Built-in | Yes | Heaviest, best base |
| 5 | **Whisper medium** | Yes | ~45% | 5 GB GPU | Yes (segmented) | Built-in | Yes | Budget option |
| 6 | **Whisper small** | Yes | ~60% | 2 GB GPU | Yes (segmented) | Built-in | Yes | Low resource |
| 7 | **Whisper base** | Yes | ~97% | 1 GB GPU | Yes (segmented) | Built-in | Yes | Too inaccurate for Urdu |
| 8 | **Whisper tiny** | Yes | ~117% | 1 GB GPU | Yes (segmented) | Built-in | Yes | Unusable for Urdu |
| 9 | **MLX-Whisper large-v3-turbo** | Yes | ~35-40% | Unified (Mac) | Yes (segmented) | Built-in (MLX) | Yes | Apple Silicon only, ~50x RT |
| 10 | **Whisper.cpp** (ggml) | Yes | Same as Whisper | CPU (Q4/Q8) | Yes (segmented) | No (custom needed) | Yes | CPU-optimized, GGML quantized |
| 11 | **Meta MMS (mms-1b-all)** | Yes (native) | Better than Whisper on read speech | 20 GB RAM | No (10x slower) | No (custom needed) | Yes | 1100+ languages, too slow for real-time |
| 12 | **SeamlessM4T v2** | Yes | Beats Whisper on read speech | ~10 GB | No | No (custom needed) | Yes | Whisper wins on conversational |
| 13 | **Sherpa-ONNX (omnilingual)** | Yes (1600 langs) | Unverified | ~2 GB CPU | Yes (true streaming) | No (WebSocket wrapper) | Yes | Best CPU-only streaming option |
| 14 | **wav2vec2 Urdu** (kingabzpro) | Yes (fine-tuned) | Lower than Whisper | ~3 GB | Partial (CTC) | No (custom needed) | Yes | Outperformed by Whisper |
| 15 | **WhisperX** | Yes | Same as Whisper | Same as Whisper | No (batch) | No | Yes | Redundant with pipecat VAD |
| 16 | **SenseVoice** (Alibaba) | No | N/A | ~2 GB | Yes | No | Yes | 5x faster than Whisper, no Urdu |
| 17 | **NVIDIA Canary/Parakeet** | No | N/A | GPU | Yes | Yes (Riva) | Yes | No Urdu, 25 European langs only |
| 18 | **Vosk** | No (no model) | N/A | 50 MB | Yes (true streaming) | No | Yes | No Urdu model exists |
| 19 | **Moonshine** | No (unconfirmed) | N/A | <1 GB | Yes | No | Yes | Edge-focused, no Urdu |
| 20 | **Google USM/Chirp** | Yes | N/A | N/A | Yes | No | **No (cloud only)** | Not open-source |

---

## Detailed Analysis: Top Candidates

### 1. Faster-Whisper with Urdu Fine-tuned Model (TOP PICK)

**Why:** Zero integration effort -- pipecat 0.0.85 already ships with `WhisperSTTService` using Faster-Whisper backend. Urdu language already mapped.

**Models on HuggingFace:**
- `kingabzpro/whisper-large-v3-turbo-urdu` -- WER ~26.23% on Common Voice 17.0
- `kingabzpro/whisper-large-v3-urdu` -- WER ~22% (heavier)

**Performance:**
- 4x faster than vanilla PyTorch Whisper
- INT8 quantization cuts VRAM by ~50% with minimal accuracy loss
- large-v3-turbo: RTFx of 216x on GPU (comfortably real-time)
- Segment-based (not true streaming) but pipecat's VAD handles chunking

**Integration code:**
```python
from pipecat.services.whisper.stt import WhisperSTTService
stt = WhisperSTTService(
    model="kingabzpro/whisper-large-v3-turbo-urdu",
    device="auto",
    compute_type="int8_float16",
    language=Language.UR,
)
```

**Install:** `pip install pipecat-ai[whisper]`

---

### 2. MLX-Whisper (Apple Silicon)

**Why:** Native Apple Silicon acceleration via MLX framework. pipecat has built-in `WhisperSTTServiceMLX`.

**Performance:** ~50x real-time on M2 Ultra with unified memory.

**Limitation:** No fine-tuned Urdu MLX model exists yet. Base model quality ~35-40% WER.

---

### 3. Sherpa-ONNX (CPU-only backup)

**Why:** True streaming ASR via ONNX Runtime, runs on CPU without GPU. 1600+ languages including Urdu.

**Model:** `sherpa-onnx-omnilingual-asr-1600-languages` (300M and 1B variants, INT8 quantized)

**Limitation:** Urdu quality unverified, needs custom pipecat wrapper.

---

### 4. Meta MMS

**Why:** 1100+ languages, halves Whisper's WER on FLEURS benchmark for many languages.

**Limitation:** 20 GB RAM, ~10x slower than Faster-Whisper. NOT practical for real-time voice conversations.

---

## Key Insight: Medical Terminology

No open-source Urdu medical ASR model exists. Best path:
1. Start with `kingabzpro/whisper-large-v3-turbo-urdu`
2. Fine-tune further on Urdu medical speech data if available
3. Use LLM-based post-processing to correct medical term transcription errors

---

## Sources
- [Benchmarking Whisper for Low-Resource Speech Recognition](https://aclanthology.org/2025.chipsal-1.20/)
- [WER We Stand: Benchmarking Urdu ASR Models](https://arxiv.org/html/2409.11252v1)
- [kingabzpro/whisper-large-v3-turbo-urdu](https://huggingface.co/kingabzpro/whisper-large-v3-turbo-urdu)
- [kingabzpro/whisper-large-v3-urdu](https://huggingface.co/kingabzpro/whisper-large-v3-urdu)
- [Faster-Whisper GitHub](https://github.com/SYSTRAN/faster-whisper)
- [Sherpa-ONNX GitHub](https://github.com/k2-fsa/sherpa-onnx)
- [Meta MMS](https://huggingface.co/docs/transformers/model_doc/mms)
- [Pipecat Whisper STT docs](https://docs.pipecat.ai/server/services/stt/whisper)
