# Open-Source TTS (Text-to-Speech) Models for Urdu

> Research for replacing OpenAI TTS / ElevenLabs in AI Clinical Triage System
> Date: 2026-04-09

## Current Setup
- **Primary:** OpenAI tts-1 (nova voice, paid API)
- **Secondary:** ElevenLabs eleven_flash_v2_5 (paid API)
- **Config:** 24kHz output, mono, hot-swappable via TTSProxy
- **Framework:** pipecat 0.0.85 with `OpenAITTSService`

---

## All TTS Models Evaluated

| # | Model | Urdu Support | Urdu Voices | Quality | Latency | VRAM | Streaming | Pipecat Integration | Truly Self-Hosted |
|---|-------|-------------|-------------|---------|---------|------|-----------|-------------------|-------------------|
| 1 | **Edge TTS (openai-edge-tts)** | Yes (native) | UzmaNeural (F), AsadNeural (M) | Production-grade | ~100-200ms | 0 GB | Yes | Drop-in (OpenAI API) | No (calls MS servers, free) |
| 2 | **Orpheus TTS (Urdu fine-tuned)** | Yes (community) | mahwizzzz/orpheus-urdu-tts | Good (experimental) | ~200ms | 15 GB | Yes | OpenAI-compatible API | Yes |
| 3 | **Orpheus TTS (Zia Mohyeddin)** | Yes (community) | Aasher/Orpheus-Urdu-TTS | Voice cloned, some pronunciation issues | ~200ms | 15 GB | Yes | OpenAI-compatible API | Yes |
| 4 | **MMS-TTS Urdu** (Meta) | Yes (native) | Single speaker (Arabic script) | Robotic/monotone | Fast | ~1 GB | No | Custom wrapper needed | Yes |
| 5 | **SpeechT5 Urdu** (community) | Yes (fine-tuned) | Multiple community voices | Variable | Moderate | ~2 GB | No | Custom wrapper needed | Yes |
| 6 | **Fish Speech / S2** | Unconfirmed (80+ langs) | Unknown | Excellent (if supported) | <100ms | 12 GB | Yes | Built-in (pipecat) | Yes |
| 7 | **Bark** (Suno) | Partial (Hindi, not Urdu) | Hindi presets may work | Natural but inconsistent | Slow (2-10x RT) | 5-8 GB | No | No | Yes |
| 8 | **Voxtral TTS** (Mistral) | No (Hindi/Arabic, not Urdu) | N/A | Very high (beats ElevenLabs) | 70ms | 8-16 GB | Yes | No (too new) | Yes |
| 9 | **Coqui XTTS v2** | No (16 langs, no Urdu) | N/A | High | Moderate | 4-6 GB | Yes | Built-in (pipecat) | Yes |
| 10 | **Piper TTS** | No (no Urdu voice) | N/A | Good | Very fast | CPU | Yes | Built-in (pipecat) | Yes |
| 11 | **MeloTTS** | No (6 langs) | N/A | Good | Fast | ~2 GB | No | No | Yes |
| 12 | **Kokoro TTS** | No (8 langs) | N/A | Excellent (82M params) | Fast | <1 GB | Yes | Built-in (pipecat) | Yes |
| 13 | **F5-TTS** | No (EN/CN) | N/A | High | Moderate | 4-6 GB | Yes | No | Yes |
| 14 | **StyleTTS 2** | No (14 langs) | N/A | Very high | Fast | ~2 GB | No | No | Yes |
| 15 | **GPT-SoVITS** | No (CJK + EN) | N/A | High (cloning) | Moderate | 6-8 GB | No | No | Yes |
| 16 | **Dia** (Nari Labs) | No (English only) | N/A | Excellent (conversational) | Moderate | ~8 GB | Yes | No | Yes |
| 17 | **Sesame CSM** | No (English only) | N/A | Excellent | Moderate | ~8 GB | Yes | No | Yes |
| 18 | **Parler TTS** | No (English) | N/A | Good | Moderate | 4-6 GB | No | No | Yes |
| 19 | **CosyVoice 2/3** | No (9 langs) | N/A | Excellent | 150ms | 6-8 GB | Yes | No | Yes |
| 20 | **Qwen3-TTS** | No (10 langs) | N/A | Very high | 97ms | 8-12 GB | Yes | No | Yes |
| 21 | **Matcha-TTS** | No (English) | N/A | Good | Very fast | ~1 GB | No | No | Yes |
| 22 | **AllTalk TTS** | Depends on engine | N/A | Varies | Varies | Varies | Varies | No | Yes |
| 23 | **Meta Voicebox** | N/A | N/A | N/A | N/A | N/A | N/A | N/A | **Not open-source** |

---

## Detailed Analysis: Top Candidates

### 1. Edge TTS via openai-edge-tts (TOP PICK)

**Why:** Drop-in replacement for OpenAI TTS with production-quality Urdu voices, zero GPU, zero cost.

**Urdu Voices:**
- `ur-PK-UzmaNeural` (Female) -- clear, professional Pakistani Urdu
- `ur-PK-AsadNeural` (Male) -- clear, professional Pakistani Urdu

**Integration code:**
```python
return OpenAITTSService(
    api_key="dummy",  # not needed for edge-tts
    voice="ur-PK-UzmaNeural",
    model="tts-1",
    sample_rate=SAMPLE_RATE_OUT,
    base_url="http://localhost:5050/v1",
)
```

**Serve:** `docker run -d -p 5050:5050 travisvn/openai-edge-tts:latest`

**Pros:**
- Zero cost, zero GPU
- Production-quality Microsoft neural voices
- OpenAI-compatible API (drop-in for pipecat)
- Streaming support
- No model training needed

**Cons:**
- Still calls Microsoft's servers (free, no key needed)
- If Microsoft deprecates Edge TTS endpoint, it stops
- Not truly offline-capable

---

### 2. Orpheus TTS (Urdu fine-tuned) -- Best Truly Self-Hosted

**Why:** Emotion-aware TTS with Urdu fine-tuned models, fully self-hosted.

**Models on HuggingFace:**
- `mahwizzzz/orpheus-urdu-tts` -- trained on 20.4k Urdu audio samples (UAT dataset)
- `Aasher/Orpheus-Urdu-TTS-Zia-Mohyeddin` -- voice-cloned from Zia Mohyeddin. GGUF quantized available.

**Features:**
- Emotion tags: `<laugh>`, `<sigh>`, `<cough>` (useful for empathetic clinical conversations)
- ~200ms streaming latency
- OpenAI-compatible API via Orpheus-FastAPI

**Serve:** Run Orpheus-FastAPI with vLLM backend + SNAC decoder. Docker available.

**Caveat:** Urdu models are experimental. Zia Mohyeddin variant notes "cannot pronounce many Urdu words properly." May need further fine-tuning on medical vocabulary.

---

### 3. Meta MMS-TTS Urdu -- Lightweight Fallback

**Why:** Fully offline, very lightweight, native Urdu with Arabic script support.

**Models:**
- `facebook/mms-tts-urd-script_arabic` (Nastaliq/Arabic script)
- `facebook/mms-tts-urd-script_devanagari` (Devanagari script)

**Performance:** RTF ~0.067 (67ms per second of audio on GPU). Can run on CPU.

**Quality:** Functional but robotic/monotone. Single speaker, trained on religious text recordings. An Interspeech 2025 paper found custom models outperform MMS in intelligibility and naturalness.

**Serve:** HuggingFace Transformers pipeline + FastAPI, or openedai-speech Docker container.

---

### 4. SpeechT5 Urdu (Community Fine-tunes)

**Models on HuggingFace:**
- `pocketmonkey/speecht5_tts_urdu`
- `arqamwadiwala/tts_urdu`
- `TheUpperCaseGuy/Guy-Urdu-TTS`

**Quality:** Variable depending on fine-tune quality. ~1-2 GB VRAM. No streaming.

---

### 5. Fish Speech (Test Urdu Support)

**Why:** Has built-in pipecat integration, <100ms latency, excellent quality. Claims 80+ languages but Urdu unconfirmed.

**Action:** Test Urdu text with Fish Speech to verify support before investing.

---

## Key Insight: Urdu TTS is the Weakest Link

The Urdu TTS ecosystem is far behind STT and LLM. Most state-of-the-art TTS models (Kokoro, Dia, Sesame, CosyVoice, Qwen3-TTS) do NOT support Urdu. Edge TTS is the pragmatic choice until truly self-hosted options mature.

---

## Sources
- [openai-edge-tts GitHub](https://github.com/travisvn/openai-edge-tts)
- [Edge TTS Voice Samples](https://tts.travisvn.com/)
- [Orpheus TTS GitHub](https://github.com/canopyai/Orpheus-TTS)
- [Orpheus Urdu TTS](https://huggingface.co/mahwizzzz/orpheus-urdu-tts)
- [Orpheus Urdu Zia Mohyeddin](https://huggingface.co/Aasher/Orpheus-Urdu-TTS-Zia-Mohyeddin)
- [MMS-TTS Urdu](https://huggingface.co/facebook/mms-tts-urd-script_arabic)
- [Interspeech 2025 Urdu/Punjabi TTS](https://www.isca-archive.org/interspeech_2025/naseem25_interspeech.html)
- [SpeechT5 Urdu](https://huggingface.co/pocketmonkey/speecht5_tts_urdu)
- [Fish Speech GitHub](https://github.com/fishaudio/fish-speech)
