# Recommended Architectures for Self-Hosted Deployment

> Combining STT + LLM + TTS for AI Clinical Triage System
> Date: 2026-04-09

## Current System (Paid APIs)

| Component | Provider | Model | Cost |
|-----------|----------|-------|------|
| STT | Deepgram | Nova-3 | Paid API |
| LLM | OpenAI | GPT-4o-mini | Paid API |
| TTS | OpenAI / ElevenLabs | tts-1 / flash_v2.5 | Paid API |

---

## Option A: Minimal Effort, Maximum Impact

Best for: Getting off paid APIs quickly with minimal code changes.

| Component | Model | VRAM | Integration Effort | Quality |
|-----------|-------|------|-------------------|---------|
| STT | Faster-Whisper (Urdu fine-tuned) | 6 GB | Zero (pipecat built-in) | Good (~26% WER) |
| LLM | Qwen3-8B via Ollama | 6 GB | Low (pipecat built-in) | Strong Urdu + function calling |
| TTS | Edge TTS (openai-edge-tts) | 0 GB | Low (OpenAI-compatible) | Production-grade |
| **Total** | | **~12 GB** | **Minimal** | **Good** |

**Hardware:** Any GPU with 12+ GB VRAM (RTX 3060 12GB, RTX 4070, etc.) or Apple Silicon with 16+ GB unified memory.

**Caveat:** TTS still calls Microsoft's free servers (not truly offline).

---

## Option B: Fully Offline / Self-Hosted

Best for: Complete independence from any external service.

| Component | Model | VRAM | Integration Effort | Quality |
|-----------|-------|------|-------------------|---------|
| STT | Faster-Whisper (Urdu fine-tuned) | 6 GB | Zero (pipecat built-in) | Good (~26% WER) |
| LLM | Qwen3-8B via vLLM | 6 GB | Low (OpenAI-compatible) | Strong Urdu + function calling |
| TTS | Orpheus TTS (Urdu fine-tuned) | 15 GB | Medium (OpenAI-compatible) | Experimental Urdu |
| **Total** | | **~27 GB** | **Medium** | **Good (TTS experimental)** |

**Hardware:** RTX 4090 (24 GB) or dual GPU setup. On Apple Silicon: M3/M4 Max with 64+ GB unified memory.

---

## Option C: Best Urdu Quality

Best for: Maximum Urdu fluency and accuracy.

| Component | Model | VRAM | Integration Effort | Quality |
|-----------|-------|------|-------------------|---------|
| STT | Faster-Whisper large-v3-urdu | 10 GB | Zero (pipecat built-in) | Best (~22% WER) |
| LLM | Qalb-1.0-8B (Urdu specialist) | 5 GB | Medium (Ollama/vLLM) | Best Urdu (90.34 score) |
| TTS | Edge TTS | 0 GB | Low (OpenAI-compatible) | Production-grade |
| **Total** | | **~15 GB** | **Medium** | **Best Urdu** |

**Caveat:** Qalb function calling untested -- may need custom structured output handling. TTS still calls Microsoft.

---

## Option D: Ultra-Lightweight / Edge

Best for: Low-resource hardware, mobile/embedded deployment.

| Component | Model | VRAM | Integration Effort | Quality |
|-----------|-------|------|-------------------|---------|
| STT | Whisper medium (base) | 5 GB | Zero (pipecat built-in) | Fair (~45% WER) |
| LLM | Tiny Aya Fire South Asia | 2 GB | Low (Ollama) | Good Urdu, limited reasoning |
| TTS | MMS-TTS Urdu | 1 GB | Medium (custom wrapper) | Robotic but functional |
| **Total** | | **~8 GB** | **Medium** | **Functional** |

**Hardware:** Any GPU with 8+ GB or CPU-only with patience.

---

## Option E: Two-Model LLM Pipeline (Best of Both Worlds)

Best for: Maximum quality across both Urdu and clinical accuracy.

| Component | Model | VRAM | Role |
|-----------|-------|------|------|
| STT | Faster-Whisper large-v3-turbo-urdu | 6 GB | Speech recognition |
| LLM (conversation) | Qalb-1.0-8B | 5 GB | Patient-facing Urdu dialogue |
| LLM (extraction) | Qwen3-8B | 6 GB | Symptom extraction, JSON, scoring |
| TTS | Edge TTS | 0 GB | Speech output |
| **Total** | | **~17 GB** | |

**How it works:** Qalb handles the Urdu conversation, Qwen3 handles function calling and structured output. Slightly higher latency but best quality in both dimensions.

---

## Option F: Future-Proof (Speech-to-Speech)

Best for: When Qwen3.5-Omni matures with Urdu speech output.

| Component | Model | VRAM | Notes |
|-----------|-------|------|-------|
| All-in-one | Qwen3.5-Omni | 20+ GB | Native speech-to-speech in 36 langs |

**Status:** Urdu speech INPUT supported, speech output expanding. Could eventually replace entire STT + LLM + TTS pipeline with single model.

---

## Viable Urdu Options Summary

| Component | Model | Urdu Quality | VRAM | Effort | Recommended |
|-----------|-------|-------------|------|--------|-------------|
| **STT** | Faster-Whisper (Urdu fine-tuned) | ~26% WER | 6 GB | Zero (built-in) | **Yes** |
| **STT** | Faster-Whisper large-v3-urdu | ~22% WER | 10 GB | Zero (built-in) | Best accuracy |
| **STT** | Sherpa-ONNX omnilingual | Unknown | CPU | Medium | CPU-only backup |
| **STT** | Meta MMS | Strong read speech | 20 GB | High | Too slow for RT |
| **LLM** | Qwen3-8B | Strong + function calling | 6 GB | Low (Ollama) | **Yes** |
| **LLM** | Qalb-1.0-8B | Best Urdu (90.34) | 5 GB | Medium | Best Urdu quality |
| **LLM** | Lughaat-1.0-8B | Best Urdu (91.4% claimed) | 5 GB | Medium | Test it |
| **LLM** | Qwen3-30B-A3B | Very strong MoE | 20 GB | Low | Production upgrade |
| **TTS** | Edge TTS | Production-grade | 0 GB | Low (Docker) | **Yes** |
| **TTS** | Orpheus Urdu | Experimental | 15 GB | Medium | Best self-hosted |
| **TTS** | MMS-TTS Urdu | Robotic | 1 GB | Medium | Fallback only |

---

## Quick Start: Option A

```bash
# 1. Install Ollama and pull Qwen3
brew install ollama
ollama pull qwen3:8b

# 2. Start Edge TTS server
docker run -d -p 5050:5050 travisvn/openai-edge-tts:latest

# 3. Install Whisper for pipecat
pip install pipecat-ai[whisper]

# 4. Download Urdu Whisper model (auto-downloads on first use)
# Model: kingabzpro/whisper-large-v3-turbo-urdu
```

Then update `conversation_agent.py` to use local services instead of paid APIs.
