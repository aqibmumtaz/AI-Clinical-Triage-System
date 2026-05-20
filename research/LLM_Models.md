# Open-Source LLM (Large Language Model) Options for Urdu

> Research for replacing GPT-4o-mini / Claude Haiku in AI Clinical Triage System
> Date: 2026-04-09

## Current Setup
- **Primary:** GPT-4o-mini (OpenAI, paid API)
- **Secondary:** Claude Haiku 4.5 (Anthropic, paid API)
- **Features used:** Function calling (`update_patient_screening`), structured JSON output, complex system prompts, Urdu script generation
- **Framework:** pipecat 0.0.85 with `OpenAILLMService`

---

## All LLM Models Evaluated

| # | Model | Urdu Quality (Score) | Params (Active) | VRAM (Q4) | tok/s (est.) | Function Calling | Medical | Pipecat Integration | License |
|---|-------|---------------------|-----------------|-----------|-------------|-----------------|---------|-------------------|---------|
| 1 | **Qwen3-8B** | Strong (119 langs) | 8B | ~6 GB | ~40 | Yes (native) | General | Built-in (Ollama) | Apache 2.0 |
| 2 | **Qwen3-30B-A3B** (MoE) | Very strong | 3.3B active / 30.5B total | ~20 GB | ~34 | Yes (native) | General | Built-in (Ollama) | Apache 2.0 |
| 3 | **Qalb-1.0-8B-Instruct** | **90.34** (best tested) | 8B | ~5 GB | ~40 | Untested | General | Via Ollama/vLLM | Open |
| 4 | **Lughaat-1.0-8B-Instruct** | **91.4%** (claimed best) | 8B | ~5 GB | ~40 | Untested | General | Via Ollama/vLLM | Open |
| 5 | **Alif-1.0-8B-Instruct** | 87.1 | 8B | ~5 GB | ~40 | Untested | General | Via Ollama/vLLM | Open |
| 6 | **Llama 3.1 8B** (base) | Poor (45.7) | 8B | ~5 GB | ~68 | Yes | General | Built-in (Ollama) | Llama 3.1 |
| 7 | **Llama 4 Scout** (MoE) | Moderate | 17B active / 109B total | ~55 GB | ~20 | Yes | General | Via vLLM | Llama 4 |
| 8 | **Gemma 4 26B-A4B** (MoE) | Good (140+ langs) | 3.8B active / 25.2B total | ~15 GB | ~35 | Yes (native) | General | Via Ollama | Apache 2.0 |
| 9 | **DeepSeek V3** (MoE) | Good (100+ langs) | 37B active / 671B total | ~380 GB | Fast | Yes | General | Via vLLM | Open |
| 10 | **DeepSeek R1 Distilled 14B** | Fair | 14B | ~8 GB | ~30 | Limited | General | Via Ollama | Open |
| 11 | **Tiny Aya Fire (South Asia)** | Good (South Asian specialist) | 3.35B | ~2 GB | Fast | Limited | General | Via Ollama | Open |
| 12 | **Aya-23-8B** (Cohere) | Good (23 langs incl Urdu) | 8B | ~5 GB | ~40 | Limited | General | Via Ollama | CC-BY-NC |
| 13 | **Mistral 3 / Mixtral** | Poor (European focus) | 8B / 47B | 5-25 GB | ~40 | Yes | General | Built-in (Ollama) | Apache 2.0 |
| 14 | **Phi-4-mini** (Microsoft) | Fair (reduced non-English) | 3.8B | ~2 GB | ~60 | Yes | General | Via Ollama | MIT |
| 15 | **BLOOM / Bloomz** | Poor (outdated) | 3B-176B | 2-350 GB | Slow | No | General | Via vLLM | RAIL |
| 16 | **mGPT** | Poor | 1.3B | ~1 GB | Fast | No | General | Custom | MIT |
| 17 | **Yi 34B** (01.AI) | Poor (Chinese-English) | 34B | ~20 GB | ~20 | Limited | General | Via Ollama | Apache 2.0 |
| 18 | **InternLM2** | Poor (Chinese-English) | 7B-20B | 5-12 GB | ~35 | Yes | General | Via vLLM | Apache 2.0 |
| 19 | **Command R+** (Cohere) | Poor (not in 23 langs) | 104B | ~60 GB | ~10 | Yes (RAG) | General | Via vLLM | CC-BY-NC |
| 20 | **OpenBioLLM-8B** | No Urdu | 8B | ~5 GB | ~40 | Yes | **Medical (beats GPT-4)** | Via Ollama | Llama 3 |
| 21 | **OpenBioLLM-70B** | No Urdu | 70B | ~40 GB | ~10 | Yes | **Medical (best)** | Via vLLM | Llama 3 |
| 22 | **BioMistral-7B** | No Urdu | 7B | ~4 GB | ~45 | Limited | **Medical (PubMed)** | Via Ollama | Apache 2.0 |
| 23 | **Meditron-70B** | No Urdu | 70B | ~40 GB | ~10 | Limited | **Medical (48B tokens)** | Via vLLM | Llama 2 |

---

## Detailed Analysis: Top Candidates

### 1. Qwen3-8B (TOP PICK -- Best Overall)

**Why:** Best balance of Urdu quality, function calling, and deployment ease.

- **119 languages** including Urdu with strong quality
- **Native function calling** via Hermes-style tool use (critical for `update_patient_screening`)
- **Hybrid thinking/non-thinking modes** for reasoning
- Structured JSON output support
- Available on Ollama: `ollama pull qwen3:8b`

**Integration code:**
```python
# Via Ollama (pipecat native)
from pipecat.services.ollama import OLLamaLLMService
self._llm_service = OLLamaLLMService(model="qwen3:8b")

# Via vLLM (OpenAI-compatible)
self._llm_service = OpenAILLMService(
    api_key="local", model="qwen3:8b",
    base_url="http://localhost:8000/v1"
)
```

**Quantization:** GGUF (IQ2_XXS through BF16), AWQ, GPTQ all available.

---

### 2. Qalb-1.0-8B-Instruct (Best Pure Urdu)

**Why:** State-of-the-art Urdu with 90.34 weighted average across 7 Urdu tasks.

- Trained on 1.97B tokens -- 1.84B diverse Urdu text (news, literature, government docs, social media)
- Built on Llama 3.1 8B -- full compatibility with all serving frameworks
- Beats Alif (87.1) by 3.24 points, base Llama 3.1 by 44.64 points
- HuggingFace: `enstazao/Qalb-1.0-8B-Instruct`

**Caveat:** Function calling untested. May need custom structured output handling or constrained decoding via vLLM.

---

### 3. Lughaat-1.0-8B-Instruct (Potentially Best Urdu)

**Why:** Claims 91.4% average across all Urdu evaluation categories (highest of any Urdu model).

- Built on Llama 3.1 8B architecture
- HuggingFace: `muhammadnoman76/Lughaat-1.0-8B-Instruct`

**Caveat:** Less peer-reviewed than Qalb. Worth testing head-to-head.

---

### 4. Qwen3-30B-A3B (Best Quality/Efficiency MoE)

**Why:** 30B total but only 3.3B active per token (MoE). Near-30B quality at 3B inference cost.

**Requires:** ~20 GB VRAM. Best for production with good hardware.

---

### 5. Tiny Aya Fire South Asia (Ultra-Lightweight)

**Why:** Specifically designed for South Asian languages including Urdu. Only 3.35B params, runs on phones.

**Caveat:** Very small model -- may struggle with complex clinical workflow prompts and structured output.

---

## Medical-Specialized Models

No medical LLM supports Urdu natively. Options:

| Model | Medical Score | Size | Use Case |
|-------|-------------|------|----------|
| OpenBioLLM-8B/70B | Beats GPT-4 on 9 datasets | 8B/70B | Symptom validation (English pipeline) |
| BioMistral-7B | Strong PubMed training | 7B | Medical text analysis |
| Meditron-70B | 48.1B medical tokens | 70B | Medical reasoning |

**Practical approach:** Use Qwen3/Qalb for Urdu conversation, rely on general medical knowledge from pre-training.

---

## Serving Frameworks

| Framework | Best For | Concurrent Users | OpenAI API | Function Calling | Setup |
|-----------|---------|-----------------|-----------|-----------------|-------|
| **Ollama** | Dev/single-user | 4 concurrent | Yes | Yes | `brew install ollama` |
| **vLLM** | Production/multi-user | 100+ concurrent | Yes (most complete) | Yes (Hermes parser) | Docker/pip |
| **llama.cpp** | Edge/CPU/low-VRAM | 1-2 | Via wrapper | Limited | Build from source |
| **TGI** | Enterprise/HF integration | High | Yes | Yes | Docker |
| **LocalAI** | Multi-backend orchestration | Varies | Yes | Yes | Docker |
| **LM Studio** | Desktop GUI/prototyping | 1-2 | Yes (local server) | Limited | App download |

**Recommendation:** Start with Ollama for dev/testing, migrate to vLLM for production (3.2x faster at scale).

---

## Apple Silicon Performance

| Chip | Memory | 8B Model Q4 tok/s | 30B Model Q4 tok/s |
|------|--------|-------------------|-------------------|
| M2 Pro | 16-32 GB | ~30-35 | Not feasible at 16GB |
| M3 Max | 36-128 GB | ~28-35 | ~15-20 |
| M4 Max | 36-128 GB | ~18-25 | ~12-18 |

Memory bandwidth is the bottleneck on Apple Silicon, not compute.

---

## Sources
- [Qalb: State-of-the-Art Urdu LLM](https://arxiv.org/html/2601.08141v1)
- [Qalb on HuggingFace](https://huggingface.co/enstazao/Qalb-1.0-8B-Instruct)
- [Qwen3 Blog](https://qwenlm.github.io/blog/qwen3/)
- [UrduBench: Urdu Reasoning Benchmark](https://arxiv.org/html/2601.21000)
- [Alif Urdu LLM](https://huggingface.co/large-traversaal/Alif-1.0-8B-Instruct)
- [Lughaat on HuggingFace](https://huggingface.co/muhammadnoman76/Lughaat-1.0-8B-Instruct)
- [Pipecat Ollama Integration](https://docs.pipecat.ai/server/services/llm/ollama)
- [Ollama vs vLLM Benchmark 2026](https://www.sitepoint.com/ollama-vs-vllm-performance-benchmark-2026/)
- [OpenBioLLM on HuggingFace](https://huggingface.co/aaditya/Llama3-OpenBioLLM-70B)
- [BioMistral Paper](https://arxiv.org/abs/2402.10373)
- [Meditron GitHub](https://github.com/epfLLM/meditron)
