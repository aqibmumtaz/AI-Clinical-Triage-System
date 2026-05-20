# Fine-Tuning — Urdu Medical Lab Call Center Bot

Complete guide to fine-tuning STT, LLM, and TTS models for a Pakistani medical lab call center AI agent capable of handling patient calls in Urdu/Roman Urdu.

---

## Table of Contents

1. [Overview](#overview)
2. [Directory Structure](#directory-structure)
3. [Current Dataset Status](#current-dataset-status)
4. [Priority & ROI](#priority--roi)
5. [Expected Benchmarks](#expected-benchmarks)
6. [Phase 1 — LLM Fine-Tune](#phase-1--llm-fine-tune-highest-roi)
7. [Phase 2 — STT Fine-Tune](#phase-2--stt-fine-tune)
8. [Phase 3 — TTS Fine-Tune](#phase-3--tts-fine-tune-optional)
9. [Benchmark & Evaluation](#benchmark--evaluation)
10. [Platform & Cost](#platform--cost)
11. [Plan B — Cloud Fallbacks](#plan-b--cloud-fallbacks)

---

## Overview

### Goal
Train open-source models to reach near-commercial quality for a Pakistani medical lab call center:

| Component | Model | Target Quality |
|-----------|-------|----------------|
| STT | Faster-Whisper large-v3-turbo (Urdu fine-tuned) | < 20% WER on medical Urdu |
| LLM | Qwen3-8B-Instruct (QLoRA fine-tuned) | > 90% intent accuracy |
| TTS | Edge TTS `ur-PK-UzmaNeural` | MOS > 3.8 (no training needed) |

### Key Decisions
- **All training data is generic** — no specific lab name is hardcoded
- **Distillation approach** — use GPT-4o-mini or Claude to generate synthetic training data (~$0.30–$3)
- **QLoRA** for LLM (fits on 12–16 GB VRAM, not full fine-tune)
- **TTS**: Use Edge TTS out-of-the-box — it outperforms fine-tuned Orpheus for Urdu at zero cost

---

## Directory Structure

```
finetuning/
├── README.md                          ← this file
├── PLAN.md                            ← detailed execution plan
│
├── llm/
│   ├── data/
│   │   ├── raw_conversations.jsonl    ← 101 seed conversations (generated)
│   │   ├── extend_dataset.py          ← script that added new categories
│   │   ├── train.jsonl                ← 80% split (created by prepare_dataset.py)
│   │   └── eval.jsonl                 ← 20% split
│   ├── scripts/
│   │   ├── generate_data.py           ← generate 3000+ conversations via Claude/OpenAI API
│   │   ├── prepare_dataset.py         ← validate, clean, split train/eval
│   │   └── train_qlora.py             ← QLoRA fine-tune Qwen3-8B on cloud GPU
│   └── checkpoints/                   ← saved LoRA adapters after training
│
├── stt/
│   ├── data/
│   │   ├── audio/                     ← .wav files (16kHz mono)
│   │   └── transcripts.csv            ← ground truth text (audio filename, transcript)
│   ├── scripts/
│   │   ├── prepare_dataset.py         ← format audio + transcripts for Whisper training
│   │   └── train_whisper.py           ← fine-tune Whisper large-v3
│   └── checkpoints/
│
├── tts/
│   ├── data/                          ← .wav + transcript pairs for voice cloning
│   └── scripts/
│       └── train_orpheus.py           ← Orpheus LoRA fine-tune (only if Edge TTS MOS < 3.8)
│
└── benchmark/
    ├── stt_test_set/
    │   ├── audio/                     ← 100 test .wav clips
    │   └── transcripts.csv
    ├── llm_test_set/                  ← managed inside run_llm_benchmark.py (50 prompts)
    ├── tts_test_set/
    │   └── sentences.txt              ← 20 sentences for MOS evaluation
    └── run_llm_benchmark.py           ← automated 50-prompt benchmark with pass/fail
```

---

## Current Dataset Status

### LLM Dataset (`llm/data/raw_conversations.jsonl`)

| Metric | Count |
|--------|-------|
| Total conversations | **101** |
| Multi-turn (4–8 turns) | **5** |
| Single-turn | **96** |
| Lab name references | **0** (fully generic) |

### Intent Categories Covered

| Category | Coverage |
|----------|----------|
| Test booking & appointment | ✓ |
| Test prices & rates | ✓ |
| Fasting & preparation | ✓ |
| Report status & delivery | ✓ |
| Home sample collection | ✓ |
| Branch locations & timings | ✓ |
| Multiple tests same day | ✓ |
| Urgent / STAT reports | ✓ |
| Pediatric patients | ✓ |
| Elderly / bedridden patients | ✓ |
| Corporate & bulk testing | ✓ |
| Repeat / duplicate reports | ✓ |
| Confused / unclear requests | ✓ |
| Angry / frustrated patients | ✓ |
| Diabetes panel tests | ✓ |
| Thyroid panel tests | ✓ |
| Liver function tests (LFTs) | ✓ |
| Kidney function tests (KFTs) | ✓ |
| CBC blood count tests | ✓ |
| Infectious disease (dengue, malaria, typhoid) | ✓ |
| **Multi-turn booking conversations** | ✓ NEW |
| **Sehat Sahulat / insurance queries** | ✓ NEW |
| **Ramadan fasting & test timing** | ✓ NEW |
| **Dengue season surge queries** | ✓ NEW |
| **Report portal / app tech issues** | ✓ NEW |
| **Billing disputes & refunds** | ✓ NEW |
| **Disputed / incorrect test results** | ✓ NEW |
| **Female phlebotomist requests** | ✓ NEW |
| **Punjabi-mixed Urdu queries** | ✓ NEW |
| **Complaint escalation / manager requests** | ✓ NEW |
| **Antenatal / pregnancy panel** | ✓ NEW |
| **Newborn & child health tests** | ✓ NEW |
| **Prescription abbreviation confusion** | ✓ NEW |
| **Multiple family members, single visit** | ✓ NEW |
| **Result interpretation boundary** | ✓ NEW |

> **101 conversations is seed data only.** Run `generate_data.py` to scale to 3000+ before training.

---

## Priority & ROI

```
Priority 1 — LLM fine-tune   → +20% accuracy gain, $2 compute cost
Priority 2 — STT fine-tune   → +8% WER improvement, $11 compute cost
Priority 3 — TTS fine-tune   → skip — Edge TTS (free) beats fine-tuned Orpheus
```

---

## Expected Benchmarks

| Component | Baseline (no fine-tune) | After Fine-Tune | Cloud Baseline |
|-----------|------------------------|-----------------|----------------|
| STT WER | ~26–30% | ~18–22% | Deepgram ~12% |
| LLM Intent Accuracy | ~65–75% | ~88–92% | GPT-4o ~97% |
| LLM Urdu Script % | ~55–70% | ~90–95% | GPT-4o ~99% |
| TTS MOS (Orpheus) | ~3.0–3.3 | ~3.6–3.9 | ElevenLabs ~4.5 |
| TTS MOS (Edge TTS) | ~3.8–4.0 | N/A | ElevenLabs ~4.5 |

> Edge TTS baseline (3.8–4.0 MOS) beats fine-tuned Orpheus. Use Edge TTS, skip TTS training.

---

## Phase 1 — LLM Fine-Tune (Highest ROI)

### Step 1: Generate Training Data

**Option A — Claude Sonnet (recommended, ~$3 for 3000 pairs)**
```bash
export ANTHROPIC_API_KEY=your_key_here
cd finetuning/llm/scripts
python generate_data.py --num_batches 150 --append
# 150 batches × 20 conversations = 3000 more pairs
```

**Option B — OpenAI GPT-4o-mini (~$0.30 for 3000 pairs)**
```bash
# Edit generate_data.py: change provider to openai
# model = "gpt-4o-mini"   ← best value for Urdu data generation
export OPENAI_API_KEY=your_key_here
python generate_data.py --num_batches 150 --append
```

**OpenAI Model Selection:**

| Model | Urdu Quality | Cost/1M output tokens | Use When |
|-------|-------------|----------------------|----------|
| `gpt-4o-mini` | Good | $0.60 | **Default** — bulk generation |
| `gpt-4o` | Excellent | $15 | Quality issues with mini |
| `gpt-4.1-mini` | Good | $0.40 | Cheapest option |
| `gpt-4.1` | Excellent | $8 | Alternative to gpt-4o |

### Step 2: Prepare Dataset
```bash
python prepare_dataset.py \
  --input ../data/raw_conversations.jsonl \
  --train ../data/train.jsonl \
  --eval ../data/eval.jsonl \
  --split 0.8
```

Validates every conversation:
- Checks 3-message structure (system / user / assistant)
- Verifies assistant responds in Urdu script (≥10 Arabic/Urdu chars)
- Filters corrupt JSON lines

### Step 3: Fine-Tune on Cloud GPU

**Recommended: Vast.ai RTX 4090 (~$0.25/hr)**
```bash
# Upload data to cloud instance then:
pip install transformers peft bitsandbytes trl datasets accelerate
python train_qlora.py \
  --model Qwen/Qwen3-8B-Instruct \
  --train_data ../data/train.jsonl \
  --eval_data ../data/eval.jsonl \
  --output ../checkpoints \
  --epochs 3 \
  --lora_r 64 \
  --lora_alpha 128
# Time: ~8 hrs | Cost: ~$2
```

**Training configuration:**
```
Method:       QLoRA (4-bit quantized + LoRA)
VRAM needed:  12–16 GB
LoRA rank:    64 (r=64, alpha=128)
Epochs:       3
Batch size:   4 (grad accum 8 = effective 32)
LR:           2e-4 with cosine schedule
```

### Step 4: Push to HuggingFace Hub
```python
model.push_to_hub("your-username/qwen3-8b-urdu-lab")
tokenizer.push_to_hub("your-username/qwen3-8b-urdu-lab")
```

---

## Phase 2 — STT Fine-Tune

### Data Collection

| Source | Size | How to Get |
|--------|------|-----------|
| Mozilla Common Voice 17 (Urdu) | ~50 hrs | `datasets` library, HuggingFace |
| FLEURS Urdu | ~12 hrs | `datasets` library, HuggingFace |
| OpenSLR Urdu (SLR79) | ~10 hrs | openslr.org |
| Self-recorded medical terms | 2–3 hrs | Record on any microphone |

**Self-recording script — 200 medical sentences to record:**
```
"مجھے سی بی سی کروانی ہے"
"ایچ بی اے ون سی کا کیا ریٹ ہے"
"ایل ایف ٹیز کے لیے خالی پیٹ آنا ہے"
"گھر پر سیمپل لینے کی سہولت ہے"
... (200 medical lab sentences in Urdu script)
```

### Training

**Recommended: RunPod A40 48GB (~$0.40/hr)**
```bash
cd finetuning/stt/scripts
python prepare_dataset.py --audio_dir ../data/audio --transcripts ../data/transcripts.csv
python train_whisper.py \
  --model openai/whisper-large-v3 \
  --language ur \
  --data_dir ../data/processed \
  --output ../checkpoints \
  --epochs 15
# Time: ~28 hrs | Cost: ~$11
```

### Audio Requirements
```
Format:      WAV, 16kHz, mono
Min length:  1 second
Max length:  30 seconds
Encoding:    16-bit PCM
Noise:       Clean recording, no background noise
Minimum:     10 hours total (65+ hours recommended)
```

---

## Phase 3 — TTS Fine-Tune (Optional)

> **Recommended: Skip this phase.** Edge TTS `ur-PK-UzmaNeural` (free, zero VRAM) scores MOS 3.8–4.0 which **exceeds** fine-tuned Orpheus (MOS 3.6–3.9).

**Only proceed if:**
- Human MOS evaluation scores Edge TTS below 3.5
- You need a specific cloned voice (e.g., a branded voice for the lab)

### If proceeding:
```bash
# Collect 3–5 hours of clean Urdu speech from target speaker
# 24kHz WAV, quiet room, consistent microphone distance
cd finetuning/tts/scripts
python train_orpheus.py
# Time: ~6 hrs on A40 | Cost: ~$2.40
```

---

## Benchmark & Evaluation

### Run LLM Benchmark (50 prompts, automated)
```bash
cd finetuning/benchmark

# Baseline — before fine-tuning:
python run_llm_benchmark.py --model Qwen/Qwen3-8B-Instruct

# After fine-tuning:
python run_llm_benchmark.py \
  --model ../llm/checkpoints/final \
  --base_model Qwen/Qwen3-8B-Instruct \
  --is_lora
```

**Pass thresholds:**
```
Urdu script output > 95%    → PASS
Keyword accuracy   > 80%    → Minimum pass
Keyword accuracy   > 90%    → Target pass
```

### STT Benchmark (WER)
```bash
# Build test set: 100 audio clips + ground truth transcripts
# Place in benchmark/stt_test_set/audio/ and transcripts.csv
python run_stt_benchmark.py --model_path ../stt/checkpoints/final
# Target: WER < 20%
```

### TTS Benchmark (MOS)
```
1. Generate 20 audio samples using the TTS model
2. Have 5 native Urdu speakers rate each on scale 1–5
3. Average all scores = MOS
4. Target: MOS > 3.8
```

---

## Platform & Cost

### Training Compute

| Phase | Platform | GPU | Time | Cost |
|-------|----------|-----|------|------|
| LLM data generation | Anthropic/OpenAI API | — | 2 hrs | ~$0.30–$3 |
| LLM fine-tune | Vast.ai | RTX 4090 24GB | ~8 hrs | ~$2 |
| STT fine-tune | RunPod | A40 48GB | ~28 hrs | ~$11 |
| TTS fine-tune (optional) | RunPod | A40 48GB | ~6 hrs | ~$2.40 |
| **Total** | | | **~38 hrs** | **~$15–$18** |

### RunPod Setup
```bash
# 1. Go to runpod.io → Deploy → GPU Cloud
# 2. Select A40 48GB, PyTorch 2.x template
# 3. Upload dataset via SCP or HuggingFace datasets
# 4. Run training script
# 5. Download checkpoints → terminate pod
```

### Vast.ai Setup
```bash
# 1. Go to vast.ai → Search → filter RTX 4090, PyTorch
# 2. Rent instance
# 3. SSH in: ssh -p PORT root@host
# 4. Upload data, run training
# 5. Download checkpoints → destroy instance
```

---

## Plan B — Cloud Fallbacks

If fine-tuned models do not hit benchmark targets:

| Component | Trigger | Fallback | Notes |
|-----------|---------|----------|-------|
| STT WER > 25% after fine-tune | Switch to Deepgram Nova-3 | Already in main codebase — zero dev work |
| STT WER 18–25% | Hybrid: Deepgram for medical terms | Confidence threshold routing |
| LLM < 80% accuracy | Switch to GPT-4o-mini | ~$0.01/call |
| LLM Urdu script < 85% | Add strict system prompt + retry | Prompt engineering fix |
| TTS MOS < 3.5 | Switch to Edge TTS `ur-PK-UzmaNeural` | Free, zero setup |
| TTS MOS < 4.0 | ElevenLabs with Urdu voice | Already in main codebase |

### Hybrid Routing Strategy (Recommended Production Approach)
```
Open-source handles:    ~80% of calls (general queries, FAQs, bookings)
Cloud API handles:      ~20% of calls (low-confidence STT, complex edge cases)
Result:                 80% cost reduction, no quality compromise
```

---

## Data Format Reference

### LLM Training Data (JSONL)
```json
{"messages": [
  {"role": "system", "content": "آپ ایک طبی تشخیصی لیبارٹری کے کال سینٹر میں..."},
  {"role": "user",   "content": "CBC ka rate kya hai?"},
  {"role": "assistant", "content": "سی بی سی کی قیمت ۵۰۰ روپے ہے۔"}
]}
```

For multi-turn, extend the messages array:
```json
{"messages": [
  {"role": "system",    "content": "..."},
  {"role": "user",      "content": "hello CBC karwani hai"},
  {"role": "assistant", "content": "کیا آپ خالی پیٹ ہیں؟"},
  {"role": "user",      "content": "nahi naashta kar liya"},
  {"role": "assistant", "content": "کوئی بات نہیں، سی بی سی کے لیے..."}
]}
```

### STT Training Data (CSV)
```csv
audio,sentence
audio/clip001.wav,مجھے سی بی سی کروانی ہے
audio/clip002.wav,ایچ بی اے ون سی کا کیا ریٹ ہے
```

---

## Quick Start Checklist

```
□ Step 1:  Run prepare_dataset.py on existing 101 seed conversations
□ Step 2:  Benchmark BASE Qwen3-8B (record baseline score)
□ Step 3:  Generate 3000 conversations via Claude/OpenAI API (~$3)
□ Step 4:  Fine-tune LLM on Vast.ai (~$2, ~8 hrs)
□ Step 5:  Benchmark fine-tuned LLM → compare to baseline
□ Step 6:  If LLM passes → collect STT audio data
□ Step 7:  Fine-tune STT on RunPod (~$11, ~28 hrs overnight)
□ Step 8:  Benchmark STT WER
□ Step 9:  Test Edge TTS MOS — if > 3.8 → skip TTS training
□ Step 10: Integration test full pipeline
□ Step 11: Deploy — use cloud fallback for any failing component
```
