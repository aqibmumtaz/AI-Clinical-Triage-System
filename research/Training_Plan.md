# Urdu ASR Model Training Plan

> Train custom Urdu STT models on 1000 hours of Chughtai Lab call center recordings
> Date: 2026-04-16

## Data Overview

| Metric | Value |
|--------|-------|
| Calls per day | 1,000 |
| Collection period | 30 days |
| Total calls | 30,000 |
| Avg call duration | 2 min |
| Total raw audio | **1,000 hours** |
| Estimated usable audio | **~600 hours** (after filtering) |

---

## Models to Train

| # | Model | Type | Why Train It |
|---|-------|------|-------------|
| 1 | **Whisper large-v3** | Fine-tune | Best accuracy, reference model |
| 2 | **Whisper large-v3-turbo** | Fine-tune | Best speed/accuracy tradeoff for production |
| 3 | **Faster-Whisper** | Convert (not trained) | Convert trained Whisper → CTranslate2 for pipecat |
| 4 | **Qwen2-Audio-7B-Instruct** | Fine-tune | LLM-based ASR, understands context, handles noisy audio |

---

## Phase 1: Data Preparation Pipeline

### Step 1.1 — Batch Transcribe with Deepgram Nova-3

**Input:** 30,000 raw call recordings (MP3)
**Output:** JSON transcriptions with utterance-level timestamps + confidence

```
recordings/
├── day_01/
│   ├── call_0001.mp3
│   ├── call_0002.mp3
│   └── ...
├── day_02/
└── ...
```

**Cost:** ~$258 (60,000 min × $0.0043/min)
**Time:** ~6-12 hours with async batch API

### Step 1.2 — Segment into Utterances

Split each call into utterance-level chunks using Deepgram timestamps.

**Filters applied:**
- Duration: 1s < chunk < 30s
- Confidence: > 0.50
- Remove silence-only, music, ring tones
- Remove chunks with overlapping speakers (crosstalk)

### Step 1.3 — Audio Normalization

All chunks converted to:
- **Format:** WAV (PCM 16-bit)
- **Sample rate:** 16,000 Hz (Whisper native)
- **Channels:** Mono
- **Loudness:** Normalized to -23 LUFS

### Step 1.4 — Quality Tiers

| Tier | Confidence | Human Review | Use For |
|------|-----------|-------------|---------|
| **Gold** (top 10%) | > 95% + human corrected | Yes | Evaluation set + high-quality train |
| **Silver** (middle 60%) | 70-95% | Spot-check | Main training data |
| **Bronze** (bottom 30%) | 50-70% | No | Weak supervision training |
| **Rejected** | < 50% | No | Discarded |

**Human review target:** 100-200 hours (Gold tier) = ~$200-600

### Step 1.5 — Train/Validation/Test Split

| Split | % | Hours | Purpose |
|-------|---|-------|---------|
| Train | 90% | ~540 hrs | Model training |
| Validation | 5% | ~30 hrs | Hyperparameter tuning |
| Test | 5% | ~30 hrs | Final evaluation (Gold tier only) |

**Important:** Split by call (not by utterance) to prevent data leakage. All utterances from one call go into the same split.

---

## Phase 2: Data Formats Per Model

### Format A — Whisper (large-v3 & large-v3-turbo)

HuggingFace Datasets format with `metadata.jsonl`:

```jsonl
{"audio": "audio/chunk_0001.wav", "sentence": "السلام علیکم"}
{"audio": "audio/chunk_0002.wav", "sentence": "ڈاکٹر صاحب سے بات کرنی ہے"}
```

**Directory structure:**
```
whisper_dataset/
├── train/
│   ├── audio/
│   │   ├── chunk_000001.wav
│   │   └── ...
│   └── metadata.jsonl
├── validation/
│   ├── audio/
│   │   └── ...
│   └── metadata.jsonl
└── test/
    ├── audio/
    │   └── ...
    └── metadata.jsonl
```

### Format B — Qwen2-Audio-7B

Chat-style JSON with audio paths (HuggingFace SFT format):

```jsonl
{
  "messages": [
    {
      "role": "system",
      "content": "You are a speech recognition assistant. Transcribe the given Urdu audio accurately in Urdu script."
    },
    {
      "role": "user",
      "content": [
        {"type": "audio", "audio_url": "audio/chunk_0001.wav"}
      ]
    },
    {
      "role": "assistant",
      "content": "السلام علیکم ڈاکٹر صاحب سے بات کرنی ہے"
    }
  ]
}
```

**Directory structure:**
```
qwen_dataset/
├── train.jsonl
├── validation.jsonl
├── test.jsonl
└── audio/
    ├── chunk_000001.wav
    └── ...
```

### Format C — Faster-Whisper (CTranslate2 Conversion)

No separate training data. Convert the trained Whisper checkpoint:

```bash
ct2-opus-converter --model trained_whisper_model --output_dir faster_whisper_model --quantization float16
```

---

## Phase 3: Training Configuration

### Model 1 — Whisper large-v3

| Parameter | Value |
|-----------|-------|
| Base model | `openai/whisper-large-v3` |
| Parameters | 1.55B |
| GPU | 1× A100 80GB |
| VRAM usage | ~40 GB (FP16 + gradient checkpointing) |
| Batch size | 16 (per device) |
| Gradient accumulation | 2 (effective batch 32) |
| Learning rate | 1e-5 (with warmup) |
| Warmup steps | 500 |
| Epochs | 3-5 |
| Max audio length | 30s |
| Training time | ~5-7 days on 1× A100 |
| FP16 | Yes |
| Gradient checkpointing | Yes |
| Eval strategy | Every 1000 steps |
| Save strategy | Every 1000 steps, keep best 3 |
| Metric | WER (Word Error Rate) |

**Key training arguments:**
```python
training_args = Seq2SeqTrainingArguments(
    output_dir="./whisper-large-v3-urdu-chughtai",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=2,
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=50000,      # ~3 epochs over 600 hrs
    gradient_checkpointing=True,
    fp16=True,
    eval_strategy="steps",
    eval_steps=1000,
    save_steps=1000,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    logging_steps=25,
    dataloader_num_workers=8,
    predict_with_generate=True,
    generation_max_length=225,
    report_to=["tensorboard"],
)
```

### Model 2 — Whisper large-v3-turbo

| Parameter | Value |
|-----------|-------|
| Base model | `openai/whisper-large-v3-turbo` |
| Parameters | 809M |
| GPU | 1× A100 80GB (or RTX 4090 24GB with grad checkpointing) |
| VRAM usage | ~20 GB (FP16) |
| Batch size | 32 (per device) |
| Gradient accumulation | 1 (effective batch 32) |
| Learning rate | 1e-5 |
| Warmup steps | 500 |
| Epochs | 3-5 |
| Training time | ~3-4 days on 1× A100 |

Same training script as large-v3, just change `model_name_or_path`.

### Model 3 — Faster-Whisper (Conversion, not training)

After training Whisper large-v3-turbo:

```bash
# Install
pip install ctranslate2[cuda]

# Convert to CTranslate2 (Faster-Whisper format)
ct2-opus-converter \
    --model ./whisper-large-v3-turbo-urdu-chughtai/best_checkpoint \
    --output_dir ./faster-whisper-urdu-chughtai \
    --quantization float16    # or int8_float16 for smaller/faster

# Test
from faster_whisper import WhisperModel
model = WhisperModel("./faster-whisper-urdu-chughtai", device="cuda", compute_type="float16")
segments, info = model.transcribe("test.wav", language="ur")
```

**Quantization options:**
| Type | Size | Speed | Accuracy Loss |
|------|------|-------|--------------|
| float16 | 100% | 2x vs PyTorch | None |
| int8_float16 | ~50% | 3x vs PyTorch | Minimal |
| int8 | ~50% | 4x vs PyTorch | Slight |

### Model 4 — Qwen2-Audio-7B-Instruct

| Parameter | Value |
|-----------|-------|
| Base model | `Qwen/Qwen2-Audio-7B-Instruct` |
| Parameters | 7B (audio encoder + LLM) |
| GPU | 1× A100 80GB (or 2× RTX 4090 with DeepSpeed) |
| VRAM usage | ~50 GB (FP16 + LoRA) |
| Fine-tuning method | **LoRA** (rank 64, alpha 128) |
| Batch size | 4 (per device) |
| Gradient accumulation | 8 (effective batch 32) |
| Learning rate | 2e-5 |
| Warmup ratio | 0.03 |
| Epochs | 2-3 |
| Training time | ~7-10 days on 1× A100 |
| Framework | **LLaMA-Factory** or **Unsloth** |

**Why Qwen2-Audio over Whisper?**
- Understands conversational context (not just audio → text)
- Can handle noisy call center audio with LLM reasoning
- Can be prompted: "Transcribe this Urdu medical call center audio"
- Handles code-switching (Urdu + English medical terms) natively
- Single model for ASR + understanding

**LoRA config:**
```python
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)
```

---

## Phase 4: Training Schedule

| Week | Activity | Output |
|------|----------|--------|
| **Week 1** | Batch transcribe 30K calls with Deepgram | Raw transcriptions |
| **Week 1-2** | Segment, filter, normalize audio | Clean dataset (~600 hrs) |
| **Week 2** | Human review Gold tier (100 hrs) | Corrected test/eval set |
| **Week 2-3** | Train Whisper large-v3-turbo (fastest) | First model checkpoint |
| **Week 3** | Evaluate, tune hyperparams | WER baseline |
| **Week 3-4** | Train Whisper large-v3 (best accuracy) | Best accuracy model |
| **Week 4** | Convert to Faster-Whisper (CTranslate2) | Production-ready model |
| **Week 4-5** | Train Qwen2-Audio-7B (LoRA) | LLM-based ASR model |
| **Week 5** | Benchmark all models, select best | Final model selection |

---

## Phase 5: Evaluation

### Metrics

| Metric | What It Measures |
|--------|-----------------|
| **WER** (Word Error Rate) | Overall transcription accuracy |
| **CER** (Character Error Rate) | Character-level accuracy (better for Urdu script) |
| **RTF** (Real-Time Factor) | Speed: RTF < 1 means faster than real-time |
| **TTFA** (Time to First Audio) | Latency before first transcription appears |

### Benchmark Against

| Model | Type | Baseline |
|-------|------|----------|
| Deepgram Nova-3 (current) | Paid API | ~95% confidence (our recordings) |
| Whisper large-v3 (base, no fine-tune) | Open source | ~30-35% WER Urdu |
| kingabzpro/whisper-large-v3-turbo-urdu | Community fine-tune | ~26% WER |
| **Your fine-tuned model** | Custom | **Target: 10-15% WER** |

### Test Protocol

1. Reserve 30 hours of Gold-tier (human-corrected) data as test set
2. Run each model on test set
3. Compute WER, CER, RTF
4. Test on edge cases: noisy audio, fast speech, medical terms, numbers, names
5. A/B test in live pipecat pipeline

---

## Phase 6: Production Deployment

### Recommended Production Model

```
Whisper large-v3-turbo (fine-tuned) → Convert to Faster-Whisper (int8_float16)
```

**Integration with pipecat (zero code change):**
```python
from pipecat.services.whisper.stt import WhisperSTTService

stt = WhisperSTTService(
    model="./faster-whisper-urdu-chughtai",    # local path
    device="cuda",
    compute_type="int8_float16",
    language=Language.UR,
)
```

### Fallback Strategy

```
Primary:   Faster-Whisper (local, fine-tuned)
Fallback:  Deepgram Nova-3 (API, if local fails)
```

---

## Cost Summary

| Item | Cost |
|------|------|
| Deepgram transcription (30K calls) | ~$258 |
| Human review (100-200 hrs) | ~$200-600 |
| GPU training — A100 80GB × 3 weeks | ~$600-1,200 |
| **Total** | **~$1,058 — $2,058** |

---

## Hardware Options

| Option | Spec | Monthly Cost | Training Speed |
|--------|------|-------------|---------------|
| **Lambda Labs 1× A100 80GB** | 80 GB VRAM, 30 vCPUs | ~$930/mo | Baseline |
| **RunPod 1× A100 80GB** | On-demand | ~$1.29/hr = ~$930/mo | Same |
| **Vast.ai 1× A100 80GB** | Community | ~$0.90/hr = ~$650/mo | Same |
| **RunPod 1× RTX 4090** | 24 GB VRAM | ~$0.40/hr = ~$290/mo | 2x slower (turbo only) |
| **Google Colab Pro+** | 1× A100 40GB | $50/mo | Limited hours |

---

## Files to Create

| Script | Purpose |
|--------|---------|
| `batch_transcribe.py` | Transcribe 30K calls via Deepgram batch API |
| `prepare_training_data.py` | Segment, filter, normalize → training format |
| `prepare_qwen_data.py` | Convert to Qwen2-Audio chat format |
| `train_whisper.py` | Fine-tune Whisper large-v3 / turbo |
| `convert_faster_whisper.py` | Convert to CTranslate2 |
| `train_qwen_audio.py` | Fine-tune Qwen2-Audio with LoRA |
| `evaluate.py` | Benchmark all models on test set |
| `apply_corrections.py` | Apply human corrections from review.tsv |
