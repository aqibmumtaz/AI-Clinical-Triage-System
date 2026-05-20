# Urdu ASR Training Plan — Iterative Self-Training

> Train custom Whisper + Qwen2-Audio models on 1000 hours of Chughtai Lab call center recordings
> Using iterative self-training to maximize quality while minimizing human correction cost

---

## Data Source

| Metric | Value |
|--------|-------|
| Calls per day | 1,000 |
| Collection period | 30 days |
| Total calls | 30,000 |
| Avg call duration | 2 min |
| Total raw audio | 1,000 hours |
| Estimated usable (after filtering) | ~600-700 hours |

---

## Phase 1 — Raw Data Collection & Auto-Transcription

**Duration:** Week 1
**Cost:** ~$258

### Step 1.1 — Collect & Organize Recordings

```
training/input/recordings/
├── day_01/   # 1000 calls
├── day_02/   # 1000 calls
└── ...       # 30 days
```

- Audio format: MP3 (telephone quality)
- Organize by date folder
- Log metadata: call ID, timestamp, duration, caller number (hashed)

### Step 1.2 — Batch Transcribe with Deepgram Nova-3

**Script:** `training/scripts/batch_transcribe.py`

```bash
python training/scripts/batch_transcribe.py \
    --input_dir training/input/recordings \
    --output_dir training/data/transcriptions \
    --max_concurrent 20 --resume
```

- 60,000 minutes × $0.0043/min = ~$258
- ~6-12 hours with 20 concurrent API calls
- Output: JSON with utterance timestamps + confidence scores
- Resume capability if interrupted

### Step 1.3 — Segment into Utterance Chunks

**Script:** `training/scripts/prepare_training_data.py`

Split each call into utterance-level WAV chunks:
- Format: 16kHz, mono, PCM WAV
- Loudness normalized to -23 LUFS
- Duration: 1s — 30s per chunk
- Filter: drop confidence < 50%

**Expected output:** ~200,000 — 300,000 chunks

---

## Phase 2 — Data Quality Tiers & Filtering

**Duration:** Week 1-2

### Step 2.1 — Tier by Confidence

| Tier | Confidence Range | Estimated Volume | Use |
|------|-----------------|------------------|-----|
| **Gold candidate** | > 95% | ~150 hrs | Human review → Gold set |
| **Silver** | 80% — 95% | ~300 hrs | Main training data |
| **Bronze** | 60% — 80% | ~150 hrs | Weak supervision (Round 1 only) |
| **Rejected** | < 60% | ~100 hrs | Discarded |

### Step 2.2 — Domain Vocabulary Check

Flag utterances missing expected Chughtai Lab vocabulary:

**Expected keywords (should appear frequently):**
```
چغتائی، لیب، رپورٹ، ٹیسٹ، ڈاکٹر، پیشنٹ، ایپلیکیشن، ڈاؤن لوڈ،
کنیکٹ، ٹرانسفر، نمبر، کارڈ، بکنگ، ہوم سیمپل، ایج، ایڈریس
```

If these words are absent where expected → flag for review.

### Step 2.3 — LLM Coherence Check (Automated)

Use Qwen3-8B or GPT-4o-mini to score each utterance:

```
Prompt: "Rate this Urdu transcription 1-5 for grammatical and semantic coherence:
         '{utterance_text}'
         Reply with just the number."
```

- Score 1-2 → flag for human review
- Score 3-5 → pass

**Cost:** ~$5-10 for 300K utterances via local Qwen3-8B (free) or GPT-4o-mini API

### Step 2.4 — Split by Call (Not Utterance)

Prevent data leakage — all utterances from one call go to same split:

| Split | % of Calls | Purpose |
|-------|-----------|---------|
| Train | 85% | Model training |
| Validation | 5% | Hyperparameter tuning |
| Test-Auto | 5% | Auto-labeled test (quick checks) |
| Test-Gold | 5% | Human-corrected test (final eval) |

**Important:** Test-Gold split is ONLY from Gold tier, ONLY human-corrected.

---

## Phase 3 — Human Correction (Gold Set)

**Duration:** Week 2-3
**Cost:** ~$400-750

### What to Correct

| Priority | Source | Volume | Why |
|----------|--------|--------|-----|
| 1 | Test-Gold split (all) | ~30 hrs | Final evaluation MUST be correct |
| 2 | LLM flagged (coherence 1-2) | ~20 hrs | Worst errors, highest impact |
| 3 | Confidence 60-80% with domain keywords missing | ~20 hrs | Domain-specific errors |
| 4 | Random sample from Silver tier | ~30 hrs | Verify quality assumption |
| **Total** | | **~100 hrs** | |

### How to Correct

1. Transcriber listens to audio chunk
2. Types correct Urdu text in `corrected` column of review.tsv
3. Mark unclear/unusable chunks as `[SKIP]`
4. Tool: Any audio player + spreadsheet editor

### Hiring Transcribers

| Option | Cost | Speed |
|--------|------|-------|
| Local Urdu speakers (Fiverr/Upwork) | $1-2/hr of audio | 1-2 weeks |
| In-house team (if available) | Internal cost | Fastest |
| Specialized ASR data company | $2-5/hr of audio | 1 week |

### Apply Corrections

**Script:** `training/scripts/apply_corrections.py`

```bash
python training/scripts/apply_corrections.py \
    --review training/data/review.tsv \
    --metadata training/data/metadata.jsonl
```

---

## Phase 4 — Round 1 Training (Noisy Labels)

**Duration:** Week 3-4
**Cost:** ~$150-250 (GPU rental)

### Step 4.1 — Prepare Whisper Dataset

**Script:** `training/scripts/prepare_whisper_data.py`

```bash
python training/scripts/prepare_whisper_data.py \
    --transcriptions_dir training/data/transcriptions \
    --output_dir training/data/whisper_dataset_v1 \
    --min_confidence 0.60
```

Include Gold + Silver + Bronze tiers (~600 hrs).

### Step 4.2 — Train Whisper large-v3-turbo (Round 1)

**Script:** `training/scripts/train_whisper.py`

```bash
python training/scripts/train_whisper.py \
    --model openai/whisper-large-v3-turbo \
    --dataset_dir training/data/whisper_dataset_v1 \
    --output_dir training/models/whisper-turbo-v1 \
    --batch_size 32 --epochs 3 --learning_rate 1e-5
```

| Parameter | Value |
|-----------|-------|
| Base model | openai/whisper-large-v3-turbo (809M params) |
| GPU | 1× A100 80GB |
| VRAM | ~20 GB |
| Training time | ~3-4 days |
| Data | ~600 hrs (noisy labels) |

### Step 4.3 — Evaluate Round 1

```bash
python training/scripts/evaluate.py \
    --test_dir training/data/whisper_dataset_v1/test \
    --models \
        deepgram:api \
        whisper-base:openai/whisper-large-v3-turbo \
        whisper-v1:training/models/whisper-turbo-v1/best
```

**Expected results:**

| Model | Expected WER |
|-------|-------------|
| Deepgram Nova-3 | ~12-15% (on this domain) |
| Whisper turbo (base, no fine-tune) | ~30-35% |
| **Whisper turbo v1 (your fine-tune)** | **~20-25%** |

---

## Phase 5 — Iterative Label Cleaning

**Duration:** Week 4
**Cost:** ~$50-100 (GPU time only)

### Step 5.1 — Re-transcribe ALL Audio with Whisper v1

```bash
python training/scripts/retranscribe.py \
    --model training/models/whisper-turbo-v1/best \
    --input_dir training/input/recordings \
    --output_dir training/data/whisper_v1_transcriptions
```

Now you have TWO transcriptions for every audio chunk:
1. **Deepgram labels** (original)
2. **Whisper v1 labels** (your model)

### Step 5.2 — Agreement Filter

Compare both transcriptions for each chunk:

```
CER = character_error_rate(deepgram_text, whisper_v1_text)

if CER < 0.10:   → HIGH confidence (both agree)     → keep
if CER 0.10-0.30: → MEDIUM (minor diff)              → use Whisper v1 label
if CER > 0.30:   → DISAGREEMENT                      → discard or human review
```

**Expected outcome:**

| Agreement Level | Volume | Action |
|----------------|--------|--------|
| Both agree (CER < 10%) | ~400 hrs | High-confidence training data |
| Minor differences (CER 10-30%) | ~150 hrs | Use Whisper v1 label |
| Major disagreement (CER > 30%) | ~50 hrs | Discard or human review |

### Step 5.3 — Build Cleaned Dataset v2

- Agreement chunks → ~400 hrs (best labels)
- Whisper v1 preferred → ~150 hrs (good labels)
- Human-corrected Gold set → ~100 hrs (perfect labels)
- **Total: ~650 hrs of improved labels**

---

## Phase 6 — Round 2 Training (Clean Labels)

**Duration:** Week 4-5
**Cost:** ~$300-500 (GPU rental)

### Step 6.1 — Train Whisper large-v3-turbo v2

```bash
python training/scripts/train_whisper.py \
    --model openai/whisper-large-v3-turbo \
    --dataset_dir training/data/whisper_dataset_v2 \
    --output_dir training/models/whisper-turbo-v2 \
    --batch_size 32 --epochs 3 --learning_rate 5e-6
```

Lower learning rate (5e-6) — labels are cleaner, don't need aggressive learning.

### Step 6.2 — Train Whisper large-v3 v2 (Best Accuracy)

```bash
python training/scripts/train_whisper.py \
    --model openai/whisper-large-v3 \
    --dataset_dir training/data/whisper_dataset_v2 \
    --output_dir training/models/whisper-large-v2 \
    --batch_size 16 --epochs 3 --learning_rate 5e-6
```

| Parameter | Whisper Turbo v2 | Whisper Large v2 |
|-----------|-----------------|-----------------|
| Params | 809M | 1.55B |
| VRAM | ~20 GB | ~40 GB |
| Training | ~3-4 days | ~5-7 days |
| Expected WER | **14-18%** | **10-15%** |

### Step 6.3 — Train Qwen2-Audio-7B (LoRA)

```bash
python training/scripts/prepare_qwen_data.py \
    --whisper_dir training/data/whisper_dataset_v2 \
    --output_dir training/data/qwen_dataset_v2

python training/scripts/train_qwen_audio.py \
    --dataset_dir training/data/qwen_dataset_v2 \
    --output_dir training/models/qwen2-audio-urdu \
    --lora_rank 64 --epochs 2 --load_in_4bit
```

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen2-Audio-7B-Instruct |
| Method | LoRA (rank 64) + QLoRA (4-bit) |
| VRAM | ~24 GB (fits RTX 4090) |
| Training | ~7-10 days |
| Expected WER | **12-16%** |

### Step 6.4 — Convert to Faster-Whisper

```bash
python training/scripts/convert_faster_whisper.py \
    --model_dir training/models/whisper-turbo-v2/best \
    --output_dir training/models/faster-whisper-urdu \
    --quantization int8_float16
```

---

## Phase 7 — Final Evaluation & Selection

**Duration:** Week 5

### Benchmark All Models

```bash
python training/scripts/evaluate.py \
    --test_dir training/data/whisper_dataset_v2/test \
    --models \
        deepgram:api \
        whisper-turbo-v1:training/models/whisper-turbo-v1/best \
        whisper-turbo-v2:training/models/whisper-turbo-v2/best \
        whisper-large-v2:training/models/whisper-large-v2/best \
        faster-whisper:training/models/faster-whisper-urdu
```

### Expected Final Results

| Model | WER | RTF | Latency | Self-Hosted |
|-------|-----|-----|---------|-------------|
| Deepgram Nova-3 (current paid) | ~12-15% | API | ~500ms | No |
| Whisper turbo v1 (Round 1, noisy) | ~20-25% | 0.05 | ~200ms | Yes |
| **Whisper turbo v2 (Round 2, clean)** | **~14-18%** | **0.05** | **~200ms** | **Yes** |
| **Whisper large v2 (Round 2, clean)** | **~10-15%** | **0.15** | **~500ms** | **Yes** |
| **Faster-Whisper (production)** | **~14-18%** | **0.02** | **~100ms** | **Yes** |
| Qwen2-Audio-7B | ~12-16% | 0.30 | ~800ms | Yes |

### Edge Case Test Checklist

- [ ] Medical terminology (ٹیسٹ، رپورٹ، ڈاکٹر، پیشنٹ)
- [ ] Chughtai-specific vocabulary (چغتائی لیب، ہوم سیمپل، بکنگ)
- [ ] Numbers and amounts (روپے، نمبر، ایج)
- [ ] Names (patient names, doctor names, addresses)
- [ ] Fast speech / overlapping speakers
- [ ] Noisy phone connections
- [ ] Code-switching (Urdu + English mixed)

### Production Deployment

**Primary:** Faster-Whisper (from turbo v2) → pipecat built-in integration
**Fallback:** Deepgram Nova-3 API (if local model fails)

---

## Phase 8 — Optional Round 3+ (Diminishing Returns)

Repeat Phase 5-6 if WER target not met:
- Re-transcribe with v2 model
- 3-way agreement filter (Deepgram + v1 + v2)
- Train v3 on triple-agreement data

Each round improves WER by ~2-5%. Stop when improvement < 1%.

---

## Full Timeline

| Week | Phase | Activity | Output |
|------|-------|----------|--------|
| 1 | 1 | Collect + Deepgram batch transcribe | 30K transcriptions |
| 1-2 | 2 | Segment, filter, tier, LLM coherence check | ~600 hrs clean chunks |
| 2-3 | 3 | Human correct 100 hrs (Gold set) | Corrected test/eval set |
| 3-4 | 4 | Train Whisper turbo v1 (noisy labels) | First model (~22% WER) |
| 4 | 5 | Re-transcribe + agreement filter | Cleaned dataset v2 |
| 4-5 | 6 | Train turbo v2, large v2, Qwen2-Audio | Final models (~14% WER) |
| 5 | 7 | Benchmark, select, deploy | Production model |

---

## Cost Summary

| Item | Cost |
|------|------|
| Deepgram batch transcription (30K calls) | $258 |
| Human correction (100 hrs × $2-5/hr) | $200-500 |
| GPU rental — Round 1 (A100 × 4 days) | $125-200 |
| GPU rental — Round 2 (A100 × 10 days) | $300-500 |
| LLM coherence check (optional) | $5-10 |
| **Total** | **$888 — $1,468** |

---

## Hardware Options

| Option | Spec | Cost/hr | Monthly |
|--------|------|---------|---------|
| Lambda Labs 1× A100 80GB | Best value dedicated | $1.29/hr | ~$930/mo |
| RunPod 1× A100 80GB | On-demand | $1.29/hr | ~$930/mo |
| Vast.ai 1× A100 80GB | Community (cheapest) | ~$0.90/hr | ~$650/mo |
| RunPod 1× RTX 4090 | Budget (turbo only) | ~$0.40/hr | ~$290/mo |

---

## Scripts Reference

| Script | Phase | Purpose |
|--------|-------|---------|
| `batch_transcribe.py` | 1 | Transcribe 30K calls via Deepgram |
| `prepare_training_data.py` | 2 | Segment into utterance chunks |
| `prepare_whisper_data.py` | 2 | Create train/val/test splits |
| `apply_corrections.py` | 3 | Apply human corrections from review.tsv |
| `train_whisper.py` | 4, 6 | Fine-tune Whisper models |
| `convert_faster_whisper.py` | 6 | Convert to CTranslate2 / Faster-Whisper |
| `prepare_qwen_data.py` | 6 | Convert to Qwen2-Audio chat format |
| `train_qwen_audio.py` | 6 | Fine-tune Qwen2-Audio with LoRA |
| `evaluate.py` | 7 | Benchmark all models on test set |
| `retranscribe.py` | 5 | Re-transcribe with trained model (to create) |
| `agreement_filter.py` | 5 | Compare labels, build cleaned dataset (to create) |

---

## Key Principles

1. **Volume beats perfection** — 600 hrs of 90% correct > 50 hrs of 100% correct
2. **Iterate** — Each training round produces better labels for the next
3. **Agreement = confidence** — When two models agree, the label is almost certainly correct
4. **Human effort on disagreements** — Only pay humans to fix what machines can't agree on
5. **Test set must be perfect** — Never compromise on Gold test set quality
6. **Split by call, not utterance** — Prevents data leakage between train/test
7. **Domain matters** — A model trained on Chughtai calls will beat any general Urdu model on Chughtai calls
