"""
Prepare training data for Whisper fine-tuning.
Takes batch transcription output → segments into utterances → creates HuggingFace dataset.

Usage:
  python prepare_whisper_data.py \
      --transcriptions_dir ../input/recordings/transcriptions \
      --output_dir ../finetuning/whisper_dataset \
      --train_ratio 0.90 --val_ratio 0.05 --test_ratio 0.05

Output:
  whisper_dataset/
  ├── train/
  │   ├── audio/chunk_000001.wav
  │   └── metadata.jsonl
  ├── validation/
  │   ├── audio/...
  │   └── metadata.jsonl
  └── test/
      ├── audio/...
      └── metadata.jsonl
"""

import os, sys, json, subprocess, argparse, hashlib, random, pathlib, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SAMPLE_RATE = 16000
MIN_DURATION = 1.0
MAX_DURATION = 30.0
MIN_CONFIDENCE = 0.50
TARGET_LUFS = -23


def split_audio(input_path: str, output_path: str, start: float, end: float):
    duration = end - start
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ss", str(start), "-t", str(duration),
        "-ar", str(SAMPLE_RATE), "-ac", "1",
        "-af", f"loudnorm=I={TARGET_LUFS}:TP=-1.5:LRA=11",
        "-f", "wav", output_path,
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def call_id_from_path(filepath: str) -> str:
    return hashlib.md5(filepath.encode()).hexdigest()[:12]


def load_transcriptions(transcriptions_dir: str) -> list[dict]:
    all_results = []
    for fname in sorted(os.listdir(transcriptions_dir)):
        if not fname.endswith(".json") or fname == "progress.json":
            continue
        fpath = os.path.join(transcriptions_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                all_results.extend(data)
            else:
                all_results.append(data)
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Prepare Whisper training data")
    parser.add_argument("--transcriptions_dir", required=True)
    parser.add_argument("--output_dir", default="whisper_dataset")
    parser.add_argument("--train_ratio", type=float, default=0.90)
    parser.add_argument("--val_ratio", type=float, default=0.05)
    parser.add_argument("--test_ratio", type=float, default=0.05)
    parser.add_argument("--min_confidence", type=float, default=MIN_CONFIDENCE)
    parser.add_argument("--min_duration", type=float, default=MIN_DURATION)
    parser.add_argument("--max_duration", type=float, default=MAX_DURATION)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    transcriptions = load_transcriptions(args.transcriptions_dir)
    log.info(f"Loaded {len(transcriptions)} transcription files")

    call_ids = list({call_id_from_path(t["file"]) for t in transcriptions if "error" not in t})
    random.shuffle(call_ids)

    n = len(call_ids)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)

    train_ids = set(call_ids[:n_train])
    val_ids = set(call_ids[n_train:n_train + n_val])
    test_ids = set(call_ids[n_train + n_val:])

    log.info(f"Split: {len(train_ids)} train / {len(val_ids)} val / {len(test_ids)} test calls")

    splits = {"train": train_ids, "validation": val_ids, "test": test_ids}
    stats = {s: {"chunks": 0, "duration": 0, "skipped": 0} for s in splits}

    for split_name in splits:
        os.makedirs(os.path.join(args.output_dir, split_name, "audio"), exist_ok=True)

    chunk_counter = 0
    metadata_files = {s: open(os.path.join(args.output_dir, s, "metadata.jsonl"), "w", encoding="utf-8") for s in splits}

    try:
        for entry in transcriptions:
            if "error" in entry:
                continue

            filepath = entry["file"]
            cid = call_id_from_path(filepath)

            split_name = None
            for s, ids in splits.items():
                if cid in ids:
                    split_name = s
                    break
            if split_name is None:
                continue

            if not os.path.exists(filepath):
                log.warning(f"Audio not found: {filepath}")
                continue

            utterances = entry.get("utterances", [])
            if not utterances:
                fr = entry.get("full_response", {}).get("results", {})
                utterances = fr.get("utterances", [])

            for utt in utterances:
                start = utt["start"]
                end = utt["end"]
                text = utt["transcript"].strip()
                confidence = utt["confidence"]
                duration = end - start

                if not text or duration < args.min_duration or duration > args.max_duration:
                    stats[split_name]["skipped"] += 1
                    continue
                if confidence < args.min_confidence:
                    stats[split_name]["skipped"] += 1
                    continue

                chunk_counter += 1
                chunk_filename = f"chunk_{chunk_counter:06d}.wav"
                chunk_path = os.path.join(args.output_dir, split_name, "audio", chunk_filename)

                try:
                    split_audio(filepath, chunk_path, start, end)
                except Exception as e:
                    log.warning(f"ffmpeg error: {e}")
                    stats[split_name]["skipped"] += 1
                    continue

                row = json.dumps(
                    {"audio": f"audio/{chunk_filename}", "sentence": text},
                    ensure_ascii=False,
                )
                metadata_files[split_name].write(row + "\n")

                stats[split_name]["chunks"] += 1
                stats[split_name]["duration"] += duration

                if chunk_counter % 1000 == 0:
                    log.info(f"Processed {chunk_counter} chunks...")

    finally:
        for f in metadata_files.values():
            f.close()

    log.info("=" * 60)
    log.info("WHISPER DATASET PREPARATION COMPLETE")
    log.info("=" * 60)
    for s in splits:
        hrs = stats[s]["duration"] / 3600
        log.info(f"  {s:12s}: {stats[s]['chunks']:6d} chunks, {hrs:.1f} hrs, {stats[s]['skipped']} skipped")
    log.info(f"  Total chunks: {chunk_counter}")
    log.info(f"  Output: {args.output_dir}")


if __name__ == "__main__":
    main()
