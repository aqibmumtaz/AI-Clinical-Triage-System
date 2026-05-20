"""
Prepare Urdu ASR training data from call recordings.
Splits long recordings into utterance-level chunks + metadata.jsonl for Whisper fine-tuning.

Usage:
  python prepare_training_data.py [recordings_folder]
  Default: input/recordings/3551

Output structure:
  training_data/
  ├── audio/           # WAV chunks (16kHz mono)
  ├── metadata.jsonl   # {"audio": "audio/chunk_0001.wav", "sentence": "..."}
  └── review.tsv       # For manual review/correction
"""

import os, sys, json, subprocess, pathlib

RECORDINGS_DIR = os.path.join(os.path.dirname(__file__), "input", "recordings", "3551")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "training_data")
SAMPLE_RATE = 16000
MIN_DURATION = 1.0     # skip chunks shorter than 1s
MAX_DURATION = 30.0    # skip chunks longer than 30s
MIN_CONFIDENCE = 0.50  # skip low-confidence utterances


def split_audio(input_path: str, output_path: str, start: float, end: float):
    """Extract a segment from audio file using ffmpeg, output 16kHz mono WAV."""
    duration = end - start
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ss", str(start), "-t", str(duration),
        "-ar", str(SAMPLE_RATE), "-ac", "1", "-f", "wav",
        output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)


def main():
    recordings_dir = sys.argv[1] if len(sys.argv) > 1 else RECORDINGS_DIR
    recordings_dir = os.path.abspath(recordings_dir)

    transcriptions_path = os.path.join(recordings_dir, "transcriptions.json")
    if not os.path.exists(transcriptions_path):
        print(f"transcriptions.json not found in {recordings_dir}")
        print("Run transcribe_recordings.py first.")
        sys.exit(1)

    with open(transcriptions_path, "r", encoding="utf-8") as f:
        transcriptions = json.load(f)

    audio_dir = os.path.join(OUTPUT_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    metadata = []
    skipped = {"short": 0, "long": 0, "low_conf": 0, "empty": 0}
    chunk_id = 0

    for entry in transcriptions:
        if "error" in entry:
            print(f"Skipping {entry['file']} (error)")
            continue

        filepath = os.path.join(recordings_dir, entry["file"])
        if not os.path.exists(filepath):
            print(f"Audio file not found: {filepath}")
            continue

        utterances = entry.get("full_response", {}).get("results", {}).get("utterances", [])
        if not utterances:
            print(f"No utterances in {entry['file']}")
            continue

        print(f"\nProcessing: {entry['file']}")
        print(f"  Utterances: {len(utterances)}")

        for utt in utterances:
            start = utt["start"]
            end = utt["end"]
            text = utt["transcript"].strip()
            confidence = utt["confidence"]
            duration = end - start

            if not text:
                skipped["empty"] += 1
                continue
            if duration < MIN_DURATION:
                skipped["short"] += 1
                continue
            if duration > MAX_DURATION:
                skipped["long"] += 1
                continue
            if confidence < MIN_CONFIDENCE:
                skipped["low_conf"] += 1
                continue

            chunk_id += 1
            chunk_filename = f"chunk_{chunk_id:04d}.wav"
            chunk_path = os.path.join(audio_dir, chunk_filename)

            try:
                split_audio(filepath, chunk_path, start, end)
            except subprocess.CalledProcessError as e:
                print(f"  ffmpeg error on chunk {chunk_id}: {e}")
                continue

            metadata.append({
                "audio": f"audio/{chunk_filename}",
                "sentence": text,
                "duration": round(duration, 2),
                "confidence": round(confidence, 4),
                "source_file": entry["file"],
                "start": start,
                "end": end,
            })

            print(f"  [{chunk_id:04d}] {start:.1f}-{end:.1f}s ({confidence:.0%}) {text[:60]}...")

    # --- Write metadata.jsonl (HuggingFace Whisper format) ---
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.jsonl")
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in metadata:
            row = {"audio": item["audio"], "sentence": item["sentence"]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # --- Write review.tsv (for manual correction) ---
    review_path = os.path.join(OUTPUT_DIR, "review.tsv")
    with open(review_path, "w", encoding="utf-8") as f:
        f.write("chunk_file\tduration\tconfidence\tsentence\tcorrected\n")
        for item in metadata:
            f.write(f"{item['audio']}\t{item['duration']}\t{item['confidence']}\t{item['sentence']}\t\n")

    # --- Write full metadata with source info ---
    full_metadata_path = os.path.join(OUTPUT_DIR, "metadata_full.json")
    with open(full_metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # --- Summary ---
    total_duration = sum(item["duration"] for item in metadata)
    avg_confidence = sum(item["confidence"] for item in metadata) / len(metadata) if metadata else 0

    print("\n" + "=" * 60)
    print("TRAINING DATA SUMMARY")
    print("=" * 60)
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"  Total chunks:     {len(metadata)}")
    print(f"  Total duration:   {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"  Avg confidence:   {avg_confidence:.2%}")
    print(f"  Avg chunk length: {total_duration/len(metadata):.1f}s" if metadata else "")
    print(f"\n  Skipped:")
    print(f"    Too short (<{MIN_DURATION}s): {skipped['short']}")
    print(f"    Too long (>{MAX_DURATION}s):  {skipped['long']}")
    print(f"    Low confidence (<{MIN_CONFIDENCE}): {skipped['low_conf']}")
    print(f"    Empty text:     {skipped['empty']}")
    print(f"\n  Files created:")
    print(f"    {metadata_path}")
    print(f"    {review_path}")
    print(f"    {full_metadata_path}")
    print(f"    {len(metadata)} WAV files in {audio_dir}/")
    print("\n  NEXT STEP:")
    print("    Open review.tsv and fill the 'corrected' column")
    print("    for any utterances where Deepgram got it wrong.")
    print("    Then run: python apply_corrections.py")


if __name__ == "__main__":
    main()
