"""
Prepare training data for Qwen2-Audio-7B fine-tuning.
Converts Whisper-format dataset → Qwen2-Audio chat-style JSONL.

Usage:
  python prepare_qwen_data.py \
      --whisper_dir ../finetuning/whisper_dataset \
      --output_dir ../finetuning/qwen_dataset

Output:
  qwen_dataset/
  ├── train.jsonl
  ├── validation.jsonl
  ├── test.jsonl
  └── audio/ (symlinked or copied)
"""

import os, json, argparse, shutil, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an expert Urdu speech recognition system for a medical call center. "
    "Transcribe the given audio accurately in Urdu script (اردو رسم الخط). "
    "Use proper Urdu punctuation. Keep English medical terms, names, and numbers as-is. "
    "Do not translate or transliterate — output pure Urdu script."
)


def convert_split(whisper_dir: str, split: str, output_dir: str):
    metadata_path = os.path.join(whisper_dir, split, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        log.warning(f"No {split} split found at {metadata_path}")
        return 0

    audio_src = os.path.join(whisper_dir, split, "audio")
    audio_dst = os.path.join(output_dir, "audio", split)
    os.makedirs(audio_dst, exist_ok=True)

    output_path = os.path.join(output_dir, f"{split}.jsonl")
    count = 0

    with open(metadata_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            entry = json.loads(line.strip())
            audio_filename = os.path.basename(entry["audio"])
            sentence = entry["sentence"]

            src_audio = os.path.join(audio_src, audio_filename)
            dst_audio = os.path.join(audio_dst, audio_filename)
            if os.path.exists(src_audio) and not os.path.exists(dst_audio):
                shutil.copy2(src_audio, dst_audio)

            qwen_entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT,
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "audio", "audio_url": f"audio/{split}/{audio_filename}"}
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": sentence,
                    },
                ]
            }

            fout.write(json.dumps(qwen_entry, ensure_ascii=False) + "\n")
            count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Convert Whisper dataset to Qwen2-Audio format")
    parser.add_argument("--whisper_dir", required=True, help="Whisper dataset directory")
    parser.add_argument("--output_dir", default="qwen_dataset", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for split in ["train", "validation", "test"]:
        count = convert_split(args.whisper_dir, split, args.output_dir)
        log.info(f"{split}: {count} samples")

    log.info(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
