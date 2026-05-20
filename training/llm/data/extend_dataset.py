"""
extend_dataset.py
Merge one or more seed JSONL files into raw_conversations.jsonl.
Data lives in separate .jsonl files — no conversations are hardcoded here.

Seed files in this directory:
  seed_new_categories.jsonl   — 40 conversations covering new intent categories
                                (insurance, Ramadan, billing, complaints, etc.)

Usage:
  # Merge all seed files into raw_conversations.jsonl
  python extend_dataset.py

  # Merge a specific seed file
  python extend_dataset.py --source seed_new_categories.jsonl

  # Merge into a different target
  python extend_dataset.py --target my_dataset.jsonl
"""

import json
import argparse
from pathlib import Path


def load_jsonl(path: str) -> list[dict]:
    data = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Skipping line {i} in {path}: {e}")
    return data


def print_stats(path: str) -> None:
    data = load_jsonl(path)
    multi = sum(1 for d in data if len(d.get("messages", [])) > 3)
    print(f"\nDataset stats: {path}")
    print(f"  Total conversations : {len(data)}")
    print(f"  Multi-turn          : {multi}")
    print(f"  Single-turn         : {len(data) - multi}")


def main():
    parser = argparse.ArgumentParser(
        description="Merge seed JSONL files into main dataset"
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Seed JSONL file to merge (default: all seed_*.jsonl files)",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="raw_conversations.jsonl",
        help="Target dataset file (default: raw_conversations.jsonl)",
    )
    args = parser.parse_args()

    target_path = Path(args.target)

    # Determine source files
    if args.source:
        sources = [Path(args.source)]
    else:
        sources = sorted(Path(".").glob("seed_*.jsonl"))

    if not sources:
        print("No seed files found. Create seed_*.jsonl files with conversation data.")
        return

    # Load existing target to avoid duplicates
    existing = set()
    if target_path.exists():
        for d in load_jsonl(str(target_path)):
            msgs = d.get("messages", [])
            # Use first user message as dedup key
            key = next((m["content"] for m in msgs if m["role"] == "user"), None)
            if key:
                existing.add(key)
        print(f"Existing conversations in {target_path}: {len(existing)}")

    total_added = 0
    total_skipped = 0

    with open(target_path, "a", encoding="utf-8") as out:
        for src in sources:
            data = load_jsonl(str(src))
            added = 0
            for d in data:
                msgs = d.get("messages", [])
                key = next((m["content"] for m in msgs if m["role"] == "user"), None)
                if key in existing:
                    total_skipped += 1
                    continue
                out.write(json.dumps(d, ensure_ascii=False) + "\n")
                existing.add(key)
                added += 1
            print(
                f"  {src.name}: {added} added, {len(data) - added} duplicates skipped"
            )
            total_added += added

    print(f"\nTotal added   : {total_added}")
    print(f"Total skipped : {total_skipped} (duplicates)")
    print_stats(str(target_path))


if __name__ == "__main__":
    main()


# ── HOW TO ADD MORE SEED DATA ────────────────────────────────────────────────
# 1. Create a new file: seed_<category>.jsonl
# 2. Each line is one conversation in this format:
#    {"messages": [
#      {"role": "system",    "content": "آپ ایک طبی تشخیصی لیبارٹری ..."},
#      {"role": "user",      "content": "patient message in Roman Urdu"},
#      {"role": "assistant", "content": "response in Urdu script (نستعلیق)"}
#    ]}
# 3. Run: python extend_dataset.py --source seed_<category>.jsonl
# ─────────────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    main()
