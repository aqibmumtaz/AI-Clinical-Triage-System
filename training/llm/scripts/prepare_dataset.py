"""
prepare_dataset.py
Validate, clean and split JSONL training data into train/eval sets.
Usage: python prepare_dataset.py --input ../data/raw_conversations.jsonl
"""

import json
import random
import argparse
from pathlib import Path


def validate_conversation(conv: dict) -> tuple[bool, str]:
    """Check a conversation is valid for training (single-turn and multi-turn)."""
    if "messages" not in conv:
        return False, "missing 'messages' key"
    msgs = conv["messages"]
    # Must have at least system + user + assistant
    if len(msgs) < 3:
        return False, f"too few messages: {len(msgs)}"
    if msgs[0]["role"] != "system":
        return False, "first message must be system"
    # All messages after system must alternate user/assistant and end on assistant
    turn_msgs = msgs[1:]
    if turn_msgs[-1]["role"] != "assistant":
        return False, "last message must be assistant"
    for i, msg in enumerate(turn_msgs):
        expected = "user" if i % 2 == 0 else "assistant"
        if msg["role"] != expected:
            return (
                False,
                f"message {i+2} role should be '{expected}', got '{msg['role']}'",
            )
        if len(msg["content"].strip()) < 3:
            return False, f"message {i+2} content too short"
    # Check at least one assistant turn is in Urdu script
    assistant_texts = [m["content"] for m in msgs if m["role"] == "assistant"]
    all_urdu_chars = sum(
        1 for t in assistant_texts for c in t if "\u0600" <= c <= "\u06ff"
    )
    if all_urdu_chars < 10:
        return False, "no assistant response in Urdu script"
    return True, "ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="../data/raw_conversations.jsonl")
    parser.add_argument("--train", default="../data/train.jsonl")
    parser.add_argument("--eval", default="../data/eval.jsonl")
    parser.add_argument("--split", type=float, default=0.8, help="Train split ratio")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    conversations = []
    skipped = 0

    with open(input_path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                conv = json.loads(line)
                valid, reason = validate_conversation(conv)
                if valid:
                    conversations.append(conv)
                else:
                    print(f"  Line {i} skipped: {reason}")
                    skipped += 1
            except json.JSONDecodeError as e:
                print(f"  Line {i} JSON error: {e}")
                skipped += 1

    print(f"\nLoaded: {len(conversations)} valid, {skipped} skipped")

    # Shuffle and split
    random.shuffle(conversations)
    split_idx = int(len(conversations) * args.split)
    train_set = conversations[:split_idx]
    eval_set = conversations[split_idx:]

    # Write train
    train_path = Path(args.train)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    with open(train_path, "w", encoding="utf-8") as f:
        for conv in train_set:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    # Write eval
    eval_path = Path(args.eval)
    with open(eval_path, "w", encoding="utf-8") as f:
        for conv in eval_set:
            f.write(json.dumps(conv, ensure_ascii=False) + "\n")

    print(f"Train: {len(train_set)} examples → {train_path}")
    print(f"Eval:  {len(eval_set)} examples  → {eval_path}")
    print(f"\nReady for training. Run train_qlora.py next.")


if __name__ == "__main__":
    main()
