"""
generate_data.py
Generate Urdu lab call center training conversations using Claude or OpenAI.

Usage:
  # Highest accuracy — Claude Opus (Anthropic, ~$15 for 3000 pairs)
  python generate_data.py --provider anthropic --model claude-opus-4-5 --num_batches 150 --append

  # Best value — GPT-4.1 (OpenAI, ~$4 for 3000 pairs)
  python generate_data.py --provider openai --model gpt-4.1 --num_batches 150 --append

  # Cheapest — GPT-4.1-mini (OpenAI, ~$0.30 for 3000 pairs)
  python generate_data.py --provider openai --model gpt-4.1-mini --num_batches 150 --append

Model comparison (highest to lowest accuracy):
  anthropic  claude-opus-4-5       Best Urdu quality        ~$15 / 3000 pairs
  openai     gpt-4.1               Excellent Urdu           ~$4  / 3000 pairs
  anthropic  claude-sonnet-4-5     Very good Urdu           ~$3  / 3000 pairs
  openai     gpt-4o                Excellent Urdu           ~$8  / 3000 pairs
  openai     gpt-4.1-mini          Good Urdu                ~$0.30 / 3000 pairs
  openai     gpt-4o-mini           Good Urdu                ~$0.60 / 3000 pairs
"""

import json
import argparse
import time
import os
from pathlib import Path

# Load finetuning/.env (two levels up from this script)
try:
    from dotenv import load_dotenv

    _env_path = Path(__file__).resolve().parents[2] / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
        print(f"Loaded .env from {_env_path}")
    else:
        print(f"No .env found at {_env_path} — using shell environment")
except ImportError:
    print(
        "python-dotenv not installed — using shell environment (pip install python-dotenv)"
    )

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    import openai
except ImportError:
    openai = None

SYSTEM_PROMPT = """آپ ایک طبی تشخیصی لیبارٹری کے کال سینٹر میں ایک مددگار اور پیشہ ورانہ ریسیپشنسٹ ہیں۔
آپ مریضوں کی ٹیسٹ بکنگ، قیمتوں، تیاری ہدایات، رپورٹ کی صورتحال، اور گھر پر سیمپل کلیکشن میں مدد کرتے ہیں۔
ہمیشہ اردو رسم الخط میں مختصر اور واضح جواب دیں۔"""

INTENT_CATEGORIES = [
    # Original categories
    "test booking and appointment",
    "test prices and rates",
    "fasting and preparation instructions",
    "report status and delivery",
    "home sample collection booking",
    "branch locations and timings",
    "multiple tests same day",
    "urgent/STAT reports",
    "pediatric patients (children)",
    "elderly or bedridden patients",
    "corporate and bulk testing",
    "repeat or duplicate reports",
    "confused or unclear patient requests",
    "angry or frustrated patients",
    "diabetes panel tests",
    "thyroid panel tests",
    "liver function tests",
    "kidney function tests",
    "blood count CBC tests",
    "infectious disease tests (dengue, malaria, typhoid)",
    # New categories covering real lab call center gaps
    "multi-turn booking conversation 4 to 6 turns with name address and test collection",
    "Sehat Sahulat card and health insurance panel queries",
    "Ramadan fasting roza and test timing confusion",
    "dengue season surge high volume queries",
    "report portal app login password and technical issues",
    "billing dispute wrong charge refund and payment failure",
    "disputed or incorrect test results patient wants retest",
    "female phlebotomist request and gender sensitive home collection",
    "Punjabi mixed Urdu patient queries",
    "complaint escalation and requesting manager",
    "antenatal pregnancy full panel and maternal health tests",
    "newborn screening and child growth hormone tests",
    "prescription abbreviation confusion TFTs FBS PP TORCH metabolic panel",
    "multiple family members single home visit booking",
    "test result interpretation boundary patient asks meaning of values",
]

GENERATION_PROMPT_TEMPLATE = """Generate exactly 20 realistic Urdu medical lab call center conversations.
Focus on intent category: {intent}

Rules:
1. Patient speaks in Roman Urdu mixed with English medical terms (code-switching)
   Example: "mujhe CBC karwani hai" or "HbA1c ka kya rate hai?"
2. Lab receptionist responds ONLY in Urdu script (نستعلیق) - never Roman Urdu
3. For multi-turn intents: include 4-6 back-and-forth turns as a full messages array
   For single-turn intents: one patient turn + one receptionist response
4. NEVER mention any specific lab name (no Chugtai, no Chughatai, no City Lab, no lab name at all)
5. Use realistic Pakistani medical lab context (Lahore, Karachi, Islamabad cities)
6. Include specific test names: CBC, HbA1c, LFTs, KFTs, TSH, ESR, CRP, Vitamin D, etc.
7. Make responses helpful, professional, and concise
8. For multi-turn: show realistic call flow — clarify → inform → upsell home collection → collect name/address → confirm

Return EXACTLY 20 JSON objects, one per line (JSONL format).
For single-turn:
{{"messages": [{{"role": "system", "content": "{system}"}}, {{"role": "user", "content": "PATIENT_MSG"}}, {{"role": "assistant", "content": "RESPONSE_IN_URDU"}}]}}
For multi-turn:
{{"messages": [{{"role": "system", "content": "{system}"}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}, {{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}]}}

Return ONLY the JSONL, no other text."""


def generate_batch_anthropic(
    client, model: str, intent: str, batch_num: int
) -> list[dict]:
    """Generate one batch using Anthropic (Claude)."""
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        intent=intent, system=SYSTEM_PROMPT.replace('"', '\\"').replace("\n", " ")
    )
    try:
        message = client.messages.create(
            model=model, max_tokens=4096, messages=[{"role": "user", "content": prompt}]
        )
        return _parse_jsonl(message.content[0].text, batch_num, intent)
    except Exception as e:
        print(f"  Batch {batch_num} failed (Anthropic): {e}")
        return []


def generate_batch_openai(
    client, model: str, intent: str, batch_num: int
) -> list[dict]:
    """Generate one batch using OpenAI."""
    prompt = GENERATION_PROMPT_TEMPLATE.format(
        intent=intent, system=SYSTEM_PROMPT.replace('"', '\\"').replace("\n", " ")
    )
    try:
        response = client.chat.completions.create(
            model=model, max_tokens=4096, messages=[{"role": "user", "content": prompt}]
        )
        return _parse_jsonl(response.choices[0].message.content, batch_num, intent)
    except Exception as e:
        print(f"  Batch {batch_num} failed (OpenAI): {e}")
        return []


def _parse_jsonl(content: str, batch_num: int, intent: str) -> list[dict]:
    """Parse JSONL response from any provider."""
    conversations = []
    for line in content.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            conv = json.loads(line)
            if "messages" in conv and len(conv["messages"]) >= 3:
                conversations.append(conv)
        except json.JSONDecodeError:
            continue
    print(f"  Batch {batch_num}: {len(conversations)} conversations for '{intent}'")
    return conversations


def main():
    parser = argparse.ArgumentParser(
        description="Generate Urdu lab call center training data"
    )
    parser.add_argument(
        "--provider",
        choices=["anthropic", "openai"],
        default="anthropic",
        help="API provider (default: anthropic)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model name. Defaults: anthropic=claude-opus-4-5, openai=gpt-4.1",
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=60,
        help="Number of batches (20 conversations each)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../data/raw_conversations.jsonl",
        help="Output JSONL file path",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing file instead of overwriting",
    )
    args = parser.parse_args()

    # Set default models — highest accuracy for each provider
    if args.model is None:
        args.model = "claude-opus-4-5" if args.provider == "anthropic" else "gpt-4.1"

    # Build client
    if args.provider == "anthropic":
        if anthropic is None:
            raise ImportError("pip install anthropic")
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("Set ANTHROPIC_API_KEY environment variable")
        client = anthropic.Anthropic(api_key=api_key)
    else:
        if openai is None:
            raise ImportError("pip install openai")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Set OPENAI_API_KEY environment variable")
        client = openai.OpenAI(api_key=api_key)

    print(f"Provider: {args.provider} | Model: {args.model}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if args.append else "w"
    total = 0

    print(
        f"Generating {args.num_batches} batches (~{args.num_batches * 20} conversations)..."
    )
    print(f"Output: {output_path}")

    with open(output_path, mode, encoding="utf-8") as f:
        for i in range(args.num_batches):
            intent = INTENT_CATEGORIES[i % len(INTENT_CATEGORIES)]
            if args.provider == "anthropic":
                conversations = generate_batch_anthropic(
                    client, args.model, intent, i + 1
                )
            else:
                conversations = generate_batch_openai(client, args.model, intent, i + 1)

            for conv in conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

            total += len(conversations)

            # Rate limit: pause every 10 batches
            if (i + 1) % 10 == 0:
                print(f"Progress: {total} conversations written. Pausing 5s...")
                time.sleep(5)

    cost_per_conv = {
        "claude-opus-4-5": 0.015,
        "claude-sonnet-4-5": 0.005,
        "gpt-4.1": 0.004,
        "gpt-4o": 0.008,
        "gpt-4.1-mini": 0.0001,
        "gpt-4o-mini": 0.0002,
    }
    est_cost = total * cost_per_conv.get(args.model, 0.005)
    print(f"\nDone. Total conversations: {total}")
    print(f"Saved to: {output_path}")
    print(f"Estimated cost: ~${est_cost:.2f} ({args.provider} / {args.model})")
    print(f"Next step: cd ../scripts && python prepare_dataset.py")


if __name__ == "__main__":
    main()
