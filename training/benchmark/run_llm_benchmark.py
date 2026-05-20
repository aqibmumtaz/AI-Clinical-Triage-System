"""
run_llm_benchmark.py
Benchmark a fine-tuned (or base) Qwen3 model on 50 Urdu lab intents.
Usage:
  # Test base model (before fine-tune):
  python run_llm_benchmark.py --model Qwen/Qwen3-8B-Instruct

  # Test fine-tuned model (after fine-tune):
  python run_llm_benchmark.py --model ../llm/checkpoints/final --is_lora
"""

import json
import argparse
import re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

SYSTEM_PROMPT = """آپ ایک طبی تشخیصی لیبارٹری کے کال سینٹر میں ایک مددگار اور پیشہ ورانہ ریسیپشنسٹ ہیں۔
آپ مریضوں کی ٹیسٹ بکنگ، قیمتوں، تیاری ہدایات، رپورٹ کی صورتحال، اور گھر پر سیمپل کلیکشن میں مدد کرتے ہیں۔
ہمیشہ اردو رسم الخط میں مختصر اور واضح جواب دیں۔"""

# 50 test prompts covering all intents
TEST_PROMPTS = [
    # Test booking (10)
    {
        "prompt": "CBC karwani hai, kya appointment chahiye?",
        "intent": "booking",
        "expected_keywords": ["appointment", "وقت", "آئیں", "ضروری"],
    },
    {
        "prompt": "kal subah LFTs book karwane hain",
        "intent": "booking",
        "expected_keywords": ["خالی پیٹ", "فاقہ", "صبح"],
    },
    {
        "prompt": "HbA1c ke liye slot available hai?",
        "intent": "booking",
        "expected_keywords": ["ایچ بی اے", "وقت", "آ سکتے"],
    },
    {
        "prompt": "home collection book karni hai kal ke liye",
        "intent": "booking",
        "expected_keywords": ["گھر", "وزٹ", "سیمپل", "صبح"],
    },
    {
        "prompt": "mujhe KFTs karwane hain, kitne baje aaun?",
        "intent": "booking",
        "expected_keywords": ["خالی پیٹ", "صبح", "آئیں"],
    },
    {
        "prompt": "blood tests ke liye kya lana hoga?",
        "intent": "booking",
        "expected_keywords": ["سلپ", "prescription", "ڈاکٹر", "لائیں"],
    },
    {
        "prompt": "thyroid test ke liye kab aaun?",
        "intent": "booking",
        "expected_keywords": ["خالی پیٹ", "صبح", "TSH"],
    },
    {
        "prompt": "urgent report chahiye aaj, kya ho sakta hai?",
        "intent": "booking",
        "expected_keywords": ["ارجنٹ", "STAT", "اضافی", "گھنٹے"],
    },
    {
        "prompt": "diabetes ke sare tests ek din mein ho sakte hain?",
        "intent": "booking",
        "expected_keywords": ["جی", "فاقہ", "ایک ہی دن"],
    },
    {
        "prompt": "bache ka blood test karwana hai, 3 saal ka hai",
        "intent": "booking",
        "expected_keywords": ["بچے", "تجربہ", "گھر"],
    },
    # Prices (10)
    {
        "prompt": "CBC ka rate kya hai?",
        "intent": "price",
        "expected_keywords": ["روپے", "500", "قیمت"],
    },
    {
        "prompt": "LFTs kitne mein hoti hain?",
        "intent": "price",
        "expected_keywords": ["روپے", "1800", "قیمت"],
    },
    {
        "prompt": "HbA1c ka kharcha kya hai?",
        "intent": "price",
        "expected_keywords": ["روپے", "1200"],
    },
    {
        "prompt": "Vitamin D test kitne ka hai?",
        "intent": "price",
        "expected_keywords": ["روپے", "2500"],
    },
    {
        "prompt": "TSH aur T3 T4 total kitna hoga?",
        "intent": "price",
        "expected_keywords": ["روپے", "1800", "تھائیرائیڈ"],
    },
    {
        "prompt": "dengue test ka rate batao",
        "intent": "price",
        "expected_keywords": ["روپے", "NS1", "IgM", "قیمت"],
    },
    {
        "prompt": "blood sugar fasting aur random dono ka kitna hoga?",
        "intent": "price",
        "expected_keywords": ["روپے", "250", "450"],
    },
    {
        "prompt": "KFTs ka total rate kya hai?",
        "intent": "price",
        "expected_keywords": ["روپے", "1500"],
    },
    {
        "prompt": "home collection pe extra charge lagta hai?",
        "intent": "price",
        "expected_keywords": ["300", "اضافی", "وزٹ"],
    },
    {
        "prompt": "HBSAG aur Anti HCV dono ka rate?",
        "intent": "price",
        "expected_keywords": ["روپے", "400", "500"],
    },
    # Preparation (10)
    {
        "prompt": "LFTs ke liye khaali pait aana zaroori hai?",
        "intent": "preparation",
        "expected_keywords": ["خالی پیٹ", "فاقہ", "گھنٹے", "پانی"],
    },
    {
        "prompt": "lipid profile ke liye kitne ghante ka faqa?",
        "intent": "preparation",
        "expected_keywords": ["12", "گھنٹے", "فاقہ", "پانی"],
    },
    {
        "prompt": "TSH test ke liye thyroid ki dawa band karni hogi?",
        "intent": "preparation",
        "expected_keywords": ["دوائی", "سیمپل", "بعد"],
    },
    {
        "prompt": "urine test ke liye sample kaise lein?",
        "intent": "preparation",
        "expected_keywords": ["صبح", "پہلا", "پیشاب", "ڈبہ"],
    },
    {
        "prompt": "HbA1c ke liye kuch khaana pina band?",
        "intent": "preparation",
        "expected_keywords": ["ضروری نہیں", "خالی پیٹ"],
    },
    {
        "prompt": "blood sugar test ke liye kya karna hoga?",
        "intent": "preparation",
        "expected_keywords": ["فاسٹنگ", "فاقہ", "8"],
    },
    {
        "prompt": "cortisol test ke liye koi special time?",
        "intent": "preparation",
        "expected_keywords": ["صبح", "8", "9", "وقت", "ضروری"],
    },
    {
        "prompt": "OGTT test mein kitna time lagta hai?",
        "intent": "preparation",
        "expected_keywords": ["2", "3", "گھنٹے", "گلوکوز"],
    },
    {
        "prompt": "stool sample ghar se la sakte hain?",
        "intent": "preparation",
        "expected_keywords": ["جی", "ڈبہ", "2 گھنٹے", "جمع"],
    },
    {
        "prompt": "Vitamin D ke liye khaali pait aana hai?",
        "intent": "preparation",
        "expected_keywords": ["ضروری نہیں", "کسی بھی وقت"],
    },
    # Report status (10)
    {
        "prompt": "meri report kab tak ready hogi? subah test diya tha",
        "intent": "report",
        "expected_keywords": ["شام", "SMS", "اطلاع", "تیار"],
    },
    {
        "prompt": "report WhatsApp pe mil sakti hai?",
        "intent": "report",
        "expected_keywords": ["جی", "WhatsApp", "بھجوا"],
    },
    {
        "prompt": "slip number 12345 ki report check karo",
        "intent": "report",
        "expected_keywords": ["چیک", "نام", "slip"],
    },
    {
        "prompt": "3 din pehle test diya tha, still report nahi mili",
        "intent": "report",
        "expected_keywords": ["معذرت", "slip", "نام", "چیک"],
    },
    {
        "prompt": "purani report dobara chahiye 2 mahine pehle ki",
        "intent": "report",
        "expected_keywords": ["محفوظ", "نام", "تاریخ", "مفت"],
    },
    {
        "prompt": "report mein normal range nahi thi",
        "intent": "report",
        "expected_keywords": ["معذرت", "درست", "مفت", "بھجوا"],
    },
    {
        "prompt": "CBC ki report kab milti hai?",
        "intent": "report",
        "expected_keywords": ["گھنٹے", "3", "4", "شام"],
    },
    {
        "prompt": "email pe report chahiye",
        "intent": "report",
        "expected_keywords": ["جی", "ای میل", "رجسٹر", "بھجوا"],
    },
    {
        "prompt": "Vitamin D report kab tak ayegi?",
        "intent": "report",
        "expected_keywords": ["اگلے دن", "دوپہر"],
    },
    {
        "prompt": "blood culture ka result kab milega?",
        "intent": "report",
        "expected_keywords": ["3", "5", "دن", "بیکٹیریا"],
    },
    # Edge cases (10)
    {
        "prompt": "doctor ka parcha gum ho gaya, kya tests karwana tha yaad nahi",
        "intent": "edge",
        "expected_keywords": ["ڈاکٹر", "فون", "بیماری", "بتائیں"],
    },
    {
        "prompt": "mere husband hospital mein hain, wahan sample le sakte ho?",
        "intent": "edge",
        "expected_keywords": ["ہسپتال", "وارڈ", "300", "وزٹ"],
    },
    {
        "prompt": "kya aap MRI bhi karte hain?",
        "intent": "edge",
        "expected_keywords": ["نہیں", "laboratory", "diagnostic", "hospital"],
    },
    {
        "prompt": "meri report mein HB 9 hai, kya yeh theek hai?",
        "intent": "edge",
        "expected_keywords": ["ڈاکٹر", "تشریح", "دکھائیں"],
    },
    {
        "prompt": "discount milega agar 10 tests karwaaun?",
        "intent": "edge",
        "expected_keywords": ["چھوٹ", "discount", "فیصد", "فہرست"],
    },
    {
        "prompt": "kya aap NABL accredited hain?",
        "intent": "edge",
        "expected_keywords": ["NABL", "ISO", "منظوری", "accredited"],
    },
    {
        "prompt": "card se payment ho sakti hai?",
        "intent": "edge",
        "expected_keywords": ["card", "JazzCash", "Easypaisa", "ادائیگی"],
    },
    {
        "prompt": "pathologist se milna hai report samajhni hai",
        "intent": "edge",
        "expected_keywords": ["pathologist", "مشاورت", "پیر", "مفت"],
    },
    {
        "prompt": "sunday ko bhi khule ho?",
        "intent": "edge",
        "expected_keywords": ["جی", "اتوار", "ساتوں", "دن"],
    },
    {
        "prompt": "main pregnant hun, ghar par sample lena safe hai?",
        "intent": "edge",
        "expected_keywords": ["گھر", "محفوظ", "sیمپل", "phlebotomist"],
    },
]


def is_urdu_script(text: str, threshold: int = 10) -> bool:
    """Check if text contains enough Urdu script characters."""
    return sum(1 for c in text if "\u0600" <= c <= "\u06ff") >= threshold


def check_keywords(response: str, keywords: list[str]) -> bool:
    """Check if any expected keyword appears in the response."""
    response_lower = response.lower()
    return any(kw.lower() in response_lower for kw in keywords)


def run_inference(model, tokenizer, prompt: str, device: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(
        outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
    )
    return response.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="Qwen/Qwen3-8B-Instruct", help="Model path or HF name"
    )
    parser.add_argument(
        "--base_model",
        default="Qwen/Qwen3-8B-Instruct",
        help="Base model (if --is_lora)",
    )
    parser.add_argument("--is_lora", action="store_true", help="Load as LoRA adapter")
    parser.add_argument("--output", default="benchmark_results.json")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Loading model: {args.model}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model if args.is_lora else args.model, trust_remote_code=True
    )

    base = AutoModelForCausalLM.from_pretrained(
        args.base_model if args.is_lora else args.model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    if args.is_lora:
        model = PeftModel.from_pretrained(base, args.model)
        print("LoRA adapter loaded")
    else:
        model = base

    model.eval()

    results = []
    urdu_count = 0
    keyword_count = 0
    intent_scores = {}

    print(f"\nRunning {len(TEST_PROMPTS)} benchmark prompts...\n")

    for i, test in enumerate(TEST_PROMPTS, 1):
        response = run_inference(model, tokenizer, test["prompt"], device)
        urdu_ok = is_urdu_script(response)
        keyword_ok = check_keywords(response, test["expected_keywords"])

        if urdu_ok:
            urdu_count += 1
        if keyword_ok:
            keyword_count += 1

        intent = test["intent"]
        if intent not in intent_scores:
            intent_scores[intent] = {"total": 0, "keyword_pass": 0}
        intent_scores[intent]["total"] += 1
        if keyword_ok:
            intent_scores[intent]["keyword_pass"] += 1

        status = "✓" if (urdu_ok and keyword_ok) else "✗"
        print(f"  {i:2}. [{status}] {test['prompt'][:50]}")
        if not urdu_ok:
            print(f"      ⚠ Not in Urdu script")
        if not keyword_ok:
            print(f"      ⚠ Missing expected keywords: {test['expected_keywords'][:2]}")

        results.append(
            {
                "prompt": test["prompt"],
                "intent": test["intent"],
                "response": response,
                "urdu_script": urdu_ok,
                "keyword_match": keyword_ok,
            }
        )

    total = len(TEST_PROMPTS)
    print(f"\n{'='*50}")
    print(f"BENCHMARK RESULTS")
    print(f"{'='*50}")
    print(f"Urdu script output:  {urdu_count}/{total} ({100*urdu_count//total}%)")
    print(f"Keyword accuracy:    {keyword_count}/{total} ({100*keyword_count//total}%)")
    print(
        f"Overall pass rate:   {sum(1 for r in results if r['urdu_script'] and r['keyword_match'])}/{total}"
    )
    print(f"\nBy intent:")
    for intent, s in intent_scores.items():
        pct = 100 * s["keyword_pass"] // s["total"]
        print(f"  {intent:12}: {s['keyword_pass']}/{s['total']} ({pct}%)")

    print(f"\nTargets:")
    print(f"  Urdu script > 95%:  {'PASS' if urdu_count/total >= 0.95 else 'FAIL'}")
    print(f"  Keyword acc > 80%:  {'PASS' if keyword_count/total >= 0.80 else 'FAIL'}")
    print(f"  Keyword acc > 90%:  {'PASS' if keyword_count/total >= 0.90 else 'FAIL'}")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": {
                    "model": args.model,
                    "total": total,
                    "urdu_pct": round(100 * urdu_count / total, 1),
                    "keyword_pct": round(100 * keyword_count / total, 1),
                },
                "results": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nFull results saved to: {args.output}")


if __name__ == "__main__":
    main()
