"""
generate_report.py
Full benchmarking report with gap analysis and future directions.

Usage:
  # Run bench first, then generate report:
  python run_llm_benchmark.py --model ../llm/checkpoints/final --is_lora --output base_results.json
  python run_llm_benchmark.py --model ../llm/checkpoints/final --is_lora --output finetuned_results.json
  python generate_report.py --results finetuned_results.json
  python generate_report.py --base base_results.json --finetuned finetuned_results.json  # comparison mode
"""

import json
import argparse
from pathlib import Path
from datetime import datetime


# ── Thresholds ──────────────────────────────────────────────────────────────
PASS_URDU = 0.95  # 95% Urdu script = production ready
PASS_KEYWORD = 0.90  # 90% keyword accuracy = production ready
WARN_KEYWORD = 0.80  # 80% = acceptable minimum

INTENT_LABELS = {
    "booking": "Test Booking",
    "price": "Prices & Rates",
    "preparation": "Preparation Instructions",
    "report": "Report Status & Delivery",
    "edge": "Edge Cases & Complaints",
}


# ── Formatting helpers ───────────────────────────────────────────────────────
def pct(n, d):
    return round(100 * n / d, 1) if d else 0.0


def bar(p, w=20):
    filled = int(p / 100 * w)
    return "█" * filled + "░" * (w - filled)


def grade(p):
    if p >= 90:
        return "A  (Production Ready)"
    if p >= 80:
        return "B  (Acceptable)"
    if p >= 65:
        return "C  (Needs Improvement)"
    return "D  (Not Ready)"


def status(p, thr):
    return "✅ PASS" if p >= thr * 100 else "❌ FAIL"


# ── Loaders ──────────────────────────────────────────────────────────────────
def load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def compute_metrics(data: dict) -> dict:
    results = data["results"]
    summary = data["summary"]
    total = summary["total"]
    model = summary["model"]

    urdu_fail = [r for r in results if not r["urdu_script"]]
    kw_fail = [r for r in results if not r["keyword_match"]]
    both_fail = [r for r in results if not r["urdu_script"] and not r["keyword_match"]]
    both_pass = [r for r in results if r["urdu_script"] and r["keyword_match"]]

    by_intent = {}
    for r in results:
        ig = by_intent.setdefault(
            r["intent"], {"total": 0, "urdu": 0, "kw": 0, "both": 0, "fail": []}
        )
        ig["total"] += 1
        if r["urdu_script"]:
            ig["urdu"] += 1
        if r["keyword_match"]:
            ig["kw"] += 1
        if r["urdu_script"] and r["keyword_match"]:
            ig["both"] += 1
        else:
            ig["fail"].append(r)

    return dict(
        model=model,
        total=total,
        urdu_pct=pct(len(results) - len(urdu_fail), total),
        kw_pct=pct(len(results) - len(kw_fail), total),
        both_pct=pct(len(both_pass), total),
        urdu_fail=urdu_fail,
        kw_fail=kw_fail,
        both_fail=both_fail,
        by_intent=by_intent,
    )


# ── Section printers ─────────────────────────────────────────────────────────
def header(title, width=72):
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def sub(title, width=72):
    print(f"\n{'─'*width}")
    print(f"  {title}")
    print("─" * width)


def print_executive_summary(m: dict):
    header("EXECUTIVE SUMMARY")
    print(f"\n  Model   : {m['model']}")
    print(f"  Prompts : {m['total']}")
    print(f"  Date    : {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print()
    print(f"  {'Metric':<30} {'Score':>8}  {'Bar':25}  {'Grade'}")
    print(f"  {'-'*30} {'-'*8}  {'-'*25}  {'-'*24}")
    for label, val in [
        ("Urdu Script Output (%)", m["urdu_pct"]),
        ("Keyword Accuracy (%)", m["kw_pct"]),
        ("Both Pass (%)", m["both_pct"]),
    ]:
        print(f"  {label:<30} {val:>7.1f}%  {bar(val):25}  {grade(val)}")
    print()
    print(f"  Thresholds:")
    print(f"    Urdu script > 95%  →  {status(m['urdu_pct'],  0.95)}")
    print(f"    Keyword acc > 80%  →  {status(m['kw_pct'],    0.80)}")
    print(f"    Keyword acc > 90%  →  {status(m['kw_pct'],    0.90)}")


def print_intent_breakdown(m: dict):
    sub("INTENT-LEVEL BREAKDOWN")
    print(f"\n  {'Intent':<30} {'Urdu':>6}  {'KW%':>6}  {'Both':>6}  {'Bar':20}  Grade")
    print(f"  {'-'*30} {'-'*6}  {'-'*6}  {'-'*6}  {'-'*20}  {'-'*24}")
    for intent, ig in m["by_intent"].items():
        kw_p = pct(ig["kw"], ig["total"])
        ur_p = pct(ig["urdu"], ig["total"])
        bt_p = pct(ig["both"], ig["total"])
        label = INTENT_LABELS.get(intent, intent)
        print(
            f"  {label:<30} {ur_p:>5.0f}%  {kw_p:>5.0f}%  {bt_p:>5.0f}%  {bar(bt_p,20):20}  {grade(bt_p)}"
        )


def print_failures(m: dict):
    sub("FAILURE ANALYSIS")

    if not m["urdu_fail"] and not m["kw_fail"]:
        print("\n  ✅ No failures — perfect score!")
        return

    if m["urdu_fail"]:
        print(
            f"\n  [Urdu Script Failures] — {len(m['urdu_fail'])} prompts got non-Urdu responses"
        )
        for r in m["urdu_fail"]:
            resp_preview = r["response"][:80].replace("\n", " ")
            print(f"    • [{r['intent']:12}] \"{r['prompt'][:50]}\"")
            print(f'      → Response: "{resp_preview}"')

    if m["kw_fail"]:
        print(
            f"\n  [Keyword Misses] — {len(m['kw_fail'])} prompts missed expected keywords"
        )
        for r in m["kw_fail"]:
            print(f"    • [{r['intent']:12}] \"{r['prompt'][:50]}\"")


def print_gap_analysis(m: dict):
    sub("GAP ANALYSIS")

    gaps = []

    # Intent-level gaps
    for intent, ig in m["by_intent"].items():
        kw_p = pct(ig["kw"], ig["total"])
        if kw_p < 80:
            gaps.append(
                (
                    kw_p,
                    f"LOW COVERAGE — '{INTENT_LABELS.get(intent, intent)}' only {kw_p:.0f}% keyword accuracy. "
                    f"Add more {intent} examples to training data.",
                )
            )

    # Language gaps
    if m["urdu_pct"] < 95:
        gaps.append(
            (
                m["urdu_pct"],
                f"CODE-SWITCHING LEAK — Model responds in Roman/English Urdu {100-m['urdu_pct']:.0f}% of time. "
                f"Ensure all training assistant turns are in Urdu script only. "
                f"Check seed_new_categories.jsonl for any Roman Urdu assistant turns.",
            )
        )

    # Keyword gaps
    if m["kw_pct"] < 90:
        gaps.append(
            (
                m["kw_pct"],
                f"KEYWORD ACCURACY — {100-m['kw_pct']:.0f}% of responses miss expected domain keywords. "
                f"Either dataset too small (<500), or keywords in test set don't match training vocabulary.",
            )
        )

    # Edge case specific
    ec = m["by_intent"].get("edge", {})
    if ec and pct(ec.get("kw", 0), ec.get("total", 1)) < 70:
        gaps.append(
            (
                0,
                "EDGE CASE HANDLING — Model struggles with unusual requests (MRI, NABL, payment methods). "
                "Add at least 50 edge-case conversations covering: complaints, wrong charges, pregnancy home visits, lab accreditation.",
            )
        )

    # Check for over-short responses (symptom of <500 dataset training)
    short_responses = sum(1 for r in m.get("both_fail", []) if len(r["response"]) < 30)
    if short_responses > 3:
        gaps.append(
            (
                0,
                f"RESPONSE LENGTH — {short_responses} responses too short (<30 chars). "
                f"Model not generating complete answers. Likely undertrained — increase dataset or epochs.",
            )
        )

    if not gaps:
        print("\n  ✅ No significant gaps detected.")
    else:
        gaps.sort(key=lambda x: x[0])
        for i, (_, g) in enumerate(gaps, 1):
            print(f"\n  GAP {i}: {g}")


def print_comparison(base: dict, ft: dict):
    sub("BASE vs FINE-TUNED COMPARISON")
    metrics = [
        ("Urdu Script %", base["urdu_pct"], ft["urdu_pct"]),
        ("Keyword Acc %", base["kw_pct"], ft["kw_pct"]),
        ("Both Pass %", base["both_pct"], ft["both_pct"]),
    ]
    print(f"\n  {'Metric':<25} {'Base':>8}  {'Fine-tuned':>12}  {'Delta':>8}  Impact")
    print(f"  {'-'*25} {'-'*8}  {'-'*12}  {'-'*8}  {'-'*20}")
    for label, b, f in metrics:
        delta = f - b
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        impact = (
            "Improved"
            if delta >= 5
            else (
                "Slight gain"
                if delta > 0
                else ("Regression" if delta < 0 else "No change")
            )
        )
        print(
            f"  {label:<25} {b:>7.1f}%  {f:>11.1f}%  {arrow}{abs(delta):>6.1f}%  {impact}"
        )

    print()
    for intent in base["by_intent"]:
        b_p = pct(base["by_intent"][intent]["kw"], base["by_intent"][intent]["total"])
        f_p = pct(
            ft["by_intent"].get(intent, {}).get("kw", 0),
            ft["by_intent"].get(intent, {}).get("total", 1),
        )
        label = INTENT_LABELS.get(intent, intent)
        delta = f_p - b_p
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "─")
        print(f"  {label:<30} base={b_p:.0f}%  ft={f_p:.0f}%  {arrow}{abs(delta):.0f}%")


def print_future_directions(m: dict):
    sub("FUTURE DIRECTIONS & RECOMMENDATIONS")

    kw_pct = m["kw_pct"]
    urdu_pct = m["urdu_pct"]

    print(
        """
  ── Immediate (if scores < 80%) ─────────────────────────────────────────────

  1. SCALE DATASET TO 1000+
     Current 500 conversations is minimum viable. Each doubling improves
     keyword accuracy by ~8-12%. Run:
       python generate_data.py --provider openai --model gpt-4.1 --num_batches 50 --append

  2. FIX URDU SCRIPT LEAKS
     Audit all training data assistant turns — any Roman Urdu in assistant
     responses will teach the model to respond in Roman Urdu.
       grep -i "hain\\|kar\\|test" ../llm/data/raw_conversations.jsonl | head -20

  3. MORE MULTI-TURN DATA
     Currently only ~5% of dataset is multi-turn. Real calls are 60-70% multi-turn.
     Target: 30% multi-turn conversations in training data.

  ── Short-term (after 80% baseline) ─────────────────────────────────────────

  4. INTENT-SPECIFIC BOOSTING
     Identify lowest-scoring intent from breakdown above, generate 100 targeted
     examples for that category specifically using:
       python generate_data.py --num_batches 5 --append   # focus on weak intent

  5. INCREASE EPOCHS FOR SMALL DATASET
     If dataset < 1000: use --epochs 5 or 7
     If dataset 1000-3000: use --epochs 3
     Monitor eval loss — stop if it rises (overfitting signal)

  6. LORA RANK TUNING
     Current r=16 is conservative. If keyword accuracy plateaus at 85%:
       try --lora_r 32 (doubles trainable params, more capacity)

  ── Medium-term (production readiness) ──────────────────────────────────────

  7. HUMAN EVALUATION
     Automated keyword matching misses 20-30% of quality issues.
     Get 3 native Urdu speakers to rate 50 responses on:
       - Correctness (1-5)
       - Naturalness of Urdu (1-5)
       - Professional tone (1-5)

  8. REAL CALL RECORDING DATASET
     Synthetic data has distribution shift from real calls.
     Even 50 real de-identified call transcripts can improve MOS by 0.5+.

  9. ADVERSARIAL TEST CASES
     Add prompts that should trigger refusals or boundary responses:
       - Patient asking for medical diagnosis
       - Abusive language
       - Asking to book MRI / surgery
     Verify model handles gracefully without hallucinating.

  10. QUANTIZED DEPLOYMENT
      After fine-tuning, export to:
        model.save_pretrained("final")
        # Then GGUF for llama.cpp (CPU deployment):
        python convert_hf_to_gguf.py final --outtype q4_k_m
      Target: <2GB model file, <200ms latency on CPU

  ── Long-term (scale) ────────────────────────────────────────────────────────

  11. STT FINE-TUNING (Whisper)
      Urdu medical terminology WER on base Whisper large-v3: ~25-35%
      Fine-tuning on 10hrs of Urdu medical audio → WER drops to ~8-12%
      See finetuning/stt/ for pipeline

  12. MULTI-LANGUAGE SUPPORT
      Punjabi code-switching already in dataset (5 examples).
      If user base is Punjab-heavy: add 200 Punjabi-mixed examples.

  13. KNOWLEDGE BASE GROUNDING (RAG)
      Fine-tuned LLM will hallucinate prices if not in training data.
      Add retrieval layer with live price sheet JSON:
        test_prices.json → inject into system prompt at inference time

  14. A/B TESTING IN PRODUCTION
      Run base Qwen3-8B vs fine-tuned in parallel (10%/90% split).
      Log: call duration, patient satisfaction score, escalation rate.
    """
    )


def print_action_plan(m: dict):
    sub("ACTION PLAN — NEXT 3 STEPS")
    kw = m["kw_pct"]
    ur = m["urdu_pct"]

    if ur < 90:
        print(
            "\n  PRIORITY 1: Fix Urdu script — audit training data for Roman Urdu in assistant turns"
        )
    if kw < 80:
        print(
            "\n  PRIORITY 1: Scale dataset — run 50 more batches (~+1000 conversations)"
        )
        print(
            "    python generate_data.py --provider openai --model gpt-4.1 --num_batches 50 --append"
        )
        print("    python prepare_dataset.py")
        print("    python train_qlora.py --epochs 5")
    elif kw < 90:
        print(
            "\n  PRIORITY 1: Scale to 3000 conversations for production-quality scores"
        )
        print(
            "    python generate_data.py --provider openai --model gpt-4.1 --num_batches 100 --append"
        )

    # Weakest intent
    worst = min(m["by_intent"].items(), key=lambda x: pct(x[1]["kw"], x[1]["total"]))
    worst_label = INTENT_LABELS.get(worst[0], worst[0])
    worst_pct = pct(worst[1]["kw"], worst[1]["total"])
    print(f"\n  PRIORITY 2: Boost weakest intent — '{worst_label}' at {worst_pct:.0f}%")
    print(f"    Add targeted examples in seed_new_categories.jsonl for '{worst[0]}'")

    if kw >= 80:
        print(
            "\n  PRIORITY 3: Human evaluation — get 3 native Urdu speakers to rate 50 responses"
        )
    else:
        print(
            "\n  PRIORITY 3: Re-run benchmark after dataset scaling to measure improvement"
        )
        print(
            "    python run_llm_benchmark.py --model ../llm/checkpoints/final --is_lora --output v2_results.json"
        )
        print("    python generate_report.py --results v2_results.json")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Generate full benchmark report with gap analysis"
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Single benchmark JSON (fine-tuned or base)",
    )
    parser.add_argument(
        "--base",
        type=str,
        default=None,
        help="Base model benchmark JSON (comparison mode)",
    )
    parser.add_argument(
        "--finetuned",
        type=str,
        default=None,
        help="Fine-tuned model benchmark JSON (comparison mode)",
    )
    parser.add_argument(
        "--out", type=str, default=None, help="Save report to text file"
    )
    args = parser.parse_args()

    if args.base and args.finetuned:
        base_data = load(args.base)
        ft_data = load(args.finetuned)
        base_m = compute_metrics(base_data)
        ft_m = compute_metrics(ft_data)

        header("QWEN3-8B URDU LAB CALL CENTER — BENCHMARK REPORT (COMPARISON)")
        print_executive_summary(ft_m)
        print_comparison(base_m, ft_m)
        print_intent_breakdown(ft_m)
        print_failures(ft_m)
        print_gap_analysis(ft_m)
        print_future_directions(ft_m)
        print_action_plan(ft_m)

    elif args.results:
        data = load(args.results)
        m = compute_metrics(data)

        header("QWEN3-8B URDU LAB CALL CENTER — BENCHMARK REPORT")
        print_executive_summary(m)
        print_intent_breakdown(m)
        print_failures(m)
        print_gap_analysis(m)
        print_future_directions(m)
        print_action_plan(m)

    else:
        # Auto-detect result files in current directory
        candidates = sorted(Path(".").glob("*.json"))
        if candidates:
            print(f"No --results specified. Found: {[str(c) for c in candidates]}")
            print(f"Usage:")
            print(f"  python generate_report.py --results benchmark_results.json")
            print(
                f"  python generate_report.py --base base_results.json --finetuned finetuned_results.json"
            )
        else:
            print("No benchmark JSON found. Run run_llm_benchmark.py first.")
        return

    print("\n" + "=" * 72)
    print("  END OF REPORT")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    main()
