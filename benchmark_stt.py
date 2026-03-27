"""
STT Benchmark Suite — Test accuracy, latency, and performance across providers.
Generates test audio via TTS, sends to each STT provider, and compares results.
"""

import os
import sys
import json
import asyncio
import time
import wave
import io
import numpy as np
from dotenv import load_dotenv
from dataclasses import dataclass, field

load_dotenv()

SAMPLE_RATE = 16000

# ─────────────────────────────────────────────
# Test Cases
# ─────────────────────────────────────────────

TEST_CASES = {
    "urdu": [
        {"id": "UR-01", "category": "Greeting", "text": "السلام علیکم، آپ کیسے ہیں؟", "difficulty": "easy"},
        {"id": "UR-02", "category": "Common", "text": "میں ٹھیک ہوں، شکریہ", "difficulty": "easy"},
        {"id": "UR-03", "category": "Medical", "text": "مریض کو بخار ہے اور سر درد بھی ہے", "difficulty": "medium"},
        {"id": "UR-04", "category": "Medical", "text": "بلڈ پریشر چیک کریں اور شوگر ٹیسٹ بھی", "difficulty": "medium"},
        {"id": "UR-05", "category": "Long", "text": "مریض کل رات سے بہت بیمار ہے، اسے بخار ہے، کھانسی ہے اور گلے میں درد بھی ہے", "difficulty": "hard"},
        {"id": "UR-06", "category": "Code-Switch", "text": "پیشنٹ کا blood pressure بہت high ہے", "difficulty": "hard"},
        {"id": "UR-07", "category": "Numbers", "text": "مریض کی عمر پچاس سال ہے اور وزن ستر کلو ہے", "difficulty": "medium"},
        {"id": "UR-08", "category": "Medical", "text": "ذیابیطس کی دوائی صبح اور شام لیں", "difficulty": "hard"},
        {"id": "UR-09", "category": "Conversational", "text": "آج موسم بہت اچھا ہے، باہر چلتے ہیں", "difficulty": "easy"},
        {"id": "UR-10", "category": "Medical", "text": "ایکسرے رپورٹ نارمل ہے لیکن خون کے ٹیسٹ میں انفیکشن ہے", "difficulty": "hard"},
    ],
    "english": [
        {"id": "EN-01", "category": "Greeting", "text": "Hello, how are you doing today?", "difficulty": "easy"},
        {"id": "EN-02", "category": "Common", "text": "I am fine, thank you very much", "difficulty": "easy"},
        {"id": "EN-03", "category": "Medical", "text": "The patient has a fever and headache", "difficulty": "medium"},
        {"id": "EN-04", "category": "Medical", "text": "Check blood pressure and run a sugar test", "difficulty": "medium"},
        {"id": "EN-05", "category": "Long", "text": "The patient has been very sick since last night with fever, cough, and sore throat", "difficulty": "hard"},
        {"id": "EN-06", "category": "Medical", "text": "Patient presents with acute myocardial infarction and requires immediate intervention", "difficulty": "hard"},
        {"id": "EN-07", "category": "Numbers", "text": "The patient is fifty years old and weighs seventy kilograms", "difficulty": "medium"},
        {"id": "EN-08", "category": "Medical", "text": "Take the diabetes medication in the morning and evening", "difficulty": "medium"},
        {"id": "EN-09", "category": "Conversational", "text": "The weather is very nice today, let's go outside", "difficulty": "easy"},
        {"id": "EN-10", "category": "Medical", "text": "X-ray report is normal but blood test shows infection", "difficulty": "hard"},
    ],
}


@dataclass
class BenchmarkResult:
    provider: str
    test_id: str
    expected: str
    actual: str
    latency_ms: float = 0.0
    word_error_rate: float = 0.0
    character_error_rate: float = 0.0
    passed: bool = False


def calculate_wer(reference: str, hypothesis: str) -> float:
    """Calculate Word Error Rate using dynamic programming."""
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()

    if len(ref_words) == 0:
        return 1.0 if len(hyp_words) > 0 else 0.0

    d = np.zeros((len(ref_words) + 1, len(hyp_words) + 1), dtype=np.uint32)
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j

    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,      # deletion
                    d[i][j - 1] + 1,      # insertion
                    d[i - 1][j - 1] + 1,  # substitution
                )

    return d[len(ref_words)][len(hyp_words)] / len(ref_words)


def calculate_cer(reference: str, hypothesis: str) -> float:
    """Calculate Character Error Rate."""
    ref_chars = list(reference.lower().replace(" ", ""))
    hyp_chars = list(hypothesis.lower().replace(" ", ""))

    if len(ref_chars) == 0:
        return 1.0 if len(hyp_chars) > 0 else 0.0

    d = np.zeros((len(ref_chars) + 1, len(hyp_chars) + 1), dtype=np.uint32)
    for i in range(len(ref_chars) + 1):
        d[i][0] = i
    for j in range(len(hyp_chars) + 1):
        d[0][j] = j

    for i in range(1, len(ref_chars) + 1):
        for j in range(1, len(hyp_chars) + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                d[i][j] = d[i - 1][j - 1]
            else:
                d[i][j] = min(
                    d[i - 1][j] + 1,
                    d[i][j - 1] + 1,
                    d[i - 1][j - 1] + 1,
                )

    return d[len(ref_chars)][len(hyp_chars)] / len(ref_chars)


# ─────────────────────────────────────────────
# Provider Transcription Functions
# ─────────────────────────────────────────────

async def transcribe_deepgram(audio_bytes: bytes, language: str, api_key: str) -> tuple[str, float]:
    """Send audio to Deepgram and get transcription with latency."""
    import websockets

    url = (
        f"wss://api.deepgram.com/v1/listen"
        f"?model=nova-3"
        f"&language={language}"
        f"&encoding=linear16"
        f"&sample_rate={SAMPLE_RATE}"
        f"&channels=1"
        f"&punctuate=true"
    )
    headers = {"Authorization": f"Token {api_key}"}

    start = time.monotonic()
    transcript = ""

    try:
        async with websockets.connect(url, extra_headers=headers) as ws:
            await ws.send(audio_bytes)
            await ws.send(json.dumps({"type": "CloseStream"}))

            async for msg in ws:
                data = json.loads(msg)
                ch = data.get("channel", {})
                alt = ch.get("alternatives", [{}])[0]
                text = alt.get("transcript", "")
                is_final = data.get("is_final", False)
                if text and is_final:
                    transcript += text + " "
    except Exception as e:
        transcript = f"[Error: {e}]"

    latency = (time.monotonic() - start) * 1000
    return transcript.strip(), latency


async def transcribe_soniox(audio_bytes: bytes, language: str, api_key: str) -> tuple[str, float]:
    """Send audio to Soniox and get transcription."""
    import websockets

    start = time.monotonic()
    transcript = ""

    try:
        async with websockets.connect("wss://api.soniox.com/transcribe-websocket") as ws:
            config = {
                "api_key": api_key,
                "sample_rate_hertz": SAMPLE_RATE,
                "audio_format": "pcm_s16le",
                "num_audio_channels": 1,
                "language_hints": [language],
                "model": "soniox-multilingual",
            }
            await ws.send(json.dumps(config))
            await ws.send(audio_bytes)
            await ws.send(json.dumps({"type": "eof"}))

            async for msg in ws:
                data = json.loads(msg)
                words = data.get("words", [])
                if words:
                    final_words = [w["text"] for w in words if w.get("is_final", True)]
                    if final_words:
                        transcript += " ".join(final_words) + " "
                if data.get("finished"):
                    break
    except Exception as e:
        transcript = f"[Error: {e}]"

    latency = (time.monotonic() - start) * 1000
    return transcript.strip(), latency


def transcribe_whisper_api(audio_bytes: bytes, language: str) -> tuple[str, float]:
    """Send audio to OpenAI Whisper API."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(audio_bytes)
        wav_buffer.seek(0)
        wav_buffer.name = "audio.wav"

        start = time.monotonic()
        result = client.audio.transcriptions.create(
            model="whisper-1",
            file=wav_buffer,
            language=language,
        )
        latency = (time.monotonic() - start) * 1000
        return result.text.strip(), latency
    except Exception as e:
        return f"[Error: {e}]", 0.0


# ─────────────────────────────────────────────
# Generate Test Audio (TTS)
# ─────────────────────────────────────────────

def generate_test_audio_tts(text: str, language: str = "ur") -> bytes:
    """Generate test audio using OpenAI TTS, returns PCM int16 bytes."""
    try:
        from openai import OpenAI
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text,
            response_format="pcm",
            speed=0.9,
        )

        # OpenAI PCM is 24kHz, resample to 16kHz
        pcm_24k = np.frombuffer(response.content, dtype=np.int16)
        # Simple downsample from 24kHz to 16kHz (take every 1.5th sample)
        indices = np.round(np.arange(0, len(pcm_24k), 24000 / SAMPLE_RATE)).astype(int)
        indices = indices[indices < len(pcm_24k)]
        pcm_16k = pcm_24k[indices]

        return pcm_16k.tobytes()
    except Exception as e:
        print(f"TTS Error: {e}")
        # Return 3 seconds of silence as fallback
        return np.zeros(SAMPLE_RATE * 3, dtype=np.int16).tobytes()


# ─────────────────────────────────────────────
# Run Benchmark
# ─────────────────────────────────────────────

async def run_benchmark(language: str = "urdu", use_tts: bool = True):
    """Run full benchmark suite."""
    test_cases = TEST_CASES.get(language, [])
    if not test_cases:
        print(f"No test cases for language: {language}")
        return

    lang_code_map = {
        "urdu": {"deepgram": "ur", "soniox": "ur", "whisper": "ur"},
        "english": {"deepgram": "en", "soniox": "en", "whisper": "en"},
    }
    lang_codes = lang_code_map.get(language, lang_code_map["urdu"])

    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "")
    soniox_key = os.getenv("SONIOX_API_KEY", "")
    openai_key = os.getenv("OPENAI_API_KEY", "")

    results: list[BenchmarkResult] = []

    print(f"\n{'='*80}")
    print(f"  STT BENCHMARK — Language: {language.upper()}")
    print(f"  Test Cases: {len(test_cases)}")
    print(f"  Providers: Deepgram={'✓' if deepgram_key else '✗'}  "
          f"Soniox={'✓' if soniox_key else '✗'}  "
          f"Whisper={'✓' if openai_key else '✗'}")
    print(f"{'='*80}\n")

    for i, tc in enumerate(test_cases):
        print(f"[{i+1}/{len(test_cases)}] {tc['id']} ({tc['category']}, {tc['difficulty']})")
        print(f"  Expected: {tc['text']}")

        # Generate or load test audio
        if use_tts:
            print("  Generating audio via TTS...")
            audio_bytes = generate_test_audio_tts(tc["text"], lang_codes["whisper"])
        else:
            audio_bytes = np.zeros(SAMPLE_RATE * 3, dtype=np.int16).tobytes()

        # Test each provider
        if deepgram_key:
            text, latency = await transcribe_deepgram(audio_bytes, lang_codes["deepgram"], deepgram_key)
            wer = calculate_wer(tc["text"], text)
            cer = calculate_cer(tc["text"], text)
            result = BenchmarkResult(
                provider="Deepgram Nova-3",
                test_id=tc["id"],
                expected=tc["text"],
                actual=text,
                latency_ms=latency,
                word_error_rate=wer,
                character_error_rate=cer,
                passed=wer < 0.5,
            )
            results.append(result)
            print(f"  Deepgram:  {text[:80]}{'...' if len(text)>80 else ''}")
            print(f"             WER={wer:.2%} CER={cer:.2%} Latency={latency:.0f}ms")

        if soniox_key:
            text, latency = await transcribe_soniox(audio_bytes, lang_codes["soniox"], soniox_key)
            wer = calculate_wer(tc["text"], text)
            cer = calculate_cer(tc["text"], text)
            result = BenchmarkResult(
                provider="Soniox",
                test_id=tc["id"],
                expected=tc["text"],
                actual=text,
                latency_ms=latency,
                word_error_rate=wer,
                character_error_rate=cer,
                passed=wer < 0.5,
            )
            results.append(result)
            print(f"  Soniox:    {text[:80]}{'...' if len(text)>80 else ''}")
            print(f"             WER={wer:.2%} CER={cer:.2%} Latency={latency:.0f}ms")

        if openai_key:
            text, latency = transcribe_whisper_api(audio_bytes, lang_codes["whisper"])
            wer = calculate_wer(tc["text"], text)
            cer = calculate_cer(tc["text"], text)
            result = BenchmarkResult(
                provider="Whisper API",
                test_id=tc["id"],
                expected=tc["text"],
                actual=text,
                latency_ms=latency,
                word_error_rate=wer,
                character_error_rate=cer,
                passed=wer < 0.5,
            )
            results.append(result)
            print(f"  Whisper:   {text[:80]}{'...' if len(text)>80 else ''}")
            print(f"             WER={wer:.2%} CER={cer:.2%} Latency={latency:.0f}ms")

        print()

    # ─────────────────────────────────────────
    # Generate Summary Report
    # ─────────────────────────────────────────
    print_report(results, language)
    save_report(results, language)


def print_report(results: list[BenchmarkResult], language: str):
    """Print formatted benchmark report."""
    providers = sorted(set(r.provider for r in results))

    print(f"\n{'='*100}")
    print(f"  BENCHMARK RESULTS — {language.upper()}")
    print(f"{'='*100}")

    # Summary table
    print(f"\n{'Provider':<20} {'Avg WER':<12} {'Avg CER':<12} {'Avg Latency':<15} {'Pass Rate':<12} {'Tests':<8}")
    print(f"{'-'*79}")

    for provider in providers:
        pr = [r for r in results if r.provider == provider]
        avg_wer = sum(r.word_error_rate for r in pr) / len(pr) if pr else 0
        avg_cer = sum(r.character_error_rate for r in pr) / len(pr) if pr else 0
        avg_lat = sum(r.latency_ms for r in pr) / len(pr) if pr else 0
        pass_rate = sum(1 for r in pr if r.passed) / len(pr) * 100 if pr else 0
        print(f"{provider:<20} {avg_wer:<12.2%} {avg_cer:<12.2%} {avg_lat:<15.0f}ms {pass_rate:<12.1f}% {len(pr):<8}")

    # By category
    categories = sorted(set(r.test_id.split("-")[0] + "-" + str(i) for i, r in enumerate(results)))
    print(f"\n{'─'*100}")
    print(f"  BY DIFFICULTY")
    print(f"{'─'*100}")

    for difficulty in ["easy", "medium", "hard"]:
        test_ids_for_diff = [tc["id"] for tc in TEST_CASES.get(language, []) if tc["difficulty"] == difficulty]
        print(f"\n  [{difficulty.upper()}]")
        print(f"  {'Provider':<20} {'Avg WER':<12} {'Avg CER':<12} {'Pass Rate':<12}")
        print(f"  {'-'*56}")
        for provider in providers:
            pr = [r for r in results if r.provider == provider and r.test_id in test_ids_for_diff]
            if pr:
                avg_wer = sum(r.word_error_rate for r in pr) / len(pr)
                avg_cer = sum(r.character_error_rate for r in pr) / len(pr)
                pass_rate = sum(1 for r in pr if r.passed) / len(pr) * 100
                print(f"  {provider:<20} {avg_wer:<12.2%} {avg_cer:<12.2%} {pass_rate:<12.1f}%")

    # By category
    print(f"\n{'─'*100}")
    print(f"  BY CATEGORY")
    print(f"{'─'*100}")

    for category in sorted(set(tc["category"] for tc in TEST_CASES.get(language, []))):
        test_ids_for_cat = [tc["id"] for tc in TEST_CASES.get(language, []) if tc["category"] == category]
        print(f"\n  [{category.upper()}]")
        print(f"  {'Provider':<20} {'Avg WER':<12} {'Pass Rate':<12}")
        print(f"  {'-'*44}")
        for provider in providers:
            pr = [r for r in results if r.provider == provider and r.test_id in test_ids_for_cat]
            if pr:
                avg_wer = sum(r.word_error_rate for r in pr) / len(pr)
                pass_rate = sum(1 for r in pr if r.passed) / len(pr) * 100
                print(f"  {provider:<20} {avg_wer:<12.2%} {pass_rate:<12.1f}%")

    # Performance table
    print(f"\n{'─'*100}")
    print(f"  PERFORMANCE (LATENCY)")
    print(f"{'─'*100}")
    print(f"\n  {'Provider':<20} {'Min':<12} {'Avg':<12} {'Max':<12} {'P95':<12}")
    print(f"  {'-'*56}")

    for provider in providers:
        pr = [r for r in results if r.provider == provider]
        lats = sorted(r.latency_ms for r in pr)
        if lats:
            p95_idx = int(len(lats) * 0.95)
            print(f"  {provider:<20} {lats[0]:<12.0f} {sum(lats)/len(lats):<12.0f} {lats[-1]:<12.0f} {lats[min(p95_idx, len(lats)-1)]:<12.0f}")

    print(f"\n{'='*100}\n")


def save_report(results: list[BenchmarkResult], language: str):
    """Save results to JSON."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "output")
    os.makedirs(output_dir, exist_ok=True)

    report_path = os.path.join(output_dir, f"benchmark_{language}.json")
    data = {
        "language": language,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": [
            {
                "provider": r.provider,
                "test_id": r.test_id,
                "expected": r.expected,
                "actual": r.actual,
                "latency_ms": r.latency_ms,
                "word_error_rate": float(r.word_error_rate),
                "character_error_rate": float(r.character_error_rate),
                "passed": bool(r.passed),
            }
            for r in results
        ],
    }

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="STT Benchmark Suite")
    parser.add_argument("--language", default="urdu", choices=["urdu", "english"])
    parser.add_argument("--no-tts", action="store_true", help="Skip TTS audio generation (use silence)")
    args = parser.parse_args()

    asyncio.run(run_benchmark(language=args.language, use_tts=not args.no_tts))
