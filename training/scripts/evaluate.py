"""
Benchmark all trained models on the test set.

Usage:
  python evaluate.py \
      --test_dir ./whisper_dataset/test \
      --models whisper:./whisper-large-v3-urdu/best \
              whisper-turbo:./whisper-large-v3-turbo-urdu/best \
              faster:./faster-whisper-urdu \
              deepgram:api

Output: benchmark_results.json + comparison table
"""

import os, json, time, argparse, logging
import evaluate as hf_evaluate

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def load_test_data(test_dir: str) -> list[dict]:
    metadata_path = os.path.join(test_dir, "metadata.jsonl")
    samples = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line.strip())
            entry["audio_path"] = os.path.join(test_dir, entry["audio"])
            samples.append(entry)
    return samples


def eval_whisper_hf(model_path: str, samples: list[dict]) -> dict:
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    import librosa, torch

    processor = WhisperProcessor.from_pretrained(model_path)
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    if torch.cuda.is_available():
        model = model.to("cuda")
    model.eval()

    predictions = []
    total_audio_dur = 0
    total_inference_time = 0

    for s in samples:
        audio, sr = librosa.load(s["audio_path"], sr=16000)
        total_audio_dur += len(audio) / sr

        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        start = time.time()
        with torch.no_grad():
            predicted_ids = model.generate(**inputs, language="ur", task="transcribe")
        elapsed = time.time() - start
        total_inference_time += elapsed

        text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        predictions.append(text)

    return {
        "predictions": predictions,
        "total_audio_seconds": total_audio_dur,
        "total_inference_seconds": total_inference_time,
        "rtf": total_inference_time / total_audio_dur if total_audio_dur > 0 else 0,
    }


def eval_faster_whisper(model_path: str, samples: list[dict]) -> dict:
    from faster_whisper import WhisperModel

    model = WhisperModel(model_path, device="auto", compute_type="auto")

    predictions = []
    total_audio_dur = 0
    total_inference_time = 0

    for s in samples:
        start = time.time()
        segments, info = model.transcribe(s["audio_path"], language="ur", beam_size=5)
        text = "".join(seg.text for seg in segments)
        elapsed = time.time() - start

        total_audio_dur += info.duration
        total_inference_time += elapsed
        predictions.append(text)

    return {
        "predictions": predictions,
        "total_audio_seconds": total_audio_dur,
        "total_inference_seconds": total_inference_time,
        "rtf": total_inference_time / total_audio_dur if total_audio_dur > 0 else 0,
    }


def eval_deepgram(samples: list[dict]) -> dict:
    import asyncio
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

    from deepgram import DeepgramClient, PrerecordedOptions

    api_key = os.getenv("DEEPGRAM_API_KEY", "")
    client = DeepgramClient(api_key)

    predictions = []
    total_audio_dur = 0
    total_inference_time = 0

    async def transcribe(filepath):
        with open(filepath, "rb") as f:
            audio_data = f.read()
        options = PrerecordedOptions(model="nova-3", language="ur", smart_format=True)
        response = await client.listen.asyncrest.v("1").transcribe_file(
            {"buffer": audio_data, "mimetype": "audio/wav"}, options,
        )
        return response.to_dict()

    for s in samples:
        start = time.time()
        result = asyncio.run(transcribe(s["audio_path"]))
        elapsed = time.time() - start

        channels = result.get("results", {}).get("channels", [])
        text = channels[0]["alternatives"][0]["transcript"] if channels else ""
        duration = result.get("metadata", {}).get("duration", 0)

        predictions.append(text)
        total_audio_dur += duration
        total_inference_time += elapsed

    return {
        "predictions": predictions,
        "total_audio_seconds": total_audio_dur,
        "total_inference_seconds": total_inference_time,
        "rtf": total_inference_time / total_audio_dur if total_audio_dur > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True, help="Test dataset dir with metadata.jsonl")
    parser.add_argument("--models", nargs="+", required=True,
                        help="Model specs: name:path (e.g. whisper:./model faster:./model deepgram:api)")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit samples for quick test")
    args = parser.parse_args()

    samples = load_test_data(args.test_dir)
    if args.max_samples:
        samples = samples[:args.max_samples]

    references = [s["sentence"] for s in samples]
    log.info(f"Evaluating on {len(samples)} test samples")

    wer_metric = hf_evaluate.load("wer")
    cer_metric = hf_evaluate.load("cer")

    results = {}

    for model_spec in args.models:
        name, path = model_spec.split(":", 1)
        log.info(f"\nEvaluating: {name} ({path})")

        if name.startswith("faster"):
            output = eval_faster_whisper(path, samples)
        elif name == "deepgram":
            output = eval_deepgram(samples)
        else:
            output = eval_whisper_hf(path, samples)

        wer = wer_metric.compute(predictions=output["predictions"], references=references)
        cer = cer_metric.compute(predictions=output["predictions"], references=references)

        results[name] = {
            "wer": round(wer * 100, 2),
            "cer": round(cer * 100, 2),
            "rtf": round(output["rtf"], 4),
            "total_audio_hours": round(output["total_audio_seconds"] / 3600, 2),
            "total_inference_seconds": round(output["total_inference_seconds"], 1),
        }

        log.info(f"  WER: {results[name]['wer']:.2f}%")
        log.info(f"  CER: {results[name]['cer']:.2f}%")
        log.info(f"  RTF: {results[name]['rtf']:.4f}")

    log.info("\n" + "=" * 70)
    log.info(f"{'Model':<25} {'WER%':>8} {'CER%':>8} {'RTF':>8}")
    log.info("-" * 70)
    for name, r in sorted(results.items(), key=lambda x: x[1]["wer"]):
        log.info(f"{name:<25} {r['wer']:>8.2f} {r['cer']:>8.2f} {r['rtf']:>8.4f}")

    output_path = os.path.join(os.path.dirname(args.test_dir), "benchmark_results.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
