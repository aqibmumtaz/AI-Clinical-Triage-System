"""
Transcribe Urdu audio recordings using Faster-Whisper (local).
Usage: python transcribe_whisper.py [folder_path] [--model large-v3-turbo]
Default folder: ../input/recordings/3551
"""

import os, sys, json, time, pathlib, argparse

DEFAULT_FOLDER = os.path.join(os.path.dirname(__file__), "..", "input", "recordings", "3551")
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm", ".mp4"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=DEFAULT_FOLDER)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--compute_type", default="auto")
    args = parser.parse_args()

    from faster_whisper import WhisperModel

    folder = os.path.abspath(args.folder)
    files = sorted(
        f for f in os.listdir(folder)
        if pathlib.Path(f).suffix.lower() in AUDIO_EXTENSIONS
    )

    print(f"Found {len(files)} audio file(s) in {folder}")
    print(f"Model: whisper-{args.model} | Device: {args.device}")
    print(f"Loading model (first run downloads ~3GB)...")

    model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
    print("Model loaded.\n" + "=" * 80)

    all_results = []

    for i, filename in enumerate(files, 1):
        filepath = os.path.join(folder, filename)
        print(f"\n[{i}/{len(files)}] Transcribing: {filename}")
        print("-" * 60)

        start_time = time.time()
        segments, info = model.transcribe(
            filepath,
            language="ur",
            beam_size=5,
            best_of=5,
            temperature=0.0,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        seg_list = []
        full_text = []

        for seg in segments:
            seg_list.append({
                "start": round(seg.start, 1),
                "end": round(seg.end, 1),
                "text": seg.text.strip(),
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
            })
            full_text.append(seg.text.strip())
            print(f"  [{seg.start:.1f}s - {seg.end:.1f}s] {seg.text.strip()}")

        elapsed = time.time() - start_time
        rtf = elapsed / info.duration if info.duration > 0 else 0
        transcript = " ".join(full_text)

        print(f"\nDuration: {info.duration:.1f}s | Inference: {elapsed:.1f}s | RTF: {rtf:.3f}")
        print(f"Language: {info.language} (prob={info.language_probability:.2f})")
        print(f"\nFull transcript:\n{transcript}")

        all_results.append({
            "file": filename,
            "transcript": transcript,
            "duration": round(info.duration, 1),
            "inference_time": round(elapsed, 1),
            "rtf": round(rtf, 3),
            "language_prob": round(info.language_probability, 2),
            "segments": seg_list,
        })

    output_path = os.path.join(folder, f"transcriptions_whisper_{args.model}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_path}")
    print(f"Total files: {len(files)}, Model: whisper-{args.model}")


if __name__ == "__main__":
    main()
