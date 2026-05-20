"""
Transcribe Urdu audio recordings using OpenAI gpt-4o-transcribe.
Usage: python transcribe_openai.py [folder_path]
Default folder: ../input/recordings/3551
"""

import os, sys, json, pathlib
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", "..", ".env"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
assert len(OPENAI_API_KEY) > 0, "OPENAI_API_KEY not found in .env"

DEFAULT_FOLDER = os.path.join(os.path.dirname(__file__), "..", "input", "recordings", "3551")
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm", ".mp4"}


def transcribe_file(filepath: str) -> dict:
    from openai import OpenAI

    client = OpenAI(api_key=OPENAI_API_KEY)

    with open(filepath, "rb") as f:
        response = client.audio.transcriptions.create(
            model="gpt-4o-transcribe",
            file=f,
            language="ur",
            response_format="json",
        )

    return response.model_dump()


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FOLDER
    folder = os.path.abspath(folder)

    if not os.path.isdir(folder):
        print(f"Folder not found: {folder}")
        sys.exit(1)

    files = sorted(
        f for f in os.listdir(folder)
        if pathlib.Path(f).suffix.lower() in AUDIO_EXTENSIONS
    )

    if not files:
        print(f"No audio files found in {folder}")
        sys.exit(1)

    print(f"Found {len(files)} audio file(s) in {folder}")
    print(f"Model: gpt-4o-transcribe | Language: ur")
    print("=" * 80)

    all_results = []

    for i, filename in enumerate(files, 1):
        filepath = os.path.join(folder, filename)
        print(f"\n[{i}/{len(files)}] Transcribing: {filename}")
        print("-" * 60)

        try:
            result = transcribe_file(filepath)
            transcript = result.get("text", "")
            duration = result.get("duration", 0)

            print(f"Duration: {duration:.1f}s")
            print(f"Transcript:\n{transcript}\n")

            segments = result.get("segments", [])
            if segments:
                print("Segments:")
                for seg in segments:
                    start = seg.get("start", 0)
                    end = seg.get("end", 0)
                    text = seg.get("text", "").strip()
                    print(f"  [{start:.1f}s - {end:.1f}s] {text}")

            all_results.append({
                "file": filename,
                "transcript": transcript,
                "duration": duration,
                "segments": segments,
                "full_response": result,
            })

        except Exception as e:
            print(f"Error: {e}")
            all_results.append({"file": filename, "error": str(e)})

    output_path = os.path.join(folder, "transcriptions_openai.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_path}")
    print(f"Total files: {len(files)}, Success: {sum(1 for r in all_results if 'error' not in r)}")


if __name__ == "__main__":
    main()
