"""
Transcribe Urdu audio recordings using Deepgram Nova-3.
Usage: python transcribe_recordings.py [folder_path]
Default folder: input/recordings/3551
"""

import os, sys, json, asyncio, pathlib
from dotenv import load_dotenv

load_dotenv()

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
assert len(DEEPGRAM_API_KEY) > 0, "DEEPGRAM_API_KEY not found in .env"

DEFAULT_FOLDER = os.path.join(os.path.dirname(__file__), "input", "recordings", "3551")
AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm", ".mp4"}


async def transcribe_file(filepath: str) -> dict:
    from deepgram import DeepgramClient, PrerecordedOptions

    client = DeepgramClient(DEEPGRAM_API_KEY)

    with open(filepath, "rb") as f:
        audio_data = f.read()

    options = PrerecordedOptions(
        model="nova-3",
        language="ur",
        smart_format=True,
        punctuate=True,
        utterances=True,
        paragraphs=True,
    )

    response = await client.listen.asyncrest.v("1").transcribe_file(
        {"buffer": audio_data, "mimetype": _mimetype(filepath)},
        options,
    )
    return response.to_dict()


def _mimetype(filepath: str) -> str:
    ext = pathlib.Path(filepath).suffix.lower()
    return {
        ".mp3": "audio/mpeg",
        ".wav": "audio/wav",
        ".ogg": "audio/ogg",
        ".flac": "audio/flac",
        ".m4a": "audio/mp4",
        ".webm": "audio/webm",
        ".mp4": "audio/mp4",
    }.get(ext, "audio/mpeg")


async def main():
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

    print(f"Found {len(files)} audio file(s) in {folder}\n")
    print("=" * 80)

    all_results = []

    for i, filename in enumerate(files, 1):
        filepath = os.path.join(folder, filename)
        print(f"\n[{i}/{len(files)}] Transcribing: {filename}")
        print("-" * 60)

        try:
            result = await transcribe_file(filepath)
            channels = result.get("results", {}).get("channels", [])

            if channels:
                transcript = channels[0]["alternatives"][0]["transcript"]
                confidence = channels[0]["alternatives"][0]["confidence"]
                print(f"Confidence: {confidence:.2%}")
                print(f"Transcript:\n{transcript}\n")

                utterances = result.get("results", {}).get("utterances", [])
                if utterances:
                    print("Utterances:")
                    for u in utterances:
                        start = u["start"]
                        end = u["end"]
                        text = u["transcript"]
                        conf = u["confidence"]
                        print(f"  [{start:.1f}s - {end:.1f}s] ({conf:.2%}) {text}")
            else:
                print("No transcription returned.")
                transcript = ""
                confidence = 0

            all_results.append({
                "file": filename,
                "transcript": transcript,
                "confidence": confidence,
                "full_response": result,
            })

        except Exception as e:
            print(f"Error: {e}")
            all_results.append({"file": filename, "error": str(e)})

    output_path = os.path.join(folder, "transcriptions.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 80)
    print(f"Results saved to: {output_path}")
    print(f"Total files: {len(files)}, Success: {sum(1 for r in all_results if 'error' not in r)}")


if __name__ == "__main__":
    asyncio.run(main())
