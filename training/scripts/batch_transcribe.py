"""
Batch transcribe all call recordings using Deepgram Nova-3.
Handles 30,000+ calls with async concurrency, retry logic, and progress tracking.

Usage:
  python batch_transcribe.py --input_dir /path/to/recordings --output_dir /path/to/transcriptions
  python batch_transcribe.py --input_dir ../input/recordings --max_concurrent 20
"""

import os, sys, json, asyncio, argparse, pathlib, time, logging
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY", "")
assert len(DEEPGRAM_API_KEY) > 0, "DEEPGRAM_API_KEY not found in .env"

AUDIO_EXTENSIONS = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".webm", ".mp4"}
MAX_RETRIES = 3

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def find_audio_files(input_dir: str) -> list[str]:
    files = []
    for root, _, filenames in os.walk(input_dir):
        for f in filenames:
            if pathlib.Path(f).suffix.lower() in AUDIO_EXTENSIONS:
                files.append(os.path.join(root, f))
    return sorted(files)


def mimetype(filepath: str) -> str:
    ext = pathlib.Path(filepath).suffix.lower()
    return {
        ".mp3": "audio/mpeg", ".wav": "audio/wav", ".ogg": "audio/ogg",
        ".flac": "audio/flac", ".m4a": "audio/mp4", ".webm": "audio/webm",
        ".mp4": "audio/mp4",
    }.get(ext, "audio/mpeg")


async def transcribe_one(client, filepath: str, sem: asyncio.Semaphore) -> dict:
    from deepgram import PrerecordedOptions

    async with sem:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
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
                    {"buffer": audio_data, "mimetype": mimetype(filepath)},
                    options,
                )
                result = response.to_dict()

                channels = result.get("results", {}).get("channels", [])
                transcript = ""
                confidence = 0.0
                if channels:
                    transcript = channels[0]["alternatives"][0]["transcript"]
                    confidence = channels[0]["alternatives"][0]["confidence"]

                return {
                    "file": filepath,
                    "transcript": transcript,
                    "confidence": confidence,
                    "utterances": result.get("results", {}).get("utterances", []),
                    "duration": result.get("metadata", {}).get("duration", 0),
                    "full_response": result,
                }

            except Exception as e:
                if attempt < MAX_RETRIES:
                    wait = 2 ** attempt
                    log.warning(f"Retry {attempt}/{MAX_RETRIES} for {filepath}: {e} (waiting {wait}s)")
                    await asyncio.sleep(wait)
                else:
                    log.error(f"Failed after {MAX_RETRIES} attempts: {filepath}: {e}")
                    return {"file": filepath, "error": str(e)}


async def main():
    parser = argparse.ArgumentParser(description="Batch transcribe Urdu audio with Deepgram Nova-3")
    parser.add_argument("--input_dir", required=True, help="Directory containing audio files (recursive)")
    parser.add_argument("--output_dir", default=None, help="Output directory (default: input_dir/transcriptions)")
    parser.add_argument("--max_concurrent", type=int, default=10, help="Max concurrent API calls")
    parser.add_argument("--resume", action="store_true", help="Skip already transcribed files")
    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    output_dir = args.output_dir or os.path.join(input_dir, "transcriptions")
    os.makedirs(output_dir, exist_ok=True)

    files = find_audio_files(input_dir)
    log.info(f"Found {len(files)} audio files in {input_dir}")

    if not files:
        log.error("No audio files found.")
        sys.exit(1)

    progress_file = os.path.join(output_dir, "progress.json")
    completed = set()
    if args.resume and os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            completed = set(json.load(f))
        log.info(f"Resuming: {len(completed)} already done, {len(files) - len(completed)} remaining")

    files_to_process = [f for f in files if f not in completed]

    from deepgram import DeepgramClient
    client = DeepgramClient(DEEPGRAM_API_KEY)
    sem = asyncio.Semaphore(args.max_concurrent)

    results = []
    total = len(files_to_process)
    done = 0
    start_time = time.time()

    batch_size = 100
    for batch_start in range(0, total, batch_size):
        batch = files_to_process[batch_start:batch_start + batch_size]
        tasks = [transcribe_one(client, f, sem) for f in batch]
        batch_results = await asyncio.gather(*tasks)

        for r in batch_results:
            results.append(r)
            completed.add(r["file"])
            done += 1

            if done % 50 == 0:
                elapsed = time.time() - start_time
                rate = done / elapsed * 3600
                eta = (total - done) / (done / elapsed) if done > 0 else 0
                log.info(f"Progress: {done}/{total} ({done/total:.1%}) | Rate: {rate:.0f}/hr | ETA: {eta/60:.0f}min")

        with open(progress_file, "w") as f:
            json.dump(list(completed), f)

        chunk_file = os.path.join(output_dir, f"batch_{batch_start:06d}.json")
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump(batch_results, f, ensure_ascii=False, indent=2)

    all_results_file = os.path.join(output_dir, "all_transcriptions.json")
    with open(all_results_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    elapsed = time.time() - start_time
    success = sum(1 for r in results if "error" not in r)
    total_duration = sum(r.get("duration", 0) for r in results if "error" not in r)

    log.info("=" * 60)
    log.info("BATCH TRANSCRIPTION COMPLETE")
    log.info(f"  Total files:    {total}")
    log.info(f"  Successful:     {success}")
    log.info(f"  Failed:         {total - success}")
    log.info(f"  Total audio:    {total_duration/3600:.1f} hours")
    log.info(f"  Wall time:      {elapsed/3600:.1f} hours")
    log.info(f"  Output:         {all_results_file}")


if __name__ == "__main__":
    asyncio.run(main())
