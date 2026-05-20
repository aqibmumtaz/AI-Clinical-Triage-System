"""
Convert fine-tuned Whisper model to Faster-Whisper (CTranslate2) format.

Usage:
  # Float16 (best accuracy)
  python convert_faster_whisper.py \
      --model_dir ./whisper-large-v3-turbo-urdu/best \
      --output_dir ./faster-whisper-urdu \
      --quantization float16

  # Int8+Float16 (smaller, faster, minimal accuracy loss)
  python convert_faster_whisper.py \
      --model_dir ./whisper-large-v3-turbo-urdu/best \
      --output_dir ./faster-whisper-urdu-int8 \
      --quantization int8_float16

  # Test the converted model
  python convert_faster_whisper.py \
      --model_dir ./faster-whisper-urdu \
      --test_audio ../input/recordings/3551/some_file.mp3

Requirements:
  pip install ctranslate2 faster-whisper transformers
"""

import os, sys, argparse, logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def convert(model_dir: str, output_dir: str, quantization: str):
    import ctranslate2

    log.info(f"Converting {model_dir} → {output_dir} (quantization={quantization})")

    converter = ctranslate2.converters.TransformersConverter(
        model_name_or_path=model_dir,
    )
    converter.convert(
        output_dir=output_dir,
        quantization=quantization,
        force=True,
    )

    # Copy tokenizer and processor files needed by faster-whisper
    import shutil
    for fname in ["tokenizer.json", "preprocessor_config.json", "config.json",
                   "generation_config.json", "special_tokens_map.json",
                   "vocab.json", "merges.txt", "added_tokens.json",
                   "normalizer.json"]:
        src = os.path.join(model_dir, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(output_dir, fname))

    log.info(f"Conversion complete: {output_dir}")

    model_size_mb = sum(
        os.path.getsize(os.path.join(output_dir, f))
        for f in os.listdir(output_dir)
        if os.path.isfile(os.path.join(output_dir, f))
    ) / (1024 * 1024)
    log.info(f"Model size: {model_size_mb:.0f} MB")


def test_model(model_dir: str, audio_path: str):
    from faster_whisper import WhisperModel
    import time

    log.info(f"Loading model from {model_dir}")
    model = WhisperModel(model_dir, device="auto", compute_type="auto")

    log.info(f"Transcribing: {audio_path}")
    start = time.time()
    segments, info = model.transcribe(audio_path, language="ur", beam_size=5)

    print(f"\nLanguage: {info.language} (prob={info.language_probability:.2f})")
    print(f"Duration: {info.duration:.1f}s\n")

    full_text = []
    for seg in segments:
        print(f"  [{seg.start:.1f}s → {seg.end:.1f}s] {seg.text}")
        full_text.append(seg.text)

    elapsed = time.time() - start
    rtf = elapsed / info.duration if info.duration > 0 else 0
    print(f"\nRTF: {rtf:.3f} ({'real-time' if rtf < 1 else 'slower than real-time'})")
    print(f"Full transcript:\n{''.join(full_text)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--quantization", default="float16",
                        choices=["float16", "int8_float16", "int8", "float32"])
    parser.add_argument("--test_audio", default=None, help="Test audio file path")
    args = parser.parse_args()

    if args.test_audio:
        test_model(args.model_dir, args.test_audio)
    else:
        if not args.output_dir:
            args.output_dir = args.model_dir + f"-ct2-{args.quantization}"
        convert(args.model_dir, args.output_dir, args.quantization)


if __name__ == "__main__":
    main()
