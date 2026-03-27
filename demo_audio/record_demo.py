"""Record patient audio clips for demo video."""
import sounddevice as sd
import numpy as np
import wave
import os

SAMPLE_RATE = 16000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "input")

PROMPTS = [
    ("01_greeting.wav", "Say: السلام علیکم (Assalam o Alaikum)"),
    ("02_name.wav", "Say: میرا نام عمران ہے (Mera naam Imran hai)"),
    ("03_symptom.wav", "Say: مجھے سینے میں درد ہو رہا ہے (Mujhe seene mein dard ho raha hai)"),
    ("04_duration.wav", "Say: کل رات سے (Kal raat se)"),
    ("05_confirm.wav", "Say: جی ہاں، بلڈ پریشر نارمل ہے (Ji haan, blood pressure normal hai)"),
    ("06_thanks.wav", "Say: شکریہ (Shukriya)"),
]


def record_clip(filename, prompt):
    filepath = os.path.join(OUTPUT_DIR, filename)
    print(f"\n{'='*60}")
    print(f"  CLIP: {filename}")
    print(f"  {prompt}")
    print(f"{'='*60}")
    input("  Press ENTER when ready to record (5 seconds)... ")

    print("  🔴 RECORDING... speak now!")
    audio = sd.rec(int(5 * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    print("  ✅ Done!")

    with wave.open(filepath, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())

    print(f"  Saved: {filepath}")


if __name__ == "__main__":
    print("\n🎤 DEMO AUDIO RECORDER")
    print("You will record 6 short clips (5 seconds each).\n")

    for filename, prompt in PROMPTS:
        record_clip(filename, prompt)

    print(f"\n✅ All {len(PROMPTS)} clips recorded in: {OUTPUT_DIR}")
    print("Files:", [p[0] for p in PROMPTS])
