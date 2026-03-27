"""
Quick diagnostic: test mic → Deepgram → LLM → TTS pipeline step by step.
"""
import os, sys, json, asyncio, time, numpy as np
from dotenv import load_dotenv
load_dotenv()

SAMPLE_RATE = 16000
DEEPGRAM_KEY = os.getenv("DEEPGRAM_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

# ── Step 1: Test which mic works ──
def test_mic():
    import sounddevice as sd
    print("\n=== STEP 1: MIC TEST ===")
    print(f"Available devices:\n{sd.query_devices()}")
    print(f"\nDefault input: {sd.default.device[0]}")

    for dev_id in [1, 3]:  # External Mic, MacBook Mic
        name = sd.query_devices(dev_id)['name']
        print(f"\nTesting device {dev_id}: {name} — speak now for 3 seconds...")
        try:
            audio = sd.rec(int(SAMPLE_RATE * 3), samplerate=SAMPLE_RATE, channels=1, dtype='float32', device=dev_id)
            sd.wait()
            rms = np.sqrt(np.mean(audio ** 2))
            peak = np.max(np.abs(audio))
            print(f"  RMS={rms:.6f}, Peak={peak:.6f}")
            if rms > 0.01:
                print(f"  ✅ Speech detected on device {dev_id}!")
                return dev_id, audio
            else:
                print(f"  ❌ No speech detected (too quiet)")
        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Fallback: try default device
    print(f"\nTrying default device — speak now for 3 seconds...")
    audio = sd.rec(int(SAMPLE_RATE * 3), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
    sd.wait()
    rms = np.sqrt(np.mean(audio ** 2))
    print(f"  RMS={rms:.6f}")
    return sd.default.device[0], audio


# ── Step 2: Test Deepgram with recorded audio ──
async def test_deepgram(audio_data):
    import websockets
    print("\n=== STEP 2: DEEPGRAM STT TEST ===")
    if not DEEPGRAM_KEY:
        print("  ❌ No DEEPGRAM_API_KEY in .env")
        return None

    url = (
        f"wss://api.deepgram.com/v1/listen"
        f"?model=nova-3&language=ur"
        f"&encoding=linear16&sample_rate={SAMPLE_RATE}&channels=1"
        f"&punctuate=true"
    )
    headers = {"Authorization": f"Token {DEEPGRAM_KEY}"}

    try:
        ws = await websockets.connect(url, extra_headers=headers)
        print("  ✅ Connected to Deepgram")

        # Send audio
        audio_bytes = (audio_data[:, 0] * 32768).astype(np.int16).tobytes()
        chunk_size = 8000  # 0.25s chunks
        for i in range(0, len(audio_bytes), chunk_size):
            await ws.send(audio_bytes[i:i+chunk_size])
            await asyncio.sleep(0.05)

        # Send close
        await ws.send(json.dumps({"type": "CloseStream"}))

        # Collect results
        transcript = ""
        async for msg in ws:
            data = json.loads(msg)
            alt = data.get("channel", {}).get("alternatives", [{}])[0]
            text = alt.get("transcript", "").strip()
            is_final = data.get("is_final", False)
            if text:
                print(f"  Deepgram: '{text}' (is_final={is_final})")
                if is_final:
                    transcript += " " + text

        await ws.close()
        transcript = transcript.strip()
        if transcript:
            print(f"  ✅ Final transcript: '{transcript}'")
        else:
            print("  ❌ No transcript received")
        return transcript

    except Exception as e:
        print(f"  ❌ Deepgram error: {e}")
        return None


# ── Step 3: Test LLM ──
def test_llm(text):
    print("\n=== STEP 3: LLM TEST ===")
    if not text:
        text = "السلام علیکم"
        print(f"  (Using fallback text: '{text}')")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "آپ ایک مددگار اردو اسسٹنٹ ہیں۔ ہمیشہ اردو میں جواب دیں۔"},
                {"role": "user", "content": text},
            ],
            max_tokens=200,
        )
        reply = response.choices[0].message.content.strip()
        print(f"  ✅ LLM reply: '{reply}'")
        return reply
    except Exception as e:
        print(f"  ❌ LLM error: {e}")
        return None


# ── Step 4: Test TTS ──
def test_tts(text):
    print("\n=== STEP 4: TTS TEST ===")
    if not text:
        text = "وعلیکم السلام"
        print(f"  (Using fallback text: '{text}')")

    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_KEY)
    try:
        response = client.audio.speech.create(
            model="tts-1", voice="alloy", input=text, response_format="mp3",
        )
        path = os.path.join(os.path.dirname(__file__), "output", "test_response.mp3")
        response.stream_to_file(path)
        size = os.path.getsize(path)
        print(f"  ✅ TTS audio saved: {path} ({size} bytes)")
        return path
    except Exception as e:
        print(f"  ❌ TTS error: {e}")
        return None


# ── Run all ──
if __name__ == "__main__":
    print("=" * 50)
    print("CONVERSATION AGENT PIPELINE DIAGNOSTIC")
    print("=" * 50)

    dev_id, audio = test_mic()
    print(f"\n  → Using device {dev_id}")

    transcript = asyncio.run(test_deepgram(audio))
    reply = test_llm(transcript)
    test_tts(reply)

    print("\n" + "=" * 50)
    print("DIAGNOSTIC COMPLETE")
    print(f"  Working mic device: {dev_id}")
    print("=" * 50)
