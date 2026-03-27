"""
Real-Time STT Demo — Compare Soniox, Deepgram Nova-3, Google Chirp 3, and Whisper
FastAPI + WebSocket backend with custom HTML/CSS/JS frontend.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import asyncio
import threading
import queue
import time
import logging
import traceback
import numpy as np
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)
LOG_FILE = os.path.join(OUTPUT_DIR, "demo_terminal.log")
with open(LOG_FILE, "w") as f:
    f.write(f"=== STT Demo Started at {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n\n")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("stt_demo")

logging.getLogger("uvicorn").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

class TeeWriter:
    def __init__(self, original, log_file):
        self.original = original
        self.log_file = log_file
    def write(self, msg):
        self.original.write(msg)
        with open(self.log_file, "a") as f:
            f.write(msg)
    def flush(self):
        self.original.flush()
    def isatty(self):
        return False
    def fileno(self):
        return self.original.fileno()

sys.stdout = TeeWriter(sys.__stdout__, LOG_FILE)
sys.stderr = TeeWriter(sys.__stderr__, LOG_FILE)

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.25
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_DURATION)

# ─────────────────────────────────────────────
# Provider Implementations
# ─────────────────────────────────────────────

class DeepgramSTT:
    def __init__(self, api_key: str, language: str = "ur"):
        self.api_key = api_key
        self.language = language
        self.transcript = ""
        self.interim = ""
        self.ws = None
        self._running = False

    async def connect(self):
        import websockets
        logger.info("Deepgram: Connecting...")
        url = (
            f"wss://api.deepgram.com/v1/listen"
            f"?model=nova-3&language={self.language}"
            f"&encoding=linear16&sample_rate={SAMPLE_RATE}&channels=1"
            f"&punctuate=true&interim_results=true"
        )
        headers = {"Authorization": f"Token {self.api_key}"}
        self.ws = await websockets.connect(url, extra_headers=headers)
        self._running = True
        logger.info("Deepgram: Connected")

    async def send_audio(self, audio_bytes: bytes):
        if self.ws and self._running:
            await self.ws.send(audio_bytes)

    async def receive_loop(self, callback):
        try:
            async for msg in self.ws:
                data = json.loads(msg)
                alt = data.get("channel", {}).get("alternatives", [{}])[0]
                text = alt.get("transcript", "")
                is_final = data.get("is_final", False)
                if text:
                    if is_final:
                        self.transcript += text + " "
                        self.interim = ""
                        logger.info(f"Deepgram [FINAL]: {text}")
                    else:
                        self.interim = text
                    callback("deepgram", self.transcript + self.interim)
        except Exception as e:
            logger.error(f"Deepgram Error: {e}")
            callback("deepgram", f"[Error: {e}]")

    async def close(self):
        self._running = False
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "CloseStream"}))
                await self.ws.close()
            except Exception:
                pass


class SonioxSTT:
    def __init__(self, api_key: str, language: str = "ur"):
        self.api_key = api_key
        self.language = language
        self.transcript = ""
        self.interim = ""
        self.ws = None
        self._running = False

    async def connect(self):
        import websockets
        logger.info("Soniox: Connecting...")
        self.ws = await websockets.connect("wss://api.soniox.com/transcribe-websocket")
        config = {
            "api_key": self.api_key, "sample_rate_hertz": SAMPLE_RATE,
            "audio_format": "pcm_s16le", "num_audio_channels": 1,
            "model": "stt-rt-preview", "include_nonfinal": True,
        }
        await self.ws.send(json.dumps(config))
        self._running = True
        logger.info("Soniox: Connected")

    async def send_audio(self, audio_bytes: bytes):
        if self.ws and self._running:
            await self.ws.send(audio_bytes)

    async def receive_loop(self, callback):
        try:
            async for msg in self.ws:
                data = json.loads(msg)
                words = data.get("words", [])
                if words:
                    final_words = [w["text"] for w in words if w.get("is_final")]
                    non_final = [w["text"] for w in words if not w.get("is_final")]
                    if final_words:
                        self.transcript += " ".join(final_words) + " "
                        logger.info(f"Soniox [FINAL]: {' '.join(final_words)}")
                    self.interim = " ".join(non_final)
                    callback("soniox", self.transcript + self.interim)
        except Exception as e:
            logger.error(f"Soniox Error: {e}")
            callback("soniox", f"[Error: {e}]")

    async def close(self):
        self._running = False
        if self.ws:
            try:
                await self.ws.send(json.dumps({"type": "eof"}))
                await self.ws.close()
            except Exception:
                pass


class GoogleChirpSTT:
    def __init__(self, language: str = "ur-PK"):
        self.language = language
        self.transcript = ""
        self.interim = ""

    def stream_recognize(self, audio_queue: queue.Queue, callback):
        try:
            from google.cloud import speech_v2 as speech
            from google.cloud.speech_v2.types import cloud_speech

            project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "")
            client = speech.SpeechClient()
            recognition_config = cloud_speech.RecognitionConfig(
                explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                    encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                    sample_rate_hertz=SAMPLE_RATE, audio_channel_count=1,
                ),
                language_codes=[self.language], model="chirp_2",
            )
            streaming_config = cloud_speech.StreamingRecognitionConfig(
                config=recognition_config,
                streaming_features=cloud_speech.StreamingRecognitionFeatures(interim_results=True),
            )
            config_request = cloud_speech.StreamingRecognizeRequest(
                recognizer=f"projects/{project_id}/locations/global/recognizers/_",
                streaming_config=streaming_config,
            )

            def request_generator():
                yield config_request
                while True:
                    chunk = audio_queue.get()
                    if chunk is None:
                        break
                    yield cloud_speech.StreamingRecognizeRequest(audio=chunk)

            responses = client.streaming_recognize(requests=request_generator())
            for response in responses:
                for result in response.results:
                    text = result.alternatives[0].transcript
                    if result.is_final:
                        self.transcript += text + " "
                        self.interim = ""
                    else:
                        self.interim = text
                    callback("google", self.transcript + self.interim)
        except ImportError:
            callback("google", "[Install: pip install google-cloud-speech]")
        except Exception as e:
            callback("google", f"[Error: {e}]")


class WhisperSTT:
    HALLUCINATION_PHRASES = {
        "شکریہ", "ٹھیک ہے", "اللہ حافظ", "بسم اللہ",
        "السلام علیکم", "وعلیکم السلام",
        "thanks for watching", "thank you", "subscribe",
        "like and subscribe", "bye bye", "goodbye",
        "please subscribe", "see you next time",
        "امید ہے", "بہتر بہتر بہتر",
    }

    def __init__(self, model_size: str = "base", language: str = "ur"):
        self.model_size = model_size
        self.language = language
        self.transcript = ""
        self.buffer = np.array([], dtype=np.float32)
        self.buffer_duration = 3.0
        self.client = None

    def _get_client(self):
        if self.client is None:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))
            except ImportError:
                return None
        return self.client

    def _is_speech(self, audio, threshold=0.01):
        return float(np.sqrt(np.mean(audio ** 2))) > threshold

    def _is_hallucination(self, text):
        t = text.strip()
        if t in self.HALLUCINATION_PHRASES:
            return True
        words = t.split()
        if len(words) >= 3:
            from collections import Counter
            c = Counter(words).most_common(1)[0][1]
            if c >= 4 and c / len(words) > 0.6:
                return True
        return False

    def process_chunk(self, audio_float32, callback):
        self.buffer = np.concatenate([self.buffer, audio_float32])
        if len(self.buffer) >= int(SAMPLE_RATE * self.buffer_duration):
            if not self._is_speech(self.buffer):
                self.buffer = np.array([], dtype=np.float32)
                return
            client = self._get_client()
            if not client:
                callback("whisper", "[Error: openai not installed]")
                self.buffer = np.array([], dtype=np.float32)
                return
            try:
                import io, wave
                audio_int16 = (self.buffer * 32768).astype(np.int16)
                wav_buf = io.BytesIO()
                with wave.open(wav_buf, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(audio_int16.tobytes())
                wav_buf.seek(0)
                wav_buf.name = "audio.wav"
                result = client.audio.transcriptions.create(model="whisper-1", file=wav_buf, language=self.language)
                text = result.text.strip()
                if text and not self._is_hallucination(text):
                    self.transcript += text + " "
                    logger.info(f"Whisper [FINAL]: {text}")
            except Exception as e:
                logger.error(f"Whisper Error: {e}")
            self.buffer = np.array([], dtype=np.float32)
            callback("whisper", self.transcript)


# ─────────────────────────────────────────────
# Demo Controller
# ─────────────────────────────────────────────

class RealtimeSTTDemo:
    def __init__(self):
        self.is_recording = False
        self.results = {"deepgram": "", "soniox": "", "google": "", "whisper": ""}
        self.audio_queues = {
            "deepgram": asyncio.Queue(), "soniox": asyncio.Queue(),
            "google": queue.Queue(), "whisper": queue.Queue(),
        }
        self.deepgram = None
        self.soniox = None
        self.google = None
        self.whisper_stt = None
        self.loop = None
        self._lock = threading.Lock()
        self._ws_clients = set()
        self._app_loop = None

    def update_result(self, provider: str, text: str):
        with self._lock:
            self.results[provider] = text
        self._broadcast_sync({"type": "transcript", "provider": provider, "text": text})

    def _broadcast_sync(self, msg):
        for ws in list(self._ws_clients):
            try:
                asyncio.run_coroutine_threadsafe(ws.send_json(msg), self._app_loop)
            except Exception:
                pass

    async def _run_deepgram(self, api_key, language):
        try:
            self.deepgram = DeepgramSTT(api_key, language)
            await self.deepgram.connect()
            async def sender():
                while self.is_recording:
                    try:
                        chunk = await asyncio.wait_for(self.audio_queues["deepgram"].get(), timeout=0.1)
                        await self.deepgram.send_audio(chunk)
                    except asyncio.TimeoutError:
                        continue
            await asyncio.gather(sender(), self.deepgram.receive_loop(self.update_result))
        except Exception as e:
            self.update_result("deepgram", f"[Error: {e}]")

    async def _run_soniox(self, api_key, language):
        try:
            self.soniox = SonioxSTT(api_key, language)
            await self.soniox.connect()
            async def sender():
                while self.is_recording:
                    try:
                        chunk = await asyncio.wait_for(self.audio_queues["soniox"].get(), timeout=0.1)
                        await self.soniox.send_audio(chunk)
                    except asyncio.TimeoutError:
                        continue
            await asyncio.gather(sender(), self.soniox.receive_loop(self.update_result))
        except Exception as e:
            self.update_result("soniox", f"[Error: {e}]")

    def _run_google(self, language):
        self.google = GoogleChirpSTT(language)
        self.google.stream_recognize(self.audio_queues["google"], self.update_result)

    def _run_whisper(self, model_size, language):
        self.whisper_stt = WhisperSTT(model_size, language)
        while self.is_recording:
            try:
                chunk = self.audio_queues["whisper"].get(timeout=0.5)
                audio_float = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
                self.whisper_stt.process_chunk(audio_float, self.update_result)
            except queue.Empty:
                continue

    def _capture_mic(self):
        import sounddevice as sd
        logger.info("Mic: Starting capture...")

        def audio_callback(indata, frames, time_info, status):
            if not self.is_recording:
                raise sd.CallbackAbort()
            audio_bytes = (indata[:, 0] * 32768).astype(np.int16).tobytes()
            if self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self.audio_queues["deepgram"].put(audio_bytes), self.loop)
                asyncio.run_coroutine_threadsafe(self.audio_queues["soniox"].put(audio_bytes), self.loop)
            self.audio_queues["google"].put(audio_bytes)
            self.audio_queues["whisper"].put(audio_bytes)

            rms = float(np.sqrt(np.mean(indata[:, 0] ** 2)))
            self._broadcast_sync({"type": "mic_rms", "rms": round(rms, 5)})

        try:
            with sd.InputStream(
                samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                blocksize=CHUNK_SAMPLES, callback=audio_callback,
            ):
                while self.is_recording:
                    time.sleep(0.1)
        except Exception as e:
            logger.error(f"Mic Error: {e}")

    def start(self, config):
        if self.is_recording:
            return
        language = config.get("language", "Urdu")
        whisper_model = config.get("whisper_model", "base")
        providers = config.get("providers", {})

        logger.info(f"=== Recording started | Language: {language} | Providers: {providers} ===")
        self.is_recording = True
        self.results = {k: "" for k in self.results}

        lang_map = {
            "Urdu": {"deepgram": "ur", "soniox": "ur", "google": "ur-PK", "whisper": "ur"},
            "English": {"deepgram": "en", "soniox": "en", "google": "en-US", "whisper": "en"},
            "Hindi": {"deepgram": "hi", "soniox": "hi", "google": "hi-IN", "whisper": "hi"},
            "Arabic": {"deepgram": "ar", "soniox": "ar", "google": "ar-SA", "whisper": "ar"},
            "Malay": {"deepgram": "ms", "soniox": "ms", "google": "ms-MY", "whisper": "ms"},
        }
        langs = lang_map.get(language, lang_map["Urdu"])
        deepgram_key = os.getenv("DEEPGRAM_API_KEY", "")
        soniox_key = os.getenv("SONIOX_API_KEY", "")

        self.loop = asyncio.new_event_loop()

        def run_async():
            asyncio.set_event_loop(self.loop)
            tasks = []
            if providers.get("deepgram") and deepgram_key:
                tasks.append(self._run_deepgram(deepgram_key, langs["deepgram"]))
            elif providers.get("deepgram"):
                self.update_result("deepgram", "[No API key]")
            if providers.get("soniox") and soniox_key:
                tasks.append(self._run_soniox(soniox_key, langs["soniox"]))
            elif providers.get("soniox"):
                self.update_result("soniox", "[No API key]")
            if tasks:
                self.loop.run_until_complete(asyncio.gather(*tasks))

        threading.Thread(target=run_async, daemon=True).start()

        if providers.get("google"):
            threading.Thread(target=self._run_google, args=(langs["google"],), daemon=True).start()
        if providers.get("whisper"):
            threading.Thread(target=self._run_whisper, args=(whisper_model, langs["whisper"]), daemon=True).start()

        threading.Thread(target=self._capture_mic, daemon=True).start()
        self._broadcast_sync({"type": "status", "status": "recording"})

    def stop(self):
        logger.info("=== Recording stopped ===")
        self.is_recording = False
        try:
            if self.deepgram and self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self.deepgram.close(), self.loop)
        except Exception:
            pass
        try:
            if self.soniox and self.loop and self.loop.is_running():
                asyncio.run_coroutine_threadsafe(self.soniox.close(), self.loop)
        except Exception:
            pass
        self.audio_queues["google"].put(None)

        time.sleep(0.5)
        self.deepgram = None
        self.soniox = None
        self.google = None
        self.whisper_stt = None
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        self.loop = None
        self.audio_queues = {
            "deepgram": asyncio.Queue(), "soniox": asyncio.Queue(),
            "google": queue.Queue(), "whisper": queue.Queue(),
        }
        self._broadcast_sync({"type": "status", "status": "stopped"})


# ─────────────────────────────────────────────
# FastAPI App
# ─────────────────────────────────────────────

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

app = FastAPI(title="Real-Time STT Demo")
demo_app = RealtimeSTTDemo()


_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, "templates", "realtime_stt_demo.html")
with open(_TEMPLATE_PATH, "r", encoding="utf-8") as _f:
    HTML_PAGE = _f.read()
logger.info(f"Loaded HTML template from {_TEMPLATE_PATH} ({len(HTML_PAGE)} chars)")




@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    demo_app._ws_clients.add(websocket)
    demo_app._app_loop = asyncio.get_event_loop()
    logger.info(f"WS: Client connected (total: {len(demo_app._ws_clients)})")

    # Send current state
    status = "recording" if demo_app.is_recording else "stopped"
    await websocket.send_json({"type": "status", "status": status})
    for provider, text in demo_app.results.items():
        if text:
            await websocket.send_json({"type": "transcript", "provider": provider, "text": text})

    try:
        while True:
            data = await websocket.receive_json()
            action = data.get("action")
            if action == "start":
                demo_app.start(data)
            elif action == "stop":
                threading.Thread(target=demo_app.stop, daemon=True).start()
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WS Error: {e}")
    finally:
        demo_app._ws_clients.discard(websocket)
        logger.info(f"WS: Client disconnected (total: {len(demo_app._ws_clients)})")


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting Real-Time STT Demo on port 7870")
    uvicorn.run(app, host="0.0.0.0", port=7870, log_level="warning")
