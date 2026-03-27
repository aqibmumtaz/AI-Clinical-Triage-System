"""
Automated demo with voiceover narration + full audio capture.
  Phase 1: UI Tour — highlights each section with TTS voiceover
  Phase 2: Live Conversation — patient audio clips + bot TTS

Audio architecture (post-processing build):
  - During recording: track wall-clock timestamps for each audio event
  - After recording: decode source files (narration MP3s, patient WAVs) to PCM
  - Capture bot TTS via MediaRecorder on WebRTC audio stream directly
  - Build combined audio track with numpy at correct timestamps
  - Mux final audio + Playwright video via ffmpeg

Usage:
  1. Start conversation agent: python conversation_agent.py
  2. Run: python run_demo.py
Output: demo_final.mp4
"""
import os
import sys
import json
import wave
import time
import asyncio
import base64
import hashlib
import subprocess
import shutil
import numpy as np
from pathlib import Path

from playwright.async_api import async_playwright

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
AGENT_URL = "http://localhost:7871"
INPUT_DIR = SCRIPT_DIR / "input"
CACHE_DIR = SCRIPT_DIR / "cache"
OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_VIDEO = CACHE_DIR / "demo_recording.webm"
BUILT_AUDIO = CACHE_DIR / "demo_built_audio.wav"
FINAL_OUTPUT = OUTPUT_DIR / "demo_final.mp4"
NARRATION_DIR = CACHE_DIR / "narrations"

SAMPLE_RATE = 48000  # output audio sample rate
MAX_WAIT = 30

AUDIO_CLIPS = [
    ("trimmed_01_greeting.wav", "Patient greeting"),
    ("trimmed_02_name.wav",     "Patient name"),
    ("trimmed_03_symptom.wav",  "Patient symptom"),
    ("trimmed_04_duration.wav", "Symptom duration"),
    ("trimmed_05_confirm.wav",  "Vitals confirmation"),
    ("trimmed_06_thanks.wav",   "Patient thanks"),
]

# ── Voiceover narrations — ordered to match TOUR_STEPS ──
# Tour order: Title → Header → Doctors → Doctor Detail →
#   Agent Persona → Agent Settings → AI Models → Quick Ref →
#   Patient Screening → Flow Stepper → STT Live → Conversation →
#   Controls → Transition
NARRATIONS = [
    # 0: Title card
    "Welcome to City General Hospital's AI Powered Patient Support Desk. "
    "Let me walk you through the key features of this platform.",

    # 1: Header
    "The application header displays the hospital identity along with three "
    "AI capability badges: Voice Enabled for hands-free interaction, "
    "Multilingual support for Urdu, English and Malay, and Smart Triage "
    "for intelligent doctor referral.",

    # 2: Doctors panel
    "The Doctors On Duty panel displays all available physicians with their "
    "specialties, room numbers, and real-time availability status.",

    # 3: Doctor detail
    "Clicking a doctor reveals their complete profile: qualifications, years "
    "of experience, consultation hours, languages spoken, and areas of "
    "clinical expertise.",

    # 4: Agent persona
    "In the Settings panel, you can select an AI agent persona. Choose from "
    "Amina, Amir, or Aliza. Each agent has a unique personality, gender, "
    "and conversational style tailored for patient interaction.",

    # 5: Agent settings
    "Configure the speech to text language, personality traits like warm or "
    "professional, and the language style from formal to casual. These settings "
    "shape how the AI communicates with patients.",

    # 6: AI models
    "The AI Model section lets you select the speech recognition engine, "
    "language model, and text to speech provider. Options include Deepgram, "
    "OpenAI Whisper, GPT-4, and ElevenLabs voices.",

    # 7: Quick reference
    "Quick Reference provides a hospital directory with locations for "
    "Emergency, Pharmacy, Laboratory, Radiology, Physiotherapy, and Billing.",

    # 8: Patient screening
    "The Patient Screening card is the heart of the system. It automatically "
    "extracts patient information from the conversation in real time, including "
    "name, symptoms, duration, and vitals.",

    # 9: Flow stepper
    "The screening flow stepper visually tracks progress through five stages: "
    "collecting the patient's name, identifying symptoms, assessing duration, "
    "checking relevant vitals, and making a doctor referral. The vitals step "
    "adapts based on the patient's symptoms.",

    # 10: STT live
    "The live transcription area shows real-time speech to text output as "
    "the patient speaks, providing immediate feedback on recognition.",

    # 11: Conversation
    "The conversation panel displays the full chat history between patient "
    "and AI assistant, with right to left text support for Urdu and automatic "
    "language detection.",

    # 12: Controls
    "The session controls allow you to start a new patient session or end "
    "the current one. The status indicator shows whether the AI is listening, "
    "thinking, or speaking.",

    # 13: Transition
    "Now let's see a live conversation demo. A patient will speak in Urdu, "
    "and the AI assistant will conduct a medical screening, extract patient "
    "data, and refer to the appropriate doctor. All in real time.",
]

# ── Tour steps — Doctors first, then Agent Profile ──────────────
TOUR_STEPS = [
    {"type": "title"},
    {"selector": ".header",
     "text": "Hospital Header — AI capability badges:\nVoice Enabled  |  Multilingual  |  Smart Triage",
     "pos": "bottom"},
    # Doctors (hospital tab)
    {"action": {"click": ".right-tab:nth-child(1)"},
     "selector": "#doctorsGrid",
     "text": "Doctors On Duty — Specialties, rooms, availability",
     "pos": "left"},
    {"action": {"click": ".doctor-row:nth-child(3)"},
     "selector": "#doctorDetail",
     "text": "Doctor Profile — Qualifications, experience, expertise",
     "pos": "left"},
    # Agent profile (switch to settings tab — closes doctor detail)
    {"action": {"close_detail": True, "click": ".right-tab:nth-child(2)"},
     "selector": "#agentSummary",
     "text": "Agent Persona — Choose AI assistant (Amina / Amir / Aliza)\nwith configurable personality and speaking style",
     "pos": "left"},
    {"selector": "#agentProfileBody .settings-grid",
     "text": "Agent Settings — STT Language, Personality, Language Style",
     "pos": "left"},
    {"action": {"scroll": "#settingsBody"},
     "selector": "#settingsBody",
     "text": "AI Models — STT, LLM, TTS provider and voice selection",
     "pos": "left"},
    # Back to hospital tab for remaining items
    {"action": {"click": ".right-tab:nth-child(1)", "scroll": "#quickRefCard"},
     "selector": "#quickRefCard",
     "text": "Quick Reference — Hospital department directory",
     "pos": "left"},
    {"selector": "#patientScreeningCard",
     "text": "Patient Screening — AI extracts data from conversation in real-time",
     "pos": "right"},
    {"selector": "#flowStepper",
     "text": "Screening Flow — Name → Symptoms → Duration → Vitals → Referral",
     "pos": "bottom"},
    {"selector": "#sttLive",
     "text": "Live Transcription — Real-time speech-to-text display",
     "pos": "bottom"},
    {"selector": ".card:has(#conversation)",
     "text": "Conversation — Full chat with RTL Urdu support",
     "pos": "top-right"},
    {"selector": "#btnStart",
     "text": "Session Controls — Start/Stop with live status indicator",
     "pos": "bottom"},
    {"type": "transition"},
]


# ── CSS for highlights ──────────────────────────────────────────
HIGHLIGHT_CSS = """
<style id="demo-styles">
.demo-hl { outline: 3px solid #FF6B35 !important; outline-offset: 4px !important;
  position: relative !important; z-index: 100 !important;
  animation: dp 1.5s ease-in-out infinite !important; }
@keyframes dp { 0%,100%{outline-color:#FF6B35} 50%{outline-color:#FFD700} }
.demo-ann { position:fixed; background:rgba(15,23,42,0.92); color:white;
  padding:14px 20px; border-radius:10px; font-size:15px; font-weight:400;
  line-height:1.6; max-width:380px; z-index:10000;
  box-shadow:0 8px 32px rgba(0,0,0,0.4); pointer-events:none;
  white-space:pre-line; border-left:4px solid #FF6B35;
  animation: dfi 0.4s ease-out; }
@keyframes dfi { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }
.demo-ov { position:fixed; top:0;left:0;width:100%;height:100%;
  display:flex;align-items:center;justify-content:center;z-index:99999;
  animation:dfi 0.5s ease-out; }
.demo-bg { background:linear-gradient(135deg,#0d9488,#0e7490); }
.demo-dk { background:rgba(15,23,42,0.88); }
.demo-ctr { position:fixed;bottom:16px;right:16px;
  background:rgba(15,23,42,0.8);color:#94a3b8;
  padding:6px 14px;border-radius:20px;font-size:13px;z-index:10001;pointer-events:none; }
</style>
"""


# ── Init script: getUserMedia override for WebRTC audio injection ──
INIT_SCRIPT = """
(() => {
    // ═══ PATIENT AudioContext: feeds WebRTC via getUserMedia ═══
    let patientCtx = null, patientDest = null;

    navigator.mediaDevices.getUserMedia = async function(constraints) {
        if (constraints && constraints.audio) {
            if (!patientCtx) {
                patientCtx  = new AudioContext({ sampleRate: 16000 });
                patientDest = patientCtx.createMediaStreamDestination();

                // Keep stream active with near-silent oscillator
                const osc = patientCtx.createOscillator();
                osc.frequency.value = 0;
                const g = patientCtx.createGain();
                g.gain.value = 0.0001;
                osc.connect(g);
                g.connect(patientDest);
                osc.start();

                window.__patientCtx  = patientCtx;
                window.__patientDest = patientDest;
            }
            if (patientCtx.state === 'suspended') await patientCtx.resume();
            console.log('[DEMO] getUserMedia → injected stream');
            return patientDest.stream;
        }
        throw new Error('No audio device');
    };

    console.log('[DEMO] getUserMedia override ready');
})();
"""

# ── Post-load script: master AudioContext for narration playback ──
MASTER_SETUP_SCRIPT = """
async () => {
    const masterCtx = new AudioContext({ sampleRate: 48000 });
    if (masterCtx.state === 'suspended') await masterCtx.resume();
    window.__masterCtx = masterCtx;
    console.log('[DEMO] Master AudioContext ready for narration playback');
    return masterCtx.state;
}
"""


# ── Helper functions ────────────────────────────────────────────

def read_wav_float32_b64(filepath: str) -> tuple:
    with wave.open(filepath, 'rb') as wf:
        pcm = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
        sr = wf.getframerate()
    f32 = np.clip(pcm.astype(np.float32) / 32768.0 * 1.5, -1.0, 1.0)
    return base64.b64encode(f32.tobytes()).decode(), sr, len(pcm) / sr


def decode_audio_to_pcm(filepath: Path) -> np.ndarray:
    """Decode any audio file (MP3, WAV, WebM) to 48kHz mono float32 using ffmpeg."""
    result = subprocess.run([
        'ffmpeg', '-y', '-i', str(filepath),
        '-f', 'f32le', '-acodec', 'pcm_f32le',
        '-ar', str(SAMPLE_RATE), '-ac', '1',
        'pipe:1'
    ], capture_output=True)
    if result.returncode != 0:
        print(f"    WARNING: Failed to decode {filepath.name}")
        return np.array([], dtype=np.float32)
    return np.frombuffer(result.stdout, dtype=np.float32)


async def get_msg_count(page) -> int:
    return await page.evaluate("document.querySelectorAll('#conversation .msg').length")


async def get_status_class(page) -> str:
    return await page.evaluate("document.getElementById('statusDot')?.className || ''")


async def wait_for_response(page, prev_count: int, clip_dur: float) -> int:
    await page.wait_for_timeout(int(clip_dur * 1000) + 800)
    start = time.time()
    while time.time() - start < MAX_WAIT:
        count = await get_msg_count(page)
        status = await get_status_class(page)
        if count > prev_count and 'listening' in status:
            break
        await page.wait_for_timeout(500)
    else:
        print("     (timeout)")
    await page.wait_for_timeout(1500)
    return await get_msg_count(page)


async def clear_highlights(page):
    await page.evaluate("""() => {
        document.querySelectorAll('.demo-hl').forEach(el => el.classList.remove('demo-hl'));
        document.querySelectorAll('.demo-ann,.demo-ov,.demo-ctr').forEach(el => el.remove());
    }""")


def generate_narrations():
    """Generate TTS voiceover MP3 files. Regenerates if narration texts changed."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / "Model-End" / ".env")
    from openai import OpenAI

    NARRATION_DIR.mkdir(exist_ok=True)

    # Check if cached narrations match current text
    narr_hash = hashlib.md5("||".join(NARRATIONS).encode()).hexdigest()
    hash_file = NARRATION_DIR / ".narr_hash"
    if hash_file.exists() and hash_file.read_text().strip() == narr_hash:
        all_exist = all(
            (NARRATION_DIR / f"narr_{i:02d}.mp3").exists()
            for i in range(len(NARRATIONS))
        )
        if all_exist:
            print("  All narrations cached (hash match).")
            return

    # Text changed or files missing — regenerate all
    print("  Narration texts changed or missing, regenerating...")
    for f in NARRATION_DIR.glob("narr_*.mp3"):
        f.unlink()

    client = OpenAI()
    for i, text in enumerate(NARRATIONS):
        path = NARRATION_DIR / f"narr_{i:02d}.mp3"
        print(f"  Generating narr_{i:02d}: {text[:50]}...")
        response = client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text,
        )
        response.stream_to_file(str(path))

    hash_file.write_text(narr_hash)
    print(f"  Generated {len(NARRATIONS)} narration files.")


async def play_narration(page, index: int) -> float:
    """Play narration MP3 through browser speakers. Returns duration in seconds."""
    path = NARRATION_DIR / f"narr_{index:02d}.mp3"
    if not path.exists():
        return 0

    with open(str(path), 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()

    duration = await page.evaluate(f"""async () => {{
        const ctx = window.__masterCtx;
        if (!ctx) return 0;
        if (ctx.state === 'suspended') await ctx.resume();

        const b64 = "{b64}";
        const raw = atob(b64);
        const bytes = new Uint8Array(raw.length);
        for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
        const audioBuffer = await ctx.decodeAudioData(bytes.buffer.slice(0));

        const src = ctx.createBufferSource();
        src.buffer = audioBuffer;
        src.connect(ctx.destination);  // speakers only
        src.start();

        return audioBuffer.duration;
    }}""")
    return duration or 0


async def show_title(page):
    await page.evaluate("""() => {
        const el = document.createElement('div');
        el.className = 'demo-ov demo-bg';
        el.innerHTML = `
            <div style="text-align:center;color:white;">
                <div style="font-size:64px;margin-bottom:16px;">🏥</div>
                <h1 style="font-size:42px;margin:0;font-weight:700;">City General Hospital</h1>
                <p style="font-size:22px;margin-top:10px;opacity:0.9;">AI-Powered Patient Support Desk</p>
                <div style="margin-top:28px;display:flex;gap:12px;justify-content:center;">
                    <span style="background:rgba(255,255,255,0.2);padding:6px 16px;border-radius:20px;font-size:14px;">Voice Enabled</span>
                    <span style="background:rgba(255,255,255,0.2);padding:6px 16px;border-radius:20px;font-size:14px;">Multilingual</span>
                    <span style="background:rgba(255,255,255,0.2);padding:6px 16px;border-radius:20px;font-size:14px;">Smart Triage</span>
                </div>
                <p style="font-size:16px;margin-top:32px;opacity:0.6;">Product Demo</p>
            </div>`;
        document.body.appendChild(el);
    }""")


async def show_transition(page):
    await page.evaluate("""() => {
        const el = document.createElement('div');
        el.className = 'demo-ov demo-dk';
        el.innerHTML = `
            <div style="text-align:center;color:white;">
                <div style="font-size:48px;margin-bottom:16px;">🎙️</div>
                <h2 style="font-size:32px;margin:0;font-weight:600;">Live Conversation Demo</h2>
                <p style="font-size:18px;margin-top:12px;opacity:0.8;">
                    Patient speaks in Urdu — AI screens, extracts data,<br>
                    and refers to the right doctor in real-time
                </p>
            </div>`;
        document.body.appendChild(el);
    }""")


async def highlight_element(page, sel, text, pos, step_num, total):
    escaped = text.replace('\\', '\\\\').replace('`', '\\`').replace('$', '\\$')
    await page.evaluate(f"""() => {{
        document.querySelectorAll('.demo-hl').forEach(el => el.classList.remove('demo-hl'));
        document.querySelectorAll('.demo-ann,.demo-ctr').forEach(el => el.remove());

        const el = document.querySelector(`{sel}`);
        if (!el) return;
        el.classList.add('demo-hl');

        const rect = el.getBoundingClientRect();
        const ann = document.createElement('div');
        ann.className = 'demo-ann';
        ann.textContent = `{escaped}`;

        const pos = '{pos}';
        if (pos==='bottom') {{ ann.style.top=(rect.bottom+12)+'px'; ann.style.left=Math.max(10,rect.left)+'px'; }}
        else if (pos==='top') {{ ann.style.bottom=(window.innerHeight-rect.top+12)+'px'; ann.style.left=Math.max(10,rect.left)+'px'; }}
        else if (pos==='left') {{ ann.style.top=Math.max(10,rect.top)+'px'; ann.style.right=(window.innerWidth-rect.left+12)+'px'; }}
        else if (pos==='right') {{ ann.style.top=Math.max(10,rect.top)+'px'; ann.style.left=(rect.right+12)+'px'; }}
        else if (pos==='top-right') {{ ann.style.bottom=(window.innerHeight-rect.top+12)+'px'; ann.style.left=(rect.right-300)+'px'; }}
        document.body.appendChild(ann);

        const ctr = document.createElement('div');
        ctr.className = 'demo-ctr';
        ctr.textContent = 'UI Tour  {step_num}/{total}';
        document.body.appendChild(ctr);
    }}""")


# ── Post-processing: build audio from timestamps ──────────────

def build_audio_from_events(audio_events: list, total_duration: float) -> bool:
    """
    Build the final audio WAV from tracked timestamps + source files.
    audio_events: list of (timestamp_sec, source_type, source_path)
    Returns True if audio was built successfully.
    """
    print(f"\n  Building audio from {len(audio_events)} events over {total_duration:.1f}s...")

    total_samples = int(total_duration * SAMPLE_RATE) + SAMPLE_RATE
    output = np.zeros(total_samples, dtype=np.float32)

    for ts, src_type, src_path in audio_events:
        if not Path(src_path).exists():
            print(f"    SKIP: {src_path} not found")
            continue

        pcm = decode_audio_to_pcm(Path(src_path))
        if len(pcm) == 0:
            print(f"    SKIP: {Path(src_path).name} decode failed")
            continue

        offset_samples = int(ts * SAMPLE_RATE)
        end_samples = offset_samples + len(pcm)

        if end_samples > len(output):
            output = np.pad(output, (0, end_samples - len(output) + SAMPLE_RATE))

        segment_len = min(len(pcm), len(output) - offset_samples)
        output[offset_samples:offset_samples + segment_len] += pcm[:segment_len]

        dur = len(pcm) / SAMPLE_RATE
        print(f"    {src_type:10s} @ {ts:6.1f}s  {Path(src_path).name} ({dur:.1f}s)")

    output = np.clip(output, -1.0, 1.0)
    peak = np.max(np.abs(output))
    rms = np.sqrt(np.mean(output ** 2))
    print(f"  Audio stats: peak={peak:.4f}, RMS={rms:.4f}")

    if peak < 0.001:
        print("  WARNING: Audio appears silent!")
        return False

    pcm16 = (output * 32767).astype(np.int16)
    with wave.open(str(BUILT_AUDIO), 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm16.tobytes())

    print(f"  Built audio: {BUILT_AUDIO} ({BUILT_AUDIO.stat().st_size // 1024}KB)")
    return True


# ── Main demo ───────────────────────────────────────────────────

async def run_demo():
    print(f"\n{'='*60}")
    print("  AUTOMATED DEMO — Post-Processing Audio Build")
    print(f"{'='*60}\n")

    # Ensure cache and output dirs exist
    CACHE_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)

    # 1. Generate narrations
    print("  Generating voiceover narrations...")
    generate_narrations()

    # 2. Verify clips
    clip_data = []
    for clip_file, label in AUDIO_CLIPS:
        path = INPUT_DIR / clip_file
        if not path.exists():
            print(f"  ERROR: Missing {path}")
            sys.exit(1)
        b64, sr, dur = read_wav_float32_b64(str(path))
        clip_data.append((clip_file, label, b64, sr, dur))
        print(f"  {label}: {clip_file} ({dur:.1f}s)")

    print(f"\n  Output: {FINAL_OUTPUT}")
    print(f"{'='*60}\n")

    # Timestamp tracking for post-processing audio build
    audio_events = []  # (offset_sec, type, filepath)

    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=[
                "--use-fake-ui-for-media-stream",
                "--autoplay-policy=no-user-gesture-required",
            ]
        )
        context = await browser.new_context(
            viewport={"width": 1440, "height": 900},
            record_video_dir=str(CACHE_DIR),
            record_video_size={"width": 1440, "height": 900},
            permissions=["microphone"],
        )
        page = await context.new_page()
        await page.add_init_script(INIT_SCRIPT)

        # ── VIDEO RECORDING STARTS HERE (Playwright records from page creation) ──
        recording_start = time.time()

        # ── Intercept WebSocket messages to capture bot text + timestamps ──
        bot_responses = []  # (offset_sec, text)

        def on_ws_created(ws):
            def on_frame(payload):
                # payload is a plain string in Playwright Python
                try:
                    data = json.loads(payload)
                    if data.get('type') == 'conversation' and data.get('role') == 'medi':
                        ts = time.time() - recording_start
                        text = data.get('text', '')
                        if text:
                            bot_responses.append((ts, text))
                            print(f"    [BOT @ {ts:.1f}s] {text[:60]}...")
                except Exception:
                    pass
            ws.on('framereceived', on_frame)

        page.on('websocket', on_ws_created)

        # Open page
        print("  Loading UI...")
        await page.goto(AGENT_URL, wait_until="networkidle")
        await page.wait_for_timeout(1500)

        # Inject CSS
        await page.evaluate(f"""() => {{
            document.head.insertAdjacentHTML('beforeend', `{HIGHLIGHT_CSS}`);
        }}""")

        # Set up master AudioContext for narration playback
        master_state = await page.evaluate(MASTER_SETUP_SCRIPT)

        # Store JS recording start time (for bot audio offset calculation)
        await page.evaluate("window.__demoRecordingStartMs = Date.now()")
        print(f"  Master AudioContext: {master_state}\n")

        # ══════════════════════════════════════════════════════════
        #  PHASE 1: UI TOUR WITH VOICEOVER
        # ══════════════════════════════════════════════════════════
        print("═══ PHASE 1: UI TOUR ═══\n")

        highlight_steps = [s for s in TOUR_STEPS if s.get("type") not in ("title", "transition")]
        total_hl = len(highlight_steps)
        hl_num = 0

        for i, step in enumerate(TOUR_STEPS):
            step_type = step.get("type")

            if step_type == "title":
                print(f"  [{i+1}/{len(TOUR_STEPS)}] Title card")
                await show_title(page)
                # Wait for visual to render in video frame
                await page.wait_for_timeout(600)
                narr_offset = time.time() - recording_start
                narr_path = NARRATION_DIR / f"narr_{i:02d}.mp3"
                if narr_path.exists():
                    audio_events.append((narr_offset, "narration", str(narr_path)))
                narr_dur = await play_narration(page, i)
                await page.wait_for_timeout(max(4000, int(narr_dur * 1000) + 1000))
                await clear_highlights(page)
                continue

            if step_type == "transition":
                print(f"  [{i+1}/{len(TOUR_STEPS)}] Transition")
                await show_transition(page)
                await page.wait_for_timeout(600)
                narr_offset = time.time() - recording_start
                narr_path = NARRATION_DIR / f"narr_{i:02d}.mp3"
                if narr_path.exists():
                    audio_events.append((narr_offset, "narration", str(narr_path)))
                narr_dur = await play_narration(page, i)
                await page.wait_for_timeout(max(3000, int(narr_dur * 1000) + 1000))
                await clear_highlights(page)
                continue

            hl_num += 1

            # ── Execute actions (click, scroll, close detail) ──
            action = step.get("action")
            if action:
                # Close doctor detail panel if requested
                if action.get("close_detail"):
                    close_btn = page.locator(".detail-close")
                    if await close_btn.count() > 0:
                        try:
                            if await close_btn.first.is_visible():
                                await close_btn.first.click()
                                await page.wait_for_timeout(400)
                        except Exception:
                            pass
                # Click tab or element
                if "click" in action:
                    await page.locator(action["click"]).first.click()
                    await page.wait_for_timeout(600)
                # Scroll to element
                if "scroll" in action:
                    await page.evaluate(f"""() => {{
                        document.querySelector('{action["scroll"]}')
                            ?.scrollIntoView({{behavior:'smooth',block:'center'}});
                    }}""")
                    await page.wait_for_timeout(500)

            # ── Show highlight + annotation ──
            sel = step["selector"]
            text = step["text"]
            pos = step.get("pos", "bottom")
            await highlight_element(page, sel, text, pos, hl_num, total_hl)

            # ── IMPORTANT: Wait for visual to render in video frame ──
            # This ensures the UI change is visible BEFORE narration starts
            await page.wait_for_timeout(600)

            # ── Record timestamp and play narration ──
            narr_offset = time.time() - recording_start
            narr_path = NARRATION_DIR / f"narr_{i:02d}.mp3"
            if narr_path.exists():
                audio_events.append((narr_offset, "narration", str(narr_path)))

            narr_dur = await play_narration(page, i)
            wait_ms = max(4000, int(narr_dur * 1000) + 500)
            print(f"  [{i+1}/{len(TOUR_STEPS)}] {sel} — narr {narr_dur:.1f}s @ {narr_offset:.1f}s")
            await page.wait_for_timeout(wait_ms)

        await clear_highlights(page)
        print("\n  UI Tour complete.\n")

        # ══════════════════════════════════════════════════════════
        #  PHASE 2: CONVERSATION
        # ══════════════════════════════════════════════════════════
        print("═══ PHASE 2: CONVERSATION ═══\n")

        # Switch to Hospital tab, close doctor detail if open
        await page.locator(".right-tab:nth-child(1)").click()
        await page.wait_for_timeout(500)
        detail_close = page.locator(".detail-close")
        if await detail_close.count() > 0 and await detail_close.first.is_visible():
            await detail_close.first.click()
            await page.wait_for_timeout(300)

        # Start session
        print("  Starting session...")
        await page.locator("#btnStart").click()
        await page.wait_for_timeout(3000)

        # Resume patient AudioContext
        ctx_state = await page.evaluate("window.__patientCtx?.state || 'N/A'")
        print(f"  Patient AudioContext: {ctx_state}")
        if ctx_state == 'suspended':
            await page.evaluate("window.__patientCtx.resume()")

        # Wait for bot greeting
        print("  Waiting for bot greeting...")
        msg_count = 0
        start = time.time()
        while time.time() - start < 15:
            msg_count = await get_msg_count(page)
            status = await get_status_class(page)
            if msg_count > 0 and 'listening' in status:
                break
            await page.wait_for_timeout(500)
        print(f"  Bot greeted. Messages: {msg_count}")

        # Play clips with dynamic timing
        for i, (clip_file, label, b64, sr, dur) in enumerate(clip_data):
            print(f"\n  Clip {i+1}/6: {label} ({dur:.1f}s)")

            # Track clip timestamp
            clip_offset = time.time() - recording_start
            clip_path = INPUT_DIR / clip_file
            audio_events.append((clip_offset, "clip", str(clip_path)))

            # Inject into patient context only (WebRTC for STT)
            await page.evaluate(f"""async () => {{
                const raw = atob("{b64}");
                const bytes = new Uint8Array(raw.length);
                for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
                const f32 = new Float32Array(bytes.buffer);

                const pCtx = window.__patientCtx;
                const pDest = window.__patientDest;
                if (pCtx && pDest) {{
                    if (pCtx.state === 'suspended') await pCtx.resume();
                    const buf = pCtx.createBuffer(1, f32.length, {sr});
                    buf.getChannelData(0).set(f32);
                    const src = pCtx.createBufferSource();
                    src.buffer = buf;
                    src.connect(pDest);
                    src.start();
                }}
            }}""")

            prev = msg_count
            msg_count = await wait_for_response(page, prev, dur)
            print(f"  Messages: {prev} -> {msg_count}  (clip @ {clip_offset:.1f}s)")

        # Final wait
        print("\n  Final wait (5s)...")
        await page.wait_for_timeout(5000)

        # Screenshot
        ss = CACHE_DIR / "demo_screenshot.png"
        await page.screenshot(path=str(ss))
        final_msgs = await get_msg_count(page)
        print(f"  Screenshot saved. Total messages: {final_msgs}")

        # End session
        print("  Ending session...")
        stop_btn = page.locator("#btnStop")
        if await stop_btn.is_visible():
            await stop_btn.click()
            await page.wait_for_timeout(2000)

        # End card
        await page.evaluate("""() => {
            const el = document.createElement('div');
            el.className = 'demo-ov demo-bg';
            el.innerHTML = `
                <div style="text-align:center;color:white;">
                    <div style="font-size:48px;margin-bottom:16px;">✅</div>
                    <h2 style="font-size:32px;margin:0;">Demo Complete</h2>
                    <p style="font-size:18px;margin-top:12px;opacity:0.8;">
                        AI-Powered Patient Screening & Doctor Referral
                    </p>
                </div>`;
            document.body.appendChild(el);
        }""")
        await page.wait_for_timeout(3000)

        # Record total duration
        total_duration = time.time() - recording_start

        # ── Bot responses captured via WebSocket ──
        print(f"\n  Bot responses captured: {len(bot_responses)}")
        for ts, text in bot_responses:
            print(f"    @ {ts:.1f}s: {text[:70]}...")

        video_path = await page.video.path()
        await page.close()
        await context.close()
        await browser.close()

    # ── Copy video ──
    if video_path and Path(video_path).exists():
        if OUTPUT_VIDEO.exists():
            OUTPUT_VIDEO.unlink()
        shutil.copy2(str(video_path), str(OUTPUT_VIDEO))
        print(f"\n  Video: {OUTPUT_VIDEO} ({OUTPUT_VIDEO.stat().st_size//1024}KB)")
    else:
        print(f"\n  ERROR: No video at {video_path}")
        return

    # ══════════════════════════════════════════════════════════
    #  POST-PROCESSING: GENERATE BOT TTS FROM CAPTURED TEXT
    # ══════════════════════════════════════════════════════════
    if bot_responses:
        print(f"\n  Generating TTS for {len(bot_responses)} bot responses...")
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / "Model-End" / ".env")
        from openai import OpenAI
        client = OpenAI()

        bot_tts_dir = CACHE_DIR / "bot_tts"
        bot_tts_dir.mkdir(exist_ok=True)
        # Clean old files
        for f in bot_tts_dir.glob("bot_*.mp3"):
            f.unlink()

        for i, (ts, text) in enumerate(bot_responses):
            tts_path = bot_tts_dir / f"bot_{i:02d}.mp3"
            try:
                response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=text,
                )
                response.stream_to_file(str(tts_path))
                # Add 1s delay to account for TTS processing latency in original
                audio_events.append((ts + 1.0, "bot_tts", str(tts_path)))
                print(f"    bot_{i:02d}.mp3 @ {ts:.1f}s: {text[:50]}...")
            except Exception as e:
                print(f"    ERROR generating bot TTS {i}: {e}")
    else:
        print("\n  No bot responses captured via WebSocket.")

    # ══════════════════════════════════════════════════════════
    #  POST-PROCESSING: BUILD AUDIO FROM TIMESTAMPS
    # ══════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("  POST-PROCESSING: Building audio from source files")
    print(f"{'='*60}")
    print(f"  Total recording duration: {total_duration:.1f}s")
    print(f"  Audio events tracked: {len(audio_events)}")

    audio_built = build_audio_from_events(audio_events, total_duration)

    # ── Mux video + built audio ──
    if audio_built and BUILT_AUDIO.exists():
        print("\n  Muxing video + built audio...")
        if FINAL_OUTPUT.exists():
            FINAL_OUTPUT.unlink()

        r = subprocess.run([
            'ffmpeg', '-y',
            '-i', str(OUTPUT_VIDEO),
            '-i', str(BUILT_AUDIO),
            '-c:v', 'libx264', '-preset', 'fast',
            '-af', 'volume=3dB,alimiter=limit=0.95',
            '-c:a', 'aac', '-b:a', '192k',
            '-map', '0:v:0', '-map', '1:a:0',
            '-shortest',
            str(FINAL_OUTPUT)
        ], capture_output=True, text=True)

        if r.returncode == 0 and FINAL_OUTPUT.exists():
            r3 = subprocess.run([
                'ffmpeg', '-i', str(FINAL_OUTPUT),
                '-af', 'volumedetect', '-f', 'null', '-'
            ], capture_output=True, text=True)
            vol_info = ""
            for line in r3.stderr.split('\n'):
                if 'mean_volume' in line or 'max_volume' in line:
                    vol_info += "  " + line.strip() + "\n"

            size_kb = FINAL_OUTPUT.stat().st_size // 1024
            print(f"\n{'='*60}")
            print(f"  DEMO VIDEO: {FINAL_OUTPUT}")
            print(f"  Size: {size_kb}KB")
            if vol_info:
                print(f"  Audio levels:\n{vol_info.rstrip()}")
            narr_count = sum(1 for _, t, _ in audio_events if t == 'narration')
            clip_count = sum(1 for _, t, _ in audio_events if t == 'clip')
            bot_count = sum(1 for _, t, _ in audio_events if t == 'bot_tts')
            print(f"\n  Audio: {narr_count} narrations, {clip_count} patient clips, {bot_count} bot responses")
            print(f"{'='*60}")
        else:
            print(f"  ffmpeg mux error: {r.stderr[-400:]}")
            shutil.copy2(str(OUTPUT_VIDEO), str(FINAL_OUTPUT))
    else:
        print(f"  No audio built. Video-only output.")
        shutil.copy2(str(OUTPUT_VIDEO), str(FINAL_OUTPUT))


# ── Validation test ─────────────────────────────────────────────

async def validate_demo():
    """Quick test: verify narrations exist, clips are valid, server is up."""
    print("\n  VALIDATION TEST")
    print("  " + "─" * 40)

    errors = []

    for i in range(len(NARRATIONS)):
        path = NARRATION_DIR / f"narr_{i:02d}.mp3"
        if not path.exists() or path.stat().st_size < 1000:
            errors.append(f"Missing/empty narration: {path.name}")
        else:
            print(f"  OK narr_{i:02d}.mp3 ({path.stat().st_size//1024}KB)")

    for clip_file, label in AUDIO_CLIPS:
        path = INPUT_DIR / clip_file
        if not path.exists():
            errors.append(f"Missing clip: {clip_file}")
        else:
            with wave.open(str(path), 'rb') as wf:
                dur = wf.getnframes() / wf.getframerate()
            if dur < 0.3:
                errors.append(f"Clip too short: {clip_file} ({dur:.1f}s)")
            else:
                print(f"  OK {clip_file} ({dur:.1f}s)")

    import urllib.request
    try:
        urllib.request.urlopen(AGENT_URL, timeout=3)
        print(f"  OK Server running at {AGENT_URL}")
    except Exception:
        errors.append(f"Server not running at {AGENT_URL}")

    try:
        async with async_playwright() as p_test:
            b = await p_test.chromium.launch(headless=True)
            await b.close()
        print("  OK Playwright Chromium")
    except Exception as e:
        errors.append(f"Playwright error: {e}")

    r = subprocess.run(['ffmpeg', '-version'], capture_output=True)
    if r.returncode == 0:
        print("  OK ffmpeg")
    else:
        errors.append("ffmpeg not found")

    if len(TOUR_STEPS) != len(NARRATIONS):
        errors.append(f"TOUR_STEPS ({len(TOUR_STEPS)}) != NARRATIONS ({len(NARRATIONS)})")
    else:
        print(f"  OK {len(TOUR_STEPS)} tour steps = {len(NARRATIONS)} narrations")

    test_narr = NARRATION_DIR / "narr_00.mp3"
    if test_narr.exists():
        pcm = decode_audio_to_pcm(test_narr)
        if len(pcm) > 0:
            print(f"  OK Audio decode pipeline ({len(pcm)/SAMPLE_RATE:.1f}s)")
        else:
            errors.append("Audio decode pipeline failed")

    print("  " + "─" * 40)
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
        return False
    else:
        print("  ALL CHECKS PASSED")
        return True


if __name__ == "__main__":
    if "--validate" in sys.argv:
        asyncio.run(validate_demo())
    else:
        asyncio.run(run_demo())
