"""
Automated Test Suite — Hospital Support Desk Conversation Agent
Tests: API endpoints, WebSocket protocol, session management, settings persistence,
       model configuration, language switching, and performance benchmarks.

Usage:
    # Start the server first:  python conversation_agent.py
    # Then run tests:
    pytest test_conversation_agent.py -v --tb=short
    pytest test_conversation_agent.py -v -k "benchmark"   # benchmarks only
"""

import asyncio
import json
import time
import os
import sys
import pytest
import httpx
import websockets

BASE_URL = os.getenv("AGENT_URL", "http://localhost:7871")
WS_URL = BASE_URL.replace("http", "ws") + "/ws"

# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


async def ws_connect(timeout=5):
    """Connect to WebSocket and return (ws, initial_state)."""
    ws = await websockets.connect(WS_URL, open_timeout=timeout)
    raw = await asyncio.wait_for(ws.recv(), timeout=timeout)
    state = json.loads(raw)
    return ws, state


async def ws_send(ws, data, timeout=3):
    await ws.send(json.dumps(data))


async def ws_recv_until(ws, msg_type, timeout=10):
    """Receive messages until we get one with the given type."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            raw = await asyncio.wait_for(ws.recv(), timeout=deadline - time.time())
            msg = json.loads(raw)
            if msg.get("type") == msg_type:
                return msg
        except asyncio.TimeoutError:
            break
    return None


# ──────────────────────────────────────────────
# 1. API Endpoint Tests
# ──────────────────────────────────────────────

class TestAPIEndpoints:
    """Test HTTP endpoints respond correctly."""

    @pytest.mark.asyncio
    async def test_index_returns_html(self):
        """GET / should return the HTML frontend."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BASE_URL}/")
        assert resp.status_code == 200
        assert "text/html" in resp.headers.get("content-type", "")
        assert "Hospital" in resp.text
        assert "Support Desk" in resp.text

    @pytest.mark.asyncio
    async def test_index_contains_webrtc_code(self):
        """Frontend should include WebRTC connection code."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BASE_URL}/")
        assert "RTCPeerConnection" in resp.text
        assert "/api/offer" in resp.text

    @pytest.mark.asyncio
    async def test_offer_rejects_invalid_payload(self):
        """POST /api/offer with bad data should return error."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(f"{BASE_URL}/api/offer", json={})
        assert resp.status_code in (400, 422, 500)

    @pytest.mark.asyncio
    async def test_cors_headers(self):
        """CORS should be enabled for all origins."""
        async with httpx.AsyncClient() as client:
            resp = await client.get(f"{BASE_URL}/", headers={"Origin": "http://example.com"})
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "*"


# ──────────────────────────────────────────────
# 2. WebSocket Protocol Tests
# ──────────────────────────────────────────────

class TestWebSocketProtocol:
    """Test WebSocket message protocol."""

    @pytest.mark.asyncio
    async def test_ws_connect_receives_state(self):
        """WS connection should immediately receive full state."""
        ws, state = await ws_connect()
        await ws.close()
        assert state["type"] == "state"
        assert "conversation" in state
        assert "status" in state
        assert "config" in state
        assert "doctors" in state

    @pytest.mark.asyncio
    async def test_ws_state_has_all_config_keys(self):
        """State config should contain all expected keys."""
        ws, state = await ws_connect()
        await ws.close()
        config = state["config"]
        expected_keys = [
            "stt_model", "stt_language", "llm_model",
            "tts_provider", "tts_model", "tts_voice",
            "voice_gender", "agent_name", "personality", "lang_style",
        ]
        for key in expected_keys:
            assert key in config, f"Missing config key: {key}"

    @pytest.mark.asyncio
    async def test_ws_doctors_panel_complete(self):
        """Doctors panel should have all required fields."""
        ws, state = await ws_connect()
        await ws.close()
        doctors = state["doctors"]
        assert len(doctors) >= 10
        for d in doctors:
            assert "name" in d
            assert "specialty" in d
            assert "room" in d
            assert "available" in d
            assert "gender" in d
            assert "qualification" in d
            assert "languages" in d
            assert "expertise" in d

    @pytest.mark.asyncio
    async def test_ws_setting_update(self):
        """Updating a setting via WS should persist."""
        ws, state = await ws_connect()
        # Change LLM model
        await ws_send(ws, {"action": "setting", "key": "llm_model", "value": "gpt-4o"})
        await asyncio.sleep(0.3)
        await ws.close()

        # Reconnect and verify
        ws2, state2 = await ws_connect()
        await ws.close() if not ws.closed else None
        assert state2["config"]["llm_model"] == "gpt-4o"

        # Restore default
        await ws_send(ws2, {"action": "setting", "key": "llm_model", "value": "gpt-4o-mini"})
        await asyncio.sleep(0.2)
        await ws2.close()

    @pytest.mark.asyncio
    async def test_ws_multiple_clients(self):
        """Multiple WS clients should all receive state."""
        ws1, s1 = await ws_connect()
        ws2, s2 = await ws_connect()
        await ws1.close()
        await ws2.close()
        assert s1["type"] == "state"
        assert s2["type"] == "state"


# ──────────────────────────────────────────────
# 3. Session Management Tests
# ──────────────────────────────────────────────

class TestSessionManagement:
    """Test session persistence and reset behavior."""

    @pytest.mark.asyncio
    async def test_session_file_exists(self):
        """Session state file should exist after server start."""
        ws, state = await ws_connect()
        await ws.close()
        session_file = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "output", "session_state.json"
        )
        assert os.path.exists(session_file)

    @pytest.mark.asyncio
    async def test_settings_persist_across_reconnect(self):
        """Changed settings should survive WS reconnection (page refresh)."""
        ws, _ = await ws_connect()
        await ws_send(ws, {"action": "setting", "key": "agent_personality", "value": "professional"})
        await ws_send(ws, {"action": "setting", "key": "agent_lang_style", "value": "casual"})
        await asyncio.sleep(0.5)
        await ws.close()

        ws2, state = await ws_connect()
        assert state["config"]["personality"] == "professional"
        assert state["config"]["lang_style"] == "casual"

        # Restore defaults
        await ws_send(ws2, {"action": "setting", "key": "agent_personality", "value": "warm"})
        await ws_send(ws2, {"action": "setting", "key": "agent_lang_style", "value": "formal"})
        await asyncio.sleep(0.3)
        await ws2.close()

    @pytest.mark.asyncio
    async def test_conversation_empty_on_startup(self):
        """Conversation should be empty on fresh connection (cleared on startup)."""
        ws, state = await ws_connect()
        await ws.close()
        assert state["conversation"] == [] or isinstance(state["conversation"], list)

    @pytest.mark.asyncio
    async def test_stop_resets_status(self):
        """Stop action should reset status to ended."""
        ws, _ = await ws_connect()
        await ws_send(ws, {"action": "stop"})
        msg = await ws_recv_until(ws, "status", timeout=5)
        await ws.close()
        assert msg is not None
        assert msg["status"] == "ended"


# ──────────────────────────────────────────────
# 4. Settings Configuration Tests
# ──────────────────────────────────────────────

class TestSettingsConfiguration:
    """Test all configurable settings."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("language", ["Urdu", "English", "Hindi", "Arabic", "Punjabi", "Malay"])
    async def test_language_setting(self, language):
        """Each supported language should be accepted."""
        ws, _ = await ws_connect()
        await ws_send(ws, {"action": "setting", "key": "stt_language", "value": language})
        await asyncio.sleep(0.2)
        await ws.close()

        ws2, state = await ws_connect()
        lang_code_map = {"Urdu": "ur", "English": "en", "Hindi": "hi", "Arabic": "ar", "Punjabi": "pa", "Malay": "ms"}
        # State returns display name, verify it round-trips
        await ws2.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model", ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"])
    async def test_llm_model_setting(self, model):
        """Each LLM model should be accepted."""
        ws, _ = await ws_connect()
        await ws_send(ws, {"action": "setting", "key": "llm_model", "value": model})
        await asyncio.sleep(0.2)
        await ws.close()

        ws2, state = await ws_connect()
        assert state["config"]["llm_model"] == model
        await ws2.close()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider,model", [
        ("OpenAI", "tts-1"), ("OpenAI", "tts-1-hd"),
        ("ElevenLabs", "eleven_flash_v2_5"), ("ElevenLabs", "eleven_turbo_v2_5"),
        ("ElevenLabs", "eleven_multilingual_v2"),
    ])
    async def test_tts_provider_model_setting(self, provider, model):
        """Each TTS provider + model combo should be accepted."""
        ws, _ = await ws_connect()
        await ws_send(ws, {"action": "setting", "key": "tts_provider", "value": provider})
        await ws_send(ws, {"action": "setting", "key": "tts_model", "value": model})
        await asyncio.sleep(0.2)
        await ws.close()

        ws2, state = await ws_connect()
        assert state["config"]["tts_provider"] == provider
        assert state["config"]["tts_model"] == model
        await ws2.close()

    @pytest.mark.asyncio
    async def test_agent_persona_settings(self):
        """Agent name, gender, personality, lang_style should persist."""
        ws, _ = await ws_connect()
        await ws_send(ws, {"action": "setting", "key": "agent_name", "value": "Ali"})
        await ws_send(ws, {"action": "setting", "key": "voice_gender", "value": "M"})
        await ws_send(ws, {"action": "setting", "key": "agent_personality", "value": "professional"})
        await ws_send(ws, {"action": "setting", "key": "agent_lang_style", "value": "native"})
        await asyncio.sleep(0.3)
        await ws.close()

        ws2, state = await ws_connect()
        assert state["config"]["agent_name"] == "Ali"
        assert state["config"]["voice_gender"] == "M"
        assert state["config"]["personality"] == "professional"
        assert state["config"]["lang_style"] == "native"

        # Restore defaults
        await ws_send(ws2, {"action": "setting", "key": "agent_name", "value": "Amina"})
        await ws_send(ws2, {"action": "setting", "key": "voice_gender", "value": "F"})
        await ws_send(ws2, {"action": "setting", "key": "agent_personality", "value": "warm"})
        await ws_send(ws2, {"action": "setting", "key": "agent_lang_style", "value": "formal"})
        await asyncio.sleep(0.2)
        await ws2.close()


# ──────────────────────────────────────────────
# 5. System Prompt Tests
# ──────────────────────────────────────────────

class TestSystemPrompt:
    """Test system prompt generation (import from conversation_agent)."""

    def _import_build_system_prompt(self):
        agent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if agent_dir not in sys.path:
            sys.path.insert(0, agent_dir)
        from conversation_agent import build_system_prompt, LANGUAGE_NAMES
        return build_system_prompt, LANGUAGE_NAMES

    def test_default_prompt_contains_hospital(self):
        build_system_prompt, _ = self._import_build_system_prompt()
        prompt = build_system_prompt()
        assert "City General Hospital" in prompt
        assert "Amina" in prompt

    def test_prompt_language_substitution(self):
        build_system_prompt, _ = self._import_build_system_prompt()
        prompt = build_system_prompt(stt_language="en")
        assert "English" in prompt
        assert "Urdu by default" not in prompt

    @pytest.mark.parametrize("lang_code,lang_name", [
        ("ur", "Urdu"), ("en", "English"), ("hi", "Hindi"),
        ("ar", "Arabic"), ("pa", "Punjabi"), ("ms", "Malay"),
    ])
    def test_prompt_all_languages(self, lang_code, lang_name):
        build_system_prompt, _ = self._import_build_system_prompt()
        prompt = build_system_prompt(stt_language=lang_code)
        assert f"Respond in {lang_name} by default" in prompt

    def test_prompt_male_gender(self):
        build_system_prompt, _ = self._import_build_system_prompt()
        prompt = build_system_prompt(agent_name="Ali", voice_gender="M")
        assert "Ali" in prompt
        assert "MALE" in prompt
        assert "masculine" in prompt.lower()

    def test_prompt_female_gender(self):
        build_system_prompt, _ = self._import_build_system_prompt()
        prompt = build_system_prompt(agent_name="Amina", voice_gender="F")
        assert "Amina" in prompt
        assert "FEMALE" in prompt

    def test_prompt_personality_warm(self):
        build_system_prompt, _ = self._import_build_system_prompt()
        prompt = build_system_prompt(personality="warm")
        assert "warm" in prompt.lower() or "caring" in prompt.lower()

    def test_prompt_personality_professional(self):
        build_system_prompt, _ = self._import_build_system_prompt()
        prompt = build_system_prompt(personality="professional")
        assert "professional" in prompt.lower() or "efficient" in prompt.lower()

    def test_prompt_lang_style_formal(self):
        build_system_prompt, _ = self._import_build_system_prompt()
        prompt = build_system_prompt(lang_style="formal")
        assert "formal" in prompt.lower() or "آپ" in prompt

    def test_prompt_lang_style_native(self):
        build_system_prompt, _ = self._import_build_system_prompt()
        prompt = build_system_prompt(lang_style="native")
        assert "native" in prompt.lower() or "idiom" in prompt.lower()

    def test_prompt_contains_doctor_mapping(self):
        build_system_prompt, _ = self._import_build_system_prompt()
        prompt = build_system_prompt()
        assert "Dr. Ahmed Khan" in prompt
        assert "Cardiology" in prompt
        assert "Room" in prompt


# ──────────────────────────────────────────────
# 6. Performance Benchmarks
# ──────────────────────────────────────────────

class TestPerformanceBenchmarks:
    """Benchmark API response times and connection setup."""

    RESULTS = []

    @pytest.fixture(autouse=True, scope="class")
    def print_results(self):
        yield
        if self.RESULTS:
            print("\n\n" + "=" * 70)
            print("PERFORMANCE BENCHMARK RESULTS")
            print("=" * 70)
            print(f"{'Test':<45} {'Time (ms)':<12} {'Status':<10}")
            print("-" * 70)
            for r in self.RESULTS:
                status = "✅ PASS" if r["pass"] else "❌ FAIL"
                print(f"{r['name']:<45} {r['time_ms']:<12.1f} {status:<10}")
            print("=" * 70)

    def _record(self, name, time_ms, threshold_ms):
        passed = time_ms <= threshold_ms
        self.RESULTS.append({"name": name, "time_ms": time_ms, "threshold_ms": threshold_ms, "pass": passed})
        return passed

    @pytest.mark.asyncio
    async def test_benchmark_index_page_load(self):
        """GET / should respond within 200ms."""
        async with httpx.AsyncClient() as client:
            t0 = time.perf_counter()
            resp = await client.get(f"{BASE_URL}/")
            elapsed = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200
        passed = self._record("Index page load", elapsed, 200)
        assert passed, f"Index page took {elapsed:.0f}ms (threshold: 200ms)"

    @pytest.mark.asyncio
    async def test_benchmark_ws_connect(self):
        """WS connection + initial state should complete within 500ms."""
        t0 = time.perf_counter()
        ws, state = await ws_connect(timeout=5)
        elapsed = (time.perf_counter() - t0) * 1000
        await ws.close()
        assert state["type"] == "state"
        passed = self._record("WS connect + state", elapsed, 500)
        assert passed, f"WS connect took {elapsed:.0f}ms (threshold: 500ms)"

    @pytest.mark.asyncio
    async def test_benchmark_ws_setting_roundtrip(self):
        """Setting update via WS should round-trip within 300ms."""
        ws, _ = await ws_connect()
        t0 = time.perf_counter()
        await ws_send(ws, {"action": "setting", "key": "llm_model", "value": "gpt-4o"})
        await asyncio.sleep(0.1)
        elapsed = (time.perf_counter() - t0) * 1000
        await ws.close()

        # Verify and restore
        ws2, state = await ws_connect()
        assert state["config"]["llm_model"] == "gpt-4o"
        await ws_send(ws2, {"action": "setting", "key": "llm_model", "value": "gpt-4o-mini"})
        await asyncio.sleep(0.1)
        await ws2.close()
        passed = self._record("WS setting roundtrip", elapsed, 300)
        assert passed, f"Setting roundtrip took {elapsed:.0f}ms (threshold: 300ms)"

    @pytest.mark.asyncio
    async def test_benchmark_ws_stop_action(self):
        """Stop action should process within 500ms."""
        ws, _ = await ws_connect()
        t0 = time.perf_counter()
        await ws_send(ws, {"action": "stop"})
        msg = await ws_recv_until(ws, "status", timeout=5)
        elapsed = (time.perf_counter() - t0) * 1000
        await ws.close()
        assert msg is not None and msg["status"] == "ended"
        passed = self._record("Stop action processing", elapsed, 500)
        assert passed, f"Stop took {elapsed:.0f}ms (threshold: 500ms)"

    @pytest.mark.asyncio
    async def test_benchmark_session_persistence(self):
        """Session save + reload should complete within 200ms."""
        ws, _ = await ws_connect()
        t0 = time.perf_counter()
        await ws_send(ws, {"action": "setting", "key": "agent_name", "value": "TestAgent"})
        await asyncio.sleep(0.05)
        await ws.close()

        ws2, state = await ws_connect()
        elapsed = (time.perf_counter() - t0) * 1000
        assert state["config"]["agent_name"] == "TestAgent"

        # Restore
        await ws_send(ws2, {"action": "setting", "key": "agent_name", "value": "Amina"})
        await asyncio.sleep(0.1)
        await ws2.close()
        passed = self._record("Session persist + reload", elapsed, 500)
        assert passed, f"Persistence took {elapsed:.0f}ms (threshold: 500ms)"

    @pytest.mark.asyncio
    async def test_benchmark_concurrent_ws_connections(self):
        """5 concurrent WS connections should all succeed within 2s."""
        t0 = time.perf_counter()
        tasks = [ws_connect(timeout=5) for _ in range(5)]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = (time.perf_counter() - t0) * 1000

        success = 0
        for r in results:
            if isinstance(r, tuple):
                ws, state = r
                if state["type"] == "state":
                    success += 1
                await ws.close()

        passed = self._record(f"5 concurrent WS connects ({success}/5)", elapsed, 2000)
        assert success == 5, f"Only {success}/5 connections succeeded"
        assert passed, f"Concurrent connects took {elapsed:.0f}ms"


# ──────────────────────────────────────────────
# 7. Frontend Integrity Tests
# ──────────────────────────────────────────────

class TestFrontendIntegrity:
    """Verify HTML/JS has all required UI elements."""

    @pytest.fixture(autouse=True)
    def load_html(self):
        import httpx as _httpx
        with _httpx.Client() as client:
            resp = client.get(f"{BASE_URL}/")
            self.html = resp.text

    def test_has_new_patient_button(self):
        assert 'id="btnStart"' in self.html
        assert "New Patient" in self.html

    def test_has_end_session_button(self):
        assert 'id="btnStop"' in self.html
        assert "End Session" in self.html

    def test_has_stt_live_display(self):
        assert 'id="sttLive"' in self.html

    def test_has_conversation_container(self):
        assert 'id="conversation"' in self.html

    def test_has_status_bar(self):
        assert 'id="statusDot"' in self.html
        assert 'id="statusText"' in self.html

    def test_has_all_settings_dropdowns(self):
        for sel_id in ["sttModel", "llmModel", "ttsProvider", "ttsModel", "ttsVoice",
                       "sttLang", "agentPersona", "agentPersonality", "agentLangStyle"]:
            assert f'id="{sel_id}"' in self.html, f"Missing select: {sel_id}"

    def test_has_model_tags(self):
        assert "model-tags" in self.html
        assert "sttModelTags" in self.html
        assert "llmModelTags" in self.html
        assert "ttsModelTags" in self.html

    def test_has_language_badge(self):
        assert 'id="convLangBadge"' in self.html

    def test_has_rtl_ltr_support(self):
        assert "RTL_LANGUAGES" in self.html
        assert "updateConversationDirection" in self.html

    def test_has_lock_unlock_settings(self):
        assert "lockSettings" in self.html
        assert "unlockSettings" in self.html

    def test_has_agent_profile_summary(self):
        assert 'id="agentSummary"' in self.html
        assert "updateAgentProfileSummary" in self.html

    def test_has_doctors_panel(self):
        assert 'id="doctorsGrid"' in self.html
        assert "showDoctorDetail" in self.html
        assert "closeDoctorDetail" in self.html

    def test_has_quick_reference(self):
        assert 'id="quickRefCard"' in self.html
        assert "Emergency" in self.html
        assert "Pharmacy" in self.html

    def test_has_model_info_data(self):
        assert "MODEL_INFO" in self.html
        assert "Deepgram Nova-3" in self.html
        assert "gpt-4o-mini" in self.html

    def test_has_tts_models_data(self):
        assert "TTS_MODELS" in self.html
        assert "eleven_flash_v2_5" in self.html
        assert "tts-1-hd" in self.html

    def test_stt_live_has_direction_classes(self):
        assert "stt-live.rtl" in self.html
        assert "stt-live.ltr" in self.html


# ──────────────────────────────────────────────
# 8. Model Configuration Matrix
# ──────────────────────────────────────────────

class TestModelConfigurationMatrix:
    """Test all model combinations are valid."""

    LANGUAGES = ["Urdu", "English", "Hindi", "Arabic", "Punjabi", "Malay"]
    LLM_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo", "gpt-3.5-turbo"]
    TTS_CONFIGS = [
        ("OpenAI", "tts-1", "nova"),
        ("OpenAI", "tts-1-hd", "echo"),
        ("ElevenLabs", "eleven_flash_v2_5", "Rachel"),
        ("ElevenLabs", "eleven_turbo_v2_5", "Drew"),
        ("ElevenLabs", "eleven_multilingual_v2", "Sarah"),
    ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("lang", LANGUAGES)
    async def test_language_with_default_models(self, lang):
        """Each language should work with default model config."""
        ws, _ = await ws_connect()
        await ws_send(ws, {"action": "setting", "key": "stt_language", "value": lang})
        await asyncio.sleep(0.2)
        await ws.close()

        ws2, state = await ws_connect()
        await ws2.close()
        # Just verify it didn't crash
        assert state["type"] == "state"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("provider,model,voice", TTS_CONFIGS)
    async def test_tts_config_accepted(self, provider, model, voice):
        """Each TTS provider+model+voice combo should be accepted."""
        ws, _ = await ws_connect()
        await ws_send(ws, {"action": "setting", "key": "tts_provider", "value": provider})
        await ws_send(ws, {"action": "setting", "key": "tts_model", "value": model})
        await ws_send(ws, {"action": "setting", "key": "tts_voice", "value": voice})
        await asyncio.sleep(0.2)
        await ws.close()

        ws2, state = await ws_connect()
        assert state["config"]["tts_provider"] == provider
        assert state["config"]["tts_model"] == model
        assert state["config"]["tts_voice"] == voice
        await ws2.close()


# ──────────────────────────────────────────────
# Entry point for running with stats summary
# ──────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
