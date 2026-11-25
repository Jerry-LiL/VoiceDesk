import json

import httpx
import numpy as np
import pytest
from app.config import Settings
from app.llm import OllamaChat, OpenAIChat
from app.stt import FasterWhisperSTT
from app.tts import MockTTS, PiperTTS
from app.wiring import build_llm, build_stt


def test_speech_libraries_are_installed():
    import faster_whisper
    import piper
    from openai import OpenAI

    assert callable(faster_whisper.WhisperModel)
    assert callable(piper.PiperVoice.load)
    assert OpenAI is not None


def test_whisper_backend_constructs_the_model(monkeypatch):
    created = {}

    class FakeModel:
        def __init__(self, model_size, device, compute_type):
            created["args"] = (model_size, device, compute_type)

        def transcribe(self, audio, language=None, vad_filter=False, beam_size=1):
            created["call"] = {
                "language": language,
                "vad_filter": vad_filter,
                "beam_size": beam_size,
                "dtype": str(audio.dtype),
            }
            segment = type("Segment", (), {"text": " kitchen "})()
            return [segment], None

    monkeypatch.setattr("faster_whisper.WhisperModel", FakeModel)
    settings = Settings(
        _env_file=None,
        stt_backend="whisper",
        whisper_model="tiny",
        whisper_device="cpu",
        whisper_compute_type="int8",
        whisper_language="en",
    )
    stt = build_stt(settings)
    assert isinstance(stt, FasterWhisperSTT)
    assert stt.transcribe(np.zeros(160, dtype=np.float32), 16000) == "kitchen"
    assert created["args"] == ("tiny", "cpu", "int8")
    assert created["call"]["vad_filter"] is True
    assert created["call"]["language"] == "en"


def test_openai_chat_sends_the_prompt(monkeypatch):
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["create"] = kwargs
            message = type("Message", (), {"content": "Spoken reply"})()
            choice = type("Choice", (), {"message": message})()
            return type("Response", (), {"choices": [choice]})()

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured["init"] = kwargs
            self.chat = FakeChat()

    monkeypatch.setattr("app.llm.OpenAI", FakeOpenAI)
    chat = OpenAIChat(
        api_key="sk-test",
        model="gpt-4o-mini",
        base_url="https://example.invalid/v1",
    )
    assert chat.complete([{"role": "user", "content": "hi"}]) == "Spoken reply"
    assert captured["init"]["api_key"] == "sk-test"
    assert captured["init"]["base_url"] == "https://example.invalid/v1"
    assert captured["create"]["model"] == "gpt-4o-mini"
    assert captured["create"]["temperature"] == 0.4
    assert captured["create"]["messages"][0]["content"] == "hi"


def test_openai_backend_requires_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = Settings(_env_file=None, llm_backend="openai", openai_api_key="")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        build_llm(settings)


def test_ollama_posts_a_chat_request():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["body"] = json.loads(request.content.decode())
        return httpx.Response(
            200, json={"message": {"role": "assistant", "content": "Hello from Ollama"}}
        )

    client = httpx.Client(
        base_url="http://ollama",
        transport=httpx.MockTransport(handler),
    )
    chat = OllamaChat("http://ollama", "llama3.2", timeout=10, client=client)
    assert chat.complete([{"role": "user", "content": "hi"}]) == "Hello from Ollama"
    assert seen["path"] == "/api/chat"
    assert seen["body"]["model"] == "llama3.2"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"][0]["content"] == "hi"


def test_ollama_error_status_raises():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"error": "down"})

    client = httpx.Client(
        base_url="http://ollama",
        transport=httpx.MockTransport(handler),
    )
    chat = OllamaChat("http://ollama", "llama3.2", client=client)
    try:
        chat.complete([{"role": "user", "content": "hi"}])
    except httpx.HTTPStatusError:
        return
    raise AssertionError("expected a status error")


def test_piper_requires_a_model_file(tmp_path):
    missing = tmp_path / "missing.onnx"
    try:
        PiperTTS(str(missing))
    except RuntimeError as exc:
        assert "Piper model not found" in str(exc)
        return
    raise AssertionError("expected a missing-model error")


def test_piper_writes_a_wav(monkeypatch, tmp_path):
    model = tmp_path / "voice.onnx"
    model.write_bytes(b"not a real model")

    class FakeVoice:
        def synthesize_wav(self, text, wav_file, syn_config=None, set_wav_format=True):
            assert text == "hello"
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(16000)
            wav_file.writeframes(b"\x01\x00" * 80)

    class FakePiper:
        @staticmethod
        def load(path, use_cuda=False):
            assert str(path).endswith("voice.onnx")
            assert use_cuda is False
            return FakeVoice()

    monkeypatch.setattr("piper.PiperVoice", FakePiper)
    wav = PiperTTS(str(model)).synthesize("hello")
    assert wav[:4] == b"RIFF"


def test_mock_tts_is_a_wav():
    wav = MockTTS().synthesize("hello")
    assert wav[:4] == b"RIFF"
    assert len(wav) > 44
