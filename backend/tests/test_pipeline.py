import json

from app.prompts import static_fallback
from app.stt import MockSTT

from tests.helpers import make_pipeline, tone_pcm


class Boom:
    def complete(self, messages):
        raise RuntimeError("network down")


class RecordingChat:
    def __init__(self):
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        return "Noted."


class BrokenTTS:
    def synthesize(self, text):
        raise RuntimeError("piper failed")


def test_text_turn_updates_memory_and_logs_latency(tmp_path):
    path = tmp_path / "latency.jsonl"
    pipeline = make_pipeline(path=str(path))
    session = pipeline.new_session()
    result = pipeline.run_text(session, "hello")
    assert result.intent == "greeting"
    assert result.used_fallback is False
    assert "VoiceDesk" in result.reply
    assert result.audio[:4] == b"RIFF"
    assert len(session.memory) == 1
    assert result.latency.stt_ms == 0.0
    assert result.latency.llm_ms >= 0
    assert result.latency.total_ms >= 0
    line = json.loads(path.read_text(encoding="utf-8").strip())
    assert line["event"] == "turn_latency"
    assert line["intent"] == "greeting"
    assert "hello" not in path.read_text(encoding="utf-8")
    assert line["transcript_chars"] == len("hello")


def test_follow_up_keeps_the_earlier_turn():
    chat = RecordingChat()
    pipeline = make_pipeline(llm=chat)
    session = pipeline.new_session()
    pipeline.run_text(session, "remind me to buy milk")
    pipeline.run_text(session, "what about tomorrow?")
    second = chat.messages[1][0]["content"]
    assert "buy milk" in second
    assert chat.messages[1][-1]["content"] == "what about tomorrow?"


def test_unknown_intent_asks_for_clarification():
    pipeline = make_pipeline()
    result = pipeline.run_text(pipeline.new_session(), "purple bicycle warranty")
    assert result.used_fallback is True
    assert result.fallback_reason == "unknown_intent"
    assert "didn't quite catch" in result.reply


def test_model_failure_uses_the_static_line():
    pipeline = make_pipeline(llm=Boom())
    result = pipeline.run_text(pipeline.new_session(), "hello")
    assert result.used_fallback is True
    assert result.fallback_reason == "llm_error"
    assert result.reply == static_fallback("greeting")
    assert result.audio[:4] == b"RIFF"


def test_silence_skips_the_model():
    chat = RecordingChat()
    pipeline = make_pipeline(llm=chat)
    session = pipeline.new_session()
    silence = bytes(16000 * 2)  # one second of zeros at 16 kHz
    result = pipeline.run_audio(session, silence, 16000)
    assert result.empty is True
    assert chat.messages == []
    assert len(session.memory) == 0
    assert result.latency.empty is True


def test_short_clip_is_empty():
    pipeline = make_pipeline()
    result = pipeline.run_audio(pipeline.new_session(), tone_pcm(seconds=0.05), 16000)
    assert result.empty is True


def test_audio_turn_transcribes_and_speaks():
    stt = MockSTT("hello there")
    pipeline = make_pipeline()
    pipeline._stt = stt
    result = pipeline.run_audio(pipeline.new_session(), tone_pcm(), 16000)
    assert stt.calls == 1
    assert result.transcript == "hello there"
    assert result.intent == "greeting"
    assert result.empty is False


def test_48k_audio_is_resampled_before_transcription():
    stt = MockSTT("hello there")
    pipeline = make_pipeline()
    pipeline._stt = stt
    result = pipeline.run_audio(pipeline.new_session(), tone_pcm(rate=48000), 48000)
    assert result.transcript == "hello there"


def test_tts_failure_still_returns_the_text():
    pipeline = make_pipeline()
    pipeline._tts = BrokenTTS()
    result = pipeline.run_text(pipeline.new_session(), "what time is it?")
    assert result.reply.startswith("You asked")
    assert result.audio == b""
    assert result.latency.audio_bytes == 0
