import json

from app.api import create_app
from app.config import Settings
from fastapi.testclient import TestClient

from tests.helpers import make_pipeline, tone_pcm


class RecordingChat:
    def __init__(self):
        self.messages = []

    def complete(self, messages):
        self.messages.append(messages)
        return "Noted."


def read_until_done(ws):
    messages = []
    audio = bytearray()
    while True:
        message = ws.receive()
        if message["type"] != "websocket.send":
            raise AssertionError(message)
        if message.get("text") is not None:
            payload = json.loads(message["text"])
            messages.append(payload)
            if payload["type"] == "done":
                break
        elif message.get("bytes") is not None:
            audio.extend(message["bytes"])
        else:
            raise AssertionError(message)
    return messages, bytes(audio)


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {
        "stt_backend": "mock",
        "llm_backend": "mock",
        "tts_backend": "mock",
    }
    assert client.get("/").json()["name"] == "VoiceDesk"


def test_text_turn_over_the_socket(client):
    with client.websocket_connect("/ws") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "ready"
        assert ready["sample_rate"] == 16000
        ws.send_json({"type": "text", "text": "hello"})
        messages, audio = read_until_done(ws)
    kinds = [item["type"] for item in messages]
    assert kinds == ["transcript", "status", "reply", "status", "latency", "done"]
    assert messages[0]["text"] == "hello"
    assert messages[1]["stage"] == "thinking"
    assert messages[2]["intent"] == "greeting"
    assert messages[2]["used_fallback"] is False
    assert "VoiceDesk" in messages[2]["text"]
    assert messages[3]["stage"] == "speaking"
    assert messages[4]["event"] == "turn_latency"
    assert messages[4]["stt_ms"] == 0
    assert audio[:4] == b"RIFF"


def test_unknown_text_uses_fallback(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "text", "text": "purple bicycle warranty"})
        messages, _audio = read_until_done(ws)
    reply = next(item for item in messages if item["type"] == "reply")
    assert reply["used_fallback"] is True
    assert reply["intent"] == "unknown"
    assert reply["fallback_reason"] == "unknown_intent"


def test_pcm_turn(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "sample_rate": 16000})
        assert ws.receive_json()["stage"] == "listening"
        ws.send_bytes(tone_pcm())
        ws.send_json({"type": "stop"})
        messages, audio = read_until_done(ws)
    kinds = [item["type"] for item in messages]
    assert kinds[0] == "status"
    assert messages[0]["stage"] == "transcribing"
    transcript = next(item for item in messages if item["type"] == "transcript")
    assert transcript["text"] == "hello there"
    reply = next(item for item in messages if item["type"] == "reply")
    assert reply["intent"] == "greeting"
    assert audio[:4] == b"RIFF"


def test_silence_does_not_invent_a_transcript(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "start", "sample_rate": 16000})
        ws.receive_json()
        ws.send_bytes(bytes(16000 * 2))
        ws.send_json({"type": "stop"})
        messages, audio = read_until_done(ws)
    assert messages[0]["stage"] == "transcribing"
    assert messages[1]["type"] == "empty"
    assert audio == b""
    assert messages[-1]["type"] == "done"


def test_reset_drops_memory(tmp_path):
    chat = RecordingChat()
    pipeline = make_pipeline(llm=chat, path=str(tmp_path / "latency.jsonl"))
    app = create_app(Settings(_env_file=None), pipeline)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        ws.receive_json()
        for _ in range(2):
            ws.send_json({"type": "text", "text": "hello"})
            read_until_done(ws)
        ws.send_json({"type": "reset"})
        assert ws.receive_json()["type"] == "reset"
        ws.send_json({"type": "text", "text": "hello"})
        read_until_done(ws)
    assert "no earlier conversation" in chat.messages[0][0]["content"]
    assert "Earlier turns" in chat.messages[1][0]["content"]
    assert "no earlier conversation" in chat.messages[2][0]["content"]


def test_blank_text_is_rejected(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        ws.send_json({"type": "text", "text": "   "})
        messages, _audio = read_until_done(ws)
    assert messages[0]["type"] == "error"
    assert messages[-1]["type"] == "done"
