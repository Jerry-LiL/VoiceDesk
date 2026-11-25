import asyncio
import json
import logging
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.config import Settings, configure_logging
from app.pipeline import VoicePipeline
from app.wiring import build_pipeline

log = logging.getLogger("voicedesk.api")

EMPTY_SPEECH = "I didn't catch any speech. Hold the button a little longer."


def create_app(
    settings: Settings | None = None,
    pipeline: VoicePipeline | None = None,
) -> FastAPI:
    settings = settings or Settings()
    configure_logging(settings.log_level)
    if pipeline is None:
        pipeline = build_pipeline(settings)

    app = FastAPI(title="VoiceDesk", version="0.1.0")
    app.state.settings = settings
    app.state.pipeline = pipeline

    @app.get("/")
    def root() -> dict:
        return {"name": "VoiceDesk", "ws": "/ws"}

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        return {
            "stt_backend": settings.stt_backend,
            "llm_backend": settings.llm_backend,
            "tts_backend": settings.tts_backend,
        }

    @app.websocket("/ws")
    async def socket(websocket: WebSocket) -> None:
        await handle_socket(websocket, pipeline, settings)

    return app


async def handle_socket(
    websocket: WebSocket,
    pipeline: VoicePipeline,
    settings: Settings,
) -> None:
    await websocket.accept()
    session = pipeline.new_session()
    recording = False
    busy = False
    sample_rate = 16000
    pcm = bytearray()
    log.info("session %s connected", session.session_id)
    await websocket.send_json(
        {"type": "ready", "session_id": session.session_id, "sample_rate": 16000}
    )
    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message.get("bytes") is not None:
                if recording:
                    room = int(sample_rate * 2 * settings.max_audio_seconds) - len(pcm)
                    if room > 0:
                        pcm.extend(message["bytes"][:room])
                continue
            if message.get("text") is None:
                continue
            try:
                data = json.loads(message["text"])
            except json.JSONDecodeError:
                await websocket.send_json(
                    {"type": "error", "message": "That message was not valid JSON."}
                )
                continue
            if not isinstance(data, dict):
                await websocket.send_json(
                    {"type": "error", "message": "That message was not a JSON object."}
                )
                continue
            kind = data.get("type")
            if busy and kind in {"start", "stop", "text"}:
                continue
            if kind == "start":
                try:
                    sample_rate = int(data.get("sample_rate") or 16000)
                except (TypeError, ValueError):
                    sample_rate = 0
                if not 8000 <= sample_rate <= 48000:
                    await websocket.send_json(
                        {
                            "type": "error",
                            "message": "Sample rate must be between 8000 and 48000.",
                        }
                    )
                    continue
                recording = True
                pcm.clear()
                await websocket.send_json({"type": "status", "stage": "listening"})
            elif kind == "stop":
                if not recording:
                    await websocket.send_json(
                        {"type": "error", "message": "No recording in progress."}
                    )
                    await websocket.send_json({"type": "done"})
                    continue
                recording = False
                audio = bytes(pcm)
                pcm.clear()
                busy = True
                try:
                    await handle_audio(websocket, pipeline, session, audio, sample_rate)
                finally:
                    busy = False
            elif kind == "text":
                busy = True
                try:
                    await handle_text(websocket, pipeline, session, str(data.get("text") or ""))
                finally:
                    busy = False
            elif kind == "reset":
                recording = False
                pcm.clear()
                session.memory.clear()
                session.turn_index = 0
                await websocket.send_json({"type": "reset"})
            else:
                await websocket.send_json({"type": "error", "message": "Unknown message type."})
    except WebSocketDisconnect:
        return
    finally:
        log.info("session %s closed", session.session_id)


async def handle_text(websocket, pipeline, session, text: str) -> None:
    cleaned = " ".join(text.strip().split())
    if not cleaned:
        await websocket.send_json({"type": "error", "message": "Say or type something first."})
        await websocket.send_json({"type": "done"})
        return
    started = time.perf_counter()
    await send_reply(websocket, pipeline, session, cleaned, stt_ms=0.0, started=started)


async def handle_audio(websocket, pipeline, session, pcm: bytes, sample_rate: int) -> None:
    await websocket.send_json({"type": "status", "stage": "transcribing"})
    try:
        heard = await asyncio.to_thread(pipeline.transcribe, pcm, sample_rate)
    except Exception:
        log.exception("transcription failed")
        await websocket.send_json({"type": "error", "message": "Transcription failed."})
        await websocket.send_json({"type": "done"})
        return
    if heard.silent:
        result = pipeline.commit_empty(session, heard)
        await websocket.send_json({"type": "empty", "message": EMPTY_SPEECH})
        await websocket.send_json(latency_message(result.latency))
        await websocket.send_json({"type": "done"})
        return
    await send_reply(
        websocket,
        pipeline,
        session,
        heard.text,
        stt_ms=heard.stt_ms,
        started=heard.started,
    )


async def send_reply(
    websocket,
    pipeline,
    session,
    transcript: str,
    stt_ms: float,
    started: float,
) -> None:
    await websocket.send_json({"type": "transcript", "text": transcript})
    await websocket.send_json({"type": "status", "stage": "thinking"})
    draft = await asyncio.to_thread(pipeline.compose_reply, session, transcript)
    await websocket.send_json(
        {
            "type": "reply",
            "text": draft.reply,
            "intent": draft.intent,
            "confidence": draft.confidence,
            "used_fallback": draft.used_fallback,
            "fallback_reason": draft.fallback_reason,
        }
    )
    await websocket.send_json({"type": "status", "stage": "speaking"})
    audio, tts_ms = await asyncio.to_thread(pipeline.synthesize, draft.reply)
    result = pipeline.commit(
        session,
        transcript,
        draft,
        audio,
        stt_ms=stt_ms,
        tts_ms=tts_ms,
        started=started,
    )
    if audio:
        await websocket.send_bytes(audio)
    await websocket.send_json(latency_message(result.latency))
    await websocket.send_json({"type": "done"})


def latency_message(latency) -> dict:
    payload = latency.as_dict()
    payload["type"] = "latency"
    return payload
