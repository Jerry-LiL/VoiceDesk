import json
import logging
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("voicedesk.latency")


@dataclass
class TurnLatency:
    session_id: str
    turn: int
    stt_ms: float
    llm_ms: float
    tts_ms: float
    total_ms: float
    intent: str
    matched_intent: str
    confidence: float
    used_fallback: bool
    fallback_reason: str
    transcript_chars: int
    reply_chars: int
    audio_bytes: int
    empty: bool

    def as_dict(self) -> dict:
        return {
            "event": "turn_latency",
            "session_id": self.session_id,
            "turn": self.turn,
            "stt_ms": self.stt_ms,
            "llm_ms": self.llm_ms,
            "tts_ms": self.tts_ms,
            "total_ms": self.total_ms,
            "intent": self.intent,
            "matched_intent": self.matched_intent,
            "confidence": round(self.confidence, 2),
            "used_fallback": self.used_fallback,
            "fallback_reason": self.fallback_reason,
            "transcript_chars": self.transcript_chars,
            "reply_chars": self.reply_chars,
            "audio_bytes": self.audio_bytes,
            "empty": self.empty,
        }


class LatencyLogger:
    """One JSON line per turn. The spoken text stays out of this log."""

    def __init__(self, path: str | None = None) -> None:
        self._path = path or None
        if self._path:
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: TurnLatency) -> None:
        line = json.dumps(record.as_dict(), separators=(",", ":"))
        log.info("%s", line)
        if not self._path:
            return
        with open(self._path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
