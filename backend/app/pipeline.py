import logging
import time
import uuid
from dataclasses import dataclass

from app.audio import TARGET_RATE, is_silent, pcm16_to_float32, resample_linear
from app.latency import LatencyLogger, TurnLatency
from app.memory import ConversationMemory
from app.prompts import build_prompt, static_fallback

log = logging.getLogger("voicedesk.pipeline")


def elapsed_ms(start: float, end: float | None = None) -> float:
    if end is None:
        end = time.perf_counter()
    return round((end - start) * 1000, 1)


@dataclass
class Session:
    session_id: str
    memory: ConversationMemory
    turn_index: int = 0


@dataclass
class Heard:
    text: str
    stt_ms: float
    started: float
    silent: bool


@dataclass
class Draft:
    reply: str
    intent: str
    matched_intent: str
    confidence: float
    used_fallback: bool
    fallback_reason: str
    llm_ms: float


@dataclass
class TurnResult:
    transcript: str
    reply: str
    intent: str
    matched_intent: str
    confidence: float
    used_fallback: bool
    fallback_reason: str
    audio: bytes
    latency: TurnLatency
    empty: bool


class VoicePipeline:
    def __init__(
        self,
        stt,
        llm,
        tts,
        latency_logger: LatencyLogger,
        confidence_threshold: float,
        memory_max_turns: int,
        silence_rms: float,
        min_audio_seconds: float,
    ) -> None:
        self._stt = stt
        self._llm = llm
        self._tts = tts
        self._latency = latency_logger
        self._threshold = confidence_threshold
        self._memory_max_turns = memory_max_turns
        self._silence_rms = silence_rms
        self._min_audio_seconds = min_audio_seconds

    def new_session(self) -> Session:
        return Session(
            session_id=uuid.uuid4().hex[:12],
            memory=ConversationMemory(self._memory_max_turns),
        )

    def transcribe(self, pcm: bytes, sample_rate: int) -> Heard:
        started = time.perf_counter()
        audio = resample_linear(pcm16_to_float32(pcm), sample_rate, TARGET_RATE)
        stt_started = time.perf_counter()
        if is_silent(audio, self._silence_rms, self._min_audio_seconds, TARGET_RATE):
            text = ""
        else:
            text = self._stt.transcribe(audio, TARGET_RATE).strip()
        return Heard(
            text=text,
            stt_ms=elapsed_ms(stt_started),
            started=started,
            silent=not text,
        )

    def compose_reply(self, session: Session, transcript: str) -> Draft:
        prompt = build_prompt(transcript, session.memory, self._threshold)
        started = time.perf_counter()
        intent = prompt.intent
        used_fallback = prompt.used_fallback
        reason = prompt.fallback_reason
        try:
            reply = self._llm.complete(prompt.messages).strip()
            if not reply:
                raise RuntimeError("model returned an empty reply")
        except Exception:
            log.exception("language model request failed")
            reply = static_fallback(intent)
            used_fallback = True
            reason = "llm_error"
        return Draft(
            reply=reply,
            intent=intent,
            matched_intent=prompt.matched_intent,
            confidence=prompt.confidence,
            used_fallback=used_fallback,
            fallback_reason=reason,
            llm_ms=elapsed_ms(started),
        )

    def synthesize(self, reply: str) -> tuple[bytes, float]:
        started = time.perf_counter()
        try:
            audio = self._tts.synthesize(reply)
        except Exception:
            log.exception("speech synthesis failed")
            audio = b""
        return audio, elapsed_ms(started)

    def commit(
        self,
        session: Session,
        transcript: str,
        draft: Draft,
        audio: bytes,
        stt_ms: float,
        tts_ms: float,
        started: float,
    ) -> TurnResult:
        session.turn_index += 1
        session.memory.add(transcript, draft.reply, draft.intent)
        latency = TurnLatency(
            session_id=session.session_id,
            turn=session.turn_index,
            stt_ms=stt_ms,
            llm_ms=draft.llm_ms,
            tts_ms=tts_ms,
            total_ms=elapsed_ms(started),
            intent=draft.intent,
            matched_intent=draft.matched_intent,
            confidence=draft.confidence,
            used_fallback=draft.used_fallback,
            fallback_reason=draft.fallback_reason,
            transcript_chars=len(transcript),
            reply_chars=len(draft.reply),
            audio_bytes=len(audio),
            empty=False,
        )
        self._latency.log(latency)
        return TurnResult(
            transcript=transcript,
            reply=draft.reply,
            intent=draft.intent,
            matched_intent=draft.matched_intent,
            confidence=draft.confidence,
            used_fallback=draft.used_fallback,
            fallback_reason=draft.fallback_reason,
            audio=audio,
            latency=latency,
            empty=False,
        )

    def commit_empty(self, session: Session, heard: Heard) -> TurnResult:
        session.turn_index += 1
        latency = TurnLatency(
            session_id=session.session_id,
            turn=session.turn_index,
            stt_ms=heard.stt_ms,
            llm_ms=0.0,
            tts_ms=0.0,
            total_ms=elapsed_ms(heard.started),
            intent="",
            matched_intent="",
            confidence=0.0,
            used_fallback=False,
            fallback_reason="",
            transcript_chars=0,
            reply_chars=0,
            audio_bytes=0,
            empty=True,
        )
        self._latency.log(latency)
        return TurnResult(
            transcript="",
            reply="",
            intent="",
            matched_intent="",
            confidence=0.0,
            used_fallback=False,
            fallback_reason="",
            audio=b"",
            latency=latency,
            empty=True,
        )

    def run_text(self, session: Session, text: str) -> TurnResult:
        started = time.perf_counter()
        cleaned = " ".join(text.strip().split())
        if not cleaned:
            raise ValueError("text is empty")
        draft = self.compose_reply(session, cleaned)
        audio, tts_ms = self.synthesize(draft.reply)
        return self.commit(
            session,
            cleaned,
            draft,
            audio,
            stt_ms=0.0,
            tts_ms=tts_ms,
            started=started,
        )

    def run_audio(self, session: Session, pcm: bytes, sample_rate: int) -> TurnResult:
        heard = self.transcribe(pcm, sample_rate)
        if heard.silent:
            return self.commit_empty(session, heard)
        draft = self.compose_reply(session, heard.text)
        audio, tts_ms = self.synthesize(draft.reply)
        return self.commit(
            session,
            heard.text,
            draft,
            audio,
            stt_ms=heard.stt_ms,
            tts_ms=tts_ms,
            started=heard.started,
        )
