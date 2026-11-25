from app.config import Settings
from app.latency import LatencyLogger
from app.llm import MockChat, OllamaChat, OpenAIChat, resolve_openai_key
from app.pipeline import VoicePipeline
from app.stt import FasterWhisperSTT, MockSTT
from app.tts import MockTTS, PiperTTS


def build_stt(settings: Settings):
    if settings.stt_backend == "mock":
        return MockSTT(settings.mock_transcript)
    if settings.stt_backend == "whisper":
        return FasterWhisperSTT(
            model_size=settings.whisper_model,
            device=settings.whisper_device,
            compute_type=settings.whisper_compute_type,
            language=settings.whisper_language,
        )
    raise RuntimeError("VOICEDESK_STT_BACKEND must be mock or whisper")


def build_llm(settings: Settings):
    if settings.llm_backend == "mock":
        return MockChat()
    if settings.llm_backend == "openai":
        key = resolve_openai_key(settings.openai_api_key)
        if not key:
            raise RuntimeError("OpenAI backend selected but VOICEDESK_OPENAI_API_KEY is empty")
        return OpenAIChat(
            api_key=key,
            model=settings.openai_model,
            base_url=settings.openai_base_url,
            timeout=settings.openai_timeout_seconds,
        )
    if settings.llm_backend == "ollama":
        return OllamaChat(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            timeout=settings.ollama_timeout_seconds,
        )
    raise RuntimeError("VOICEDESK_LLM_BACKEND must be mock, openai, or ollama")


def build_tts(settings: Settings):
    if settings.tts_backend == "mock":
        return MockTTS()
    if settings.tts_backend == "piper":
        if not settings.piper_model_path:
            raise RuntimeError("Piper backend selected but VOICEDESK_PIPER_MODEL_PATH is empty")
        return PiperTTS(settings.piper_model_path, use_cuda=settings.piper_use_cuda)
    raise RuntimeError("VOICEDESK_TTS_BACKEND must be mock or piper")


def build_pipeline(settings: Settings) -> VoicePipeline:
    settings.check_backends()
    return VoicePipeline(
        stt=build_stt(settings),
        llm=build_llm(settings),
        tts=build_tts(settings),
        latency_logger=LatencyLogger(settings.latency_log_path or None),
        confidence_threshold=settings.intent_confidence_threshold,
        memory_max_turns=settings.memory_max_turns,
        silence_rms=settings.silence_rms_threshold,
        min_audio_seconds=settings.min_audio_seconds,
    )
