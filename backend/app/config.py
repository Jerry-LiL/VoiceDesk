import logging

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

STT_BACKENDS = {"mock", "whisper"}
LLM_BACKENDS = {"mock", "openai", "ollama"}
TTS_BACKENDS = {"mock", "piper"}


class Settings(BaseSettings):
    """Runtime switches. Secrets come from the environment, never from the repo."""

    model_config = SettingsConfigDict(
        env_prefix="VOICEDESK_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    stt_backend: str = "mock"
    whisper_model: str = "tiny"
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"
    whisper_language: str = "en"
    mock_transcript: str = "hello there"

    llm_backend: str = "mock"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_base_url: str = ""
    openai_timeout_seconds: float = 30.0

    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.2"
    ollama_timeout_seconds: float = 60.0

    tts_backend: str = "mock"
    piper_model_path: str = ""
    piper_use_cuda: bool = False

    memory_max_turns: int = Field(default=8, ge=1, le=50)
    intent_confidence_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    silence_rms_threshold: float = 0.008
    min_audio_seconds: float = 0.2
    max_audio_seconds: float = 30.0

    latency_log_path: str = ""

    def check_backends(self) -> None:
        if self.stt_backend not in STT_BACKENDS:
            raise RuntimeError("VOICEDESK_STT_BACKEND must be mock or whisper")
        if self.llm_backend not in LLM_BACKENDS:
            raise RuntimeError("VOICEDESK_LLM_BACKEND must be mock, openai, or ollama")
        if self.tts_backend not in TTS_BACKENDS:
            raise RuntimeError("VOICEDESK_TTS_BACKEND must be mock or piper")


def configure_logging(level: str) -> None:
    named = logging.getLogger("voicedesk")
    named.setLevel(getattr(logging, level.upper(), logging.INFO))
    if not named.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        named.addHandler(handler)
    named.propagate = False
