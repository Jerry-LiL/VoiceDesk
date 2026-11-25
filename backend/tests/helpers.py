import numpy as np
from app.latency import LatencyLogger
from app.llm import MockChat
from app.pipeline import VoicePipeline
from app.stt import MockSTT
from app.tts import MockTTS


def tone_pcm(seconds=0.4, freq=440.0, rate=16000, amp=0.3) -> bytes:
    count = int(rate * seconds)
    timeline = np.arange(count, dtype=np.float32)
    samples = amp * np.sin(2 * np.pi * freq * timeline / rate)
    pcm = np.clip(samples * 32767, -32768, 32767).astype("<i2")
    return pcm.tobytes()


def make_pipeline(llm=None, path=None, memory_max_turns=8) -> VoicePipeline:
    return VoicePipeline(
        stt=MockSTT("hello there"),
        llm=llm or MockChat(),
        tts=MockTTS(),
        latency_logger=LatencyLogger(path),
        confidence_threshold=0.55,
        memory_max_turns=memory_max_turns,
        silence_rms=0.008,
        min_audio_seconds=0.2,
    )
