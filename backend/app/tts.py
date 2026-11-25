import io
import wave
from pathlib import Path

import numpy as np

from app.audio import TARGET_RATE, pcm16_wav


class MockTTS:
    """A short tone wrapped in a real WAV header.

    The browser can play it. It is not speech. Piper is the speech backend.
    """

    def synthesize(self, text: str) -> bytes:
        duration = min(1.6, 0.25 + max(len(text), 1) * 0.012)
        count = int(TARGET_RATE * duration)
        timeline = np.arange(count, dtype=np.float32)
        wave_f = 0.2 * np.sin(2 * np.pi * 440 * timeline / TARGET_RATE)
        pcm = np.clip(wave_f * 32767, -32768, 32767).astype("<i2")
        return pcm16_wav(pcm.tobytes(), TARGET_RATE)


class PiperTTS:
    def __init__(self, model_path: str, use_cuda: bool = False) -> None:
        path = Path(model_path)
        if not path.is_file():
            raise RuntimeError(f"Piper model not found: {path}")
        from piper import PiperVoice

        self._voice = PiperVoice.load(str(path), use_cuda=use_cuda)

    def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            if hasattr(self._voice, "synthesize_wav"):
                self._voice.synthesize_wav(text, wav_file)
            else:
                self._voice.synthesize(text, wav_file)
        return buffer.getvalue()
