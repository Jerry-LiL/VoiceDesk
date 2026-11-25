import io
import wave

import numpy as np

TARGET_RATE = 16000


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Little-endian 16-bit mono PCM to float32 in the range [-1, 1]."""
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    if len(pcm) % 2:
        pcm = pcm[:-1]
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    samples /= 32768.0
    return np.clip(samples, -1.0, 1.0)


def resample_linear(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate <= 0 or dst_rate <= 0:
        raise ValueError("sample rate must be positive")
    if audio.size == 0 or src_rate == dst_rate:
        return audio.astype(np.float32, copy=False)
    dst_len = int(round(audio.size * dst_rate / src_rate))
    if dst_len <= 0:
        return np.zeros(0, dtype=np.float32)
    src_x = np.linspace(0.0, 1.0, num=audio.size, endpoint=False)
    dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def is_silent(
    audio: np.ndarray,
    rms_threshold: float,
    min_seconds: float,
    sample_rate: int,
) -> bool:
    if audio.size < int(sample_rate * min_seconds):
        return True
    rms = float(np.sqrt(np.mean(np.square(audio), dtype=np.float64)))
    return rms < rms_threshold


def pcm16_wav(pcm: bytes, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm)
    return buffer.getvalue()
