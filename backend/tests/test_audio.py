import io
import wave

import numpy as np
from app.audio import is_silent, pcm16_to_float32, pcm16_wav, resample_linear

from tests.helpers import tone_pcm


def test_pcm_round_trip_drops_a_trailing_byte():
    pcm = np.array([1000, -1000], dtype="<i2").tobytes() + b"\xff"
    audio = pcm16_to_float32(pcm)
    assert audio.shape == (2,)
    assert audio[0] > 0
    assert audio[1] < 0


def test_resample_48k_to_16k():
    audio = np.ones(4800, dtype=np.float32)
    down = resample_linear(audio, 48000, 16000)
    assert down.shape == (1600,)
    assert np.allclose(down, 1.0)


def test_short_audio_counts_as_silence():
    loud = pcm16_to_float32(tone_pcm(seconds=0.05))
    assert is_silent(loud, rms_threshold=0.008, min_seconds=0.2, sample_rate=16000)


def test_quiet_audio_counts_as_silence():
    quiet = np.zeros(16000, dtype=np.float32)
    assert is_silent(quiet, rms_threshold=0.008, min_seconds=0.2, sample_rate=16000)


def test_a_tone_is_not_silence():
    loud = pcm16_to_float32(tone_pcm(seconds=0.4))
    assert not is_silent(loud, rms_threshold=0.008, min_seconds=0.2, sample_rate=16000)


def test_wav_header():
    payload = pcm16_wav(b"\x00\x00" * 160, 16000)
    with wave.open(io.BytesIO(payload), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() == 160
