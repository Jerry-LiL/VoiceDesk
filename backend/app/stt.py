import numpy as np


class MockSTT:
    """Returns a fixed phrase for any non-silent clip.

    This is the local stand-in. It does not recognize speech. The whisper
    backend below does.
    """

    def __init__(self, transcript: str = "hello there") -> None:
        self._transcript = transcript
        self.calls = 0

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        self.calls += 1
        if sample_rate <= 0 or audio.size == 0:
            return ""
        return self._transcript


class FasterWhisperSTT:
    def __init__(
        self,
        model_size: str,
        device: str,
        compute_type: str,
        language: str,
        model=None,
    ) -> None:
        self._language = language or None
        if model is None:
            from faster_whisper import WhisperModel

            model = WhisperModel(model_size, device=device, compute_type=compute_type)
        self._model = model

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        # Callers resample to 16 kHz before this. faster-whisper expects that rate.
        if sample_rate != 16000:
            raise ValueError("faster-whisper expects 16 kHz audio")
        segments, _info = self._model.transcribe(
            audio,
            language=self._language,
            vad_filter=True,
            beam_size=1,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()
