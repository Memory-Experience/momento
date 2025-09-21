from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Segment:
    """Class representing a transcription segment."""

    def __init__(
        self, text: str, start: float, end: float, no_speech_prob: float = 0.0
    ):
        self.text = text
        self.start = start
        self.end = end
        self.no_speech_prob = no_speech_prob


class TranscriberInterface(ABC):
    """Base interface for transcriber implementations."""

    @abstractmethod
    def initialize(self):
        """Initialize the transcriber model."""
        pass

    @abstractmethod
    def transcribe(
        self, audio: np.ndarray, language: str | None = None
    ) -> tuple[list[Segment], Any]:
        """
        Transcribe audio data.

        Args:
            audio: Audio data as numpy array
            language: Optional language code

        Returns:
            Tuple of (segments, info)
        """
        pass

    @abstractmethod
    def cleanup(self):
        """Clean up resources."""
        pass
