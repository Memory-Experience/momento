from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class Segment:
    """
    Represents a single transcription segment with timing information.

    Contains the transcribed text along with temporal boundaries and
    confidence metrics.
    """

    def __init__(
        self, text: str, start: float, end: float, no_speech_prob: float = 0.0
    ):
        """
        Initialize a transcription segment.

        Args:
            text (str): The transcribed text content
            start (float): Start time of the segment in seconds
            end (float): End time of the segment in seconds
            no_speech_prob (float): Probability that the segment contains no
                speech (default: 0.0)
        """
        self.text = text
        self.start = start
        self.end = end
        self.no_speech_prob = no_speech_prob


class TranscriberInterface(ABC):
    """
    Abstract base interface for speech-to-text transcription implementations.

    Defines the contract for transcriber classes that convert audio data
    into text segments with timing information.
    """

    @abstractmethod
    def initialize(self):
        """
        Initialize the transcriber model.

        Loads necessary models and resources. Should be called before
        the first transcription.
        """
        pass

    @abstractmethod
    def transcribe(
        self, audio: np.ndarray, language: str | None = None
    ) -> tuple[list[Segment], Any]:
        """
        Transcribe audio data.

        Args:
            audio (np.ndarray): Audio data as numpy array (16000 Hz, mono, float32)
            language (str | None): Optional language code

        Returns:
            Tuple of (segments, info) where segments may be empty if the
            implementation requires more audio before producing output
        """
        pass

    @abstractmethod
    def reset_state(self):
        """
        Reset internal state for a new transcription session.

        Implementations that maintain buffering or streaming state should
        override this to clear buffers and reset session-specific state.
        Default implementation does nothing (for stateless transcribers).
        """
        pass

    @abstractmethod
    def cleanup(self):
        """
        Clean up resources.

        Releases any resources held by the transcriber, such as GPU memory
        or model handles. Should be called when the transcriber is no longer needed.
        """
        pass
