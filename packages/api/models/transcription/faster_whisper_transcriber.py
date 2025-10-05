import logging
from typing import Any

import numpy as np
import torch
from faster_whisper import WhisperModel

from .transcriber_interface import Segment, TranscriberInterface


class FasterWhisperTranscriber(TranscriberInterface):
    """Implementation of the TranscriberInterface using Faster Whisper."""

    def __init__(
        self,
        model_size_or_path: str = "small.en",
        device: str | None = None,
        compute_type: str | None = None,
        vad_filter: bool = True,
        vad_parameters: dict[str, Any] | None = None,
    ):
        """
        Initialize the FasterWhisperTranscriber.

        Args:
            model_size_or_path (str): Model size or path to model
            device (str | None): Device to use (cuda or cpu)
            compute_type (str | None): Compute type for the model
            vad_filter (bool): Whether to use voice activity detection
            vad_parameters (dict[str, Any] | None): Parameters for VAD
        """
        self.model_size_or_path = model_size_or_path
        self.vad_filter = vad_filter
        # Use 'threshold' instead of 'onset' for VAD parameters
        self.vad_parameters = vad_parameters or {"threshold": 0.5}
        self.model: WhisperModel | None = None

        # Determine device and compute type
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if compute_type is None:
            if self.device == "cuda":
                major, _ = torch.cuda.get_device_capability(self.device)
                self.compute_type = "float16" if major >= 7 else "float32"
            else:
                self.compute_type = "int8"
        else:
            self.compute_type = compute_type

        logging.info(f"Using Device={self.device} with precision {self.compute_type}")

    def initialize(self):
        """Initialize the Whisper model."""
        if self.model is None:
            try:
                self.model = WhisperModel(
                    self.model_size_or_path,
                    device=self.device,
                    compute_type=self.compute_type,
                    local_files_only=False,
                )
                logging.info(
                    "Successfully loaded Faster Whisper model:"
                    + self.model_size_or_path
                )

                # Verify that sampling rate matches our expectation (16000)
                if (
                    hasattr(self.model, "feature_extractor")
                    and hasattr(self.model.feature_extractor, "sampling_rate")
                    and self.model.feature_extractor.sampling_rate != 16000
                ):
                    logging.warning(
                        "Model expects"
                        + str(self.model.feature_extractor.sampling_rate)
                        + "Hz, but input is 16000 Hz. Resampling may occur."
                    )

            except Exception as e:
                logging.error(f"Failed to load Faster Whisper model: {e}")
                raise

    def transcribe(
        self, audio: np.ndarray, language: str | None = None
    ) -> tuple[list[Segment], Any]:
        """
        Transcribe audio data.

        Args:
            audio (np.ndarray): Audio data as numpy array (16000 Hz, mono)
            language (str | None): Optional language code

        Returns:
            Tuple of (segments, info)
        """
        if self.model is None:
            self.initialize()  # Ensure model is loaded
        if self.model is None:
            raise RuntimeError("Model not initialized")

        result, info = self.model.transcribe(
            audio,
            language=language,
            vad_filter=self.vad_filter,
            vad_parameters=self.vad_parameters if self.vad_filter else None,
        )

        # Convert model-specific segments to our common Segment format
        segments = []
        for seg in result:
            segments.append(
                Segment(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    no_speech_prob=getattr(
                        seg, "no_speech_prob", 0.0
                    ),  # Use getattr for compatibility
                )
            )

        return segments, info

    def cleanup(self):
        """Clean up resources."""
        # Faster Whisper doesn't require explicit cleanup
        # but we'll set the model to None to allow GC
        self.model = None
