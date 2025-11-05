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
        buffer_duration: float = 4.0,
        overlap_duration: float = 0.1,
        sample_rate: int = 16000,
    ):
        """
        Initialize the FasterWhisperTranscriber.

        Args:
            model_size_or_path (str): Model size or path to model
            device (str | None): Device to use (cuda or cpu)
            compute_type (str | None): Compute type for the model
            vad_filter (bool): Whether to use voice activity detection
            vad_parameters (dict[str, Any] | None): Parameters for VAD
            buffer_duration (float): Duration in seconds to buffer before transcribing
            overlap_duration (float): Duration in seconds to keep as overlap
            sample_rate (int): Audio sample rate in Hz
        """
        self.model_size_or_path = model_size_or_path
        self.vad_filter = vad_filter
        # Use 'threshold' instead of 'onset' for VAD parameters
        self.vad_parameters = vad_parameters or {"threshold": 0.5}
        self.model: WhisperModel | None = None

        # Buffering configuration
        self.sample_rate = sample_rate
        self.buffer_duration = buffer_duration
        self.overlap_duration = overlap_duration
        self.buffer_threshold_bytes = int(
            buffer_duration * sample_rate * 2
        )  # int16 = 2 bytes
        self.overlap_bytes = int(overlap_duration * sample_rate * 2)

        # Internal buffer state (bytes in int16 format)
        self.buffer = bytearray()

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
        Transcribe audio data with internal buffering.

        Args:
            audio (np.ndarray): Audio data as numpy array (16000 Hz, mono, float32)
            language (str | None): Optional language code

        Returns:
            Tuple of (segments, info). Segments may be empty if buffer not yet full.
        """
        if self.model is None:
            self.initialize()  # Ensure model is loaded
        if self.model is None:
            raise RuntimeError("Model not initialized")

        # Convert float32 audio to int16 bytes and add to buffer
        audio_int16 = (audio * 32768.0).astype(np.int16)
        audio_bytes = audio_int16.tobytes()
        self.buffer += audio_bytes

        # Only process when buffer reaches threshold
        if len(self.buffer) < self.buffer_threshold_bytes:
            return [], {}  # Not enough data yet

        # Convert buffered bytes back to float32 array for transcription
        audio_array = (
            np.frombuffer(self.buffer, dtype=np.int16).astype(np.float32) / 32768.0
        )

        # Transcribe the buffered audio
        result, info = self.model.transcribe(
            audio_array,
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

        # Keep overlap for continuity
        self.buffer = (
            self.buffer[-self.overlap_bytes :]
            if self.overlap_bytes > 0
            else bytearray()
        )

        return segments, info

    def reset_state(self):
        """Reset internal buffer state for a new transcription session."""
        self.buffer = bytearray()

    def cleanup(self):
        """Clean up resources."""
        # Faster Whisper doesn't require explicit cleanup
        # but we'll set the model to None to allow GC
        self.model = None
