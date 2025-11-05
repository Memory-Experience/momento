# This code was originally in simul_streaming/simulstreaming_whisper.py.
# It has been adapted for this projects interfaces.
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from .transcriber_interface import Segment, TranscriberInterface
from .simul_whisper.config import AlignAttConfig
from .simul_whisper.simul_whisper import PaddedAlignAttWhisper

logger = logging.getLogger(__name__)


class SimulStreamingWhisperTranscriber(TranscriberInterface):
    """
    SimulStreaming-backed implementation of TranscriberInterface.

    This class wraps SimulStreaming's PaddedAlignAttWhisper and emulates the
    online "process_iter" loop internally to produce timestamped segments.
    """

    SAMPLING_RATE = 16000
    # Each attention frame is ~20ms on large-v3 (as in SimulStreaming)
    ATTEN_FRAME_SEC = 0.02

    def __init__(
        self,
        model_path: str = "models/transcription/simul_whisper/"
        "whisper/assets/large-v3-turbo.pt",
        *,
        # decoding
        beams: int = 1,
        decoder_type: str | None = None,  # "greedy" | "beam" | None (auto from beams)
        task: str = "transcribe",  # or "translate"
        # buffering / stopping
        audio_max_len: float = 30.0,
        audio_min_len: float = 0.0,
        frame_threshold: int = 25,
        # CIF end-of-word gate (optional)
        cif_ckpt_path: str | None = None,
        never_fire: bool = False,
        # prompting / context
        init_prompt: str | None = None,
        static_init_prompt: str | None = None,
        max_context_tokens: int | None = None,
        # logging
        logdir: str | None = None,
        # chunking inside `transcribe` (seconds)
        segment_length: float = 0.5,
        # language handling
        default_language: str = "auto",
    ):
        self.model_path = model_path
        self.beams = beams
        self.decoder_type = self._resolve_decoder_type(decoder_type, beams)
        self.task = task
        self.audio_max_len = audio_max_len
        self.audio_min_len = audio_min_len
        self.frame_threshold = frame_threshold
        self.cif_ckpt_path = cif_ckpt_path
        self.never_fire = never_fire
        self.init_prompt = init_prompt
        self.static_init_prompt = static_init_prompt
        self.max_context_tokens = max_context_tokens
        self.logdir = logdir
        self.segment_length = float(segment_length)
        self.default_language = default_language

        self.model: PaddedAlignAttWhisper | None = None

        # Internal buffer for accumulating small chunks
        self._audio_buffer: list[np.ndarray] = []
        self._buffer_duration = 0.0
        self._min_chunk_duration = (
            segment_length  # Process when we have this much audio
        )

        # Online-loop state
        self._reset_online_state()

    # --------------------------- TranscriberInterface ---------------------------

    def initialize(self):
        """Load Whisper + set up the SimulStreaming wrapper."""
        if self.model is not None:
            return

        cfg = AlignAttConfig(
            model_path=self.model_path,
            language=self.default_language,
            # SimulStreaming policy knobs
            audio_max_len=self.audio_max_len,
            audio_min_len=self.audio_min_len,
            frame_threshold=self.frame_threshold,
            # CIF / truncation
            cif_ckpt_path=self.cif_ckpt_path,
            never_fire=self.never_fire,
            # decoding
            decoder_type=self.decoder_type,
            beam_size=self.beams,
            task=self.task,
            # context/prompt
            init_prompt=self.init_prompt,
            static_init_prompt=self.static_init_prompt,
            max_context_tokens=self.max_context_tokens,
            # debug logging
            logdir=self.logdir,
        )
        self.model = PaddedAlignAttWhisper(cfg)
        self._reset_online_state()
        # Hard reset model buffers for a clean start
        self.model.refresh_segment(complete=True)

        logger.info(
            "SimulStreamingWhisperTranscriber initialized: "
            f"model={self.model_path}, beams={self.beams}, "
            f"decoder={self.decoder_type}, task={self.task}"
        )

    def transcribe(
        self, audio: np.ndarray, language: str | None = None
    ) -> tuple[list[Segment], Any]:
        """
        Transcribe audio incrementally with internal buffering.

        Accumulates incoming audio chunks and processes them in batches
        of segment_length duration. State persists across calls within
        a session until reset_state() is called.

        Args:
            audio (np.ndarray): Audio data as numpy array (16000 Hz, mono, float32)
            language (str | None): Optional language code

        Returns:
            Tuple of (segments, info). Segments may be empty if not enough
            audio has been accumulated yet.
        """
        self._ensure_initialized()

        # Optional language override (only on first call with language set)
        if language and not hasattr(self.model, "detected_language"):
            self.model.create_tokenizer(language)
            self.model.detected_language = language

        # Normalize input
        if audio is None or audio.size == 0:
            return [], {}

        if audio.ndim != 1:
            audio = audio.reshape(-1)
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # Add to internal buffer
        self._audio_buffer.append(audio)
        self._buffer_duration += len(audio) / self.SAMPLING_RATE

        # Only process when we have enough audio
        if self._buffer_duration < self._min_chunk_duration:
            return [], {}

        # Concatenate buffered audio
        buffered_audio = np.concatenate(self._audio_buffer)
        self._audio_buffer = []
        self._buffer_duration = 0.0

        segments: list[Segment] = []

        # Process the accumulated audio
        self._end_sec += len(buffered_audio) / self.SAMPLING_RATE

        # Insert audio and maybe evict old audio from model's ring buffer
        removed_sec = self.model.insert_audio(torch.from_numpy(buffered_audio))
        self._audio_buffer_offset_sec += removed_sec

        # Decode incrementally (not final)
        tokens, generation = self.model.infer(is_last=False)

        # Hide incomplete unicode to avoid '�' glitches
        tokens = self._hide_incomplete_unicode(tokens)

        # Turn tokens into (start, end, word) using attention frames
        ts_words = self._timestamped_text(tokens, generation)

        text = self.model.tokenizer.decode(tokens) if tokens else ""

        if text:
            # Map local frame times to absolute audio times
            beg = (
                ts_words[0][0] + self._audio_buffer_offset_sec
                if ts_words
                else self._last_ts_end
            )
            beg = max(beg, self._last_ts_beg + 1e-6)  # keep non-decreasing starts
            end = (
                ts_words[-1][1] + self._audio_buffer_offset_sec
                if ts_words
                else self._end_sec
            )
            end = max(end, self._last_ts_end + 1e-6)

            self._last_ts_beg, self._last_ts_end = beg, end

            segments.append(
                Segment(
                    text=text,
                    start=beg,
                    end=end,
                    no_speech_prob=float(generation.get("no_speech_prob", 0.0))
                    if generation
                    else 0.0,
                )
            )

        info = {
            "language": getattr(
                self.model, "detected_language", language or self.default_language
            ),
            "beams": self.beams,
            "decoder_type": self.decoder_type,
            "task": self.task,
            "frame_threshold": self.frame_threshold,
            "audio_max_len": self.audio_max_len,
            "audio_min_len": self.audio_min_len,
            "model_path": self.model_path,
        }
        return segments, info

    def reset_state(self):
        """Reset internal state for a new transcription session."""
        self._reset_online_state()
        self._audio_buffer = []
        self._buffer_duration = 0.0
        if self.model is not None:
            self.model.refresh_segment(complete=True)

    def cleanup(self):
        """Release model to allow GC."""
        self.model = None

    # ------------------------------- Internals ---------------------------------

    def _resolve_decoder_type(self, decoder: str | None, beams: int) -> str:
        if beams > 1:
            if decoder in (None, "beam"):
                return "beam"
            raise ValueError("Invalid decoder for beams > 1: use 'beam'.")
        # beams == 1
        if decoder is None:
            return "greedy"
        if decoder not in ("greedy", "beam"):
            raise ValueError("decoder_type must be 'greedy' or 'beam'.")
        return decoder

    def _ensure_initialized(self):
        if self.model is None:
            self.initialize()

    def _reset_online_state(self):
        # Absolute stream position (seconds)
        self._offset_sec = 0.0
        self._end_sec = self._offset_sec
        # How much audio the model evicted from its internal buffer (seconds)
        self._audio_buffer_offset_sec = self._offset_sec
        # Running timestamps (non-decreasing guards)
        self._last_ts_beg = -1.0
        self._last_ts_end = -1.0
        # Buffer for a trailing, incomplete unicode token (adapted)
        # :contentReference[oaicite:4]{index=4}
        self._unicode_buffer: list[int] = []

    # ---- Routines adapted from SimulStreaming's SimulWhisperOnline ----
    # (timestamping and unicode handling mirror upstream behavior)
    # :contentReference[oaicite:5]{index=5}

    def _timestamped_text(
        self, tokens: list[int], generation: Any
    ) -> list[tuple[float, float, str]]:
        """Return [(start_sec, end_sec, word_text), ...] for the *new* tokens."""
        if not generation:
            return []

        pr = generation["progress"]
        # Prefer precomputed word splits if available; otherwise recompute now.
        if "result" not in generation or self._unicode_buffer:
            split_words, split_tokens = self.model.tokenizer.split_to_word_tokens(
                tokens
            )
        else:
            split_words = generation["result"]["split_words"]
            split_tokens = generation["result"]["split_tokens"]

        # One attention frame per decode step; take most-attended frame of last row
        frames = [p["most_attended_frames"][0] for p in pr]

        # If we hid an incomplete unicode, pad frames so alignment stays
        # consistent (adapted) :contentReference[oaicite:6]{index=6}
        if frames and self._unicode_buffer:
            frames = [frames[0]] * len(self._unicode_buffer) + frames

        toks = tokens.copy()
        out: list[tuple[float, float, str]] = []
        for word, token_ids in zip(split_words, split_tokens, strict=True):
            begin_frame = None
            for t_id in token_ids:
                t, f = toks.pop(0), frames.pop(0)
                if t != t_id:
                    raise ValueError(
                        "Token mismatch during timestamping: "
                        f"{t} != {t_id} at frame {f}."
                    )
                if begin_frame is None:
                    begin_frame = f
            end_frame = f  # last one from the loop above
            out.append((
                begin_frame * self.ATTEN_FRAME_SEC,
                end_frame * self.ATTEN_FRAME_SEC,
                word,
            ))
            logger.debug("TS-WORD:\t%s", " ".join(map(str, out[-1])))
        return out

    def _hide_incomplete_unicode(self, tokens: list[int]) -> list[int]:
        """
        If the last token would decode to an incomplete unicode sequence
        (would render '�'), hold it back and prepend it on the next
        iteration (adapted). :contentReference[oaicite:7]{index=7}
        """
        if self._unicode_buffer:
            tokens = self._unicode_buffer + tokens
            self._unicode_buffer = []

        chars, _ = self.model.tokenizer.split_tokens_on_unicode(tokens)
        if chars and chars[-1].endswith("�"):
            # Keep last token for the next iteration
            self._unicode_buffer = tokens[-1:]
            return tokens[:-1]
        return tokens
