import json
import logging
import os
from datetime import datetime

from domain.memory import Memory
from pydub import AudioSegment

from .repository_interface import Repository

# Constants
FILE_URI_SCHEME = "file://"


class FileRepository(Repository):
    """Repository implementation that stores memories in the filesystem."""

    def __init__(self, storage_dir: str = "recordings", sample_rate: int = 16000):
        self.storage_dir = storage_dir
        self.sample_rate = sample_rate

        # Ensure the storage directory exists
        if not os.path.exists(self.storage_dir):
            os.makedirs(self.storage_dir)

        logging.info(f"Initialized FileRepository with storage dir: {self.storage_dir}")

    async def save(self, memory: Memory) -> str:
        """Save a memory to files (audio + transcript)."""
        # Generate ID if not present
        if memory.id is None:
            memory.id = memory.timestamp.strftime("%Y%m%d_%H%M%S")

        # Save audio file
        audio_filename = f"{memory.id}.wav"
        audio_filepath = os.path.join(self.storage_dir, audio_filename)

        try:
            # PCM signed 16-bit little-endian format
            audio_segment = AudioSegment(
                data=memory.audio_data,
                sample_width=2,  # 2 bytes for s16le
                frame_rate=self.sample_rate,
                channels=1,  # Mono audio
            )
            audio_segment.export(audio_filepath, format="wav")
            logging.info(f"Audio saved to {audio_filepath}")

            # Save transcription
            transcript_filename = f"{memory.id}.txt"
            transcript_filepath = os.path.join(self.storage_dir, transcript_filename)
            with open(transcript_filepath, "w") as f:
                f.write(memory.get_text())
            logging.info(f"Transcription saved to {transcript_filepath}")

            # Save metadata (could be used for searching later)
            metadata_filename = f"{memory.id}.json"
            metadata_filepath = os.path.join(self.storage_dir, metadata_filename)
            with open(metadata_filepath, "w") as f:
                json.dump(
                    {
                        "id": memory.id,
                        "timestamp": memory.timestamp.isoformat(),
                        "duration_seconds": len(audio_segment) / 1000,
                        "has_transcription": bool(memory.transcription),
                    },
                    f,
                )

            return f"{FILE_URI_SCHEME}{audio_filepath}"

        except Exception as e:
            logging.error(f"Failed to save memory: {e}")
            raise

    async def find_by_uri(self, uri: str) -> Memory | None:
        """Find a memory by its URI."""
        if not uri.startswith(FILE_URI_SCHEME):
            raise ValueError(f"Unsupported URI scheme: {uri}")
        audio_filepath = uri[len(FILE_URI_SCHEME) :]

        # Security check: Ensure the file is within the storage directory
        audio_abs_path = os.path.abspath(audio_filepath)
        storage_abs_path = os.path.abspath(self.storage_dir)
        if not audio_abs_path.startswith(storage_abs_path):
            logging.warning(
                "Security violation: Attempted to access file "
                f"outside storage directory: {audio_filepath}"
            )
            raise ValueError(
                "Access denied: Cannot access files outside the storage directory"
            )

        id = os.path.splitext(os.path.basename(audio_filepath))[0]
        transcript_filepath = os.path.join(self.storage_dir, f"{id}.txt")

        if not os.path.exists(audio_filepath):
            return None

        try:
            # Load audio data
            audio_segment = AudioSegment.from_wav(audio_filepath)
            audio_data = audio_segment.raw_data

            # Load transcription if available
            transcription = []
            if os.path.exists(transcript_filepath):
                with open(transcript_filepath) as f:
                    transcription = [f.read()]

            # Parse timestamp from ID or use file creation time
            try:
                timestamp = datetime.strptime(id, "%Y%m%d_%H%M%S")
            except ValueError:
                timestamp = datetime.fromtimestamp(os.path.getctime(audio_filepath))

            return Memory(
                id=id,
                timestamp=timestamp,
                audio_data=audio_data,
                transcription=transcription,
            )

        except Exception as e:
            logging.error(f"Failed to load memory {id}: {e}")
            return None
