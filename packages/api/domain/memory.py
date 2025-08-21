from dataclasses import dataclass
from datetime import datetime


@dataclass
class Memory:
    """
    Domain model representing a recorded memory with audio and transcription.
    """

    id: str | None
    timestamp: datetime
    audio_data: bytes
    transcription: list[str]

    @classmethod
    def create(cls, audio_data: bytes, transcription: list[str]) -> "Memory":
        """
        Factory method to create a new Memory instance.

        Args:
            audio_data: Raw audio bytes
            transcription: List of transcribed text segments

        Returns:
            A new Memory instance
        """
        return cls(
            id=None,  # ID will be assigned when persisted
            timestamp=datetime.now(),
            audio_data=audio_data,
            transcription=transcription,
        )

    def get_text(self) -> str:
        """Get the full transcription as a single string."""
        return "".join(self.transcription)
