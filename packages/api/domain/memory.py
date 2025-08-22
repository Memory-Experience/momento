from dataclasses import dataclass
from datetime import datetime


@dataclass
class Memory:
    """
    Domain model representing a recorded memory with audio and transcription.
    """

    id: str | None
    timestamp: datetime | None
    audio_data: bytes | None
    text: list[str]

    @classmethod
    def create(
        cls,
        id=None,
        timestamp: datetime = None,
        audio_data: bytes = None,
        text: list[str] = None,
    ) -> "Memory":
        """
        Factory method to create a new Memory instance.

        Args:
            audio_data: Raw audio bytes
            transcription: List of transcribed text segments

        Returns:
            A new Memory instance
        """
        return cls(
            id=id,  # ID will be assigned when persisted
            timestamp=timestamp or datetime.now(),
            audio_data=audio_data,
            text=text or [],
        )
