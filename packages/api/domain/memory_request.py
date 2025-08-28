from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID


class MemoryType(Enum):
    """
    Type of memory operation, aligned with SessionType in the proto definition.
    """

    MEMORY = 0  # For storing memories (memorization)
    QUESTION = 1  # For retrieving memories (recall)


@dataclass
class MemoryRequest:
    """
    Domain model representing a recorded memory with audio and transcription.
    """

    id: UUID | None
    timestamp: datetime | None
    audio_data: bytes | None
    text: list[str]
    memory_type: MemoryType

    @classmethod
    def create(
        cls,
        id=None,
        timestamp: datetime = None,
        audio_data: bytes = None,
        text: list[str] = None,
        memory_type: MemoryType = MemoryType.MEMORY,
    ) -> "MemoryRequest":
        """
        Factory method to create a new Memory instance.

        Args:
            id: Optional ID for the memory
            timestamp: Optional timestamp (defaults to now)
            audio_data: Raw audio bytes
            text: List of transcribed text segments
            memory_type: Type of memory operation (MEMORY or QUESTION)

        Returns:
            A new Memory instance
        """
        return cls(
            id=id,  # ID will be assigned when persisted
            timestamp=timestamp or datetime.now(),
            audio_data=audio_data,
            text=text or [],
            memory_type=memory_type,
        )
