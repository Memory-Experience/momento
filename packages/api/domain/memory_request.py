from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from protos.generated.py import stt_pb2


class MemoryType(Enum):
    """
    Type of memory operation, aligned with SessionType in the proto definition.
    """

    MEMORY = 0  # For storing memories (memorization)
    QUESTION = 1  # For retrieving memories (recall)
    ANSWER = 2  # Answer to a question


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
        id: UUID | None = None,
        timestamp: datetime | None = None,
        audio_data: bytes | None = None,
        text: list[str] | None = None,
        memory_type: MemoryType = MemoryType.MEMORY,
    ) -> "MemoryRequest":
        """Factory method to create a new Memory instance."""
        return cls(
            id=id or uuid4(),
            timestamp=timestamp or datetime.now(),
            audio_data=audio_data,
            text=text or [],
            memory_type=memory_type,
        )

    def to_chunk(
        self, session_id: str, chunk_type: stt_pb2.ChunkType | None = None
    ) -> stt_pb2.MemoryChunk:
        """
        Convert this memory request to a protobuf MemoryChunk message.

        Args:
            session_id: Session ID to include in metadata
            chunk_type: Override the default chunk type mapping. If None, it will
                be derived from memory_type.

        Returns:
            MemoryChunk protobuf message
        """
        # Map memory type to chunk type if not explicitly provided
        if chunk_type is None:
            chunk_type_map = {
                MemoryType.MEMORY: stt_pb2.ChunkType.MEMORY,
                MemoryType.QUESTION: stt_pb2.ChunkType.QUESTION,
                MemoryType.ANSWER: stt_pb2.ChunkType.ANSWER,
            }
            chunk_type = chunk_type_map.get(self.memory_type, stt_pb2.ChunkType.MEMORY)

        # Create metadata
        metadata = stt_pb2.ChunkMetadata(
            session_id=session_id,
            memory_id=str(self.id) if self.id else "",
            type=chunk_type,
        )

        # Create the chunk with either text or audio
        chunk = stt_pb2.MemoryChunk(metadata=metadata)

        if self.audio_data:
            chunk.audio_data = self.audio_data
        elif self.text:
            # Join all text segments
            chunk.text_data = " ".join(self.text)

        return chunk
