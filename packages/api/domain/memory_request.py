from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from uuid import UUID, uuid4

from protos.generated.py import stt_pb2


class MemoryType(Enum):
    """
    Enumeration of memory operation types.

    Defines the purpose of a memory request, aligned with ChunkType
    in the protobuf definition.

    Attributes:
        MEMORY: For storing memories (memorization)
        QUESTION: For retrieving memories (recall/query)
        ANSWER: For LLM-generated answers to questions
    """

    MEMORY = 0
    QUESTION = 1
    ANSWER = 2


@dataclass
class MemoryRequest:
    """
    Domain model representing a memory with audio and/or text content.

    Encapsulates all data for a memory operation, including optional audio data,
    transcribed or input text, timestamps, and the operation type. Used throughout
    the system for storage, retrieval, and question answering.
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
        """
        Factory method to create a new MemoryRequest instance.

        Automatically generates UUID and timestamp if not provided.

        Parameters:
            id (UUID | None): Unique identifier (auto-generated if None)
            timestamp (datetime | None): Creation timestamp (current time if None)
            audio_data (bytes | None): Raw audio bytes (optional)
            text (list[str] | None): List of text segments (optional)
            memory_type (MemoryType): Type of memory operation (default: MEMORY)

        Returns:
            MemoryRequest: New memory request instance
        """
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

        Parameters:
            session_id (str): Session ID to include in metadata
            chunk_type (stt_pb2.ChunkType | None): Override the default
                chunk type mapping. If None, it will be derived from
                memory_type.

        Returns:
            stt_pb2.MemoryChunk: Protobuf message containing the memory data
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
