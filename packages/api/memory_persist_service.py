from collections.abc import AsyncIterator

from .domain.memory_request import MemoryRequest, MemoryType
from protos.generated.py.stt_pb2 import MemoryChunk, ChunkMetadata, ChunkType
from .dependency_container import Container
import logging


class MemoryPersistService:
    """
    Service for persisting memories from streaming audio/text input.

    This service handles the gRPC streaming protocol for memory storage,
    accumulating chunks of audio and transcription data, then persisting
    them to both the file system and vector store for later retrieval.
    """

    def __init__(self, dependencies: Container):
        """
        Initialize the memory persist service.

        Args:
            dependencies (Container): Container with vector_store and persistence
                service dependencies
        """
        self.vector_store_service = dependencies.vector_store
        self.persistence_service = dependencies.persistence

    async def StoreMemory(
        self, request_iterator, context
    ) -> AsyncIterator[MemoryChunk]:
        """
        Store memory from a stream of MemoryChunk messages.

        This gRPC method processes streaming audio and text data, accumulating
        chunks until a final marker is received, then persisting the complete
        memory to storage and indexing it in the vector store.

        Args:
            request_iterator (AsyncIterator[MemoryChunk]): Async iterator
                yielding MemoryChunk protobuf messages
            context (grpc.aio.ServicerContext): gRPC context for the
                request

        Yields:
            protos.generated.py.stt_pb2.MemoryChunk: Confirmation messages
                with the saved memory ID
        """
        # Local variables per connection - no shared state!
        session_id = None
        memory_id = None
        audio_data = bytearray()
        transcription: list[str] = []

        async for chunk in request_iterator:
            logging.debug(f"<<<Received memory chunk: {chunk}")

            # Extract metadata
            if chunk.metadata:
                session_id = chunk.metadata.session_id or session_id
                memory_id = chunk.metadata.memory_id or memory_id

            # Accumulate audio data and transcription text
            if chunk.audio_data:
                audio_data.extend(chunk.audio_data)
            if chunk.text_data:
                transcription.append(chunk.text_data)

            # If this is the final chunk, process and save the memory
            if chunk.metadata and chunk.metadata.is_final:
                async for response in self._process_memory_saving(
                    audio_data, transcription, session_id, memory_id
                ):
                    yield response

                # Reset for next memory within the same connection
                audio_data = bytearray()
                transcription = []
                session_id = None
                memory_id = None

    async def _process_memory_saving(
        self, audio_data, transcription, session_id, memory_id
    ):
        """
        Save a memory and send confirmation to client.

        Args:
            audio_data (bytearray): Raw audio bytes to persist
            transcription (list[str]): List of transcribed text segments
            session_id (str): Session identifier for the client connection
            memory_id (str): Original memory identifier from the client (if provided)

        Yields:
            MemoryChunk: Confirmation message containing the saved memory's UUID
        """
        if not audio_data and not transcription:
            return

        # Create memory object
        memory = MemoryRequest.create(
            audio_data=bytes(audio_data) if audio_data else None,
            text=transcription,
            memory_type=MemoryType.MEMORY,
        )

        # Save and index the memory
        uri = await self.persistence_service.save_memory(memory)

        logging.info(f"Memory saved with URI: {uri}")
        await self.vector_store_service.index_memory(memory)
        new_memory_id = str(memory.id)

        # Send confirmation with memory ID back to client
        yield MemoryChunk(
            text_data=f"Memory saved with ID: {new_memory_id}",
            metadata=ChunkMetadata(
                session_id=session_id,
                memory_id=new_memory_id,
                type=ChunkType.MEMORY,
                is_final=True,
            ),
        )
        logging.info(f"Sent memory confirmation with ID: {new_memory_id}")
