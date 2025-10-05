from .domain.memory_request import MemoryRequest, MemoryType
from protos.generated.py.stt_pb2 import MemoryChunk, ChunkMetadata, ChunkType
from .dependency_container import Container
import logging


class MemoryPersistService:
    def __init__(self, dependencies: Container):
        self.vector_store_service = dependencies.vector_store
        self.persistence_service = dependencies.persistence

    async def StoreMemory(self, request_iterator, context):
        """gRPC method to store memory from a stream of MemoryChunk messages."""
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
        """Save a memory and send confirmation to client."""
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
