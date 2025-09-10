import asyncio
import logging
import sys

import grpc
import numpy as np
from domain.memory_request import MemoryRequest, MemoryType
from persistence.persistence_service import PersistenceService
from persistence.repositories.file_repository import FileRepository
from protos.generated.py import stt_pb2, stt_pb2_grpc
from rag.llm_rag_service import LLMRAGService
from vector_store.repositories.qdrant_vector_store_repository import (
    InMemoryQdrantVectorStoreRepository,
)
from vector_store.repositories.vector_store_repository_interface import (
    VectorStoreRepository,
)
from vector_store.vector_store_service import VectorStoreService

from models.character_text_chunker import CharacterTextChunker
from models.llm.qwen3 import Qwen3
from models.embedding.qwen3_embedding import Qwen3EmbeddingModel
from models.transcription.faster_whisper_transcriber import FasterWhisperTranscriber

_cleanup_coroutines = []

# Constants
PORT = 50051
RECORDINGS_DIR = "recordings"
SAMPLE_RATE = 16000


class TranscriptionServiceServicer(stt_pb2_grpc.TranscriptionServiceServicer):
    """Provides methods that implement functionality of transcription service."""

    def __init__(self):
        # Initialize transcription model
        self.transcriber = FasterWhisperTranscriber()
        self.transcriber.initialize()

        # Initialize vector store
        embedding_model = Qwen3EmbeddingModel()
        text_chunker = CharacterTextChunker()
        vector_store_repo: VectorStoreRepository = InMemoryQdrantVectorStoreRepository(
            embedding_model, text_chunker
        )
        self.vector_store_service = VectorStoreService(vector_store_repo)

        # Initialize LLM model and RAG service
        self.llm_model = Qwen3()
        self.rag_service = LLMRAGService(llm_model=self.llm_model)

        # Initialize the persistence service
        repository = FileRepository(storage_dir=RECORDINGS_DIR, sample_rate=SAMPLE_RATE)
        self.persistence_service = PersistenceService(repository)

    async def Transcribe(self, request_iterator, context):
        """Bidirectional streaming RPC for audio transcription."""
        logging.info("Client connected.")

        # State tracking
        audio_data = bytearray()
        transcription = []
        buffer = b""
        session_type = stt_pb2.ChunkType.MEMORY  # Default type
        session_id = None
        memory_id = None
        first_chunk = True
        question_completed = False
        transcription_complete = False

        try:
            logging.info("Received a new transcription request.")

            # Process incoming chunks
            async for chunk in request_iterator:
                # Handle session metadata from first chunk
                if first_chunk and chunk.metadata:
                    session_type = chunk.metadata.type
                    session_id = chunk.metadata.session_id
                    memory_id = chunk.metadata.memory_id
                    logging.info(
                        f"Session {session_id} started with type: {session_type}, "
                        f"memory ID: {memory_id}"
                    )
                    first_chunk = False

                # Handle final marker - this is our new main control flow
                if chunk.metadata and chunk.metadata.is_final:
                    logging.info("Received explicit final marker from client")

                    # Mark transcription as complete if not already done
                    if not transcription_complete:
                        transcription_complete = True

                        # Send a final transcript marker back to client
                        yield stt_pb2.MemoryChunk(
                            text_data="",
                            metadata=stt_pb2.ChunkMetadata(
                                session_id=session_id,
                                memory_id=memory_id,
                                type=stt_pb2.ChunkType.TRANSCRIPT,
                                is_final=True,
                            ),
                        )

                        # Process based on session type
                        if session_type == stt_pb2.ChunkType.MEMORY:
                            # Save memory and return ID to client
                            async for res in self._process_memory_saving(
                                audio_data, transcription, session_id, memory_id
                            ):
                                yield res
                        elif (
                            session_type == stt_pb2.ChunkType.QUESTION
                            and not question_completed
                        ):
                            # Process question and stream answer
                            async for res in self._process_question(
                                audio_data, transcription, session_id, memory_id
                            ):
                                yield res
                            question_completed = True

                    continue

                # Handle audio input
                if chunk.HasField("audio_data"):
                    buffer += chunk.audio_data
                    audio_data += chunk.audio_data

                    # Process audio when buffer is large enough
                    if len(buffer) >= SAMPLE_RATE * 4:
                        # Convert and transcribe audio
                        audio_array = (
                            np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
                            / 32768.0
                        )
                        segments, _ = self.transcriber.transcribe(audio_array)
                        transcription.extend(segment.text for segment in segments)

                        # Send transcript chunks to client
                        for segment in segments:
                            yield stt_pb2.MemoryChunk(
                                text_data=segment.text,
                                metadata=stt_pb2.ChunkMetadata(
                                    session_id=session_id,
                                    memory_id=memory_id,
                                    type=stt_pb2.ChunkType.TRANSCRIPT,
                                    is_final=False,
                                ),
                            )

                        # Keep a small overlap for continuity
                        buffer = buffer[-1600:]  # Keep last 0.1 seconds

                # Handle direct text input
                elif chunk.HasField("text_data"):
                    logging.info(f"Received text input: {chunk.text_data}")
                    transcription.append(chunk.text_data)

                    # Echo text back as transcript
                    yield stt_pb2.MemoryChunk(
                        text_data=chunk.text_data,
                        metadata=stt_pb2.ChunkMetadata(
                            session_id=session_id,
                            memory_id=memory_id,
                            type=stt_pb2.ChunkType.TRANSCRIPT,
                            is_final=False,
                        ),
                    )

        except grpc.aio.AioRpcError as e:
            logging.error(f"Error during transcription: {e}")
        finally:
            logging.info("Client disconnected.")
            # We no longer save memory in the finally block
            # All saving is now done when a final marker is received

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
        yield stt_pb2.MemoryChunk(
            text_data=f"Memory saved with ID: {new_memory_id}",
            metadata=stt_pb2.ChunkMetadata(
                session_id=session_id,
                memory_id=new_memory_id,
                type=stt_pb2.ChunkType.MEMORY,
                is_final=True,
            ),
        )
        logging.info(f"Sent memory confirmation with ID: {new_memory_id}")

    async def _process_question(self, audio_data, transcription, session_id, memory_id):
        """Process a question, fetch context, and stream answer."""
        full_transcription = " ".join(transcription)
        logging.info(f"Processing question: {full_transcription}")

        # Create question memory object
        question_memory = MemoryRequest.create(
            audio_data=bytes(audio_data) if audio_data else None,
            text=transcription,
            memory_type=MemoryType.QUESTION,
        )

        # Get memory context
        memory_context = await self.vector_store_service.search(
            question_memory, limit=5
        )

        # Generate answer
        response_generator = self.rag_service.answer_question(
            query=question_memory,
            memory_context=memory_context,
            chunk_size_tokens=8,
        )

        # Stream memory context first if available
        if memory_context and memory_context.memories:
            logging.info(
                f"Sending {len(memory_context.memories)} memories from context"
            )
            for memory in memory_context.memories.values():
                # Create the memory chunk using the factory method
                memory_chunk = memory.to_chunk(
                    session_id=session_id, chunk_type=stt_pb2.ChunkType.MEMORY
                )

                # Then add the score from memory_context
                if memory.id in memory_context.scores:
                    memory_chunk.metadata.score = float(
                        memory_context.scores[memory.id]
                    )

                yield memory_chunk

        # Stream answer chunks
        try:
            last_chunk = None
            async for response_chunk in response_generator:
                if response_chunk.response and response_chunk.response.text:
                    chunks_count = len(response_chunk.response.text)
                    for i, text_segment in enumerate(response_chunk.response.text):
                        if text_segment.strip():
                            is_final = (
                                i == chunks_count - 1
                                and response_chunk.metadata.get("is_final", False)
                            )

                            answer_chunk = stt_pb2.MemoryChunk(
                                text_data=text_segment,
                                metadata=stt_pb2.ChunkMetadata(
                                    session_id=session_id,
                                    memory_id=str(response_chunk.response.id),
                                    type=stt_pb2.ChunkType.ANSWER,
                                    is_final=is_final,
                                ),
                            )
                            last_chunk = answer_chunk
                            yield answer_chunk

            # If the generator finishes and the last chunk sent was not final,
            # -> send a final marker.
            if last_chunk and not last_chunk.metadata.is_final:
                final_marker = stt_pb2.MemoryChunk(
                    metadata=stt_pb2.ChunkMetadata(
                        session_id=session_id,
                        memory_id=last_chunk.metadata.memory_id,
                        type=stt_pb2.ChunkType.ANSWER,
                        is_final=True,
                    )
                )
                yield final_marker
                logging.info("Sent final answer marker because stream ended.")

            logging.info("Answer streaming completed")
        except Exception as e:
            logging.error(f"Error streaming answer: {e}")
            raise e


async def serve() -> None:
    """Starts the gRPC server."""
    server = grpc.aio.server()
    stt_pb2_grpc.add_TranscriptionServiceServicer_to_server(
        TranscriptionServiceServicer(), server
    )
    listen_addr = f"[::]:{PORT}"
    server.add_insecure_port(listen_addr)
    logging.info("🚀 SST Microservice is running on %s", listen_addr)
    await server.start()

    async def server_graceful_shutdown():
        logging.info("Starting graceful shutdown...")
        await server.stop(5)

    _cleanup_coroutines.append(server_graceful_shutdown())
    await server.wait_for_termination()


if __name__ == "__main__":
    # check if args contain -v
    log_level = logging.INFO
    if "-v" in sys.argv:
        log_level = logging.DEBUG

    logging.basicConfig(level=log_level)
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(serve())
    finally:
        logging.info("Cleaning up...")
        loop.run_until_complete(*_cleanup_coroutines)
        loop.close()
