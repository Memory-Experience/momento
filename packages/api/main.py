import asyncio
import logging
import sys

import grpc
import numpy as np
from domain.memory_request import MemoryRequest, MemoryType
from models.character_text_chunker import CharacterTextChunker
from models.transcription.faster_whisper_transcriber import FasterWhisperTranscriber
from persistence.persistence_service import PersistenceService
from persistence.repositories.file_repository import FileRepository
from protos.generated.py import stt_pb2, stt_pb2_grpc
from rag.rag_service import SimpleRAGService
from tests.vector_store.test_qdrant_vector_store_repository import MockEmbeddingModel
from vector_store.repositories.qdrant_vector_store_repository import (
    InMemoryQdrantVectorStoreRepository,
)
from vector_store.repositories.vector_store_repository_interface import (
    VectorStoreRepository,
)
from vector_store.vector_store_service import VectorStoreService

_cleanup_coroutines = []

# Constants
PORT = 50051
RECORDINGS_DIR = "recordings"
SAMPLE_RATE = 16000


class TranscriptionServiceServicer(stt_pb2_grpc.TranscriptionServiceServicer):
    """Provides methods that implement functionality of transcription service."""

    def __init__(self):
        self.transcriber = FasterWhisperTranscriber()
        self.transcriber.initialize()

        embedding_model = MockEmbeddingModel()
        text_chunker = CharacterTextChunker()
        vector_store_repo: VectorStoreRepository = InMemoryQdrantVectorStoreRepository(
            embedding_model, text_chunker
        )
        self.vector_store_service = VectorStoreService(vector_store_repo)

        self.rag_service = SimpleRAGService()

        # Initialize the persistence service with a file repository
        repository = FileRepository(storage_dir=RECORDINGS_DIR, sample_rate=SAMPLE_RATE)
        self.persistence_service = PersistenceService(repository)

    async def Transcribe(self, request_iterator, context):
        """Bidirectional streaming RPC for audio transcription."""
        logging.info("Client connected.")
        audio_data = bytearray()
        transcription: list[str] = []
        session_type = stt_pb2.ChunkType.MEMORY  # Default to memory
        session_id = None
        memory_id = None

        try:
            logging.info("Received a new transcription request.")
            buffer = b""
            first_chunk = True

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

                # Handle different input types (audio or text)
                if chunk.HasField("audio_data"):
                    buffer += chunk.audio_data
                    audio_data += chunk.audio_data
                    logging.debug(
                        f"Received audio chunk of size: {len(chunk.audio_data)} bytes, "
                        + f"buffer size: {len(buffer)} bytes"
                    )
                elif chunk.HasField("text_data"):
                    # Direct text input - process immediately
                    logging.info(f"Received text input: {chunk.text_data}")
                    transcription.append(chunk.text_data)
                    # Return the text directly as a transcript
                    response = stt_pb2.MemoryChunk(
                        text_data=chunk.text_data,
                        metadata=stt_pb2.ChunkMetadata(
                            session_id=session_id,
                            memory_id=memory_id,
                            type=stt_pb2.ChunkType.TRANSCRIPT,
                        ),
                    )
                    yield response
                    continue

                # Process audio when buffer exceeds 1 second and we have audio data
                # (16000 samples for 16kHz)
                if (
                    chunk.HasField("audio_data") and len(buffer) >= SAMPLE_RATE * 4
                ):  # 2 bytes per sample
                    logging.debug("Processing audio buffer for transcription.")
                    audio_array = (
                        np.frombuffer(buffer, dtype=np.int16).astype(np.float32)
                        / 32768.0
                    )
                    segments, _ = self.transcriber.transcribe(audio_array)
                    transcription.extend(segment.text for segment in segments)

                    for segment in segments:
                        logging.debug(f"Transcribed segment: {segment.text}")
                        response = stt_pb2.MemoryChunk(
                            text_data=segment.text,
                            metadata=stt_pb2.ChunkMetadata(
                                session_id=session_id,
                                memory_id=memory_id,
                                type=stt_pb2.ChunkType.TRANSCRIPT,
                            ),
                        )
                        yield response

                    # Keep a small overlap to avoid cutting words
                    buffer = buffer[-1600:]  # Keep last 0.1 seconds

        except grpc.aio.AioRpcError as e:
            logging.error(f"Error during transcription: {e}")
        finally:
            logging.info("Client disconnected.")
            full_transcription = " ".join(transcription)

            if session_type == stt_pb2.ChunkType.MEMORY:
                # For memory sessions, save the data (audio if available)
                if audio_data or transcription:
                    # Create a domain object using the factory method
                    memory = MemoryRequest.create(
                        audio_data=bytes(audio_data) if audio_data else None,
                        text=transcription,
                        memory_type=MemoryType.MEMORY,
                    )
                    uri = await self.persistence_service.save_memory(memory)
                    logging.info(f"Memory saved with URI: {uri}")
                    await self.vector_store_service.index_memory(memory)

            elif session_type == stt_pb2.ChunkType.QUESTION:
                # Generate answer for question sessions
                logging.info(f"Processing question: {full_transcription}")

                # Create a domain object for the question as well
                question_memory = MemoryRequest.create(
                    audio_data=bytes(audio_data) if audio_data else None,
                    text=transcription,
                    memory_type=MemoryType.QUESTION,
                )

                # Use the vector store repository to search for relevant memories
                memory_context = await self.vector_store_service.search(
                    question_memory, limit=5
                )

                answer_task = asyncio.create_task(
                    self.rag_service.answer_question(question_memory, memory_context)
                )

                # Stream memory context results to client
                if memory_context and memory_context.memories:
                    logging.info(
                        f"Sending {len(memory_context.memories)} memories from context"
                    )
                    for memory in memory_context.memories.values():
                        # Convert to chunk and stream to client with MEMORY_PREVIEW type
                        memory_chunk = memory.to_chunk(
                            session_id=session_id, chunk_type=stt_pb2.ChunkType.MEMORY
                        )

                        yield memory_chunk
                        logging.debug(f"Sent memory preview: {memory.text[:50]}...")

                # Send answer back to client
                answer_request = await answer_task
                answer_chunk = answer_request.to_chunk(
                    session_id=session_id, chunk_type=stt_pb2.ChunkType.ANSWER
                )

                yield answer_chunk
                logging.info(f"Sent answer: {answer_chunk.text_data[:50]}...")


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
        # Shuts down the server with 5 seconds of grace period. During the
        # grace period, the server won't accept new connections and allow
        # existing RPCs to continue within the grace period.
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
