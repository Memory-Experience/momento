import asyncio
import logging
import sys

import grpc
import numpy as np
from domain.memory_request import MemoryRequest, MemoryType
from persistence.persistence_service import PersistenceService
from persistence.repositories.file_repository import FileRepository
from protos.generated.py import stt_pb2, stt_pb2_grpc
from rag.rag_service import SimpleRAGService
from transcriber.faster_whisper_transcriber import FasterWhisperTranscriber

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
        self.rag_service = SimpleRAGService()

        # Initialize the persistence service with a file repository
        repository = FileRepository(storage_dir=RECORDINGS_DIR, sample_rate=SAMPLE_RATE)
        self.persistence_service = PersistenceService(repository)

    async def Transcribe(self, request_iterator, context):
        """Bidirectional streaming RPC for audio transcription."""
        logging.info("Client connected.")
        audio_data = bytearray()
        transcription: list[str] = []
        session_type = stt_pb2.MEMORY  # Default to memory
        session_id = None

        try:
            logging.info("Received a new transcription request.")
            buffer = b""
            first_chunk = True

            async for chunk in request_iterator:
                # Handle session metadata from first chunk
                if first_chunk and chunk.metadata:
                    session_type = chunk.metadata.type
                    session_id = chunk.metadata.session_id
                    logging.info(
                        f"Session {session_id} started with type: {session_type}"
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
                    response = stt_pb2.StreamResponse(
                        transcript=stt_pb2.Transcript(text=chunk.text_data)
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
                        response = stt_pb2.StreamResponse(
                            transcript=stt_pb2.Transcript(text=segment.text)
                        )
                        yield response

                    # Keep a small overlap to avoid cutting words
                    buffer = buffer[-1600:]  # Keep last 0.1 seconds

        except grpc.aio.AioRpcError as e:
            logging.error(f"Error during transcription: {e}")
        finally:
            logging.info("Client disconnected.")
            full_transcription = "".join(transcription)

            if session_type == stt_pb2.MEMORY:
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
                    self.rag_service.add_memory(full_transcription, uri)

            elif session_type == stt_pb2.QUESTION:
                # Generate answer for question sessions
                logging.info(f"Processing question: {full_transcription}")

                # Create a domain object for the question as well
                question_memory = MemoryRequest.create(
                    audio_data=bytes(audio_data) if audio_data else None,
                    text=transcription,
                    memory_type=MemoryType.QUESTION,
                )
                # Save the question for future reference if needed
                await self.persistence_service.save_memory(question_memory)

                answer_text = self.rag_service.search_memories(full_transcription)

                # Send answer back to client
                answer_response = stt_pb2.StreamResponse(
                    answer=stt_pb2.Answer(text=answer_text)
                )
                yield answer_response
                logging.info(f"Sent answer: {answer_text[:50]}...")


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
