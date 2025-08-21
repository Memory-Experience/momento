import asyncio
import logging
import sys

import grpc
import numpy as np
from domain.memory import Memory
from persistence.persistence_service import PersistenceService
from persistence.repositories.file_repository import FileRepository
from protos.generated.py import stt_pb2, stt_pb2_grpc
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

        # Initialize the persistence service with a file repository
        repository = FileRepository(storage_dir=RECORDINGS_DIR, sample_rate=SAMPLE_RATE)
        self.persistence_service = PersistenceService(repository)

    async def Transcribe(self, request_iterator, context):
        """Bidirectional streaming RPC for audio transcription."""
        logging.info("Client connected.")
        audio_data = bytearray()
        transcription: list[str] = []

        try:
            logging.info("Received a new transcription request.")
            buffer = b""
            async for chunk in request_iterator:
                buffer += chunk.data
                audio_data += chunk.data
                logging.debug(
                    f"Received chunk of size: {len(chunk.data)} bytes, "
                    + f"buffer size: {len(buffer)} bytes"
                )

                # Process audio when buffer exceeds 1 second
                # (16000 samples for 16kHz)
                if len(buffer) >= SAMPLE_RATE * 4:  # 2 bytes per sample
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
            if audio_data:
                # Create a domain object using the factory method
                memory = Memory.create(bytes(audio_data), transcription)
                uri = await self.persistence_service.save_memory(memory)
                logging.info(f"Memory saved with URI: {uri}")


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
