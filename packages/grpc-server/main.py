import asyncio
import logging
import os
import sys
from datetime import datetime

import grpc
import numpy as np
import psycopg2
from dotenv import load_dotenv
from protos.generated.py import stt_pb2, stt_pb2_grpc
from pydub import AudioSegment
from transcriber.faster_whisper_transcriber import FasterWhisperTranscriber

_cleanup_coroutines = []

# Constants
PORT = 50051
RECORDINGS_DIR = "recordings"
SAMPLE_RATE = 16000


class TranscriptionServiceServicer(stt_pb2_grpc.TranscriptionServiceServicer):
    """Provides methods that implement functionality of transcription service."""

    def __init__(self, db_connection: psycopg2.extensions.connection):
        self.db_connection = db_connection
        self.transcriber = FasterWhisperTranscriber()
        self.transcriber.initialize()

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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if audio_data:
                self.save_recording(f"recording_{timestamp}.wav", audio_data)
            if transcription:
                self.save_transcription(f"transcription_{timestamp}.txt", transcription)

    def save_recording(self, file_name: str, audio_data: bytearray):
        """Saves the recorded audio to a WAV file."""
        if not os.path.exists(RECORDINGS_DIR):
            os.makedirs(RECORDINGS_DIR)

        wav_filename = os.path.join(RECORDINGS_DIR, file_name)

        try:
            # PCM signed 16-bit little-endian format
            audio_segment = AudioSegment(
                data=audio_data,
                sample_width=2,  # 2 bytes for s16le
                frame_rate=SAMPLE_RATE,
                channels=1,  # Mono audio
            )
            audio_segment.export(wav_filename, format="wav")

            logging.info(f"Recording saved to {wav_filename}")
        except Exception as e:
            logging.error(f"Failed to save recording: {e}")

    def save_transcription(self, file_name: str, transcription: list[str]) -> None:
        """Saves the transcription to a text file."""
        if not os.path.exists(RECORDINGS_DIR):
            os.makedirs(RECORDINGS_DIR)

        txt_filename = os.path.join(RECORDINGS_DIR, file_name)

        try:
            with open(txt_filename, "w") as f:
                f.writelines("".join(transcription))
            logging.info(f"Transcription saved to {txt_filename}")
        except Exception as e:
            logging.error(f"Failed to save transcription: {e}")


async def serve() -> None:
    """Starts the gRPC server."""
    conn = psycopg2.connect(
        dbname=os.getenv("DB_NAME"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
    )
    server = grpc.aio.server()
    stt_pb2_grpc.add_TranscriptionServiceServicer_to_server(
        TranscriptionServiceServicer(db_connection=conn), server
    )
    listen_addr = f"[::]:{PORT}"
    server.add_insecure_port(listen_addr)
    logging.info("🚀 SST Microservice is running on %s", listen_addr)
    await server.start()

    async def server_graceful_shutdown():
        logging.info("Starting graceful shutdown...")
        conn.close()
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

    if "-e" in sys.argv:
        env_index = sys.argv.index("-e") + 1
        if env_index < len(sys.argv):
            env_file = sys.argv[env_index]
            logging.info(f"Loading {env_file} file")
            load_dotenv(env_file)
    else:
        logging.info("Loading default .env file")
        load_dotenv()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(serve())
    finally:
        logging.info("Cleaning up...")
        loop.run_until_complete(*_cleanup_coroutines)
        loop.close()
