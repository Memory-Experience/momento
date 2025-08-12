import asyncio
import io
import logging
import os
from datetime import datetime

import grpc
from protos.generated.py import stt_pb2, stt_pb2_grpc
from pydub import AudioSegment

# Constants
PORT = 50051
RECORDINGS_DIR = "recordings"
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2  # 2 bytes = 16 bits


class TranscriptionServiceServicer(stt_pb2_grpc.TranscriptionServiceServicer):
    """Provides methods that implement functionality of transcription service."""

    async def Transcribe(self, request_iterator, context):
        """Bidirectional streaming RPC for audio transcription."""
        logging.info("Client connected.")
        audio_data = bytearray()
        transcription_count = 0

        try:
            async for audio_chunk in request_iterator:
                if audio_chunk.data:
                    audio_data.extend(audio_chunk.data)
                    logging.debug(
                        f"Received audio chunk of size: {len(audio_chunk.data)}"
                    )

                    # For MVP, send back a dummy transcription inside a StreamResponse
                    transcription_count += 1
                    response = stt_pb2.StreamResponse(
                        transcript=stt_pb2.Transcript(
                            text=f"Received chunk {transcription_count}."
                        )
                    )
                    yield response

        except grpc.aio.AioRpcError as e:
            logging.error(f"Error during transcription: {e}")
        finally:
            logging.info("Client disconnected.")
            if audio_data:
                self.save_recording(audio_data)

    def save_recording(self, audio_data: bytearray):
        """Saves the recorded audio to a WAV file."""
        if not os.path.exists(RECORDINGS_DIR):
            os.makedirs(RECORDINGS_DIR)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        webm_filename = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.webm")
        wav_filename = os.path.join(RECORDINGS_DIR, f"recording_{timestamp}.wav")

        try:
            with open(webm_filename, "wb") as f:
                f.write(audio_data)

            # Convert webm to wav using pydub
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_data), format="webm"
            )
            audio_segment.export(wav_filename, format="wav")

            logging.info(f"Recording saved to {webm_filename} and {wav_filename}")
        except Exception as e:
            logging.error(f"Failed to save recording: {e}")


async def serve() -> None:
    """Starts the gRPC server."""
    server = grpc.aio.server()
    stt_pb2_grpc.add_TranscriptionServiceServicer_to_server(
        TranscriptionServiceServicer(), server
    )
    server.add_insecure_port(f"[::]:{PORT}")
    logging.info(f"Server started on port {PORT}")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    asyncio.run(serve())
