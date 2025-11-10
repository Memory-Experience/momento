import logging
import grpc
from collections.abc import AsyncIterator

import numpy as np

from protos.generated.py import stt_pb2_grpc, stt_pb2
from .dependency_container import Container


class TranscriptionServiceServicer(stt_pb2_grpc.TranscriptionServiceServicer):
    """
    gRPC servicer for real-time audio transcription.

    Provides bidirectional streaming transcription, accepting audio chunks
    and returning transcript segments as they become available.
    """

    def __init__(self, container: Container):
        """
        Initialize the transcription servicer.

        Args:
            container (Container): Dependency container with transcriber
                and configuration
        """
        self.transcriber = container.transcriber
        self.sample_rate = container.sample_rate

    async def Transcribe(
        self, request_iterator, context
    ) -> AsyncIterator[stt_pb2.MemoryChunk]:
        """
        Bidirectional streaming RPC for audio transcription.

        Accepts streaming audio or text input and returns transcription segments
        in real-time. Handles explicit final markers to signal completion.

        Args:
            request_iterator (AsyncIterator[MemoryChunk]): Async iterator
                yielding MemoryChunk messages with audio/text data
            context (grpc.aio.ServicerContext): gRPC context for the
                request

        Yields:
            protos.generated.py.stt_pb2.MemoryChunk: Transcription
                segments as they become available
        """
        logging.info("Client connected.")

        # State tracking
        audio_data = bytearray()
        transcription = []
        session_type = stt_pb2.ChunkType.MEMORY  # Default type
        session_id = None
        memory_id = None
        first_chunk = True
        question_completed = False
        transcription_complete = False

        try:
            logging.info("Received a new transcription request.")

            # Reset transcriber state for new session
            self.transcriber.reset_state()

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
                            logging.warning(
                                "Deprecated memory handling in TranscriptionService"
                            )
                            continue
                        elif (
                            session_type == stt_pb2.ChunkType.QUESTION
                            and not question_completed
                        ):
                            logging.warning(
                                "Deprecated question handling in TranscriptionService"
                            )
                            question_completed = True

                    continue

                # Handle audio input
                if chunk.HasField("audio_data"):
                    audio_data += chunk.audio_data

                    # Convert audio bytes to float32 numpy array
                    audio_array = (
                        np.frombuffer(chunk.audio_data, dtype=np.int16).astype(
                            np.float32
                        )
                        / 32768.0
                    )

                    # Pass to transcriber (handles its own buffering)
                    segments, _ = self.transcriber.transcribe(audio_array)

                    # Accumulate transcription text
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
