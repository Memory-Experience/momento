import asyncio
import logging
from collections.abc import AsyncGenerator

import grpc
from protos.generated.py import stt_pb2, stt_pb2_grpc

CHANNEL_OPTIONS = [
    ("grpc.lb_policy_name", "pick_first"),
    ("grpc.enable_retries", 0),
    ("grpc.keepalive_timeout_ms", 10000),
    ("grpc.grpclb_call_timeout_ms", 10000),
]


async def get_user_input(queue: asyncio.Queue, stop_event: asyncio.Event) -> None:
    """Handle user input in a non-blocking way"""
    while not stop_event.is_set():
        # Run input() in a thread to avoid blocking the event loop
        user_input = await asyncio.to_thread(
            lambda: input("Enter text to transcribe (or 'bye' to quit): ")
        )

        if user_input.lower() == "bye":
            stop_event.set()
            break

        # Put the user input in the queue for the audio generator
        await queue.put(user_input)


async def generate_audio_chunks(
    queue: asyncio.Queue, stop_event: asyncio.Event
) -> AsyncGenerator[stt_pb2.AudioChunk, None]:
    """Generate audio chunks from user input via queue"""
    while not stop_event.is_set():
        try:
            # Wait for input with a timeout so we can check stop_event
            # periodically
            user_input = await asyncio.wait_for(queue.get(), timeout=0.1)
            yield stt_pb2.AudioChunk(data=user_input.encode())
            queue.task_done()
        except TimeoutError:
            # No input available yet, continue loop
            continue


async def run() -> None:
    # Create communication mechanisms
    input_queue = asyncio.Queue()
    stop_event = asyncio.Event()

    async with grpc.aio.insecure_channel(
        "localhost:50051", options=CHANNEL_OPTIONS
    ) as channel:
        stub = stt_pb2_grpc.TranscriptionServiceStub(channel)
        msg: str = ""

        # Start the user input task
        input_task = asyncio.create_task(get_user_input(input_queue, stop_event))

        try:
            # Start receiving transcriptions
            async for response in stub.Transcribe(
                generate_audio_chunks(input_queue, stop_event)
            ):
                msg += response.text + " "
                logging.debug(f"Client received: {response.text}")

        except grpc.aio.AioRpcError as e:
            logging.error("RPC error: %s", e)
        except Exception as e:
            logging.error("Error: %s", e)
        finally:
            # Ensure we clean up properly
            if not input_task.done():
                input_task.cancel()
            try:  # noqa: SIM105
                await input_task
            except asyncio.CancelledError:
                pass

        logging.info(f"Final transcription: {msg}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run())
