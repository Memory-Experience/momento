import asyncio
import logging
import sys

import grpc
from protos.generated.py import stt_pb2_grpc
from api.transcription_servicer import TranscriptionServiceServicer
from api.dependency_container import Container

_cleanup_coroutines = []

# Constants
PORT = 50051


async def serve(container: Container) -> None:
    """Starts the gRPC server."""
    server = grpc.aio.server()
    stt_pb2_grpc.add_TranscriptionServiceServicer_to_server(
        TranscriptionServiceServicer(container), server
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

    container = Container.create()

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(serve(container))
    finally:
        logging.info("Cleaning up...")
        loop.run_until_complete(*_cleanup_coroutines)
        loop.close()
