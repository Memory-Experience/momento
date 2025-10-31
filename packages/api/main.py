import logging
import sys
from typing import Any
from collections.abc import AsyncGenerator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from protos.generated.py import stt_pb2
from .transcription_servicer import TranscriptionServiceServicer
from .memory_persist_service import MemoryPersistService
from .question_answer_service import QuestionAnswerService
from .dependency_container import Container

# Constants
PORT = 8080

# Global container
app = FastAPI(title="Momento Transcription WebSocket API")

# Add CORS middleware for web frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this properly for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Active WebSocket connections
active_connections: dict[str, WebSocket] = {}


class WebSocketTranscriptionHandler:
    """
    Handles WebSocket connections and message processing using protobuf messages.

    Routes WebSocket connections to the appropriate service (transcription, memory
    storage, or question answering) based on the connection type.
    """

    def __init__(self, container: Container):
        """
        Initialize the WebSocket handler.

        Args:
            container: Dependency container with all required services
        """
        self.servicer = TranscriptionServiceServicer(container)
        self.persist_servicer = MemoryPersistService(container)
        self.qa_servicer = QuestionAnswerService(container)

    async def handle_connection(self, websocket: WebSocket, connection_type: str):
        """
        Handle a WebSocket connection of a specific type.

        Args:
            websocket: The WebSocket connection to handle
            connection_type: Type of connection ('transcribe', 'memory', or 'ask')
        """
        await websocket.accept()
        connection_id = id(websocket)
        active_connections[str(connection_id)] = websocket

        logging.info(f"WebSocket client connected: {connection_id}")

        try:
            if connection_type == "transcribe":
                await self._process_transcription(websocket, connection_id)
            elif connection_type == "memory":
                await self._process_memory(websocket, connection_id)
            elif connection_type == "ask":
                await self._process_question(websocket, connection_id)
            else:
                logging.error(f"Unknown connection type: {connection_type}")
                await websocket.close(code=1003, reason="Unknown connection type")
        except WebSocketDisconnect:
            logging.info(f"WebSocket client disconnected: {connection_id}")
        except Exception as e:
            logging.error(f"Error handling WebSocket connection {connection_id}: {e}")
        finally:
            active_connections.pop(str(connection_id), None)

    async def _message_generator(
        self, websocket: WebSocket
    ) -> AsyncGenerator[stt_pb2.MemoryChunk, Any]:
        while True:
            try:
                # Receive message (could be binary protobuf or JSON)
                data = await websocket.receive_bytes()
                memory_chunk = stt_pb2.MemoryChunk()
                memory_chunk.ParseFromString(data)
                yield memory_chunk

            except WebSocketDisconnect:
                break
            except Exception as e:
                logging.error(f"Error processing WebSocket message: {e}")
                break

    async def _process_transcription(self, websocket: WebSocket, connection_id: int):
        """
        Process incoming audio/text for transcription.

        Args:
            websocket: The WebSocket connection
            connection_id: Unique identifier for this connection
        """

        # Create a message generator that yields protobuf messages
        message_generator = self._message_generator(websocket)

        # Process messages using the transcription servicer logic
        try:
            async for response_chunk in self.servicer.Transcribe(
                message_generator, None
            ):
                logging.debug(f">>>Sending response chunk: {response_chunk}")
                await websocket.send_bytes(response_chunk.SerializeToString())
        except Exception as e:
            logging.error(f"Error in transcription processing: {e}")
            await websocket.close(code=1011, reason=f"Server error: {str(e)}")

    async def _process_memory(self, websocket: WebSocket, connection_id: int):
        """
        Process incoming memory storage requests.

        Args:
            websocket: The WebSocket connection
            connection_id: Unique identifier for this connection
        """

        # Create a message generator that yields protobuf messages
        message_generator = self._message_generator(websocket)

        # Process messages using the memory store logic
        try:
            async for response in self.persist_servicer.StoreMemory(
                message_generator, None
            ):
                logging.debug(f">>>Sending memory response: {response}")
                await websocket.send_bytes(response.SerializeToString())
        except Exception as e:
            logging.error(f"Error in memory processing: {e}")
            await websocket.close(code=1011, reason=f"Server error: {str(e)}")

    async def _process_question(self, websocket: WebSocket, connection_id: int):
        """
        Process incoming question answering requests.

        Args:
            websocket: The WebSocket connection
            connection_id: Unique identifier for this connection
        """

        # Create a message generator that yields protobuf messages
        message_generator = self._message_generator(websocket)

        # Process messages using the memory store logic
        try:
            async for response in self.qa_servicer.AnswerQuestion(
                message_generator, None
            ):
                logging.debug(f">>>Sending memory response: {response}")
                await websocket.send_bytes(response.SerializeToString())
        except Exception as e:
            logging.error(f"Error in memory processing: {e}")
            await websocket.close(code=1011, reason=f"Server error: {str(e)}")


# Global handler instance
handler = None


@app.on_event("startup")
async def startup_event():
    """Initialize the container and handler on startup."""
    global handler
    container = Container.create()
    handler = WebSocketTranscriptionHandler(container)
    logging.info("🚀 Momento WebSocket Transcription Service initialized")


@app.websocket("/ws/transcribe")
async def websocket_transcribe(websocket: WebSocket):
    """WebSocket endpoint for transcription."""
    if handler is None:
        await websocket.close(code=1011, reason="Server not initialized")
        return

    await handler.handle_connection(websocket, "transcribe")


@app.websocket("/ws/memory")
async def websocket_memory(websocket: WebSocket):
    """WebSocket endpoint for saving memories."""
    if handler is None:
        await websocket.close(code=1011, reason="Server not initialized")
        return

    await handler.handle_connection(websocket, "memory")


@app.websocket("/ws/ask")
async def websocket_ask(websocket: WebSocket):
    """WebSocket endpoint for answering questions."""
    if handler is None:
        await websocket.close(code=1011, reason="Server not initialized")
        return

    await handler.handle_connection(websocket, "ask")


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "momento-ws"}


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Momento WebSocket API",
        "websocket_endpoint": ["/ws/transcribe"],
        "health_check": "/health",
    }


if __name__ == "__main__":
    # Check if args contain -v for verbose logging
    log_level = logging.INFO
    if "-v" in sys.argv:
        log_level = logging.DEBUG

    logging.basicConfig(level=log_level)

    # Run the FastAPI server with uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info" if log_level == logging.INFO else "debug",
        reload=bool("-r" in sys.argv),
    )
