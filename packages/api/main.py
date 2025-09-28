import logging
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from protos.generated.py import stt_pb2
from api.transcription_servicer import TranscriptionServiceServicer
from api.dependency_container import Container

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
    """Handles WebSocket connections and message processing using protobuf messages."""

    def __init__(self, container: Container):
        self.servicer = TranscriptionServiceServicer(container)

    async def handle_connection(self, websocket: WebSocket):
        """Handle a WebSocket connection."""
        await websocket.accept()
        connection_id = id(websocket)
        active_connections[str(connection_id)] = websocket

        logging.info(f"WebSocket client connected: {connection_id}")

        try:
            await self._process_messages(websocket, connection_id)
        except WebSocketDisconnect:
            logging.info(f"WebSocket client disconnected: {connection_id}")
        except Exception as e:
            logging.error(f"Error handling WebSocket connection {connection_id}: {e}")
        finally:
            active_connections.pop(str(connection_id), None)

    async def _process_messages(self, websocket: WebSocket, connection_id: int):
        """Process incoming messages from the WebSocket connection."""

        # Create a message generator that yields protobuf messages
        async def message_generator():
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

        # Process messages using the existing transcription servicer logic
        try:
            async for response_chunk in self.servicer.Transcribe(
                message_generator(), None
            ):
                logging.debug(f">>>Sending response chunk: {response_chunk}")
                await websocket.send_bytes(response_chunk.SerializeToString())
        except Exception as e:
            logging.error(f"Error in transcription processing: {e}")
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

    await handler.handle_connection(websocket)


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
        "main:app",
        host="0.0.0.0",
        port=PORT,
        log_level="info" if log_level == logging.INFO else "debug",
        reload=bool("-r" in sys.argv),
    )
