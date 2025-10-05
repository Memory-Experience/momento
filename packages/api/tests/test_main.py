from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from fastapi import WebSocketDisconnect

import api.main as main
from api.domain.memory_context import MemoryContext
from api.dependency_container import RETRIEVAL_LIMIT
from protos.generated.py.stt_pb2 import ChunkMetadata, ChunkType, MemoryChunk


# ----------------------------
# Lightweight mocked container
# ----------------------------


@dataclass
class DummyContainer:
    transcriber: object
    persistence: object
    vector_store: object
    rag: object
    threshold_filter: object
    sample_rate: int = 16000
    recordings_dir: str = "recordings"
    retrieval_limit: int = RETRIEVAL_LIMIT


@pytest.fixture
def container():
    """Single mock container with all service doubles."""
    # Transcriber
    transcriber = MagicMock()
    transcriber.initialize.return_value = None
    transcriber.transcribe.return_value = ([MagicMock(text="test")], None)

    # Persistence
    persistence = MagicMock()
    persistence.save_memory = AsyncMock(return_value="test-uri")

    # Vector store
    vector_store = MagicMock()
    vector_store.index_memory = AsyncMock(return_value=None)
    vector_store.search = AsyncMock()

    # RAG (we'll configure in the text test)
    rag = MagicMock()

    # Threshold filter
    threshold_filter = MagicMock()
    threshold_filter.filter_context = MagicMock()

    return DummyContainer(
        transcriber=transcriber,
        persistence=persistence,
        vector_store=vector_store,
        rag=rag,
        threshold_filter=threshold_filter,
    )


@pytest.fixture
def servicer(container):
    """Service under test with DI'd container."""
    return main.TranscriptionServiceServicer(container)


@pytest.fixture
def memory_servicer(container):
    """Memory persistence service under test with DI'd container."""
    return main.MemoryPersistService(container)


@pytest.fixture
def qa_servicer(container):
    """Question answering service under test with DI'd container."""
    return main.QuestionAnswerService(container)


@pytest.fixture
def websocket_handler(container):
    """WebSocket handler with mocked container."""
    return main.WebSocketTranscriptionHandler(container)


@pytest.fixture
def mock_websocket():
    """Mock WebSocket connection."""
    websocket = AsyncMock()
    websocket.accept = AsyncMock()
    websocket.close = AsyncMock()
    websocket.send_bytes = AsyncMock()
    websocket.receive_bytes = AsyncMock()
    return websocket


@pytest.fixture
def client():
    """Test client for FastAPI app."""
    return TestClient(main.app)


@pytest.fixture
def collect_responses():
    async def _collect(request_iterator, service_method, context):
        out = []
        async for r in service_method(request_iterator, context):
            out.append(r)
        return out

    return _collect


# ----------------------------
# FastAPI HTTP Endpoint Tests
# ----------------------------


def test_health_endpoint(client):
    """Test the health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "service": "momento-ws"}


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Momento WebSocket API"
    assert "/ws/transcribe" in data["websocket_endpoint"]
    assert data["health_check"] == "/health"


# ----------------------------
# WebSocket Connection Tests
# ----------------------------


@pytest.mark.asyncio
async def test_websocket_handler_init(container):
    """Test that WebSocket handler initializes correctly."""
    handler = main.WebSocketTranscriptionHandler(container)
    assert handler.servicer is not None
    assert handler.persist_servicer is not None
    assert handler.qa_servicer is not None
    assert isinstance(handler.servicer, main.TranscriptionServiceServicer)
    assert isinstance(handler.persist_servicer, main.MemoryPersistService)
    assert isinstance(handler.qa_servicer, main.QuestionAnswerService)


@pytest.mark.asyncio
async def test_transcription_connection_lifecycle(websocket_handler, mock_websocket):
    """Test WebSocket ws/transcribe connection acceptance and cleanup."""
    # Mock the message processing to avoid complexity
    with patch.object(websocket_handler, "_process_transcription") as mock_process:
        mock_process.side_effect = WebSocketDisconnect()

        await websocket_handler.handle_connection(mock_websocket, "transcribe")

        # Verify connection was accepted
        mock_websocket.accept.assert_awaited_once()

        # Verify message processing was attempted
        mock_process.assert_awaited_once()


@pytest.mark.asyncio
async def test_memory_connection_lifecycle(websocket_handler, mock_websocket):
    """Test WebSocket ws/memory connection acceptance and cleanup."""
    # Mock the message processing to avoid complexity
    with patch.object(websocket_handler, "_process_memory") as mock_process:
        mock_process.side_effect = WebSocketDisconnect()

        await websocket_handler.handle_connection(mock_websocket, "memory")

        # Verify connection was accepted
        mock_websocket.accept.assert_awaited_once()

        # Verify message processing was attempted
        mock_process.assert_awaited_once()


@pytest.mark.asyncio
async def test_websocket_connection_error_handling(websocket_handler, mock_websocket):
    """Test WebSocket error handling during connection."""
    # Mock the message processing to raise an exception
    with patch.object(websocket_handler, "_process_transcription") as mock_process:
        mock_process.side_effect = Exception("Test error")

        await websocket_handler.handle_connection(mock_websocket, "transcribe")

        # Verify connection was accepted
        mock_websocket.accept.assert_awaited_once()


# ----------------------------
# WebSocket Message Processing Tests
# ----------------------------


@pytest.mark.asyncio
async def test_websocket_message_processing_memory(
    websocket_handler, mock_websocket, container
):
    """Test WebSocket message processing for memory transcription."""
    session_id = "test_session"
    memory_id = "test-memory-id"

    # Create protobuf messages
    audio_chunk = MemoryChunk(
        audio_data=b"\x00" * (container.sample_rate * 4),
        metadata=ChunkMetadata(
            type=ChunkType.MEMORY, session_id=session_id, memory_id=memory_id
        ),
    )
    final_chunk = MemoryChunk(
        metadata=ChunkMetadata(
            type=ChunkType.MEMORY,
            session_id=session_id,
            memory_id=memory_id,
            is_final=True,
        )
    )

    # Mock WebSocket to return our test messages
    mock_websocket.receive_bytes.side_effect = [
        audio_chunk.SerializeToString(),
        final_chunk.SerializeToString(),
        WebSocketDisconnect(),  # End the loop
    ]

    # Track what gets sent back
    sent_messages = []

    async def mock_send_bytes(data):
        chunk = MemoryChunk()
        chunk.ParseFromString(data)
        sent_messages.append(chunk)

    mock_websocket.send_bytes.side_effect = mock_send_bytes

    # Run the message processing
    await websocket_handler._process_memory(mock_websocket, id(mock_websocket))

    # Verify that transcription and memory saving occurred
    assert container.persistence.save_memory.awaited

    memory_messages = [
        msg for msg in sent_messages if msg.metadata.type == ChunkType.MEMORY
    ]

    assert len(memory_messages) > 0  # Should have memory confirmation


@pytest.mark.asyncio
async def test_websocket_message_processing_question(
    websocket_handler, mock_websocket, container
):
    """Test WebSocket message processing for question answering."""
    session_id = "test_question_session"
    memory_id = "test-memory-id"

    # Setup RAG mock response
    mock_memory_req = MagicMock()
    mock_memory_req.text = ["test answer"]
    mock_memory_req.id = "answer-id"

    mock_memory_response = MagicMock()
    mock_memory_response.response = mock_memory_req
    mock_memory_response.metadata = {"is_final": True}
    mock_memory_response.tokens_used = 10

    # Vector store returns a MemoryContext
    mock_memory_context = MagicMock(spec=MemoryContext)
    mock_memory_context.memories = {}
    container.vector_store.search.return_value = mock_memory_context

    class AsyncGenMock:
        async def __aiter__(self):
            yield mock_memory_response

    container.rag.answer_question = MagicMock(return_value=AsyncGenMock())

    # Create protobuf messages
    text_chunk = MemoryChunk(
        text_data="Hello, this is a test question.",
        metadata=ChunkMetadata(
            type=ChunkType.QUESTION, session_id=session_id, memory_id=memory_id
        ),
    )
    final_chunk = MemoryChunk(
        metadata=ChunkMetadata(
            type=ChunkType.QUESTION,
            session_id=session_id,
            memory_id=memory_id,
            is_final=True,
        )
    )

    # Mock WebSocket to return our test messages
    mock_websocket.receive_bytes.side_effect = [
        text_chunk.SerializeToString(),
        final_chunk.SerializeToString(),
        WebSocketDisconnect(),  # End the loop
    ]

    # Track what gets sent back
    sent_messages = []

    async def mock_send_bytes(data):
        chunk = MemoryChunk()
        chunk.ParseFromString(data)
        sent_messages.append(chunk)

    mock_websocket.send_bytes.side_effect = mock_send_bytes

    # Run the message processing
    await websocket_handler._process_question(mock_websocket, id(mock_websocket))

    answer_messages = [
        msg for msg in sent_messages if msg.metadata.type == ChunkType.ANSWER
    ]

    assert len(answer_messages) > 0  # Should have answer responses

    # Verify RAG was called
    container.rag.answer_question.assert_called_once()


# ----------------------------
# Application Startup Tests
# ----------------------------


@pytest.mark.asyncio
async def test_startup_event():
    """Test that the startup event initializes the handler correctly."""
    with patch("api.main.Container.create") as mock_create:
        mock_container = MagicMock()
        mock_create.return_value = mock_container

        # Trigger startup event
        await main.startup_event()

        # Verify container was created and handler was initialized
        mock_create.assert_called_once()
        assert main.handler is not None
        assert isinstance(main.handler, main.WebSocketTranscriptionHandler)


# ----------------------------
# Legacy Transcription Service Tests (maintaining compatibility)
# ----------------------------


@pytest.mark.asyncio
async def test_transcribe_saves_memory(container, memory_servicer, collect_responses):
    """Test that memory transcription works through servicer (legacy compatibility)."""
    session_id = "test_session"
    memory_id = "test-memory-id"

    audio_chunk = MemoryChunk(
        audio_data=b"\x00" * (container.sample_rate * 4),
        metadata=ChunkMetadata(
            type=ChunkType.MEMORY, session_id=session_id, memory_id=memory_id
        ),
    )
    final_chunk = MemoryChunk(
        metadata=ChunkMetadata(
            type=ChunkType.MEMORY,
            session_id=session_id,
            memory_id=memory_id,
            is_final=True,
        )
    )

    async def request_iterator():
        yield audio_chunk
        yield final_chunk

    responses = await collect_responses(
        request_iterator(), memory_servicer.StoreMemory, None
    )

    container.persistence.save_memory.assert_awaited()
    assert any(r.metadata.type == ChunkType.MEMORY for r in responses)


@pytest.mark.asyncio
async def test_transcribe_text_input(container, qa_servicer, collect_responses):
    """Test text input processing through servicer (legacy compatibility)."""
    # Answer payload from RAG
    mock_memory_req = MagicMock()
    mock_memory_req.text = ["test answer"]
    mock_memory_req.id = "answer-id"

    mock_memory_response = MagicMock()
    mock_memory_response.response = mock_memory_req
    mock_memory_response.metadata = {"is_final": True}
    mock_memory_response.tokens_used = 10

    # Vector store returns a MemoryContext
    mock_memory_context = MagicMock(spec=MemoryContext)
    mock_memory_context.memories = {}
    container.vector_store.search.return_value = mock_memory_context

    class AsyncGenMock:
        async def __aiter__(self):
            yield mock_memory_response

    container.rag.answer_question = MagicMock(return_value=AsyncGenMock())

    session_id = "test_text_session"
    memory_id = "test-memory-id"
    text_chunk = MemoryChunk(
        text_data="Hello, this is a test question.",
        metadata=ChunkMetadata(
            type=ChunkType.QUESTION, session_id=session_id, memory_id=memory_id
        ),
    )
    final_chunk = MemoryChunk(
        metadata=ChunkMetadata(
            type=ChunkType.QUESTION,
            session_id=session_id,
            memory_id=memory_id,
            is_final=True,
        )
    )

    async def request_iterator():
        yield text_chunk
        yield final_chunk

    responses = await collect_responses(
        request_iterator(), qa_servicer.AnswerQuestion, None
    )

    # Then at least one ANSWER with expected text
    answer_responses = [r for r in responses if r.metadata.type == ChunkType.ANSWER]
    assert answer_responses, "No ANSWER responses found"
    assert any(r.text_data == "test answer" for r in answer_responses)

    # RAG called with the chunk size passthrough we care about
    _, kwargs = container.rag.answer_question.call_args
    assert kwargs["chunk_size_tokens"] == 8
