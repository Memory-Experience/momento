from unittest.mock import AsyncMock, MagicMock

import main
import pytest
from domain.memory_context import MemoryContext
from domain.memory_request import MemoryRequest
from protos.generated.py.stt_pb2 import ChunkMetadata, ChunkType, MemoryChunk


@pytest.fixture
def mock_services(mocker):
    """Set up common service mocks used across tests."""
    # Transcription service
    mock_transcriber = mocker.patch("main.FasterWhisperTranscriber")
    mock_transcriber.return_value.transcribe.return_value = (
        [MagicMock(text="test")],
        None,
    )
    mock_transcriber.return_value.initialize.return_value = None

    # Persistence service
    mock_persistence = mocker.patch("main.PersistenceService")
    mock_persistence.return_value.save_memory = AsyncMock(return_value="test-uri")

    # Vector store service
    mock_vector_store = mocker.patch("main.VectorStoreService")
    mock_vector_store.return_value.index_memory = AsyncMock(return_value=None)
    mock_vector_store.return_value.search = AsyncMock()  # Add this line

    # Memory domain object
    mock_memory = MagicMock()
    mock_memory_instance = MagicMock()
    mock_memory.create.return_value = mock_memory_instance
    mocker.patch("main.MemoryRequest", mock_memory)

    return {
        "transcriber": mock_transcriber,
        "persistence": mock_persistence,
        "vector_store": mock_vector_store,
        "memory": mock_memory,
        "memory_instance": mock_memory_instance,
    }


@pytest.fixture
def servicer():
    """Create a TranscriptionServiceServicer instance."""
    return main.TranscriptionServiceServicer()


@pytest.fixture
def grpc_context():
    """Create a mock gRPC context."""
    return MagicMock()


@pytest.fixture
def collect_responses():
    """Fixture to provide a function that collects responses from a gRPC stream."""

    async def _collect(request_iterator, service_method, context):
        responses = []
        async for response in service_method(request_iterator, context):
            responses.append(response)
        return responses

    return _collect


@pytest.mark.asyncio
async def test_transcribe_saves_memory(
    mock_services, servicer, grpc_context, collect_responses
):
    # Create chunk for memory session
    chunk = MemoryChunk(
        audio_data=b"\x00" * (main.SAMPLE_RATE * 4),
        metadata=ChunkMetadata(
            type=ChunkType.MEMORY, session_id="test_session", memory_id="test-memory-id"
        ),
    )

    async def request_iterator():
        yield chunk

    # Collect responses from the service
    responses = await collect_responses(
        request_iterator(), servicer.Transcribe, grpc_context
    )

    # Check that at least one response has the expected text in text_data
    assert any(r.text_data == "test" for r in responses)
    # Verify save_memory was called
    mock_services["persistence"].return_value.save_memory.assert_called()


@pytest.mark.asyncio
async def test_transcribe_text_input(
    mocker, mock_services, servicer, grpc_context, collect_responses
):
    # Create mock answer request
    mock_answer_request = MagicMock(spec=MemoryRequest)
    mock_answer_request.text = ["test answer"]
    mock_answer_request.to_chunk.return_value = MemoryChunk(
        text_data="test answer",
        metadata=ChunkMetadata(
            type=ChunkType.ANSWER, session_id="test_text_session", memory_id="answer-id"
        ),
    )

    # Create mock memory context and set it as the return value for search
    mock_memory_context = MagicMock(spec=MemoryContext)
    mock_memory_context.memories = {}  # Empty dict for this test
    mock_services["vector_store"].return_value.search.return_value = mock_memory_context

    # Patch RAG service
    mocker.patch.object(main.LLMRAGService, "__init__", return_value=None)
    mocker.patch.object(
        main.LLMRAGService,
        "answer_question",
        AsyncMock(return_value=mock_answer_request),
    )

    # Create test question chunk
    text_chunk = MemoryChunk(
        text_data="Hello, this is a test question.",
        metadata=ChunkMetadata(
            type=ChunkType.QUESTION,
            session_id="test_text_session",
            memory_id="test-memory-id",
        ),
    )

    async def request_iterator():
        yield text_chunk

    # Collect responses from the service
    responses = await collect_responses(
        request_iterator(), servicer.Transcribe, grpc_context
    )

    # First response should echo back the text input as a transcript
    assert responses[0].text_data == "Hello, this is a test question."
    assert responses[0].metadata.type == ChunkType.TRANSCRIPT

    # There should be an additional response with the answer
    assert any(r.text_data == "test answer" for r in responses)
    assert any(r.metadata.type == ChunkType.ANSWER for r in responses)


@pytest.fixture
def mock_server():
    """Create a mock gRPC server with awaitable methods."""
    server = MagicMock()

    async def _start():
        return None

    async def _wait():
        return None

    async def _stop(_):
        return None

    server.start = _start
    server.wait_for_termination = _wait
    server.add_insecure_port = MagicMock()
    server.stop = _stop

    return server


@pytest.mark.asyncio
async def test_serve_starts_server(mocker, mock_server):
    mocker.patch("main.grpc.aio.server", return_value=mock_server)
    mocker.patch(
        "main.stt_pb2_grpc.add_TranscriptionServiceServicer_to_server",
        side_effect=lambda serv, srv: None,
    )

    await main.serve()

    mock_server.add_insecure_port.assert_called()
