from unittest.mock import AsyncMock, MagicMock

import main
import pytest
from domain.memory_context import MemoryContext
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
    # Mock the LLM model to avoid initialization errors
    mock_llm = mocker.MagicMock()
    servicer.llm_model = mock_llm

    # Create a proper memory request object that will be part of the response
    mock_memory_req = mocker.MagicMock()
    mock_memory_req.text = [
        "test answer"
    ]  # This must be a list as main.py iterates through it
    mock_memory_req.id = "answer-id"

    # Create a sample memory response that will be yielded by the async generator
    mock_memory_response = mocker.MagicMock()
    mock_memory_response.response = (
        mock_memory_req  # Set our mock memory request as the response
    )
    mock_memory_response.metadata = {"is_final": True}
    mock_memory_response.tokens_used = 10

    # Create mock memory context and set it as the return value for search
    mock_memory_context = MagicMock(spec=MemoryContext)
    mock_memory_context.memories = {}  # Empty dict for this test
    mock_services["vector_store"].return_value.search.return_value = mock_memory_context

    # Create a proper async generator class that implements __aiter__
    class AsyncGenMock:
        async def __aiter__(self):
            yield mock_memory_response

    # Replace the RAG service's answer_question method with our mock
    servicer.rag_service.answer_question = mocker.MagicMock(return_value=AsyncGenMock())

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

    # Print responses for debugging
    for i, r in enumerate(responses):
        print(f"Response {i}: type={r.metadata.type}, text={r.text_data}")

    # First response should echo back the text input as a transcript
    assert responses[0].text_data == "Hello, this is a test question."
    assert responses[0].metadata.type == ChunkType.TRANSCRIPT

    # There should be an additional response with the answer
    answer_responses = [r for r in responses if r.metadata.type == ChunkType.ANSWER]
    assert len(answer_responses) > 0, "No ANSWER responses found"
    assert any(r.text_data == "test answer" for r in answer_responses), (
        "Expected 'test answer' not found"
    )

    # Verify the RAG service was called correctly
    servicer.rag_service.answer_question.assert_called_once()
    # Check that the right parameters were passed
    call_args = servicer.rag_service.answer_question.call_args
    assert call_args[1]["chunk_size_tokens"] == 8


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
    await main.serve()

    mock_server.add_insecure_port.assert_called()
    await main.serve()

    mock_server.add_insecure_port.assert_called()
