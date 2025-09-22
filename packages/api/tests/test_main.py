from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock
import pytest

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

    # RAG (we’ll configure in the text test)
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
    """Service under test with DI’d container."""
    return main.TranscriptionServiceServicer(container)


@pytest.fixture
def grpc_context():
    return MagicMock()


@pytest.fixture
def collect_responses():
    async def _collect(request_iterator, service_method, context):
        out = []
        async for r in service_method(request_iterator, context):
            out.append(r)
        return out

    return _collect


# ----------------------------
# Tests
# ----------------------------


@pytest.mark.asyncio
async def test_transcribe_saves_memory(
    container, servicer, grpc_context, collect_responses
):
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
        request_iterator(), servicer.Transcribe, grpc_context
    )

    assert any(r.text_data == "test" for r in responses)
    container.persistence.save_memory.assert_awaited()
    assert any(r.metadata.type == ChunkType.MEMORY for r in responses)


@pytest.mark.asyncio
async def test_transcribe_text_input(
    container, servicer, grpc_context, collect_responses
):
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
        request_iterator(), servicer.Transcribe, grpc_context
    )

    # First: echo transcript; second: final transcript marker
    assert responses[0].text_data == "Hello, this is a test question."
    assert responses[0].metadata.type == ChunkType.TRANSCRIPT
    assert responses[1].metadata.type == ChunkType.TRANSCRIPT
    assert responses[1].metadata.is_final is True

    # Then at least one ANSWER with expected text
    answer_responses = [r for r in responses if r.metadata.type == ChunkType.ANSWER]
    assert answer_responses, "No ANSWER responses found"
    assert any(r.text_data == "test answer" for r in answer_responses)

    # RAG called with the chunk size passthrough we care about
    _, kwargs = container.rag.answer_question.call_args
    assert kwargs["chunk_size_tokens"] == 8


@pytest.fixture
def mock_server():
    server = MagicMock()
    server.start = AsyncMock()
    server.wait_for_termination = AsyncMock()
    server.add_insecure_port = MagicMock()
    server.stop = AsyncMock()
    return server


@pytest.mark.asyncio
async def test_serve_starts_server(mocker, mock_server, container):
    mocker.patch("main.grpc.aio.server", return_value=mock_server)
    mocker.patch(
        "main.stt_pb2_grpc.add_TranscriptionServiceServicer_to_server",
        side_effect=lambda serv, srv: None,
    )

    await main.serve(container)

    mock_server.add_insecure_port.assert_called_once()
    mock_server.start.assert_awaited_once()
    mock_server.wait_for_termination.assert_awaited_once()
