from unittest.mock import AsyncMock, MagicMock

import main
import pytest
from domain.memory_context import MemoryContext
from domain.memory_request import MemoryRequest
from protos.generated.py.stt_pb2 import ChunkMetadata, ChunkType, MemoryChunk


@pytest.mark.asyncio
async def test_transcribe_saves_memory(mocker):
    # Patch FasterWhisperTranscriber and PersistenceService
    mock_transcriber = mocker.patch("main.FasterWhisperTranscriber")
    mock_transcriber.return_value.transcribe.return_value = (
        [MagicMock(text="test")],
        None,
    )
    mock_transcriber.return_value.initialize.return_value = None

    mock_persistence = mocker.patch("main.PersistenceService")
    mock_persistence.return_value.save_memory = AsyncMock(return_value=True)

    # Patch vector store service
    mock_vector_store = mocker.patch("main.VectorStoreService")
    mock_vector_store.return_value.index_memory = AsyncMock(return_value=None)

    # Patch Memory.create
    mock_memory = MagicMock()
    mock_memory_instance = MagicMock()
    mock_memory.create.return_value = mock_memory_instance
    mocker.patch("main.MemoryRequest", mock_memory)

    # Create servicer and simulate request
    servicer = main.TranscriptionServiceServicer()
    chunk = MemoryChunk(
        audio_data=b"\x00" * (main.SAMPLE_RATE * 4),
        metadata=ChunkMetadata(
            type=ChunkType.MEMORY, session_id="test_session", memory_id="test-memory-id"
        ),
    )

    async def request_iterator():
        yield chunk

    context = MagicMock()

    responses = []
    async for response in servicer.Transcribe(request_iterator(), context):
        responses.append(response)

    # Check that at least one response has the expected text in text_data
    assert any(r.text_data == "test" for r in responses)
    # Verify save_memory was called
    mock_persistence.return_value.save_memory.assert_called()


@pytest.mark.asyncio
async def test_transcribe_text_input(mocker):
    # Create mock answer request and memory context
    mock_answer_request = MagicMock(spec=MemoryRequest)
    mock_answer_request.text = ["test answer"]
    _mock_memory_context = MagicMock(spec=MemoryContext)

    # Patch RAG service
    mocker.patch.object(main.SimpleRAGService, "__init__", return_value=None)
    mocker.patch.object(
        main.SimpleRAGService,
        "answer_question",
        AsyncMock(return_value=mock_answer_request.text[0]),
    )

    # Patch PersistenceService
    mock_persistence = mocker.patch("main.PersistenceService")
    mock_persistence.return_value.save_memory = AsyncMock(return_value="test-uri")

    # Patch MemoryRequest.create
    mock_memory = MagicMock()
    mock_memory_instance = MagicMock()
    mock_memory.create.return_value = mock_memory_instance
    mocker.patch("main.MemoryRequest", mock_memory)

    # Create servicer and simulate text request
    servicer = main.TranscriptionServiceServicer()
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

    context = MagicMock()

    responses = []
    async for response in servicer.Transcribe(request_iterator(), context):
        responses.append(response)

    # First response should echo back the text input as a transcript
    assert responses[0].text_data == "Hello, this is a test question."
    assert responses[0].metadata.type == ChunkType.TRANSCRIPT

    # There should be an additional response with the answer
    assert any(r.text_data == "test answer" for r in responses)
    assert any(r.metadata.type == ChunkType.ANSWER for r in responses)

    # Verify save_memory was called with text data
    mock_persistence.return_value.save_memory.assert_called()


@pytest.mark.asyncio
async def test_serve_starts_server(mocker):
    mock_server = MagicMock()

    # Make server methods awaitable
    async def _start():
        return None

    async def _wait():
        return None

    async def _stop(_):
        return None

    mock_server.start = _start
    mock_server.wait_for_termination = _wait
    mock_server.add_insecure_port = MagicMock()
    mock_server.stop = _stop

    mocker.patch("main.grpc.aio.server", return_value=mock_server)
    mocker.patch(
        "main.stt_pb2_grpc.add_TranscriptionServiceServicer_to_server",
        side_effect=lambda serv, srv: None,
    )

    await main.serve()

    mock_server.add_insecure_port.assert_called()
