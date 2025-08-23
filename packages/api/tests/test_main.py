from unittest.mock import AsyncMock, MagicMock

import main
import pytest
from protos.generated.py.stt_pb2 import InputChunk


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

    # Patch Memory.create
    mock_memory = MagicMock()
    mock_memory_instance = MagicMock()
    mock_memory.create.return_value = mock_memory_instance
    mocker.patch("main.MemoryRequest", mock_memory)

    # Create servicer and simulate request
    servicer = main.TranscriptionServiceServicer()
    chunk = InputChunk(
        audio_data=b"\x00" * (main.SAMPLE_RATE * 4),
        metadata=main.stt_pb2.SessionMetadata(
            type=main.stt_pb2.MEMORY, session_id="test_session"
        ),
    )

    async def request_iterator():
        yield chunk

    context = MagicMock()

    responses = []
    async for response in servicer.Transcribe(request_iterator(), context):
        responses.append(response)

    assert any(r.transcript.text == "test" for r in responses)
    mock_persistence.return_value.save_memory.assert_called()


@pytest.mark.asyncio
async def test_transcribe_text_input(mocker):
    # Patch RAG service
    mocker.patch.object(main.SimpleRAGService, "__init__", return_value=None)
    mocker.patch.object(main.SimpleRAGService, "add_memory", return_value=None)
    mocker.patch.object(
        main.SimpleRAGService, "search_memories", return_value="test answer"
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
    text_chunk = InputChunk(
        text_data="Hello, this is a test question.",
        metadata=main.stt_pb2.SessionMetadata(
            type=main.stt_pb2.QUESTION, session_id="test_text_session"
        ),
    )

    async def request_iterator():
        yield text_chunk

    context = MagicMock()

    responses = []
    async for response in servicer.Transcribe(request_iterator(), context):
        responses.append(response)

    # First response should be the direct transcript from the text input
    assert responses[0].transcript.text == "Hello, this is a test question."

    # Verify save_memory was called with text data
    mock_persistence.return_value.save_memory.assert_called_once()


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
