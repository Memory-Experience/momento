from unittest.mock import AsyncMock, MagicMock

import main
import pytest
from protos.generated.py.stt_pb2 import AudioChunk


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
    chunk = AudioChunk(
        data=b"\x00" * (main.SAMPLE_RATE * 4),
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
