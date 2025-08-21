import os
import sys
import types
from unittest.mock import AsyncMock, MagicMock

import pytest

# Ensure we can import the api package in flat layout
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Stub protos modules to avoid requiring generated code installation
protos_mod = types.ModuleType("protos")
protos_generated_mod = types.ModuleType("protos.generated")
protos_generated_py_mod = types.ModuleType("protos.generated.py")

stt_pb2 = types.ModuleType("stt_pb2")


class Transcript:
    def __init__(self, text: str):
        self.text = text


class StreamResponse:
    def __init__(self, transcript):
        self.transcript = transcript


stt_pb2.Transcript = Transcript
stt_pb2.StreamResponse = StreamResponse

stt_pb2_grpc = types.ModuleType("stt_pb2_grpc")


class TranscriptionServiceServicer:
    pass


def add_TranscriptionServiceServicer_to_server(servicer, server):
    return None


stt_pb2_grpc.TranscriptionServiceServicer = TranscriptionServiceServicer
stt_pb2_grpc.add_TranscriptionServiceServicer_to_server = (
    add_TranscriptionServiceServicer_to_server
)

sys.modules["protos"] = protos_mod
sys.modules["protos.generated"] = protos_generated_mod
sys.modules["protos.generated.py"] = protos_generated_py_mod
sys.modules["protos.generated.py.stt_pb2"] = stt_pb2
sys.modules["protos.generated.py.stt_pb2_grpc"] = stt_pb2_grpc

# Stub pydub to avoid audioop issues on Python 3.13 and unnecessary imports
pydub_mod = types.ModuleType("pydub")


class AudioSegment:
    pass


pydub_mod.AudioSegment = AudioSegment
sys.modules["pydub"] = pydub_mod

import main  # noqa: E402  # import after stubbing heavy deps


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
    mocker.patch("main.Memory", mock_memory)

    # Create servicer and simulate request
    servicer = main.TranscriptionServiceServicer()
    chunk = MagicMock()
    chunk.data = b"\x00" * (main.SAMPLE_RATE * 4)

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
