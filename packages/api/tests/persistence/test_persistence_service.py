from unittest.mock import AsyncMock, MagicMock

import pytest
from ...domain.memory_request import MemoryRequest
from ...persistence.persistence_service import PersistenceService


@pytest.fixture
def mock_repository():
    mock_repo = MagicMock()
    mock_repo.save = AsyncMock(return_value="mock://memory_id")
    mock_repo.find_by_uri = AsyncMock(
        return_value=MemoryRequest.create(
            audio_data=b"audio data", text=["hello", "world"]
        )
    )
    return mock_repo


@pytest.fixture
def persistence_service(mock_repository):
    return PersistenceService(mock_repository)


@pytest.fixture
def sample_memory():
    return MemoryRequest.create(audio_data=b"audio data", text=["hello", "world"])


@pytest.mark.asyncio
async def test_save_memory_calls_repository(
    persistence_service, mock_repository, sample_memory
):
    uri = await persistence_service.save_memory(sample_memory)

    # Assert repository save method was called with the memory
    mock_repository.save.assert_called_once_with(sample_memory)
    # Assert the URI returned by the service is the same returned by the repository
    assert uri == "mock://memory_id"


@pytest.mark.asyncio
async def test_load_memory_calls_repository(persistence_service, mock_repository):
    uri = "mock://memory_id"
    await persistence_service.load_memory(uri)

    # Assert repository find_by_uri method was called with the URI
    mock_repository.find_by_uri.assert_called_once_with(uri)
    # We don't need to check the returned memory, as that's just what the mock returns
