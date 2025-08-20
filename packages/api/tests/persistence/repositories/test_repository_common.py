import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from domain.memory import Memory
from persistence.persistence_service import PersistenceService


@pytest.fixture
def sample_memory():
    return Memory.create(b"audio data", ["hello", "world"])


def test_save_memory_calls_repository(sample_memory):
    repo = MagicMock()
    repo.save = AsyncMock(return_value="in_memory://mock")
    service = PersistenceService(repo)
    asyncio.run(service.save_memory(sample_memory))
    repo.save.assert_awaited_once_with(sample_memory)
