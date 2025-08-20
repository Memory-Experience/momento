import asyncio
import os
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from domain.memory import Memory

from packages.api.persistence.persistence_service import PersistenceService


class DummyRepository:
    def __init__(self):
        self.saved = []
        self._counter = 0

    async def save(self, memory):
        # assign a deterministic id when missing
        if getattr(memory, "id", None) is None:
            self._counter += 1
            memory.id = f"mem-{self._counter}"
        self.saved.append(memory)
        return True

    # helper methods for tests
    def load(self, memory_id):
        return next((m for m in self.saved if m.id == memory_id), None)

    def list(self):
        return list(self.saved)


@pytest.fixture
def dummy_repository():
    return DummyRepository()


@pytest.fixture
def persistence_service(dummy_repository):
    return PersistenceService(dummy_repository)


@pytest.fixture
def sample_memory():
    return Memory.create(b"audio data", ["hello", "world"])


def test_save_memory(persistence_service, sample_memory):
    result = asyncio.run(persistence_service.save_memory(sample_memory))
    assert result is True
    assert sample_memory in persistence_service.repository.saved


def test_load_memory(persistence_service, sample_memory):
    asyncio.run(persistence_service.save_memory(sample_memory))
    loaded = persistence_service.repository.load(sample_memory.id)
    assert loaded == sample_memory


def test_list_memories(persistence_service, sample_memory):
    asyncio.run(persistence_service.save_memory(sample_memory))
    memories = persistence_service.repository.list()
    assert sample_memory in memories
    assert isinstance(memories, list)


def test_save_memory_calls_repository(sample_memory):
    repo = MagicMock()
    repo.save = AsyncMock(return_value=True)
    service = PersistenceService(repo)
    asyncio.run(service.save_memory(sample_memory))
    repo.save.assert_awaited_once_with(sample_memory)
