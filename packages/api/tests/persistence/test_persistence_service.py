import asyncio

import pytest
from domain.memory import Memory

from packages.api.persistence.persistence_service import PersistenceService


class DummyRepository:
    def __init__(self):
        self.saved = []

    async def save(self, memory):
        if getattr(memory, "id", None) is None:
            memory.id = "dummy_id"
        self.saved.append(memory)
        return "dummy://" + memory.id

    async def find_by_uri(self, uri):
        # expects uri like dummy://<id>
        id = uri.split("dummy://")[-1]
        for m in self.saved:
            if getattr(m, "id", None) == id:
                return m
        return None


@pytest.fixture
def dummy_repository():
    return DummyRepository()


@pytest.fixture
def persistence_service(dummy_repository):
    return PersistenceService(dummy_repository)


@pytest.fixture
def sample_memory():
    return Memory.create(b"audio data", ["hello", "world"])


def test_save_and_load_memory(persistence_service, sample_memory):
    uri = asyncio.run(persistence_service.save_memory(sample_memory))
    assert uri.startswith("dummy://")
    loaded = asyncio.run(persistence_service.load_memory(uri))
    assert loaded == sample_memory
