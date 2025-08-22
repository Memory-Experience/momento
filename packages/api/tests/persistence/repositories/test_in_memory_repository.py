import asyncio

import pytest
from persistence.persistence_service import PersistenceService
from persistence.repositories.in_memory_repository import (
    InMemoryRepository,
)

from packages.api.domain.memory_request import MemoryRequest


@pytest.fixture
def in_memory_repository():
    return InMemoryRepository()


@pytest.fixture
def persistence_service_in_memory(in_memory_repository):
    return PersistenceService(in_memory_repository)


@pytest.fixture
def sample_memory():
    return MemoryRequest.create(audio_data=b"audio data", text=["hello", "world"])


def test_save_memory_in_memory(persistence_service_in_memory, sample_memory):
    uri = asyncio.run(persistence_service_in_memory.save_memory(sample_memory))
    assert uri.startswith("in_memory://")
    loaded = asyncio.run(persistence_service_in_memory.load_memory(uri))
    assert loaded == sample_memory


def test_load_memory_in_memory(persistence_service_in_memory, sample_memory):
    uri = asyncio.run(persistence_service_in_memory.save_memory(sample_memory))
    loaded = asyncio.run(persistence_service_in_memory.load_memory(uri))
    assert loaded == sample_memory
