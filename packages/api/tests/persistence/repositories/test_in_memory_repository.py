import asyncio
from uuid import uuid4

import pytest
from ....domain.memory_request import MemoryRequest
from ....persistence.repositories.in_memory_repository import (
    IN_MEMORY_URI_SCHEME,
    InMemoryRepository,
)


@pytest.fixture
def in_memory_repository():
    return InMemoryRepository()


@pytest.fixture
def sample_memory():
    return MemoryRequest.create(
        id=uuid4(), audio_data=b"audio data", text=["hello", "world"]
    )


def test_save_memory_in_memory(in_memory_repository, sample_memory):
    uri = asyncio.run(in_memory_repository.save(sample_memory))
    assert uri.startswith(IN_MEMORY_URI_SCHEME)
    loaded = asyncio.run(in_memory_repository.find_by_uri(uri))
    assert loaded == sample_memory
