import asyncio

import pytest
from domain.memory import Memory
from persistence.persistence_service import PersistenceService
from persistence.repositories.file_repository import FileRepository


@pytest.fixture
def file_repository(tmp_path):
    # Use a temp directory for file repo
    return FileRepository(storage_dir=str(tmp_path))


@pytest.fixture
def persistence_service_file(file_repository):
    return PersistenceService(file_repository)


@pytest.fixture
def sample_memory():
    return Memory.create(b"audio data", ["hello", "world"])


def test_save_memory_file(persistence_service_file, sample_memory):
    uri = asyncio.run(persistence_service_file.save_memory(sample_memory))
    assert uri.startswith("file://")
    loaded = asyncio.run(persistence_service_file.load_memory(uri))
    assert loaded is not None
    assert loaded.id == sample_memory.id
