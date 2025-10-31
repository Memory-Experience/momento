import asyncio

import pytest
from ....domain.memory_request import MemoryRequest
from ....persistence.repositories.file_repository import FILE_URI_SCHEME, FileRepository


@pytest.fixture
def file_repository(tmp_path):
    # Use a temp directory for file repo
    return FileRepository(storage_dir=str(tmp_path))


@pytest.fixture
def sample_memory():
    return MemoryRequest.create(audio_data=b"audio data", text=["hello", "world"])


def test_save_memory_file(file_repository, sample_memory):
    uri = asyncio.run(file_repository.save(sample_memory))
    assert uri.startswith(FILE_URI_SCHEME)
    loaded = asyncio.run(file_repository.find_by_uri(uri))
    assert loaded is not None
    assert loaded.id == sample_memory.id
