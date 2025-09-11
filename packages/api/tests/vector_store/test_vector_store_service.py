from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from api.domain.memory_context import MemoryContext
from api.domain.memory_request import MemoryRequest
from api.vector_store.vector_store_service import VectorStoreService


@pytest.fixture
def mock_repository():
    """Fixture for a mock repository."""
    repository = AsyncMock()

    # Set up the methods we'll use
    repository.index_memory = AsyncMock()
    repository.search_similar = AsyncMock()
    repository.delete_memory = AsyncMock()
    repository.list_memories = AsyncMock()

    return repository


@pytest.fixture
def vector_store_service(mock_repository):
    """Fixture for the vector store service with mocked repository."""
    return VectorStoreService(repository=mock_repository)


@pytest.fixture
def sample_memory():
    """Fixture for a sample memory object."""
    return MemoryRequest.create(
        id="test-memory-123",
        audio_data=b"dummy audio data",
        text="This is a test memory.",
    )


@pytest.mark.asyncio
async def test_index_memory_calls_repository(
    vector_store_service, mock_repository, sample_memory
):
    """Test that index_memory calls the repository's index_memory method."""
    # Call the service method
    await vector_store_service.index_memory(sample_memory)

    # Assert the repository method was called with the correct arguments
    mock_repository.index_memory.assert_called_once_with(sample_memory)


@pytest.mark.asyncio
async def test_search_calls_repository(
    vector_store_service, mock_repository, sample_memory
):
    """Test that search calls the repository's search_similar method."""
    # Set up mock return value
    mock_results = MemoryContext.create(query_memory=sample_memory)
    mock_repository.search_similar.return_value = mock_results

    # Call the service method
    query = sample_memory
    limit = 10
    results = await vector_store_service.search(query, limit)

    # Assert the repository method was called with the correct arguments
    mock_repository.search_similar.assert_called_once_with(
        query=query,
        limit=limit,
    )

    # Assert the results are passed through
    assert results == mock_results


@pytest.mark.asyncio
async def test_delete_memory_calls_repository(vector_store_service, mock_repository):
    """Test that delete_memory calls the repository's delete_memory method."""
    # Call the service method
    memory_id = "test-id-123"
    await vector_store_service.delete_memory(memory_id)

    # Assert the repository method was called with the correct arguments
    mock_repository.delete_memory.assert_called_once_with(memory_id)


@pytest.mark.asyncio
async def test_list_memories_calls_repository(vector_store_service, mock_repository):
    """Test that list_memories calls the repository's list_memories method."""
    # Set up mock return value
    mock_memories = [
        MemoryRequest.create(id="mem-1", text=["Test 1"]),
        MemoryRequest.create(id="mem-2", text=["Test 2"]),
    ]
    mock_repository.list_memories.return_value = mock_memories, None

    # Call the service method
    limit = 50
    offset = uuid4()
    memories = await vector_store_service.list_memories(limit, offset)

    # Assert the repository method was called with the correct arguments
    mock_repository.list_memories.assert_called_once_with(
        limit=limit,
        offset=offset,
    )

    # Assert the results are passed through
    assert memories == mock_memories
