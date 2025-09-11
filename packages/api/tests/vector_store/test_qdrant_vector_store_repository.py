import hashlib
from datetime import datetime

import pytest
import pytest_asyncio
from domain.memory_context import MemoryContext
from domain.memory_request import MemoryRequest, MemoryType
from vector_store.repositories.qdrant_vector_store_repository import (
    InMemoryQdrantVectorStoreRepository,
)
from vector_store.repositories.vector_store_repository_interface import (
    FilterCondition,
    FilterOperator,
    VectorStoreRepository,
)

from models.embedding.embedding_model_interface import EmbeddingModel
from models.text_chunker_interface import TextChunker


class MockEmbeddingModel(EmbeddingModel):
    """
    Mock implementation of the EmbeddingModel interface for testing.
    Uses a simple strategy of creating vector embeddings based on the text content.
    """

    def get_vector_size(self) -> int:
        return 5  # Small vectors for testing

    async def embed_text(self, text: str) -> list[float]:
        """Creates a deterministic embedding based on the text content."""
        if not text:
            return [0.0] * 5

        # Create a simple embedding based on text statistics
        chars = len(text)
        words = len(text.split())
        unique = len(set(text.lower()))

        hash_value = int(hashlib.md5(text.encode()).hexdigest(), 16)

        # Create a simple 5-dimensional vector
        vec = [
            float(chars % 10) / 10,  # Length mod 10, normalized
            float(words % 10) / 10,  # Word count mod 10, normalized
            float(unique % 10) / 10,  # Unique chars mod 10, normalized
            float(sum(ord(c) for c in text[:10]) % 10)
            / 10,  # Sum of first 10 char codes, normalized
            float(hash_value % 1000) / 1000,  # Hash of text, normalized
        ]

        return vec


class SimpleTextChunker(TextChunker):
    """
    Simple implementation of the TextChunker interface for testing.
    Splits text by sentences with a maximum chunk size.
    """

    def __init__(self, max_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size

    def chunk_text(self, text: str) -> list[str]:
        """Split text into chunks by sentences, respecting max chunk size."""
        # Simple sentence splitting
        sentences = [
            s.strip()
            for s in text.replace("!", ".").replace("?", ".").split(".")
            if s.strip()
        ]

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) > self.max_chunk_size:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks


@pytest.fixture
def embedding_model():
    return MockEmbeddingModel()


@pytest.fixture
def text_chunker():
    return SimpleTextChunker(max_chunk_size=50)


@pytest.fixture
def test_memories():
    return [
        MemoryRequest.create(
            text=[
                f"This is test memory {i}. "
                f"It contains some information about topic {i % 3}."
            ],
            memory_type=MemoryType.MEMORY if i % 2 == 0 else MemoryType.QUESTION,
            timestamp=datetime.now(),
        )
        for i in range(5)
    ]


@pytest.fixture
def long_text_memory():
    return MemoryRequest.create(
        text=[
            "This is a long text that should be split into multiple chunks. "
            "It contains several sentences about different topics. "
            "The first topic is about machine learning and AI. "
            "The second topic relates to data storage and retrieval. "
            "The third section discusses vector embeddings for text. "
            "Finally, we talk about testing software components."
        ],
        memory_type=MemoryType.MEMORY,
        timestamp=datetime.now(),
    )


@pytest_asyncio.fixture
async def repository(embedding_model, text_chunker, test_memories, long_text_memory):
    repo = InMemoryQdrantVectorStoreRepository(
        embedding_model=embedding_model, text_chunker=text_chunker
    )

    # Index all test memories
    for memory in test_memories:
        await repo.index_memory(memory)

    # Index the long text memory
    await repo.index_memory(long_text_memory)

    return repo


@pytest.mark.asyncio
async def test_index_and_get_memory(repository):
    """Test indexing a memory and retrieving it by ID."""
    # Create a new memory
    memory = MemoryRequest.create(
        text=["This is a specific test memory for retrieval testing."],
        memory_type=MemoryType.MEMORY,
        timestamp=datetime.now(),
    )

    # Index it
    await repository.index_memory(memory)

    # Retrieve it
    retrieved = await repository.get_memory(memory.id)

    # Check it's correctly retrieved
    assert retrieved is not None
    assert retrieved.id == memory.id
    assert retrieved.text == ["This is a specific test memory for retrieval testing."]
    assert retrieved.memory_type == MemoryType.MEMORY


@pytest.mark.asyncio
async def test_search_similar(repository):
    """Test searching for similar memories."""
    # Create a query memory
    query = MemoryRequest.create(
        text=["Looking for information about topic 1"],
        memory_type=MemoryType.QUESTION,
    )

    # Search for similar memories
    results = await repository.search_similar(query, limit=3)

    # Verify results
    assert isinstance(results, MemoryContext)
    assert not results.is_empty()
    assert results.query_memory == query

    # Should find memories
    memories = results.get_memory_objects()
    assert len(memories) >= 1


@pytest.mark.asyncio
async def test_search_similar_with_chunks(
    repository: VectorStoreRepository, long_text_memory
):
    """Test searching that retrieves chunk results."""
    # Create a query specifically targeting text in the long memory
    query = MemoryRequest.create(
        text=["Tell me about vector embeddings for text"],
        memory_type=MemoryType.QUESTION,
    )

    # Search with chunk results enabled (default)
    results = await repository.search_similar(query, limit=9)

    # Verify we got results
    assert not results.is_empty()

    # The long_text memory should be in the results
    long_text_id = str(long_text_memory.id)

    found_memory = next(
        (
            memory
            for memory in results.get_memory_objects()
            if str(memory.id) == long_text_id
        ),
        None,
    )

    assert found_memory is not None, (
        "Long text memory should be found when searching for 'vector embeddings'"
    )

    # Check that we have a matched text that's a chunk, not the full text
    matched_text = results.matched_texts.get(long_text_id, "")
    full_text = " ".join(long_text_memory.text)

    assert len(matched_text) < len(full_text), (
        "Matched text should be a chunk, not the full text"
    )


@pytest.mark.asyncio
async def test_search_without_chunks(repository):
    """Test searching with chunks disabled."""
    # Create a query
    query = MemoryRequest.create(
        text=["Tell me about machine learning and AI"],
        memory_type=MemoryType.QUESTION,
    )

    # Search with chunk results disabled
    results = await repository.search_similar(query, limit=3, search_chunks=False)

    # We should still get results, but matched text should be full texts only
    for memory_id, matched_text in results.matched_texts.items():
        full_text = " ".join(results.memories[memory_id].text)
        assert matched_text.strip() == full_text.strip(), (
            "When chunks are disabled, matched text should be the full text"
        )


@pytest.mark.asyncio
async def test_delete_memory(repository, test_memories):
    """Test deleting a memory."""
    # First, verify the memory exists
    memory_id = str(test_memories[0].id)
    memory = await repository.get_memory(memory_id)
    assert memory is not None

    # Delete it
    await repository.delete_memory(memory_id)

    # Verify it's gone
    memory = await repository.get_memory(memory_id)
    assert memory is None

    # Search shouldn't find it either
    query = MemoryRequest.create(
        text=["Looking for test memory 0"], memory_type=MemoryType.QUESTION
    )

    results = await repository.search_similar(query, limit=10)

    found_memory = next(
        (
            memory
            for memory in results.get_memory_objects()
            if str(memory.id) == memory_id
        ),
        None,
    )

    assert found_memory is None, "Deleted memory should not appear in search results"


@pytest.mark.asyncio
async def test_list_memories(repository, test_memories):
    """Test listing memories with filters and pagination."""
    # List all memories
    memories, offset_token = await repository.list_memories()

    # Should include both our standard test memories and the long text memory
    assert len(memories) >= len(test_memories)

    # Test with limit
    limited, offset_uuid = await repository.list_memories(limit=2)
    assert len(limited) == 2

    # Test with offset
    offset, _ = await repository.list_memories(offset=offset_uuid, limit=2)
    assert len(offset) == 2
    assert limited[0].id != offset[0].id, "Should be different memories"

    # Test with filter
    memory_filter = FilterCondition(
        field="memory_type",
        operator=FilterOperator.EQUALS,
        value=MemoryType.MEMORY.value,
    )

    filtered, _ = await repository.list_memories(filters=memory_filter)

    # All returned memories should be of type MEMORY
    for memory in filtered:
        assert memory.memory_type == MemoryType.MEMORY
