from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Union
from uuid import UUID

from ...domain.memory_context import MemoryContext
from ...domain.memory_request import MemoryRequest

from ...models.embedding.embedding_model_interface import EmbeddingModel
from ...models.text_chunker_interface import TextChunker


class FilterOperator(Enum):
    """Operators that can be used in filter conditions."""

    EQUALS = "eq"
    NOT_EQUALS = "neq"
    GREATER_THAN = "gt"
    GREATER_THAN_OR_EQUAL = "gte"
    LESS_THAN = "lt"
    LESS_THAN_OR_EQUAL = "lte"
    EXISTS = "exists"
    NOT_EXISTS = "not_exists"
    CONTAINS = "contains"


@dataclass
class FilterCondition:
    """Represents a single filter condition."""

    field: str  # Field path (supports dot notation for nested fields)
    operator: FilterOperator
    value: Any = None  # Optional for EXISTS/NOT_EXISTS operators


@dataclass
class FilterGroup:
    """Group of filter conditions with logical operator."""

    conditions: list[Union["FilterGroup", FilterCondition]]
    operator: str = "AND"  # "AND" or "OR"


# Type alias for filter argument
FilterArg = FilterCondition | FilterGroup | None


class VectorStoreRepository(ABC):
    """Repository interface for vector store operations
    focused on Memory domain objects."""

    def __init__(
        self, embedding_model: EmbeddingModel = None, text_chunker: TextChunker = None
    ):
        """
        Initialize the vector store repository.

        Args:
            embedding_model: Embedding model for vectorizing text.
                            If provided, the repository will handle text embedding.
            text_chunker: Text chunking strategy.
                         If provided, the repository will handle text chunking.
        """
        self.embedding_model = embedding_model
        self.text_chunker = text_chunker

    @abstractmethod
    async def index_memory(self, memory: MemoryRequest) -> None:
        """
        Index a memory in the vector store.
        The repository is responsible for:
        1. Converting text to embeddings (if it has an embedding model)
        2. Chunking text if appropriate (if it has a text chunker)
        3. Storing the memory and its embedding

        Args:
            memory: The Memory domain object to index
        """
        pass

    @abstractmethod
    async def index_memories_batch(
        self, memories: list[MemoryRequest], qdrant_batch_size: int = 512
    ) -> None:
        """
        Index multiple memories in batch for better performance.

        Args:
            memories: List of memories to index
        """
        pass

    @abstractmethod
    async def get_memory(self, memory_id: UUID) -> MemoryRequest | None:
        """
        Get a memory by its ID.

        Args:
            memory_id: The ID of the memory to retrieve

        Returns:
            The Memory if found, None otherwise
        """
        pass

    @abstractmethod
    async def search_similar(
        self, query: MemoryRequest, limit: int = 5, filters: FilterArg = None
    ) -> MemoryContext:
        """
        Search for memories with content similar to the query.

        Args:
            query: MemoryRequest object containing the query text
            limit: Maximum number of results to return
            filters: Optional filters to apply using the abstract filter system

        Returns:
            List of memory search results sorted by relevance
        """
        pass

    @abstractmethod
    async def delete_memory(self, memory_id: UUID) -> None:
        """
        Delete all vector embeddings for a memory.
        The repository is responsible for:
        1. Deleting the memory with the given ID
        2. Deleting any associated chunks or related documents

        Args:
            memory_id: The ID of the memory to delete
        """
        pass

    @abstractmethod
    async def list_memories(
        self, limit: int = 100, offset: int = 0, filters: FilterArg = None
    ) -> tuple[list[MemoryRequest], UUID | None]:
        """
        List memories in the vector store.

        Args:
            limit: Maximum number of memories to return
            offset: Number of memories to skip
            filters: Optional filters to apply using the abstract filter system

        Returns:
            List of Memory objects
        """
        pass
