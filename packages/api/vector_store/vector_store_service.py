import logging
from uuid import UUID

from api.domain.memory_context import MemoryContext
from api.domain.memory_request import MemoryRequest

from .repositories.vector_store_repository_interface import (
    VectorStoreRepository,
)


class VectorStoreService:
    """Service for managing vector embeddings and search operations
    using Memory domain objects."""

    def __init__(
        self,
        repository: VectorStoreRepository,
    ):
        """
        Initialize the vector store service.

        Args:
            repository: The vector store repository implementation to use.
                       The repository handles all embedding and chunking concerns.
        """
        self.repository = repository
        logging.info(
            f"Initialized VectorStoreService with {repository.__class__.__name__}"
        )

    async def index_memory(self, memory: MemoryRequest) -> None:
        """
        Index a memory in the vector store.
        The repository is responsible for converting text to embeddings
        and handling any chunking.

        Args:
            memory: The memory to index
        """
        await self.repository.index_memory(memory)
        logging.info(f"Indexed memory {memory.id}")

    async def search(
        self,
        query: MemoryRequest,
        limit: int = 5,
    ) -> MemoryContext:
        """
        Search for relevant memories using a text query.

        Args:
            query: The search query
            limit: Maximum number of results to return
            filters: Optional filters to apply

        Returns:
            MemoryContext containing the search results
        """
        # The repository is responsible for converting the query to embeddings
        context = await self.repository.search_similar(query=query, limit=limit)

        logging.info(
            f"Search for '{query.text}' "
            f"returned {len(context.get_memory_objects())} results"
        )
        return context

    async def delete_memory(self, memory_id: UUID) -> None:
        """
        Delete all vector records for a specific memory.
        The repository is responsible for deciding how to handle related chunks.

        Args:
            memory_id: The ID of the memory to delete
        """
        await self.repository.delete_memory(memory_id)
        logging.info(f"Deleted vector records for memory {memory_id}")

    async def list_memories(
        self,
        limit: int = 100,
        offset: UUID | None = None,
    ) -> list[MemoryRequest]:
        """
        List memories stored in the vector database.

        Args:
            limit: Maximum number of memories to return
            offset: Lowest memory ID to start from (for pagination)
            filters: Optional filters to apply

        Returns:
            List of Memory objects
        """
        memories, _ = await self.repository.list_memories(
            limit=limit,
            offset=offset,
        )
        logging.info(f"Listed {len(memories)} memories")
        return memories
