import logging

from domain.memory import Memory

from .repositories.repository_interface import Repository


class PersistenceService:
    """Service for handling persistence operations."""

    def __init__(self, repository: Repository):
        """
        Initialize the persistence service with a repository.

        Args:
            repository: The repository implementation to use
        """
        self.repository = repository
        logging.info(
            f"Initialized PersistenceService with {repository.__class__.__name__}"
        )

    async def save_memory(self, memory: Memory) -> str:
        """
        Save a memory.

        Args:
            memory: The memory to save

        Returns:
            The URI for the saved memory
        """
        return await self.repository.save(memory)

    async def load_memory(self, uri: str) -> Memory | None:
        """
        Load a memory by its URI.

        Args:
            uri: The memory URI

        Returns:
            The memory if found, None otherwise
        """
        return await self.repository.find_by_uri(uri)
