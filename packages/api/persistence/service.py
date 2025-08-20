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
            The identifier for the saved memory
        """
        return await self.repository.save(memory)
