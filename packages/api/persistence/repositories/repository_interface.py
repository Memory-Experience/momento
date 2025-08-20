from abc import ABC, abstractmethod

from domain.memory import Memory


class Repository(ABC):
    """Repository interface for persistence operations."""

    @abstractmethod
    async def save(self, memory: Memory) -> str:
        """
        Save a memory to the repository.

        Args:
            memory: The memory to save

        Returns:
            The URI for the saved memory
        """
        pass

    @abstractmethod
    async def find_by_uri(self, uri: str) -> Memory | None:
        """
        Find a memory by its URI.

        Args:
            uri: The memory URI

        Returns:
            The memory if found, None otherwise
        """
        pass
