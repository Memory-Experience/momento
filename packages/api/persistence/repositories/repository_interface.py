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
            The identifier for the saved memory
        """
        pass

    @abstractmethod
    async def find_by_id(self, id: str) -> Memory | None:
        """
        Find a memory by its ID.

        Args:
            id: The memory identifier

        Returns:
            The memory if found, None otherwise
        """
        pass
