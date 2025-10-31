import logging
from uuid import UUID, uuid4

from ...domain.memory_request import MemoryRequest

from .repository_interface import Repository

# Constants
IN_MEMORY_URI_SCHEME = "in_memory://"


class InMemoryRepository(Repository):
    """
    Repository implementation that stores memories in memory.

    Provides fast, temporary storage for memories without persistence.
    Useful for testing and development. All data is lost when the
    application stops.
    """

    def __init__(self):
        """Initialize the in-memory repository with an empty dictionary."""
        self.memories: dict[str, MemoryRequest] = {}
        logging.info("Initialized InMemoryRepository")

    async def save(self, memory: MemoryRequest) -> str:
        """
        Save a memory to in-memory storage.

        Args:
            memory (MemoryRequest): The memory to save

        Returns:
            str: URI for the saved memory in format "in_memory://{uuid}"
        """
        if memory.id is None:
            memory.id = uuid4()

        self.memories[memory.id] = memory
        logging.info(f"Memory saved in memory with key: {memory.id}")
        return f"{IN_MEMORY_URI_SCHEME}{memory.id}"

    async def find_by_uri(self, uri: str) -> MemoryRequest | None:
        """
        Find a memory by its URI.

        Args:
            uri (str): Memory URI in format "in_memory://{uuid}"

        Returns:
            MemoryRequest | None: The memory if found, None otherwise

        Raises:
            ValueError: If the URI doesn't have the in_memory:// scheme
        """
        if not uri.startswith(IN_MEMORY_URI_SCHEME):
            raise ValueError(f"Invalid in-memory URI: {uri}")
        id = uri[len(IN_MEMORY_URI_SCHEME) :]
        memory = self.memories.get(UUID(id))
        if memory:
            logging.info(f"Memory found with key: {id}")
        else:
            logging.info(f"Memory not found with key: {id}")
        return memory
