import logging
from uuid import UUID, uuid4

from domain.memory_request import MemoryRequest

from .repository_interface import Repository

# Constants
IN_MEMORY_URI_SCHEME = "in_memory://"


class InMemoryRepository(Repository):
    """Repository implementation that stores data in memory."""

    def __init__(self):
        self.memories: dict[str, MemoryRequest] = {}
        logging.info("Initialized InMemoryRepository")

    async def save(self, memory: MemoryRequest) -> str:
        """Save a memory to in-memory storage."""
        if memory.id is None:
            memory.id = uuid4()

        self.memories[memory.id] = memory
        logging.info(f"Memory saved in memory with key: {memory.id}")
        return f"{IN_MEMORY_URI_SCHEME}{memory.id}"

    async def find_by_uri(self, uri: str) -> MemoryRequest | None:
        """Find a memory by its URI."""
        if not uri.startswith(IN_MEMORY_URI_SCHEME):
            raise ValueError(f"Invalid in-memory URI: {uri}")
        id = uri[len(IN_MEMORY_URI_SCHEME) :]
        memory = self.memories.get(UUID(id))
        if memory:
            logging.info(f"Memory found with key: {id}")
        else:
            logging.info(f"Memory not found with key: {id}")
        return memory
