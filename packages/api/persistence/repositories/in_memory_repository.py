import logging

from domain.memory import Memory

from .repository_interface import Repository


class InMemoryRepository(Repository):
    """Repository implementation that stores data in memory."""

    def __init__(self):
        self.memories: dict[str, Memory] = {}
        logging.info("Initialized InMemoryRepository")

    async def save(self, memory: Memory) -> str:
        """Save a memory to in-memory storage."""
        if memory.id is None:
            memory.id = memory.timestamp.strftime("%Y%m%d_%H%M%S")

        self.memories[memory.id] = memory
        logging.info(f"Memory saved in memory with key: {memory.id}")
        return f"in_memory://{memory.id}"

    async def find_by_uri(self, uri: str) -> Memory | None:
        """Find a memory by its URI."""
        if not uri.startswith("in_memory://"):
            raise ValueError(f"Invalid in-memory URI: {uri}")
        id = uri[len("in_memory://") :]
        memory = self.memories.get(id)
        if memory:
            logging.info(f"Memory found with key: {id}")
        else:
            logging.info(f"Memory not found with key: {id}")
        return memory
