import logging
from datetime import datetime

from domain.memory_request import MemoryRequest
from vector_store.repositories.vector_store_repository_interface import (
    VectorStoreRepository,
)


class SimpleRAGService:
    """
    A simple RAG service that uses a vector store repository
    to return relevant memories in a fixed response.
    """

    def __init__(self, vector_store_repo: VectorStoreRepository):
        self.vector_store_repo = vector_store_repo

    async def search_memories(self, query: MemoryRequest) -> str:
        """Search memories and return a relevant answer."""
        logging.info(f"Searching memories for: {query.text}")

        # Use the vector store repository to search for relevant memories
        search_results = await self.vector_store_repo.search_similar(query, limit=5)

        if not search_results:
            return (
                "I don't have any memories that match your question. "
                + "Try asking about something you've recorded before."
            )

        # Generate answer based on retrieved memories
        answer_parts = ["Based on your memories, here's what I found:"]

        for i, memory in enumerate(search_results.memories.values(), 1):
            # Format the timestamp nicely
            timestamp = (
                datetime.fromisoformat(memory.timestamp)
                if isinstance(memory.timestamp, str)
                else memory.timestamp
            )
            formatted_date = timestamp.strftime("%B %d, %Y at %I:%M %p")

            answer_parts.append(f"{i}. From {formatted_date}: {memory.text}")

        return "\n\n".join(answer_parts)
