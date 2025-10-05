from dataclasses import dataclass, field

from api.domain.memory_request import MemoryRequest


@dataclass
class MemoryContext:
    """
    Container for multiple memories with their similarity scores.

    Holds retrieved memories along with their relevance scores and matched text
    snippets, providing context for LLM-based question answering. Supports
    filtering, sorting, and retrieval of top-k memories by relevance.
    """

    # Dictionary of memories by ID
    memories: dict[str, MemoryRequest] = field(default_factory=dict)
    # Dictionary of scores by memory ID
    scores: dict[str, float] = field(default_factory=dict)
    # Dictionary of matched text snippets by memory ID (if applicable)
    matched_texts: dict[str, str] = field(default_factory=dict)

    # The query memory object
    query_memory: MemoryRequest | None = (
        None  # The full query memory object if available
    )

    def add_memory(
        self, memory: MemoryRequest, score: float, matched_text: str
    ) -> None:
        """
        Add a memory with its similarity score to this context.

        Args:
            memory: The memory object to add
            score: Relevance/similarity score (typically 0.0 to 1.0)
            matched_text: The text snippet that matched the query

        Raises:
            ValueError: If the memory has no ID
        """
        if not memory.id:
            raise ValueError("Memory must have an ID to be added to MemoryContext")

        self.memories[memory.id] = memory
        self.scores[memory.id] = score
        self.matched_texts[memory.id] = matched_text

    def get_memory_objects(self) -> list[MemoryRequest]:
        """
        Get just the memory objects without scores.

        Returns:
            List of MemoryRequest objects
        """
        return list(self.memories.values())

    def get_memory_by_id(self, memory_id: str) -> MemoryRequest | None:
        """
        Get a specific memory by its ID.

        Args:
            memory_id: The UUID of the memory to retrieve

        Returns:
            The MemoryRequest if found, None otherwise
        """
        return self.memories.get(memory_id)

    def get_memories_with_scores(self) -> list[tuple]:
        """
        Get all memories with their scores as tuples.

        Returns:
            List of (memory, matched_text, score) tuples
        """
        return [
            (self.memories[memory_id], self.matched_texts[memory_id], score)
            for memory_id, score in self.scores.items()
        ]

    def get_top_memories(self, limit: int) -> list[tuple]:
        """Get the top N memories by score."""
        # Sort memory IDs by their scores
        sorted_ids = sorted(
            self.scores.keys(), key=lambda id: self.scores[id], reverse=True
        )[:limit]
        # Return tuples of (memory, score) for the top IDs
        return [
            (
                self.memories[memory_id],
                self.matched_texts[memory_id],
                self.scores[memory_id],
            )
            for memory_id in sorted_ids
        ]

    def is_empty(self) -> bool:
        """Check if the context has any memories."""
        return len(self.memories) == 0

    @classmethod
    def create(cls, query_memory: MemoryRequest = None) -> "MemoryContext":
        """
        Factory method to create a new MemoryContext.

        Args:
            query_memory: Optional full query memory object

        Returns:
            A new MemoryContext instance
        """
        return cls(memories={}, scores={}, matched_texts={}, query_memory=query_memory)
