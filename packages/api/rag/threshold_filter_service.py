import logging

from api.domain.memory_context import MemoryContext


class ThresholdFilterService:
    """
    Service for filtering memory contexts based on relevance threshold scores
    to improve precision of results.
    """

    def __init__(self, relevance_threshold: float = 0.7):
        """
        Initialize the threshold filter service.

        Args:
            relevance_threshold: Minimum relevance score for filtering results (0.0-1.0)
        """
        self.relevance_threshold = relevance_threshold
        logging.info(
            f"Initialized ThresholdFilterService with threshold: {relevance_threshold}"
        )

    def filter_context(self, context: MemoryContext) -> MemoryContext:
        """
        Filter a memory context based on the relevance threshold.

        Args:
            context: The memory context to filter

        Returns:
            MemoryContext containing only memories above the threshold
        """
        if not context.scores:
            logging.info("No scores available in context, returning original context")
            return context

        original_count = len(context.get_memory_objects())

        # Apply relevance threshold filter
        filtered_context = self._apply_relevance_threshold(context)

        filtered_count = len(filtered_context.get_memory_objects())

        logging.info(
            f"Filtered context from {original_count} to {filtered_count} memories "
            f"above threshold {self.relevance_threshold}"
        )

        return filtered_context

    def _apply_relevance_threshold(self, context: MemoryContext) -> MemoryContext:
        """Filter memories based on relevance threshold."""
        if not context.scores:
            return context

        # Filter memories that meet the threshold
        filtered_memories = {
            memory_id: memory
            for memory_id, memory in context.memories.items()
            if context.scores.get(memory_id, 0.0) >= self.relevance_threshold
        }

        # Filter corresponding scores
        filtered_scores = {
            memory_id: score
            for memory_id, score in context.scores.items()
            if score >= self.relevance_threshold
        }

        # Filter corresponding matched texts
        filtered_matched_texts = {
            memory_id: text
            for memory_id, text in context.matched_texts.items()
            if memory_id in filtered_memories
        }

        # Create new MemoryContext with filtered data
        filtered_context = MemoryContext(
            memories=filtered_memories,
            scores=filtered_scores,
            matched_texts=filtered_matched_texts,
            query_memory=context.query_memory,
        )

        return filtered_context

    def set_threshold(self, new_threshold: float) -> None:
        """
        Update the relevance threshold.

        Args:
            new_threshold: New threshold value (0.0-1.0)
        """
        if not 0.0 <= new_threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        old_threshold = self.relevance_threshold
        self.relevance_threshold = new_threshold
        logging.info(
            f"Updated relevance threshold from {old_threshold} to {new_threshold}"
        )

    def get_threshold(self) -> float:
        """
        Get the current relevance threshold.

        Returns:
            Current threshold value
        """
        return self.relevance_threshold
