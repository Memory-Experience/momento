import logging
from datetime import datetime


class SimpleRAGService:
    """A simple RAG service that uses naive keyword search."""

    def __init__(self):
        self.memories = []

    def add_memory(self, transcription: str, audio_filename: str | None = None):
        """Add a new memory to the RAG corpus."""
        memory = {
            "id": len(self.memories),
            "text": transcription,
            "timestamp": datetime.now().isoformat(),
            "audio_file": audio_filename,
        }
        self.memories.append(memory)
        logging.info(f"Added memory: {memory['id']}")

    def search_memories(self, query: str) -> str:
        """Search memories and return a relevant answer."""
        # Simple keyword-based search (in a real implementation, you'd use embeddings)
        query_words = set(query.lower().split())
        relevant_memories = []

        for memory in self.memories:
            memory_words = set(memory["text"].lower().split())
            # Simple overlap scoring
            overlap = len(query_words.intersection(memory_words))
            if overlap > 0:
                relevant_memories.append((memory, overlap))

        # Sort by relevance
        relevant_memories.sort(key=lambda x: x[1], reverse=True)

        if not relevant_memories:
            return (
                "I don't have any memories that match your question. "
                + "Try asking about something you've recorded before."
            )

        # Generate answer based on top relevant memories
        top_memories = relevant_memories[:3]  # Top 3 most relevant
        answer_parts = ["Based on your memories, here's what I found:"]

        for i, (memory, _) in enumerate(top_memories, 1):
            # Format the timestamp nicely
            timestamp = datetime.fromisoformat(memory["timestamp"])
            formatted_date = timestamp.strftime("%B %d, %Y at %I:%M %p")

            answer_parts.append(f"{i}. From {formatted_date}: {memory['text']}")

        return "\n\n".join(answer_parts)
