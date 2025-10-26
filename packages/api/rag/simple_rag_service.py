import logging
from datetime import datetime

from ..domain.memory_context import MemoryContext
from ..domain.memory_request import MemoryRequest, MemoryType


class SimpleRAGService:
    """
    A simple RAG service that uses a vector store repository
    to return relevant memories in a fixed response.
    """

    async def answer_question(
        self,
        query: MemoryRequest,
        memory_context: MemoryContext,
    ) -> MemoryRequest:
        """
        Search memories and return both an answer wrapped in a MemoryRequest
        and the memory context used to generate the answer.

        Args:
            query: The question as a MemoryRequest

        Returns:
            Tuple containing:
            - MemoryRequest: The answer formatted as a memory request
            - MemoryContext: The context memories used to generate the answer
        """
        logging.info(f"Searching memories for: {query.text}")

        # Generate answer based on retrieved memories
        if not memory_context.memories:
            answer_text = (
                "I don't have any memories that match your question. "
                + "Try asking about something you've recorded before."
            )
        else:
            answer_parts = ["Based on your memories, here's what I found:"]

            for i, memory in enumerate(memory_context.memories.values(), 1):
                # Format the timestamp nicely
                timestamp = (
                    datetime.fromisoformat(memory.timestamp)
                    if isinstance(memory.timestamp, str)
                    else memory.timestamp
                )
                formatted_date = timestamp.strftime("%B %d, %Y at %I:%M %p")

                answer_parts.append(f"{i}. From {formatted_date}: {memory.text}")

            answer_text = "\n\n".join(answer_parts)

        # Create a new MemoryRequest for the answer
        answer_request = MemoryRequest.create(
            text=[answer_text],
            memory_type=MemoryType.ANSWER,
        )

        return answer_request
