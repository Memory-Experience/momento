import logging
from collections.abc import AsyncIterator

from api.domain.memory_context import MemoryContext
from api.domain.memory_request import MemoryRequest

from api.models.llm.llm_model_interface import LLMModel, MemoryResponse


class LLMRAGService:
    """
    A RAG service that uses an LLM model to generate responses based on
    memories retrieved from a vector store.
    """

    def __init__(self, llm_model: LLMModel):
        """
        Initialize the RAG service with an LLM model.

        Args:
            llm_model: The LLM model to use for generating responses
        """
        self.llm_model = llm_model

    async def answer_question(
        self,
        query: MemoryRequest,
        memory_context: MemoryContext,
        chunk_size_tokens: int = 8,  # Medium value for smoother streaming
    ) -> AsyncIterator[MemoryResponse]:
        """
        Search memories and use the LLM model to generate a streaming response based on
        the retrieved context.

        Args:
            query: The question as a MemoryRequest
            memory_context: Context containing relevant memories
            chunk_size_tokens: Controls streaming granularity, higher values produce
                               smoother output with less character-by-character display

        Yields:
            MemoryResponse: Streaming chunks of the generated answer
        """
        logging.info(f"Processing question: {query.text}")

        # Join the text list into a single string for the prompt
        prompt = " ".join(query.text)

        # Log the number of memories in the context
        if memory_context and not memory_context.is_empty():
            logging.info(f"Using {len(memory_context.memories)} memories for context")

        # Stream responses directly without accumulating
        async for chunk in self.llm_model.generate_with_memory(
            prompt=prompt,
            memory_context=memory_context,
            chunk_size_tokens=chunk_size_tokens,
        ):
            # Log tokens used for monitoring
            if chunk.metadata.get("is_final", False):
                logging.info(f"LLM response completed: {chunk.tokens_used} tokens used")

            # Pass through each chunk directly to the caller
            yield chunk
