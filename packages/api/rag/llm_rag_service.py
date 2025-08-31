import logging

from domain.memory_context import MemoryContext
from domain.memory_request import MemoryRequest, MemoryType

from models.llm.llm_model_interface import LLMModel, MemoryResponse


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
        chunk_size_tokens: int = 32,  # Higher value to reduce streaming granularity
    ) -> MemoryRequest:
        """
        Search memories and use the LLM model to generate a response based on
        the retrieved context.

        Args:
            query: The question as a MemoryRequest
            memory_context: Context containing relevant memories
            chunk_size_tokens: Controls streaming granularity, higher values produce
                               smoother output with less character-by-character display

        Returns:
            MemoryRequest: The generated answer as a memory request
        """
        logging.info(f"Processing question: {query.text}")

        # Join the text list into a single string for the prompt
        prompt = " ".join(query.text)

        # Generate response using the LLM
        accumulated_text: list[str] = []  # Use a list to collect all text segments
        tokens_used = 0
        final_response: MemoryResponse | None = None

        # Stream responses and accumulate them
        async for chunk in self.llm_model.generate_with_memory(
            prompt=prompt,
            memory_context=memory_context,
            chunk_size_tokens=chunk_size_tokens,
        ):
            # Simply extend our list with the text from the response
            accumulated_text.extend(chunk.response.text)
            tokens_used = chunk.tokens_used

            # Store the final chunk for metadata
            if chunk.metadata.get("is_final", False):
                final_response = chunk
                logging.info(f"LLM response completed: {tokens_used} tokens used")

        # Always create a new MemoryRequest with the accumulated text
        # Regardless of whether we have a final response or not
        answer_request = MemoryRequest.create(
            # Use the ID and timestamp from final response if available
            id=final_response.response.id if final_response else None,
            timestamp=final_response.response.timestamp if final_response else None,
            # Always use our accumulated text which contains the complete response
            text=accumulated_text,
            memory_type=MemoryType.ANSWER,
        )

        # Log additional information
        if tokens_used > 0:
            logging.debug(
                f"Generated response using {tokens_used} "
                f"tokens from {self.llm_model.model_name}"
            )
            # Log a preview of the accumulated text for debugging
            preview = " ".join(accumulated_text)
            logging.debug(f"Response preview: {preview}")

        return answer_request
