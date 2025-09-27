from collections.abc import AsyncIterator
from api.models.llm.llm_model_interface import LLMModel, MemoryResponse
from api.domain.memory_context import MemoryContext
from api.domain.memory_request import MemoryRequest


class MemoryReciterModel(LLMModel):
    """
    Abstract base class for large language models that generate text responses.
    """

    async def generate_with_memory(
        self,
        prompt: str,
        memory_context: MemoryContext,
        chunk_size_tokens: int = 1,
    ) -> AsyncIterator[MemoryResponse]:
        """
        Generate a text response based on the prompt and conversation history.
        Returns an async iterator that yields chunks of the response.

        Args:
            prompt: What to return when no memory was found
            memory_context: Memory context object containing relevant memories
                (from vector search)
            chunk_size_tokens: Ignored for this model
        Returns:
            AsyncIterator[MemoryResponse]: Stream of exactly one response
        """
        top = memory_context.get_top_memories(limit=1)
        if top:
            best_memory, matched_text, score = top[0]
            yield MemoryResponse(
                response=best_memory, model_name="MemoryReciterModel", tokens_used=0
            )
        else:
            yield MemoryResponse(
                response=MemoryRequest.create(text=[prompt]),
                model_name="MemoryReciterModel",
                tokens_used=0,
            )
