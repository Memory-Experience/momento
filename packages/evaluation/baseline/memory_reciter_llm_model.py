from collections.abc import AsyncIterator
from api.models.llm.llm_model_interface import LLMModel, MemoryResponse
from api.domain.memory_context import MemoryContext
from api.domain.memory_request import MemoryRequest


class MemoryReciterModel(LLMModel):
    """
    Baseline LLM model that recites the top retrieved memory.

    Instead of generating new text, this model simply returns the
    best matching memory from the retrieval context. This provides
    a simple baseline for evaluating retrieval quality without
    the complexity of text generation.
    """

    async def generate_with_memory(
        self,
        prompt: str,
        memory_context: MemoryContext,
        chunk_size_tokens: int = 1,
    ) -> AsyncIterator[MemoryResponse]:
        """
        Generate response by returning best retrieved memory.

        Returns the top memory from the context, or the prompt if
        no memories were found. This provides a simple baseline that
        doesn't use an LLM for generation.

        Args:
            prompt (str): Text to return when no memory was found
            memory_context (MemoryContext): Memory context object
                containing relevant memories from vector search
            chunk_size_tokens (int): Ignored for this model

        Yields:
            MemoryResponse: Stream of exactly one response containing
                either the best memory or the prompt
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
