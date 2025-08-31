from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from domain.memory_context import MemoryContext
from domain.memory_request import MemoryRequest


class MemoryResponse:
    """Represents a response from an LLM."""

    def __init__(
        self,
        response: MemoryRequest,
        model_name: str,
        tokens_used: int,
        metadata: dict = None,
    ):
        self.model_name = model_name
        self.tokens_used = tokens_used
        self.metadata = metadata or {}
        self.response = response


class LLMModel(ABC):
    """
    Abstract base class for large language models that generate text responses.
    """

    @abstractmethod
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
            prompt: The user prompt to generate a response for
            memory_context: Memory context object containing relevant memories
                (from vector search)
            chunk_size_tokens: Number of tokens to coalesce per streaming chunk.
                Higher values reduce object churn but increase latency between
                visible updates.
                Setting a very large value effectively creates non-streaming behavior.
        Returns:
            AsyncIterator[MemoryResponse]: Stream of response chunks
        """
        pass
