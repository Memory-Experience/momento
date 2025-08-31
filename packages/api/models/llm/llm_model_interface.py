from abc import ABC, abstractmethod

from domain.memory_context import MemoryContext
from domain.memory_request import MemoryRequest


class MemoryResponse:
    """Represents a response from an LLM."""

    def __init__(
        self,
        response: MemoryRequest,
        context: MemoryContext,
        model_name: str,
        tokens_used: int,
        metadata: dict = None,
    ):
        self.model_name = model_name
        self.tokens_used = tokens_used
        self.metadata = metadata or {}
        self.response = response
        self.context = context


class LLMModel(ABC):
    """
    Abstract base class for large language models that generate text responses.
    """

    @abstractmethod
    async def generate_with_memory(
        self,
        prompt: str,
        memory_context: MemoryContext,
    ) -> MemoryResponse:
        """
        Generate a text response based on the prompt and conversation history.

        Args:
            prompt: The user prompt to generate a response for
            memory_context: Memory context object containing relevant memories
                (from vector search)
        Returns:
            LLMResponse object containing the generated text and metadata
        """
        pass
