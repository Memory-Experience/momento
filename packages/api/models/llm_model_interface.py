from abc import ABC, abstractmethod


class LLMResponse:
    """Represents a response from an LLM."""

    def __init__(
        self,
        text: str,
        model_name: str,
        tokens_used: int = 0,
        metadata: dict | None = None,
    ):
        self.text = text
        self.model_name = model_name
        self.tokens_used = tokens_used
        self.metadata = metadata or {}


class LLMModel(ABC):
    """
    Abstract base class for large language models that generate text responses.
    """

    @abstractmethod
    async def generate_with_memory(
        self,
        prompt: str,
        memory_context: list[dict[str, str]],
    ) -> LLMResponse:
        """
        Generate a text response based on the prompt and conversation history.

        Args:
            prompt: The user prompt to generate a response for
            memory_context: List of relevant memory contexts (from vector search)
        Returns:
            LLMResponse object containing the generated text and metadata
        """
        pass
