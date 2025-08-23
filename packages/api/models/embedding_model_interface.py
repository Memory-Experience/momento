from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models that
    convert text to vector representations.
    """

    @abstractmethod
    async def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text into a vector representation.

        Args:
            text: The text to embed

        Returns:
            A list of floats representing the text embedding
        """
        pass
