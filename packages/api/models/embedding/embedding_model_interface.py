from abc import ABC, abstractmethod


class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models that
    convert text to vector representations.
    """

    @abstractmethod
    def get_vector_size(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            The size of the embedding vectors produced by this model
        """
        pass

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
