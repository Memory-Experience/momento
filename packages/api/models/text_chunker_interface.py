from abc import ABC, abstractmethod


class TextChunker(ABC):
    """
    Abstract base class for text chunking strategies.
    Text chunking splits a long text into smaller, semantically meaningful segments
    for more effective embedding and retrieval.
    """

    @abstractmethod
    def chunk_text(self, text: str) -> list[str]:
        """
        Split a text into chunks according to the strategy.

        Args:
            text: The text to split into chunks

        Returns:
            A list of text chunks
        """
        pass


class ChunkerConfig:
    """Configuration options for text chunkers."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str | None = None,
    ):
        """
        Initialize the chunker configuration.

        Args:
            chunk_size: Maximum size of each chunk (characters or tokens)
            chunk_overlap: Overlap between consecutive chunks
            separator: Optional specific separator to use for chunking
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
