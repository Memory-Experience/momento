from api.models.text_chunker_interface import ChunkerConfig, TextChunker


class CharacterTextChunker(TextChunker):
    """
    A character-based text chunking strategy that splits text based on character count
    while trying to preserve sentence boundaries.
    """

    def __init__(self, config: ChunkerConfig = None):
        """
        Initialize the character text chunker.

        Args:
            config: Configuration for the chunker
        """
        self.config = config or ChunkerConfig()

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into chunks based on character count, trying to break at sentence
        boundaries when possible.

        Args:
            text: The text to split into chunks

        Returns:
            A list of text chunks
        """
        if not text or len(text) <= self.config.chunk_size:
            return [text] if text else []

        chunks = []
        start = 0
        while start < len(text):
            # Find a good break point (end of sentence or space)
            end = min(start + self.config.chunk_size, len(text))
            if end < len(text):
                # Try to find sentence end (.!?) within the last 100 chars
                for punct in [".", "!", "?"]:
                    last_punct = text.rfind(punct, start, end)
                    if last_punct > end - 100:
                        end = last_punct + 1
                        break
                else:
                    # If no sentence break, try to break at space
                    last_space = text.rfind(" ", start, end)
                    if last_space > start:
                        end = last_space + 1

            chunks.append(text[start:end].strip())
            start = end - self.config.chunk_overlap

        return chunks
