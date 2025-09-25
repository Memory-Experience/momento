from __future__ import annotations

from api.models.text_chunker_interface import TextChunker


class DummyTextChunker(TextChunker):
    """
    A chunker that returns the entire text as a single chunk.
    """

    def chunk_text(self, text: str) -> list[str]:
        return [text]
