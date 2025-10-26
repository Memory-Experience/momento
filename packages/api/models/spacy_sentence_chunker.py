from __future__ import annotations

import spacy
from spacy.language import Language

from .text_chunker_interface import ChunkerConfig, TextChunker


class SpacySentenceChunker(TextChunker):
    """
    Sentence-level text chunker using spaCy NLP pipeline.

    Returns one sentence per chunk as plain strings. The chunk_size parameter
    in config acts as a safety net for very long sentences - when a sentence
    exceeds this limit, it's split into token windows.

    This approach follows best practices for embedding models like Qwen3,
    which recommend sentence-level chunking for optimal semantic retrieval.

    Features:
        - Sentence boundary detection using spaCy
        - No overlap between sentences (clean splits)
        - Token-level windowing for oversized sentences
        - Configurable spaCy model
    """

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        *,
        model: str = "en_core_web_sm",
        nlp: Language | None = None,
    ):
        """
        Initialize the spaCy sentence chunker.

        Args:
            config (ChunkerConfig | None): Configuration (chunk_size used
                as sentence length limit)
            model (str): spaCy model name (default: "en_core_web_sm")
            nlp (Language | None): Pre-loaded spaCy Language instance
                (optional)
        """
        self.config = config or ChunkerConfig()
        self.max_sentence_tokens = None
        try:
            # If provided and > 0, use as guardrail for very long sentences
            cs = int(getattr(self.config, "chunk_size", 0))
            if cs > 0:
                self.max_sentence_tokens = cs
        except Exception:
            self.max_sentence_tokens = None

        if nlp is not None:
            self.nlp = nlp
        else:
            try:
                self.nlp = spacy.load(model)
            except Exception:
                # Fallback: blank English with a simple sentencizer
                self.nlp = spacy.blank("en")

        if not (
            self.nlp.has_pipe("senter")
            or self.nlp.has_pipe("parser")
            or self.nlp.has_pipe("sentencizer")
        ):
            self.nlp.add_pipe("sentencizer")

    def chunk_text(self, text: str) -> list[str]:
        """
        Split text into sentence-level chunks.

        Args:
            text (str): Text to split into sentences

        Returns:
            list[str]: List of sentence chunks
        """
        if not text:
            return []

        doc = self.nlp(text)
        chunks: list[str] = []
        for sent in doc.sents:
            if self.max_sentence_tokens and len(sent) > self.max_sentence_tokens:
                # Guardrail: window an extra-long sentence by tokens
                chunks.extend(self._window_tokens(sent, self.max_sentence_tokens))
            else:
                s = sent.text.strip()
                if s:
                    chunks.append(s)
        return chunks

    @staticmethod
    def _window_tokens(span, size: int) -> list[str]:
        """
        Split a long sentence into overlapping token windows.

        Uses 25% overlap to preserve coherence across window boundaries.

        Args:
            span: spaCy Span object representing the sentence
            size (int): Maximum tokens per window

        Returns:
            list[str]: List of windowed text segments
        """
        out: list[str] = []
        overlap = max(1, size // 4)
        i = 0
        n = len(span)
        while i < n:
            j = min(i + size, n)
            out.append(span[i:j].text.strip())
            if j == n:
                break
            i = j - overlap
        return out
