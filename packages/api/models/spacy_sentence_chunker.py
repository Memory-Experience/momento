from __future__ import annotations

import spacy
from spacy.language import Language

from api.models.text_chunker_interface import ChunkerConfig, TextChunker


class SpacySentenceChunker(TextChunker):
    """
    Sentence-level text chunker using spaCy.

    - Returns one sentence per chunk (as plain strings).
    - If `config.chunk_size` is set (>0), it's used ONLY as a safety net:
      very long single sentences are windowed by token count.
    - No chunk IDs, no overlaps—just clean sentence chunks.

    This mirrors the approach described in Daft's Qwen3 embedding pipeline,
    which recommends sentence-level chunking by default.
    """

    def __init__(
        self,
        config: ChunkerConfig | None = None,
        *,
        model: str = "en_core_web_sm",
        nlp: Language | None = None,
    ):
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
        Sliding token windows over a single long sentence.
        Uses 25% overlap to preserve coherence within the sentence.
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
