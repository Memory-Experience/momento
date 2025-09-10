from __future__ import annotations

import asyncio
from typing import Any

from ..llama_cpp_base import LlamaCppBase, LlamaCppConfig
from .embedding_model_interface import EmbeddingModel


class LlamaCppEmbeddingModel(LlamaCppBase, EmbeddingModel):
    """
    llama.cpp-backed embedding base that relies on LlamaCppBase for loading.
    Simply set cfg.embedding=True when constructing.
    """

    def __init__(
        self, *, cfg: LlamaCppConfig, llama_factory: Any | None = None
    ) -> None:
        super().__init__(cfg=cfg, llama_factory=llama_factory)
        self._vector_size: int | None = None

    # ---- EmbeddingModel API ----
    def get_vector_size(self) -> int:
        if self._vector_size is not None:
            return self._vector_size

        # Try llama.cpp-reported size if exposed
        try:
            n_embd = getattr(self._llm, "n_embd", None)
            size = int(n_embd()) if callable(n_embd) else int(n_embd or 0)
            if size > 0:
                self._vector_size = size
                return size
        except Exception:
            pass

        # Fallback: probe once
        try:
            out = self._llm.create_embedding(input="")
            emb = out["data"][0]["embedding"]
            self._vector_size = len(emb)
            return self._vector_size
        except Exception:
            return 0

    async def embed_text(self, text: str) -> list[float]:
        def _compute() -> list[float]:
            out = self._llm.create_embedding(input=text)
            vec = out["data"][0]["embedding"]
            if self._vector_size is None:
                self._vector_size = len(vec)
            return [float(x) for x in vec]

        return await asyncio.to_thread(_compute)
