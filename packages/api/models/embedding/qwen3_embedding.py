from __future__ import annotations

from typing import Any
from collections.abc import Sequence

from api.models.huggingface_helper import HuggingFaceHelper
from api.models.llama_cpp_base import LlamaCppConfig
from api.models.embedding.llama_cpp_embedding import LlamaCppEmbeddingModel


class Qwen3EmbeddingModel(LlamaCppEmbeddingModel):
    """
    Qwen3-Embedding-0.6B (GGUF) via llama.cpp.

    Automatically downloads a preferred quantized file from HF and
    instantiates a llama.cpp embedding model.
    """

    DEFAULT_HF_REPO_ID: str = "Qwen/Qwen3-Embedding-0.6B-GGUF"
    # Prefer widely available quants; ordered by quality -> size tradeoff
    DEFAULT_PREFERRED_QUANTS: Sequence[str] = ("Q5_0", "Q4_0", "Q8_0", "Q3_K_M", "Q2_K")

    def __init__(
        self,
        *,
        # Model resolution
        model_path: str | None = None,
        hf_repo_id: str | None = DEFAULT_HF_REPO_ID,
        download_dir: str = "models/embedding/qwen3",
        preferred_quants: Sequence[str] = DEFAULT_PREFERRED_QUANTS,
        # Runtime
        n_ctx: int = 2048,
        n_batch: int = 1024,
        n_threads: int | None = None,
        n_gpu_layers: int = -1,  # -1: offload as much as possible; 0: CPU
        seed: int = 42,
        use_mmap: bool = True,
        use_mlock: bool = True,
        allow_gpu_fallback: bool = True,
        # DI
        model_resolver: Any | None = None,
        llama_factory: Any | None = None,
    ) -> None:
        resolver = model_resolver or HuggingFaceHelper()
        resolved = resolver.ensure_local_model(
            model_path=model_path,
            hf_repo_id=hf_repo_id,
            download_dir=download_dir,
            preferred_quants=preferred_quants,
        )

        cfg = LlamaCppConfig(
            model_path=resolved,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=n_threads,
            n_gpu_layers=n_gpu_layers,
            seed=seed,
            use_mmap=use_mmap,
            use_mlock=use_mlock,
            allow_gpu_fallback=allow_gpu_fallback,
            embedding=True,
        )

        super().__init__(cfg=cfg, llama_factory=llama_factory)

    async def embed_text(self, text: str) -> list[float]:
        # Prepend "Instruct: " to the text
        text_with_prefix = f"Instruct: {text}"
        # Call the parent class's embed_text method
        return await super().embed_text(text_with_prefix)
