from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from ..huggingface_helper import HuggingFaceHelper
from ..llama_cpp_base import LlamaCppConfig
from .llama_cpp_model import LlamaCppModel
from .llm_model_interface import LLMModelBase


class Qwen3LlamaCppModel(LlamaCppModel, LLMModelBase):
    """
    Qwen3-1.7B-Instruct (GGUF) using llama.cpp. Automatically downloads
        a quantized version from HF.
    """

    DEFAULT_HF_REPO_ID = "unsloth/Qwen3-1.7B-GGUF"
    DEFAULT_PREFERRED_QUANTS: Sequence[str] = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0")

    def __init__(
        self,
        *,
        model_path: str | None = None,
        hf_repo_id: str | None = DEFAULT_HF_REPO_ID,
        download_dir: str = "models/llm/qwen3",
        preferred_quants: Sequence[str] | None = None,
        # runtime
        n_ctx: int = 4096,
        n_batch: int = 256,
        n_threads: int | None = None,
        n_gpu_layers: int = -1,
        seed: int = 42,
        use_mmap: bool = True,
        use_mlock: bool = True,
        allow_gpu_fallback: bool = True,
        # decoding / RAG
        temperature: float = 0.3,
        top_p: float = 0.9,
        top_k_memories: int = 5,
        stop: list[str] | None = None,
        system_prompt: str | None = None,
        # streaming ergonomics
        chunk_size_tokens: int = 16,
        # dependency injection
        model_resolver: Any | None = None,
    ) -> None:
        # Use default preferred quants if none provided
        if preferred_quants is None:
            preferred_quants = self.DEFAULT_PREFERRED_QUANTS

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
        )

        # Initialize LlamaCppModel
        LlamaCppModel.__init__(
            self,
            cfg=cfg,
            model_name="Qwen3-1.7B-GGUF",
            system_prompt=system_prompt,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k_memories=top_k_memories,
            chunk_size_tokens=chunk_size_tokens,
        )

        # Initialize LLMModelBase
        LLMModelBase.__init__(
            self,
            system_prompt=system_prompt,
            top_k_memories=top_k_memories,
        )
