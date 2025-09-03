from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from models.huggingface_helper import HuggingFaceHelper
from models.llm.llama_cpp_model import LlamaCppConfig, LlamaCppModel


class Qwen3(LlamaCppModel):
    """
    Qwen3-1.7B-Instruct (GGUF) using llama.cpp. Automatically downloads
        a quantized version from HF.
    """

    DEFAULT_PREFERRED_QUANTS: Sequence[str] = ("Q4_K_M", "Q4_K_S", "Q5_K_M", "Q8_0")
    DEFAULT_HF_REPO_ID = "unsloth/Qwen3-1.7B-GGUF"

    DEFAULT_SYSTEM_PROMPT = """You are a helpful assistant with access 
to the user's memories. When answering questions, ONLY use information 
provided in the context. If the answer cannot be found in the context, 
say "I don't have any memories about that." Be concise and focus only on 
information found in the relevant memories. Always cite the source of the 
information using the format [source: <memory id>] at the end of a sentence."""

    def __init__(
        self,
        *,
        model_path: str | None = None,
        hf_repo_id: str | None = DEFAULT_HF_REPO_ID,
        download_dir: str = "models/llm/qwen3",
        preferred_quants: Sequence[str] = DEFAULT_PREFERRED_QUANTS,
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
        super().__init__(
            cfg=cfg,
            model_name="Qwen3-1.7B-GGUF",
            system_prompt=system_prompt or self.DEFAULT_SYSTEM_PROMPT,
            stop=stop,
            temperature=temperature,
            top_p=top_p,
            top_k_memories=top_k_memories,
            chunk_size_tokens=chunk_size_tokens,
        )
