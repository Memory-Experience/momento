from __future__ import annotations

import multiprocessing
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llama_cpp import Llama


@dataclass(slots=True)
class LlamaCppConfig:
    model_path: str
    n_ctx: int = 4096
    n_batch: int = 256
    n_threads: int | None = None
    n_gpu_layers: int = -1  # -1: offload as much as possible; 0: CPU
    seed: int = 42
    use_mmap: bool = True
    use_mlock: bool = True
    main_gpu: int = 0
    tensor_split: Sequence[float] | None = None
    allow_gpu_fallback: bool = True  # retry on CPU if GPU offload fails
    embedding: bool = False


def _default_threads() -> int:
    try:
        return max(1, multiprocessing.cpu_count() // 2)
    except Exception:
        return 4


class LlamaCppBase:
    """
    Base class for llama.cpp-backed models.
    Handles model loading and configuration.
    No singletons; each instance owns its Llama handle.
    """

    def __init__(
        self,
        *,
        cfg: LlamaCppConfig,
        llama_factory: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self._llama_factory = llama_factory or Llama
        self._llm = self._load_model()

    def _load_model(self) -> Llama:
        path = str(Path(self.cfg.model_path).expanduser())
        kwargs: dict[str, Any] = dict(
            model_path=path,
            n_ctx=self.cfg.n_ctx,
            n_batch=self.cfg.n_batch,
            n_threads=self.cfg.n_threads
            if self.cfg.n_threads is not None
            else _default_threads(),
            seed=self.cfg.seed,
            n_gpu_layers=self.cfg.n_gpu_layers,
            use_mmap=self.cfg.use_mmap,
            use_mlock=self.cfg.use_mlock,
            embedding=self.cfg.embedding,
        )
        if self.cfg.tensor_split is not None:
            kwargs["tensor_split"] = list(self.cfg.tensor_split)
        if self.cfg.main_gpu is not None:
            kwargs["main_gpu"] = self.cfg.main_gpu

        try:
            return self._llama_factory(**kwargs)
        except Exception:
            if self.cfg.allow_gpu_fallback and self.cfg.n_gpu_layers != 0:
                kwargs["n_gpu_layers"] = 0
                return self._llama_factory(**kwargs)
            raise

    def _suggest_max_tokens(self) -> int:
        """
        Suggest a reasonable maximum number of tokens to generate based on context size.
        """
        try:
            n_ctx = self._llm.n_ctx()
        except (AttributeError, TypeError):
            # Fall back to the config value if the method isn't available
            n_ctx = self.cfg.n_ctx

        return max(256, n_ctx // 3)
