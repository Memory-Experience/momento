from __future__ import annotations

import asyncio
import multiprocessing
import threading
from collections.abc import AsyncIterator, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from domain.memory_context import MemoryContext
from domain.memory_request import MemoryRequest, MemoryType
from llama_cpp import Llama

from models.llm.llm_model_interface import LLMModel, MemoryResponse


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


def _default_threads() -> int:
    try:
        return max(1, multiprocessing.cpu_count() // 2)
    except Exception:
        return 4


class LlamaCppModel(LLMModel):
    """
    llama.cpp-backed model implementing the streaming generator API.
    No singletons; each instance owns its Llama handle. DI supported via llama_factory.
    """

    def __init__(
        self,
        *,
        cfg: LlamaCppConfig,
        model_name: str,
        system_prompt: str | None = None,
        stop: list[str] | None = None,
        temperature: float = 0.2,
        top_p: float = 0.95,
        top_k_memories: int = 5,
        chunk_size_tokens: int = 1,
        llama_factory: Any | None = None,
    ) -> None:
        self.cfg = cfg
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.stop = stop or ["</s>"]
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k_memories = int(top_k_memories)
        self.chunk_size_tokens = max(1, int(chunk_size_tokens))
        self._llama_factory = llama_factory or Llama
        self._llm = self._load_model()

    # -------- loading --------
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

    # -------- prompt building from MemoryContext --------
    def _format_memories(self, top: Iterable[tuple], max_chars: int = 1200) -> str:
        lines: list[str] = []
        for mem, matched_text, score in top:
            snippet = (matched_text or "").strip()
            if max_chars and len(snippet) > max_chars:
                snippet = snippet[: max_chars - 3] + "..."
            lines.append(f"- score={score:.4f} | id={mem.id} | {snippet}")
        return "\n".join(lines)

    def _build_messages(
        self, prompt: str, memory_context: MemoryContext | None
    ) -> list[dict[str, str]]:
        """
        Build messages for the chat completion API using the provided prompt
        and memory context. Format the context in a way that's easier for
        small models to understand.
        """
        messages: list[dict[str, str]] = [
            {"role": "system", "content": self.system_prompt}
        ]

        if memory_context and not memory_context.is_empty():
            top = memory_context.get_top_memories(limit=self.top_k_memories)

            # Format context in a clearer, more explicit way for the model
            context_parts = ["Here are relevant memories:"]

            for i, (mem, matched_text, _score) in enumerate(top, 1):
                snippet = (matched_text or "").strip()
                if len(snippet) > 1200:
                    snippet = snippet[:1197] + "..."
                context_parts.append(f"Memory #{i} (ID: {mem.id}): {snippet}")

            context_parts.append(
                "\nAnswer the question using ONLY the information above."
            )

            messages.append({
                "role": "system",
                "content": "\n\n".join(context_parts),
            })

        messages.append({"role": "user", "content": prompt})
        return messages

    def _suggest_max_tokens(self) -> int:
        """
        Suggest a reasonable maximum number of tokens to generate based on context size.
        """
        # In llama.cpp, the context size is a method
        # on the context object (_ctx.n_ctx())
        # This is the proper way to get the context size
        try:
            n_ctx = self._llm.n_ctx()
        except (AttributeError, TypeError):
            # Fall back to the config value if the method isn't available
            n_ctx = self.cfg.n_ctx

        return max(256, n_ctx // 3)

    # -------- main API: async generator --------
    async def generate_with_memory(
        self,
        prompt: str,
        memory_context: MemoryContext,
        chunk_size_tokens: int = 1,
    ) -> AsyncIterator[MemoryResponse]:
        """
        Generates responses in a streaming fashion,
        yielding multiple MemoryResponse items.
        The size of each chunk is controlled by chunk_size_tokens -
        higher values will yield fewer, larger chunks
        (reducing the streaming granularity).
        """
        # Convert prompt from list to string if needed
        if isinstance(prompt, list):
            prompt = " ".join(prompt)

        messages = self._build_messages(prompt, memory_context)

        # Override instance chunk_size if provided in method call
        # For small models, increase chunk size to avoid character-by-character output
        chunk_size = (
            max(8, int(chunk_size_tokens))
            if chunk_size_tokens != 1
            else max(8, self.chunk_size_tokens)
        )

        # streaming setup
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue(maxsize=256)

        def _producer():
            try:
                for chunk in self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=self._suggest_max_tokens(),
                    temperature=self.temperature,
                    top_p=self.top_p,
                    stream=True,
                    stop=self.stop,
                ):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, None)

        threading.Thread(target=_producer, daemon=True).start()

        # coalescing buffer
        buf_text: str = ""
        tokens_emitted = 0
        last_usage: dict[str, int] | None = None

        async def _flush(is_final: bool = False):
            nonlocal buf_text, tokens_emitted
            if not buf_text:
                return

            # Always create a single-item list for the text field in MemoryRequest
            # This ensures consistent handling in the RAG service
            yield MemoryResponse(
                response=MemoryRequest.create(
                    id=None, text=[buf_text.strip()], memory_type=MemoryType.ANSWER
                ),
                model_name=self.model_name,
                tokens_used=tokens_emitted,
                metadata={
                    "is_final": is_final,
                    "stream": True,
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "stop": list(self.stop),
                },
            )
            buf_text = ""

        # Iterate chunks; coalesce by chunk_size_tokens;
        # mark only the final yield as is_final=True
        pending_final_delta: str | None = None

        while True:
            chunk = await queue.get()
            if chunk is None:
                # End of stream — emit any remaining buffered text as final with usage.
                if pending_final_delta is not None:
                    buf_text += pending_final_delta
                    tokens_emitted += 1
                    pending_final_delta = None
                # single final flush
                async for item in _flush(is_final=True):
                    # Attach usage if we captured it
                    if last_usage:
                        item.metadata["usage"] = {
                            "prompt_tokens": int(last_usage.get("prompt_tokens", 0)),
                            "completion_tokens": int(
                                last_usage.get("completion_tokens", 0)
                            ),
                            "total_tokens": int(last_usage.get("total_tokens", 0)),
                        }
                        item.tokens_used = (
                            int(item.metadata["usage"]["total_tokens"])
                            or item.tokens_used
                        )
                    yield item
                break

            # Try to capture usage if present (llama.cpp may include it at the tail)
            usage = chunk.get("usage")
            if usage:
                last_usage = {
                    "prompt_tokens": int(usage.get("prompt_tokens", 0)),
                    "completion_tokens": int(usage.get("completion_tokens", 0)),
                    "total_tokens": int(usage.get("total_tokens", 0)),
                }

            delta = chunk["choices"][0].get("delta", {}).get("content")
            if delta is None:
                continue

            # We want to mark the *last* emitted item as final, so keep
            # a one-chunk lookbehind.
            if pending_final_delta is None:
                pending_final_delta = delta
                continue

            # Emit previous delta(s) in coalesced chunks
            buf_text += pending_final_delta
            tokens_emitted += 1
            pending_final_delta = delta

            # Coalesce by N tokens to reduce object churn
            if (tokens_emitted % chunk_size) == 0:
                async for item in _flush(is_final=False):
                    yield item
