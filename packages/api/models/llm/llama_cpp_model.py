from __future__ import annotations

import asyncio
import threading
from collections.abc import AsyncIterator
from typing import Any

from api.domain.memory_context import MemoryContext
from api.domain.memory_request import MemoryRequest, MemoryType

from api.models.llama_cpp_base import LlamaCppBase, LlamaCppConfig
from api.models.llm.llm_model_interface import LLMModel, MemoryResponse


class LlamaCppModel(LlamaCppBase, LLMModel):
    """
    Base LLM implementation using llama.cpp backend with streaming support.

    Provides async streaming text generation with configurable chunk sizes and
    memory context integration. Each instance owns its own llama.cpp handle
    for thread-safe operation. Supports dependency injection via llama_factory
    for testing.
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
        super().__init__(cfg=cfg, llama_factory=llama_factory)
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.stop = stop or ["</s>"]
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k_memories = int(top_k_memories)
        self.chunk_size_tokens = max(1, int(chunk_size_tokens))

    # -------- prompt building from MemoryContext --------
    def build_messages(
        self, prompt: str, memory_context: MemoryContext | None
    ) -> list[dict[str, str]]:
        """
        Build messages for the chat completion API.

        Subclasses must override this method to define their own formatting
        strategy for incorporating memory context into the prompt.

        Args:
            prompt: The user's prompt or question
            memory_context: Retrieved memories with relevance scores (optional)

        Returns:
            List of message dictionaries with 'role' and 'content' keys

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            "Subclasses must implement their own build_messages method"
        )

    # -------- main API: async generator --------
    async def generate_with_memory(
        self,
        prompt: str,
        memory_context: MemoryContext,
        chunk_size_tokens: int = 1,
    ) -> AsyncIterator[MemoryResponse]:
        """
        Generate streaming text responses with memory context.

        Produces an async stream of response chunks, with the final chunk marked
        as is_final=True. The chunk_size_tokens parameter controls streaming
        granularity - higher values produce fewer, larger chunks which reduces
        UI update frequency but improves perceived smoothness.

        Args:
            prompt: The user's prompt (string or list of strings)
            memory_context: Retrieved memories to use as context
            chunk_size_tokens: Number of tokens per output chunk (default: 1)
                              Higher values (e.g., 8-16) reduce character-by-character
                              streaming for smoother output

        Yields:
            MemoryResponse: Response chunks with text, metadata, and token usage.
                           The final chunk has is_final=True in metadata.
        """
        # Convert prompt from list to string if needed
        if isinstance(prompt, list):
            prompt = " ".join(prompt)

        messages = self.build_messages(prompt, memory_context)

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
                    text=[buf_text.strip()], memory_type=MemoryType.ANSWER
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
