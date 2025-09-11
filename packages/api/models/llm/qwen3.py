from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from api.models.huggingface_helper import HuggingFaceHelper
from api.models.llama_cpp_base import LlamaCppConfig
from api.models.llm.llama_cpp_model import LlamaCppModel
from api.domain.memory_context import MemoryContext


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
information using the format <source>memory.id</source> at the end of a sentence.
Make sure to always answer in second person. NEVER say: "I had ..." in first 
person. Always answer in second person!

<examples>
These are only illustrations. Do NOT treat them as actual memories. 
They are here to demonstrate the answering format.

    <example>
        <system_prompt>
        Pretend this is what a memory input looks like in 
        JSON format: {'id': '17424128-3e76-4aa3-8230-aeaae77385e0', 
        'score': 0.83, 'content': 'In the year 1723 on first of december i
          had a toast for breakfast'}
          </system_prompt>
        <users_prompt>
        What did i eat in the morning 
        of the 1st of december 1723</users_prompt>
        <your_answer>
        You had a toast for breakfast on the 
        1st of december 1723 <source>17424128-3e76-4aa3-8230-aeaae77385e0</source>
        </your_answer>
    </example>
    <example>
        <system_prompt>
        Pretend this is what a memory input looks like in JSON format: 
        {'id': '17424128-3e76-4aa3-8230-aeaae77385e0', 'score': 0.83, 
        'content': 'In the year 1723 on first of december 
        i had a toast for breakfast'}
        </system_prompt>
        <users_prompt>What is 1 + 1</users_prompt>
        <your_answer>I don't have any memories about that.</your_answer>
    </example>

Ignore all example blocks when generating an answer. 
Only use memory JSONs that appear after this point and outside of <examples>
</examples>

/nothink
"""

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

    # -------- Custom prompt building for Qwen3 --------
    def _format_memories(self, top: Iterable[tuple], max_chars: int = 1200) -> str:
        """Format memories as a JSON-like structure for better LLM comprehension."""
        memories = []
        for mem, matched_text, score in top:
            snippet = (matched_text or "").strip()
            if max_chars and len(snippet) > max_chars:
                snippet = snippet[: max_chars - 3] + "..."

            memories.append(
                f'{{ "id": "{mem.id}", "score": {score:.4f}, "content": "{snippet}" }}'
            )

        if not memories:
            return "No relevant memories found."

        return "\n".join(memories)

    def build_messages(
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

        if memory_context:
            top = memory_context.get_top_memories(limit=self.top_k_memories)

            # Format context with JSON-like memory structures
            context_parts = [
                "Here are relevant memories in JSON format:<memories>",
                self._format_memories(top, max_chars=1200),
                "</memories>\nAnswer the question using ONLY the <memories> content!",
            ]

            messages.append({
                "role": "system",
                "content": "\n\n".join(context_parts),
            })

        messages.append({"role": "user", "content": prompt})
        return messages
