import asyncio
from typing import Any, AsyncIterator, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from .llm_model_interface import LLMModelBase, MemoryResponse
from ...domain.memory_context import MemoryContext
from ...domain.memory_request import MemoryRequest

class Qwen3TransformersModel(LLMModelBase):
    """
    Minimal HF runner for Qwen3 Instruct models (e.g., Qwen3-1.7B-Instruct)
    """

    DEFAULT_HF_REPO_ID = "Qwen/Qwen3-1.7B-Instruct"

    def __init__(
        self,
        *,
        model_path: str | None = None,
        hf_repo_id: str | None = DEFAULT_HF_REPO_ID,
        # generation knobs (keep small + simple)
        max_new_tokens: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.9,
        do_sample: bool = True,
        top_k_memories: int = 5,
        system_prompt: Optional[str] = None,
        # device/dtype (simple defaults)
        device_map: Any = "auto",
        torch_dtype: Any = torch.bfloat16,  # switch to torch.float16 if your GPU needs it
        chunk_size_tokens: int = 16,        # ~how many tokens per streamed chunk
    ) -> None:
        # Initialize LLMModelBase
        LLMModelBase.__init__(
            self,
            system_prompt=system_prompt,
            top_k_memories=top_k_memories,
        )

        # Resolve model path - either use provided path or use HF repo
        # Note: Unlike GGUF models, transformers models don't need special handling
        # as AutoModelForCausalLM handles downloading and caching automatically
        if model_path:
            # Validate local path exists
            from pathlib import Path
            p = Path(model_path).expanduser()
            if not p.exists():
                raise FileNotFoundError(f"Model path not found: {p}")
            resolved_path = str(p)
        elif hf_repo_id:
            # For transformers models, we use the repo_id directly
            # HuggingFace transformers will handle downloading and caching
            resolved_path = hf_repo_id
        else:
            raise ValueError(
                "Provide either a local `model_path` or an HF `hf_repo_id` to download."
            )

        self.model_name = resolved_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.do_sample = do_sample
        self.chunk_size_tokens = max(1, int(chunk_size_tokens))

        # Load tokenizer/model from local path or HF repo
        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_path, use_fast=True, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            resolved_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
        # sane pad/eos
        self.eos_token_id = self.tokenizer.eos_token_id
        self.pad_token_id = self.tokenizer.eos_token_id

    # ---------- LLMModel API ----------
    async def generate_with_memory(
        self,
        prompt: str,
        memory_context: MemoryContext,
        chunk_size_tokens: int = 1,
    ) -> AsyncIterator[MemoryResponse]:
        # build chat prompt
        messages = self.build_messages(prompt, memory_context)
        chat_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(chat_text, return_tensors="pt").to(self.model.device)
        prompt_tokens = inputs["input_ids"].shape[-1]

        # streaming setup
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.do_sample,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id,
        )

        # run generation in a worker thread
        loop = asyncio.get_running_loop()

        def _gen():
            with torch.inference_mode():
                self.model.generate(**gen_kwargs)

        gen_task = loop.run_in_executor(None, _gen)

        # stream pieces and coalesce to ~N tokens per chunk
        target = max(1, chunk_size_tokens or self.chunk_size_tokens)
        buf, buf_tok = [], 0

        try:
            async def _aiter():
                while True:
                    tok = await loop.run_in_executor(None, next, streamer, None)
                    if tok is None:
                        break
                    yield tok

            async for piece in _aiter():
                buf.append(piece)
                buf_tok += len(
                    self.tokenizer.encode(piece, add_special_tokens=False)
                )
                if buf_tok >= target:
                    text = "".join(buf)
                    yield MemoryResponse(
                        response=MemoryRequest.create(text=[text]),
                        model_name=self.model_name,
                        tokens_used=prompt_tokens + buf_tok,
                        metadata={},
                    )
                    buf, buf_tok = [], 0

            if buf:
                text = "".join(buf)
                buf_tok += len(self.tokenizer.encode(text, add_special_tokens=False))
                yield MemoryResponse(
                    response=MemoryRequest.create(text=[text]),
                    model_name=self.model_name,
                    tokens_used=prompt_tokens + buf_tok,
                    metadata={"final": True},
                )

            await gen_task
        finally:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
