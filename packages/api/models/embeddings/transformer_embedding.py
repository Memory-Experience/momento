import asyncio

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from models.embedding_model_interface import EmbeddingModel


class HFTransformerEmbeddingModel(EmbeddingModel):
    """
    Minimal embedding model using Hugging Face Transformers.
    Defaults to a small, fast embedding model that runs on CPU.
    """

    def __init__(
        self,
        model_name: str = "intfloat/e5-small-v2",  # good quality/speed balance
        device: str | None = None,  # "cuda", "cpu", etc.
        max_length: int = 512,
        normalize: bool = True,
    ):
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # float16 on CUDA, float32 on CPU
        torch_dtype = torch.float16 if self.device.type == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length
        self._normalize = normalize

        # Cache vector size
        hidden = getattr(self.model.config, "hidden_size", None)
        if hidden is None:
            # fallback: run a tiny forward pass to detect size
            with torch.no_grad():
                toks = self.tokenizer("test", return_tensors="pt", truncation=True).to(
                    self.device
                )
                out = self.model(**toks).last_hidden_state
                hidden = out.shape[-1]
        self._dim = int(hidden)

    def get_vector_size(self) -> int:
        return self._dim

    async def embed_text(self, text: str) -> list[float]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._embed_sync, text)

    def _embed_sync(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self._dim

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                max_length=self.max_length,
                truncation=True,
                padding=False,
            ).to(self.device)

            out = self.model(**inputs)
            last_hidden = out.last_hidden_state  # [B, T, H]
            mask = inputs["attention_mask"].unsqueeze(-1)  # [B, T, 1]
            masked = last_hidden * mask
            summed = masked.sum(dim=1)  # [B, H]
            counts = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
            emb = summed / counts

            if self._normalize:
                emb = F.normalize(emb, p=2, dim=-1)

            vec = emb.squeeze(0).to(torch.float32).cpu().tolist()
            return vec
