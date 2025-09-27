from __future__ import annotations

import asyncio
from typing import List, Optional, Sequence, Tuple

import numpy as np

from sentence_transformers import CrossEncoder

class CrossEncoderScorer:
    """
    Thin wrapper around Sentence-Transformers CrossEncoder to score (query, passage)
    or (answer, gold_answer) pairs.

    Default model: 'cross-encoder/ms-marco-MiniLM-L-6-v2'
      - Trained for MS MARCO relevance; higher scores => more relevant.
      - Outputs *unbounded* real-valued scores (not cosine sims). You can optionally
        pass through a sigmoid for a 0..1 range.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,   # "cuda", "mps", or "cpu"
        max_length: int = 512,          # token budget for each pair
        normalize: Optional[str] = None # None | "sigmoid" | "zscore"
    ) -> None:
        self._model = CrossEncoder(model_name, device=device, max_length=max_length)
        self._normalize = normalize

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def _zscore(x: np.ndarray) -> np.ndarray:
        mu = float(np.mean(x))
        sd = float(np.std(x)) or 1.0
        return (x - mu) / sd

    async def score_pairs(self, pairs: Sequence[Tuple[str, str]]) -> List[float]:
        """
        Score a batch of (text_a, text_b) pairs asynchronously.
        """
        def _predict_sync() -> np.ndarray:
            scores = self._model.predict(pairs, show_progress_bar=False)
            # CrossEncoder returns np.ndarray of shape (N,) or (N, 1)
            scores = np.array(scores).reshape(-1)
            if self._normalize == "sigmoid":
                scores = self._sigmoid(scores)
            elif self._normalize == "zscore":
                scores = self._zscore(scores)
            return scores

        scores: np.ndarray = await asyncio.to_thread(_predict_sync)
        return scores.astype(float).tolist()

    async def best_of(
        self,
        text: str,
        candidates: Sequence[str],
        reduction: str = "max"  # "max" or "mean"
    ) -> float:
        if not candidates:
            return 0.0
        pairs = [(text, c or "") for c in candidates]
        scores = await self.score_pairs(pairs)
        return float(np.mean(scores) if reduction == "mean" else np.max(scores))
