from sentence_transformers import SentenceTransformer

from api.models.embedding.embedding_model_interface import EmbeddingModel


class SBertEmbeddingModel(EmbeddingModel):
    """
    Sentence-BERT embedding model implementation.

    Notes:
      - Default model is `all-MiniLM-L6-v2` (384-dim, fast & strong baseline).
      - Embedding is done via sentence-transformers' `encode`
        with normalization enabled, so cosine similarity reduces to a dot-product.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str | None = None,  # e.g. "cuda", "mps", or "cpu"
        normalize_embeddings: bool = True,
    ) -> None:
        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize_embeddings
        # Cache the dimension to avoid repeated calls
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    def get_vector_size(self) -> int:
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        """
        Asynchronously compute a sentence embedding. We run the (blocking) model.encode
        in a thread to keep the async contract safe in event-loop environments.
        """
        emb = self._model.encode(
            text or "",
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )

        return emb.astype(float).tolist()
