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
        """
        Initialize the Sentence-BERT embedding model.

        Args:
            model_name (str): HuggingFace model identifier
                (default: all-MiniLM-L6-v2)
            device (str | None): Device to run on
                ("cuda", "mps", "cpu", or None for auto)
            normalize_embeddings (bool): Whether to normalize embeddings
                to unit length
        """
        self._model = SentenceTransformer(model_name, device=device)
        self._normalize = normalize_embeddings
        # Cache the dimension to avoid repeated calls
        self._dimension = int(self._model.get_sentence_embedding_dimension())

    def get_vector_size(self) -> int:
        """
        Get the dimension of embedding vectors.

        Returns:
            int: Embedding vector dimension (384 for default model)
        """
        return self._dimension

    async def embed_text(self, text: str) -> list[float]:
        """
        Asynchronously compute a sentence embedding.

        Runs the blocking model.encode in a synchronous manner while maintaining
        the async interface for consistency with other embedding models.

        Args:
            text (str): Text to embed

        Returns:
            list[float]: Normalized embedding vector
        """
        emb = self._model.encode(
            text or "",
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
        )

        return emb.astype(float).tolist()
