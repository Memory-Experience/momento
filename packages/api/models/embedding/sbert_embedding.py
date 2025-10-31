from sentence_transformers import SentenceTransformer

from .embedding_model_interface import EmbeddingModel


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

    async def embed_texts_batch(
        self, texts: list[str], batch_size: int = 32
    ) -> list[list[float]]:
        """
        Asynchronously compute sentence embeddings for multiple texts in batch.
        This is much more efficient than calling embed_text multiple times.

        Args:
            texts: List of texts to embed

        Returns:
            List of embeddings, one per input text
        """
        if not texts:
            return []

        # SentenceTransformer.encode handles batching natively
        embeddings = self._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
            show_progress_bar=False,
            batch_size=batch_size,
        )

        # Convert to list of lists
        return [emb.astype(float).tolist() for emb in embeddings]
