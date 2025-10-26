import os
from dataclasses import dataclass

from .persistence.persistence_service import PersistenceService
from .persistence.repositories.file_repository import FileRepository
from .rag.llm_rag_service import LLMRAGService
from .rag.threshold_filter_service import ThresholdFilterService
from .vector_store.repositories.qdrant_vector_store_repository import (
    InMemoryQdrantVectorStoreRepository,
)
from .vector_store.repositories.vector_store_repository_interface import (
    VectorStoreRepository,
)
from .vector_store.vector_store_service import VectorStoreService

from .models.spacy_sentence_chunker import SpacySentenceChunker
from .models.llm.qwen3_llama_cpp_model import Qwen3LlamaCppModel
from .models.embedding.qwen3_embedding import Qwen3EmbeddingModel
from .models.transcription.faster_whisper_transcriber import FasterWhisperTranscriber
from .models.transcription.transcriber_interface import TranscriberInterface


RECORDINGS_DIR = os.path.abspath("recordings")
SAMPLE_RATE = 16000
RETRIEVAL_LIMIT = 2


@dataclass
class Container:
    """
    Dependency injection container for the Momento application.

    This container holds all the service dependencies needed by both FastAPI
    and gRPC servicers, providing a single source of truth for service configuration.
    """

    vector_store: VectorStoreService
    persistence: PersistenceService
    rag: LLMRAGService
    threshold_filter: ThresholdFilterService
    transcriber: TranscriberInterface

    sample_rate: int = SAMPLE_RATE
    recordings_dir: str = RECORDINGS_DIR
    retrieval_limit: int = RETRIEVAL_LIMIT

    @classmethod
    def create(
        cls,
        sample_rate: int = SAMPLE_RATE,
        recordings_dir: str = RECORDINGS_DIR,
        retrieval_limit: int = RETRIEVAL_LIMIT,
    ) -> "Container":
        """
        Create and initialize a Container instance with all required
        dependencies.

        Args:
            sample_rate (int): Audio sample rate in Hz (default: 16000)
            recordings_dir (str): Directory path for storing recordings
                (default: "recordings")
            retrieval_limit (int): Maximum number of memories to retrieve
                in searches (default: 2)

        Returns:
            Container: Fully initialized container with all service
                dependencies
        """
        # Transcriber
        transcriber = FasterWhisperTranscriber()
        transcriber.initialize()

        # Vector store
        embedding_model = Qwen3EmbeddingModel()
        text_chunker = SpacySentenceChunker()
        vector_store_repo: VectorStoreRepository = InMemoryQdrantVectorStoreRepository(
            embedding_model, text_chunker
        )
        vector_store_service = VectorStoreService(vector_store_repo)

        # LLM + RAG
        llm_model = Qwen3LlamaCppModel()
        rag_service = LLMRAGService(llm_model=llm_model)

        # Threshold filter service
        threshold_filter_service = ThresholdFilterService(relevance_threshold=0.0)

        # Persistence (fix swapped args)
        repository = FileRepository(
            storage_dir=recordings_dir,
            sample_rate=sample_rate,
        )
        persistence_service = PersistenceService(repository)

        return cls(
            vector_store=vector_store_service,
            persistence=persistence_service,
            rag=rag_service,
            threshold_filter=threshold_filter_service,
            transcriber=transcriber,
            sample_rate=sample_rate,
            recordings_dir=recordings_dir,
            retrieval_limit=retrieval_limit,
        )
