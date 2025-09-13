from dataclasses import dataclass

from api.persistence.persistence_service import PersistenceService
from api.persistence.repositories.file_repository import FileRepository
from api.rag.llm_rag_service import LLMRAGService
from api.rag.threshold_filter_service import ThresholdFilterService
from api.vector_store.repositories.qdrant_vector_store_repository import (
    InMemoryQdrantVectorStoreRepository,
)
from api.vector_store.repositories.vector_store_repository_interface import (
    VectorStoreRepository,
)
from api.vector_store.vector_store_service import VectorStoreService

from api.models.spacy_sentence_chunker import SpacySentenceChunker
from api.models.llm.qwen3 import Qwen3
from api.models.embedding.qwen3_embedding import Qwen3EmbeddingModel
from api.models.transcription.faster_whisper_transcriber import FasterWhisperTranscriber


RECORDINGS_DIR = "recordings"
SAMPLE_RATE = 16000


@dataclass
class Container:
    """Single argument you pass to both FastAPI and gRPC servicer."""

    vector_store: VectorStoreService
    persistence: PersistenceService
    rag: LLMRAGService
    threshold_filter: ThresholdFilterService
    transcriber: FasterWhisperTranscriber

    sample_rate: int = SAMPLE_RATE
    recordings_dir: str = RECORDINGS_DIR

    @classmethod
    def create(
        cls,
        sample_rate: int = SAMPLE_RATE,
        recordings_dir: str = RECORDINGS_DIR,
    ) -> "Container":
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
        llm_model = Qwen3()
        rag_service = LLMRAGService(llm_model=llm_model)

        # Threshold filter service
        threshold_filter_service = ThresholdFilterService(relevance_threshold=0.7)

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
        )
