import asyncio
import logging
import json

from datetime import datetime
from collections.abc import AsyncIterator

from api.dependency_container import Container
from api.transcription_servicer import TranscriptionServiceServicer
from api.persistence.persistence_service import PersistenceService
from api.persistence.repositories.in_memory_repository import InMemoryRepository
from api.rag.llm_rag_service import LLMRAGService
from api.models.transcription.faster_whisper_transcriber import FasterWhisperTranscriber
from api.models.llm.qwen3 import Qwen3
from api.models.embedding.embedding_model_interface import EmbeddingModel
from api.models.embedding.qwen3_embedding import Qwen3EmbeddingModel
from api.models.embedding.sbert_embedding import SBertEmbeddingModel
from api.rag.threshold_filter_service import ThresholdFilterService

from baseline.bm25_dataset_loader import BM25DatasetLoader
from baseline.memory_reciter_llm_model import MemoryReciterModel

from rag_evaluation_client import RAGEvaluationClient
from dataset.marco_dataset import MSMarcoDataset
from dataset.timeline_qa_dataset import TimelineQADataset
from dataset import dataset
from dataset_loader import DatasetLoader


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# gRPC Server configuration
LIMIT_DOCS = 100_000


async def baseline_configuration(dataset, dataset_dir) -> RAGEvaluationClient:
    (
        vector_store_service,
        doc_to_memory,
        memory_to_doc,
    ) = await BM25DatasetLoader.create_filled_vector_store_service(dataset, dataset_dir)

    transcriber = FasterWhisperTranscriber()
    transcriber.initialize()

    llm_model = MemoryReciterModel()
    rag_service = LLMRAGService(llm_model=llm_model)

    threshold_filter_service = ThresholdFilterService(relevance_threshold=0)
    repository = InMemoryRepository()
    persistence_service = PersistenceService(repository)

    container = Container(
        threshold_filter=threshold_filter_service,
        vector_store=vector_store_service,
        persistence=persistence_service,
        rag=rag_service,
        transcriber=transcriber,
        retrieval_limit=20,
    )

    servicer = TranscriptionServiceServicer(container)
    return RAGEvaluationClient(servicer, doc_to_memory, memory_to_doc)


async def momento_configuration(
    dataset, dataset_dir, embedding: EmbeddingModel
) -> RAGEvaluationClient:
    (
        vector_store_service,
        doc_to_memory,
        memory_to_doc,
    ) = await DatasetLoader.create_filled_vector_store_service(
        dataset, dataset_dir, embedding
    )

    transcriber = FasterWhisperTranscriber()
    transcriber.initialize()

    llm_model = Qwen3()
    rag_service = LLMRAGService(llm_model=llm_model)

    threshold_filter_service = ThresholdFilterService(relevance_threshold=0)
    repository = InMemoryRepository()
    persistence_service = PersistenceService(repository)

    container = Container(
        threshold_filter=threshold_filter_service,
        vector_store=vector_store_service,
        persistence=persistence_service,
        rag=rag_service,
        transcriber=transcriber,
        retrieval_limit=20,
    )

    servicer = TranscriptionServiceServicer(container)
    return RAGEvaluationClient(servicer, doc_to_memory, memory_to_doc)


async def dataset_configurations() -> AsyncIterator[
    tuple[str, dataset.DataFrameDataset, RAGEvaluationClient]
]:
    """
    dataset_dir = "runs/timeline_qa_sparse_baseline"
    timeline_qa_sparse = TimelineQADataset.generate(category=0)

    yield (
        dataset_dir,
        timeline_qa_sparse,
        await baseline_configuration(timeline_qa_sparse, dataset_dir),
    )

    dataset_dir = "runs/timeline_qa_medium_baseline"
    timeline_qa_medium = TimelineQADataset.generate(category=1)

    yield (
        dataset_dir,
        timeline_qa_medium,
        await baseline_configuration(timeline_qa_medium, dataset_dir),
    )

    dataset_dir = "runs/timeline_qa_dense_baseline"
    timeline_qa_dense = TimelineQADataset.generate(category=2)

    yield (
        dataset_dir,
        timeline_qa_dense,
        await baseline_configuration(timeline_qa_dense, dataset_dir),
    )
    
    dataset_dir = "runs/ms_marco_small_baseline"
    ms_marco_small = MSMarcoDataset(limit=LIMIT_DOCS)

    yield (
        dataset_dir,
        ms_marco_small,
        await baseline_configuration(ms_marco_small, dataset_dir),
    )

    dataset_dir = "runs/timeline_qa_sparse_momento"

    qwen3_embedding = Qwen3EmbeddingModel()
    yield (
        dataset_dir,
        timeline_qa_sparse,
        await momento_configuration(timeline_qa_sparse, dataset_dir, qwen3_embedding),
    )
"""
    dataset_dir = "runs/ms_marco_qna_dev_momento"
    ms_marco_small = MSMarcoDataset(limit=100)
    qwen3_embedding = Qwen3EmbeddingModel()
    yield (
        dataset_dir,
        ms_marco_small,
        await momento_configuration(ms_marco_small, dataset_dir, qwen3_embedding),
    )
    """
    dataset_dir = "runs/timeline_qa_medium_momento"

    yield (
        dataset_dir,
        timeline_qa_medium,
        await momento_configuration(timeline_qa_medium, dataset_dir, qwen3_embedding),
    )

    # Too big of a dataset
    # dataset_dir = "runs/timeline_qa_dense_momento"

    # yield (
    #    dataset_dir,
    #    timeline_qa_dense,
    #    await momento_configuration(timeline_qa_dense, dataset_dir, qwen3_embedding),
    # )

    dataset_dir = "runs/timeline_qa_sparse_sbert"

    sbert_embedding = SBertEmbeddingModel()
    yield (
        dataset_dir,
        timeline_qa_sparse,
        await momento_configuration(timeline_qa_sparse, dataset_dir, sbert_embedding),
    )

    dataset_dir = "runs/ms_marco_small_sbert"

    yield (
        dataset_dir,
        ms_marco_small,
        await momento_configuration(ms_marco_small, dataset_dir, sbert_embedding),
    )

    dataset_dir = "runs/timeline_qa_medium_sbert"

    yield (
        dataset_dir,
        timeline_qa_medium,
        await momento_configuration(timeline_qa_medium, dataset_dir, sbert_embedding),
    )"""


async def main():
    """Main evaluation function."""
    try:
        async for dataset_dir, dataset, client in dataset_configurations():
            # Run evaluation
            results = await client.run_evaluation(dataset, max_queries=10)

            # Create a timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"{dataset_dir}_{timestamp}.json"

            # Dump results to JSON file
            with open(results_file, "w") as f:
                json.dump(results, f, indent=4)

            logger.info(f"Results saved to {results_file}")

            # Print summary results
            logger.info(f"\n===== EVALUATION RESULTS FOR {dataset_dir} =====")
            logger.info(f"Documents streamed: {results['total_docs_streamed']}")
            logger.info(f"Queries evaluated: {len(results['queries'])}")
            logger.info(f"Average response time: {results['avg_response_time']:.2f}s")
            logger.info("\nRetrieval Performance:")
            logger.info(f"  Precision: {results['retrieval_metrics']['precision']:.4f}")
            logger.info(f"  Recall: {results['retrieval_metrics']['recall']:.4f}")
            logger.info(f"  F1 Score: {results['retrieval_metrics']['f1']:.4f}")
            logger.info(f"  P@1: {results['retrieval_metrics']['precision@1']:.4f}")
            logger.info(f"  P@5: {results['retrieval_metrics']['precision@5']:.4f}")
            logger.info(f"  R@5: {results['retrieval_metrics']['recall@5']:.4f}")
            logger.info(f"  MRR: {results['retrieval_metrics']['mrr']:.4f}")
            logger.info(f"  NDCG@5: {results['retrieval_metrics']['ndcg@5']:.4f}")
            logger.info(f"  MAP: {results['retrieval_metrics']['map']:.4f}")
            logger.info(f"  AQWV: {results['retrieval_metrics']['aqwv']:.4f}")

            logger.info("\nGeneration Performance (averages):")
            gm = results["generation_metrics"]
            logger.info(
                "  EM: %.4f  F1: %.4f  ROUGE-L(F1): %.4f",
                gm["exact_match"],
                gm["f1"],
                gm["rouge_l_f1"],
            )
            logger.info(
                "  Answer↔Query Relevance (F1): %.4f",
                gm["answer_relevance"],
            )
            logger.info(
                "  Faithfulness — Support Coverage: %.4f  Density: %.4f",
                gm["support_coverage"],
                gm["support_density"],
            )
            logger.info(f"  Hallucination Rate: {gm['hallucination_rate']:.4f}")

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
