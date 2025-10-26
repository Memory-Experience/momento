from __future__ import annotations
import asyncio
import logging
import json
import os

import multiprocessing as mp
from datetime import datetime
from collections.abc import AsyncIterator

from typing import AsyncIterator
import torch

from api.dependency_container import Container
from api.question_answer_service import QuestionAnswerService
from api.persistence.persistence_service import PersistenceService
from api.persistence.repositories.in_memory_repository import InMemoryRepository
from api.rag.llm_rag_service import LLMRAGService
from api.models.embedding.embedding_model_interface import EmbeddingModel
from api.models.embedding.sbert_embedding import SBertEmbeddingModel
from api.models.llm.qwen3_transformers_model import Qwen3TransformersModel
from api.rag.threshold_filter_service import ThresholdFilterService

from evaluation.baseline.bm25_dataset_loader import BM25DatasetLoader
from evaluation.baseline.memory_reciter_llm_model import MemoryReciterModel
from evaluation.baseline.dummy_transcriber import DummyTranscriber

from evaluation.rag_evaluation_client import RAGEvaluationClient
from evaluation.dataset.marco_dataset import MSMarcoDataset
from evaluation.dataset.timeline_qa_dataset import TimelineQADataset
from evaluation.dataset.open_lifelog_qa_dataset import OpenLifelogQADataset
from evaluation.dataset import dataset
from evaluation.dataset_loader import DatasetLoader


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# gRPC Server configuration
LIMIT_DOCS = 10_000_000
MAX_QUERIES = 10_000
BATCH_SIZE = 4096
DB_BATCH_SIZE = 512


async def baseline_configuration(dataset, dataset_dir) -> RAGEvaluationClient:
    (
        vector_store_service,
        doc_to_memory,
        memory_to_doc,
    ) = await BM25DatasetLoader.create_filled_vector_store_service(dataset, dataset_dir)

    transcriber = DummyTranscriber()

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

    question_anser_service = QuestionAnswerService(container)
    return RAGEvaluationClient(question_anser_service, doc_to_memory, memory_to_doc)


async def momento_configuration(
    dataset, dataset_dir, embedding: EmbeddingModel, batch_size: int = 128, db_batch_size: int = 256
) -> RAGEvaluationClient:
    (
        vector_store_service,
        doc_to_memory,
        memory_to_doc,
    ) = await DatasetLoader.create_filled_vector_store_service(
        dataset, dataset_dir, embedding, batch_size=batch_size, db_batch_size=db_batch_size
    )

    transcriber = DummyTranscriber()

    llm_model = Qwen3TransformersModel()
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

    question_anser_service = QuestionAnswerService(container)
    return RAGEvaluationClient(question_anser_service, doc_to_memory, memory_to_doc)


async def dataset_configurations() -> AsyncIterator[
    tuple[str, dataset.DataFrameDataset, RAGEvaluationClient]
]:   
    qwen3_embedding_06B = SBertEmbeddingModel(
        model_name="Qwen/Qwen3-Embedding-0.6B",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    sbert_embedding = SBertEmbeddingModel(device="cuda" if torch.cuda.is_available() else "cpu")
    
    ####################
    # MS-Marco Passage #
    ####################

    ms_marco_dev = MSMarcoDataset(limit=LIMIT_DOCS, dataset_name="msmarco-passage/dev/small")

    dataset_dir = "runs/ms_marco_passage_dev_baseline"
    yield (
        dataset_dir,
        ms_marco_dev,
        await baseline_configuration(ms_marco_dev, dataset_dir),
    )

    dataset_dir = "runs/ms_marco_passage_dev_qwen3_0.6B"
    yield (
        dataset_dir,
        ms_marco_dev,
        await momento_configuration(ms_marco_dev, dataset_dir, qwen3_embedding_06B, BATCH_SIZE),
    )

    dataset_dir = "runs/ms_marco_passage_dev_sbert"
    yield (
        dataset_dir,
        ms_marco_dev,
        await momento_configuration(ms_marco_dev, dataset_dir, sbert_embedding, BATCH_SIZE),
    )
    
    ################
    # MS-Marco QNA #
    ################

    ms_marco_qna = MSMarcoDataset(limit=LIMIT_DOCS)

    dataset_dir = "runs/ms_marco_qna_full_dev_baseline"
    yield (
        dataset_dir,
        ms_marco_qna,
        await baseline_configuration(ms_marco_qna, dataset_dir),
    )

    dataset_dir = "runs/ms_marco_qna_full_dev_qwen3_0.6B"
    yield (
        dataset_dir,
        ms_marco_qna,
        await momento_configuration(ms_marco_qna, dataset_dir, qwen3_embedding_06B, BATCH_SIZE, DB_BATCH_SIZE),
    )

    dataset_dir = "runs/ms_marco_qna_full_dev_sbert"
    yield (
        dataset_dir,
        ms_marco_qna,
        await momento_configuration(ms_marco_qna, dataset_dir, sbert_embedding, BATCH_SIZE, DB_BATCH_SIZE),
    )

    #################
    # OpenLifelogQA #
    #################

    openlifelog_qa = OpenLifelogQADataset()
    
    dataset_dir = "runs/openlifelog_qa_baseline"
    yield (
        dataset_dir,
        openlifelog_qa,
        await baseline_configuration(openlifelog_qa, dataset_dir),
    )

    dataset_dir = "runs/openlifelog_qa_qwen3_0.6B"
    yield (
        dataset_dir,
        openlifelog_qa,
        await momento_configuration(openlifelog_qa, dataset_dir, qwen3_embedding_06B, BATCH_SIZE, DB_BATCH_SIZE),
    )

    dataset_dir = "runs/openlifelog_qa_sbert"
    yield (
        dataset_dir,
        openlifelog_qa,
        await momento_configuration(openlifelog_qa, dataset_dir, sbert_embedding, BATCH_SIZE, DB_BATCH_SIZE),
    )

    ##############
    # TimelineQA #
    ##############

    timeline_qa_sparse = TimelineQADataset.generate(category=0)
    
    # Baseline configurations
    dataset_dir = "runs/timeline_qa_sparse_baseline"
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

    # Momento configurations
    dataset_dir = "runs/timeline_qa_sparse_qwen3_0.6B"

    yield (
        dataset_dir,
        timeline_qa_sparse,
        await momento_configuration(timeline_qa_sparse, dataset_dir, qwen3_embedding_06B, BATCH_SIZE, DB_BATCH_SIZE),
    )

    dataset_dir = "runs/timeline_qa_medium_qwen3_0.6B"

    yield (
        dataset_dir,
        timeline_qa_medium,
        await momento_configuration(timeline_qa_medium, dataset_dir, qwen3_embedding_06B, BATCH_SIZE, DB_BATCH_SIZE),
    )

    # SBert configurations
    dataset_dir = "runs/timeline_qa_sparse_sbert"

    yield (
        dataset_dir,
        timeline_qa_sparse,
        await momento_configuration(timeline_qa_sparse, dataset_dir, sbert_embedding, BATCH_SIZE, DB_BATCH_SIZE),
    )

    dataset_dir = "runs/timeline_qa_medium_sbert"

    yield (
        dataset_dir,
        timeline_qa_medium,
        await momento_configuration(timeline_qa_medium, dataset_dir, sbert_embedding, BATCH_SIZE, DB_BATCH_SIZE),
    )

    # Too big of a dataset
    dataset_dir = "runs/timeline_qa_dense_qwen3_0.6B"

    yield (
        dataset_dir,
        timeline_qa_dense,
        await momento_configuration(timeline_qa_dense, dataset_dir, qwen3_embedding_06B, BATCH_SIZE, DB_BATCH_SIZE),
    )

    # Too big of a dataset
    dataset_dir = "runs/timeline_qa_dense_sbert"

    yield (
        dataset_dir,
        timeline_qa_dense,
        await momento_configuration(timeline_qa_dense, dataset_dir, sbert_embedding, BATCH_SIZE, DB_BATCH_SIZE),
    )


async def main():
    """Main evaluation function."""
    print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    try:
        async for dataset_dir, dataset, client in dataset_configurations():
            # Run evaluation
            results = await client.run_evaluation(dataset, max_queries=MAX_QUERIES)

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
    mp.set_start_method("spawn", force=True)
    asyncio.run(main())


