import asyncio
import logging
import json

from datetime import datetime

from api.dependency_container import Container
from api.transcription_servicer import TranscriptionServiceServicer
from api.persistence.persistence_service import PersistenceService
from api.persistence.repositories.in_memory_repository import InMemoryRepository
from api.rag.llm_rag_service import LLMRAGService
from api.models.transcription.faster_whisper_transcriber import FasterWhisperTranscriber
from api.models.llm.qwen3 import Qwen3
from api.rag.threshold_filter_service import ThresholdFilterService

from dataset_loader import DatasetLoader
from rag_evaluation_client import RAGEvaluationClient
from dataset.marco_dataset import MSMarcoDataset


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# gRPC Server configuration
LIMIT_DOCS = 10_000


async def main():
    """Main evaluation function."""
    try:
        # Create MS MARCO dataset
        dataset_dir = "qdrant_ms_marco_small"
        dataset = MSMarcoDataset(limit=LIMIT_DOCS)
        logger.info(f"Loaded dataset: {dataset}")

        transcriber = FasterWhisperTranscriber()
        transcriber.initialize()

        (
            vector_store_service,
            doc_to_memory,
            memory_to_doc,
        ) = await DatasetLoader.create_filled_vector_store_service(dataset, dataset_dir)

        # LLM + RAG
        llm_model = Qwen3()
        rag_service = LLMRAGService(llm_model=llm_model)

        # Threshold filter service
        threshold_filter_service = ThresholdFilterService(relevance_threshold=0.7)

        # Persistence
        repository = InMemoryRepository()
        persistence_service = PersistenceService(repository)

        container = Container(
            threshold_filter=threshold_filter_service,
            vector_store=vector_store_service,
            persistence=persistence_service,
            rag=rag_service,
            transcriber=transcriber,
        )

        servicer = TranscriptionServiceServicer(container)
        # Initialize and connect client
        client = RAGEvaluationClient(servicer, doc_to_memory, memory_to_doc)

        # Run evaluation
        results = await client.run_evaluation(dataset, max_queries=20)

        # Create a timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"{dataset_dir}/evaluation_results_{timestamp}.json"

        # Dump results to JSON file
        with open(results_file, "w") as f:
            json.dump(results, f, indent=4)

        logger.info(f"Results saved to {results_file}")

        # Print summary results
        logger.info("\n===== EVALUATION RESULTS =====")
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
