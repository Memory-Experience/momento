import logging
import time
import uuid
from typing import Any
from tqdm import tqdm

import pandas as pd

from protos.generated.py import stt_pb2

from api.transcription_servicer import TranscriptionServiceServicer
from api.models.embedding.embedding_model_interface import EmbeddingModel
from api.models.embedding.sbert_embedding import SBertEmbeddingModel

from dataset.dataset import DataFrameDataset
from metrics.generation_metrics import GenerationMetrics
from metrics.retrieval_metrics import RetrievalMetrics
from metrics.cross_encoder_scorer import CrossEncoderScorer


# Configure logging
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RAGEvaluationClient:
    """Client for evaluating RAG system performance using gRPC."""

    def __init__(
        self,
        servicer: TranscriptionServiceServicer,
        doc_to_memory: dict[str, str],
        memory_to_doc: dict[str, str],
    ):
        """Initialize the evaluation client.

        Args:
            server_address: gRPC server address (host:port)
        """
        self.servicer = servicer
        self.doc_to_memory: dict[str, str] = doc_to_memory or {}
        self.memory_to_doc: dict[str, str] = memory_to_doc or {}

    async def process_query(
        self,
        query_id: str,
        query_text: str,
    ) -> tuple[dict, list[str], str, float]:
        session_id = str(uuid.uuid4())
        retrieved_docs: dict[str, str] = {}
        retrieved_doc_ids_ordered: list[str] = []
        full_answer = ""
        start_time = time.time()

        try:

            async def query_stream(
                session_id: str = session_id,
                query_id: str = query_id,
                query_text: str = query_text,
            ):
                # Send question text
                yield stt_pb2.MemoryChunk(
                    text_data=query_text,
                    metadata=stt_pb2.ChunkMetadata(
                        session_id=session_id,
                        memory_id=query_id,
                        type=stt_pb2.ChunkType.QUESTION,
                        is_final=False,
                    ),
                )
                # NEW: explicit final marker to trigger answer generation
                yield stt_pb2.MemoryChunk(
                    text_data="",
                    metadata=stt_pb2.ChunkMetadata(
                        session_id=session_id,
                        memory_id=query_id,
                        type=stt_pb2.ChunkType.QUESTION,
                        is_final=True,
                    ),
                )

            answer_chunks: list[str] = []

            async for response in self.servicer.Transcribe(query_stream(), "context"):
                if response.metadata.type == stt_pb2.ChunkType.MEMORY:
                    # Server sends retrieved memories (by *saved* memory_id)
                    mem_id = response.metadata.memory_id
                    retrieved_docs[mem_id] = response.text_data
                    retrieved_doc_ids_ordered.append(mem_id)
                elif response.metadata.type == stt_pb2.ChunkType.ANSWER:
                    answer_chunks.append(response.text_data)

            full_answer = " ".join(answer_chunks)

        except Exception as e:
            logger.error(f"Error processing query {query_id}: {str(e)}")

        response_time = time.time() - start_time
        return retrieved_docs, retrieved_doc_ids_ordered, full_answer, response_time

    def evaluate_retrieval(
        self,
        retrieved_doc_ids: list[str],
        relevant_doc_ids: list[str],
        relevance_scores: dict[str, float] | None = None,
        collection_size: int = -1,
    ) -> dict:
        """Evaluate retrieval performance using standard IR metrics.

        Args:
            retrieved_doc_ids: List of document IDs retrieved by the system
                (in rank order)
            relevant_doc_ids: List of document IDs known to be relevant
            relevance_scores: Optional mapping of doc_id to relevance score
            collection_size: Total size of document collection for AQWV

        Returns:
            Dictionary of retrieval metrics
        """
        if not relevant_doc_ids:
            return {
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "precision@1": 0,
                "precision@3": 0,
                "precision@5": 0,
                "recall@3": 0,
                "recall@5": 0,
                "mrr": 0,
                "ndcg@3": 0,
                "ndcg@5": 0,
                "aqwv": RetrievalMetrics.aqwv(
                    retrieved_doc_ids,
                    relevant_doc_ids,
                    beta=40.0,
                    collection_size=collection_size,
                ),
            }

        # Get binary relevance scores if not provided
        if relevance_scores is None:
            relevance_scores = {doc_id: 1.0 for doc_id in relevant_doc_ids}

        # Calculate standard set-based metrics
        retrieved_set = set(retrieved_doc_ids)
        relevant_set = set(relevant_doc_ids)

        true_positives = len(retrieved_set.intersection(relevant_set))
        precision = true_positives / len(retrieved_set) if retrieved_set else 0
        recall = true_positives / len(relevant_set) if relevant_set else 0
        f1 = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall)
            else 0
        )

        # Calculate rank-aware metrics using the metrics library
        precision_at_1 = RetrievalMetrics.precision_at_k(
            retrieved_doc_ids, relevant_doc_ids, 1
        )
        precision_at_3 = RetrievalMetrics.precision_at_k(
            retrieved_doc_ids, relevant_doc_ids, 3
        )
        precision_at_5 = RetrievalMetrics.precision_at_k(
            retrieved_doc_ids, relevant_doc_ids, 5
        )
        precision_at_10 = RetrievalMetrics.precision_at_k(
            retrieved_doc_ids, relevant_doc_ids, 10
        )
        precision_at_20 = RetrievalMetrics.precision_at_k(
            retrieved_doc_ids, relevant_doc_ids, 20
        )

        recall_at_1 = RetrievalMetrics.recall_at_k(
            retrieved_doc_ids, relevant_doc_ids, 1
        )
        recall_at_3 = RetrievalMetrics.recall_at_k(
            retrieved_doc_ids, relevant_doc_ids, 3
        )
        recall_at_5 = RetrievalMetrics.recall_at_k(
            retrieved_doc_ids, relevant_doc_ids, 5
        )
        recall_at_10 = RetrievalMetrics.recall_at_k(
            retrieved_doc_ids, relevant_doc_ids, 10
        )
        recall_at_20 = RetrievalMetrics.recall_at_k(
            retrieved_doc_ids, relevant_doc_ids, 20
        )

        mrr = RetrievalMetrics.mean_reciprocal_rank(retrieved_doc_ids, relevant_doc_ids)

        mrr_at_1 = RetrievalMetrics.mean_reciprocal_rank(
            retrieved_doc_ids[:1], relevant_doc_ids
        )
        mrr_at_3 = RetrievalMetrics.mean_reciprocal_rank(
            retrieved_doc_ids[:3], relevant_doc_ids
        )
        mrr_at_5 = RetrievalMetrics.mean_reciprocal_rank(
            retrieved_doc_ids[:5], relevant_doc_ids
        )
        mrr_at_10 = RetrievalMetrics.mean_reciprocal_rank(
            retrieved_doc_ids[:10], relevant_doc_ids
        )
        mrr_at_20 = RetrievalMetrics.mean_reciprocal_rank(
            retrieved_doc_ids[:20], relevant_doc_ids
        )

        # For NDCG, we need relevance scores
        ndcg_at_1 = RetrievalMetrics.ndcg_at_k(retrieved_doc_ids, relevance_scores, 1)
        ndcg_at_3 = RetrievalMetrics.ndcg_at_k(retrieved_doc_ids, relevance_scores, 3)
        ndcg_at_5 = RetrievalMetrics.ndcg_at_k(retrieved_doc_ids, relevance_scores, 5)
        ndcg_at_10 = RetrievalMetrics.ndcg_at_k(retrieved_doc_ids, relevance_scores, 10)
        ndcg_at_20 = RetrievalMetrics.ndcg_at_k(retrieved_doc_ids, relevance_scores, 20)

        # Calculate AQWV
        aqwv = RetrievalMetrics.aqwv(
            retrieved_doc_ids,
            relevant_doc_ids,
            beta=40.0,
            collection_size=collection_size,
        )

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision@1": precision_at_1,
            "precision@3": precision_at_3,
            "precision@5": precision_at_5,
            "precision@10": precision_at_10,
            "precision@20": precision_at_20,
            "recall@1": recall_at_1,
            "recall@3": recall_at_3,
            "recall@5": recall_at_5,
            "recall@10": recall_at_10,
            "recall@20": recall_at_20,
            "mrr": mrr,
            "mrr@1": mrr_at_1,
            "mrr@3": mrr_at_3,
            "mrr@5": mrr_at_5,
            "mrr@10": mrr_at_10,
            "mrr@20": mrr_at_20,
            "ndcg@1": ndcg_at_1,
            "ndcg@3": ndcg_at_3,
            "ndcg@5": ndcg_at_5,
            "ndcg@10": ndcg_at_10,
            "ndcg@20": ndcg_at_20,
            "retrieved_count": len(retrieved_set),
            "relevant_count": len(relevant_set),
            "true_positives": true_positives,
            "aqwv": aqwv,
        }

    # evaluate generation
    async def evaluate_generation(
        self,
        answer: str,
        query_text: str,
        gold_answers: list[str] | None,
        retrieved_doc_ids_ordered: list[str],
        retrieved_docs: dict[str, str],
        top_k_docs_for_faithfulness: int = 5,
    ) -> dict[str, Any]:
        """Evaluate generation along correctness, relevance and faithfulness.

        Args:
            answer: Generated answer to evaluate
            query_text: Original query text
            gold_answers: List of reference answers (if available)
            retrieved_doc_ids_ordered: List of retrieved document IDs in rank
                order
            retrieved_docs: Dictionary mapping document IDs to their content
            top_k_docs_for_faithfulness: Number of top docs to use for
                faithfulness evaluation

        Returns:
            Dictionary of generation evaluation metrics
        """
        gold_answers = gold_answers or []

        em = (
            GenerationMetrics.exact_match(answer, gold_answers) if gold_answers else 0.0
        )
        f1 = GenerationMetrics.f1(answer, gold_answers) if gold_answers else 0.0
        rouge_l = (
            GenerationMetrics.rouge_l_f1(answer, gold_answers) if gold_answers else 0.0
        )

        ans_rel = GenerationMetrics.answer_relevance_to_query(answer, query_text)

        faith = GenerationMetrics.faithfulness_signals(
            answer,
            retrieved_doc_ids_ordered,
            retrieved_docs,
            top_k_docs=top_k_docs_for_faithfulness,
        )

        sbert_model: EmbeddingModel = SBertEmbeddingModel()
        sbert_similarity = await GenerationMetrics.sbert_similarity(
            answer, gold_answers, sbert_model
        )

        cross_encoder_model = CrossEncoderScorer(normalize="sigmoid")
        cross_encoder_similarity = await GenerationMetrics.cross_encoder_similarity(
            answer, gold_answers, cross_encoder_model
        )

        return {
            "exact_match": em,
            "f1": f1,
            "rouge_l_f1": rouge_l,
            "answer_relevance": ans_rel,
            "support_coverage": faith["support_coverage"],
            "support_density": faith["support_density"],
            "hallucination_rate": faith["hallucination_rate"],
            "answer_len_tokens": len(GenerationMetrics._tokens(answer)),
            "sbert_similarity": sbert_similarity,
            "cross_encoder_similarity": cross_encoder_similarity,
        }

    async def run_evaluation(
        self,
        dataset: DataFrameDataset,
        max_queries: int | None = None,
    ) -> dict:
        """Run full evaluation workflow.

        Args:
            dataset: Dataset to use for evaluation
            max_docs: Maximum number of documents to stream
            max_queries: Maximum number of queries to evaluate

        Returns:
            Dictionary of evaluation results
        """

        # Step 2: Process queries and evaluate
        queries_df = dataset.queries
        qrels_df = dataset.qrels

        total_queries = (
            len(queries_df)
            if max_queries is None
            else min(max_queries, len(queries_df))
        )
        logger.info(f"Evaluating {total_queries} queries...")

        # Updated metrics structure to include all the new metrics
        results = {
            "queries": [],
            "retrieval_metrics": {
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "precision@1": 0,
                "precision@3": 0,
                "precision@5": 0,
                "precision@10": 0,
                "precision@20": 0,
                "recall@1": 0,
                "recall@3": 0,
                "recall@5": 0,
                "recall@10": 0,
                "recall@20": 0,
                "mrr": 0,
                "mrr@1": 0,
                "mrr@3": 0,
                "mrr@5": 0,
                "mrr@10": 0,
                "mrr@20": 0,
                "ndcg@1": 0,
                "ndcg@3": 0,
                "ndcg@5": 0,
                "ndcg@10": 0,
                "ndcg@20": 0,
                "retrieved_count": 0,
                "relevant_count": 0,
                "true_positives": 0,
                "aqwv": 0,
                "map": 0,
            },
            "generation_metrics": {
                "exact_match": 0,
                "f1": 0,
                "rouge_l_f1": 0,
                "answer_relevance": 0,
                "support_coverage": 0,
                "support_density": 0,
                "hallucination_rate": 0,
                "answer_len_tokens": 0,
                "sbert_similarity": 0,
                "cross_encoder_similarity": 0,
            },
            "response_times": [],
            "total_docs_streamed": len(dataset.docs),
        }

        # Get collection size for AQWV calculation
        collection_size = len(dataset.docs)

        # Store data for MAP calculation (per-query lists)
        retrieved_docs_per_query = []
        relevant_docs_per_query = []

        for i, (_, query) in enumerate(
            tqdm(queries_df.iterrows(), total=total_queries)
        ):
            if max_queries is not None and i >= max_queries:
                break

            query_id = str(query["id"])
            query_text = str(query["text"])

            # Get relevant documents and relevance scores for this query
            relevant_qrels = qrels_df[qrels_df["query_id"].astype(str) == query_id]
            relevant_doc_ids = relevant_qrels["doc_id"].astype(str).tolist()

            # Create relevance scores dictionary
            relevance_scores = dict(
                zip(
                    relevant_qrels["doc_id"].astype(str),
                    relevant_qrels["relevance"].astype(float),
                    strict=False,
                )
            )

            # Process query
            (
                retrieved_docs,
                retrieved_doc_ids_ordered,
                answer,
                response_time,
            ) = await self.process_query(query_id, query_text)

            retrieved_doc_ids_for_eval = [
                self.memory_to_doc.get(uuid.UUID(mem_id))
                for mem_id in retrieved_doc_ids_ordered
            ]

            # Store for MAP calculation
            retrieved_docs_per_query.append(retrieved_doc_ids_for_eval)
            relevant_docs_per_query.append(relevant_doc_ids)

            # Evaluate retrieval
            retrieval_metrics = self.evaluate_retrieval(
                retrieved_doc_ids_for_eval,  # <- mapped to dataset doc IDs
                relevant_doc_ids,
                relevance_scores,
                collection_size=collection_size,
            )

            # Evaluate generation (using gold answers if available)
            # NOTE: pull actual answer text from queries, NOT doc IDs
            gold_answers: list[str] = []
            if "answers" in queries_df.columns and pd.notna(query.get("answers")):
                val = query["answers"]
                if isinstance(val, list | tuple):
                    gold_answers = [str(x) for x in val]
                else:
                    gold_answers = [str(val)]
            elif "answer" in queries_df.columns and pd.notna(query.get("answer")):
                gold_answers = [str(query["answer"])]
            generation_metrics = await self.evaluate_generation(
                answer,
                query_text,
                gold_answers,
                retrieved_doc_ids_ordered,
                retrieved_docs,
            )

            query_result = {
                "query_id": query_id,
                "query_text": query_text,
                "retrieved_docs": retrieved_docs,  # keyed by memory_id
                "retrieved_doc_ids_ordered": retrieved_doc_ids_ordered,
                "retrieved_doc_ids_for_eval": retrieved_doc_ids_for_eval,
                "answer": answer,
                "response_time": response_time,
                "retrieval_metrics": retrieval_metrics,
                "generation_metrics": generation_metrics,
            }
            results["queries"].append(query_result)
            results["response_times"].append(response_time)

            # Update aggregated retrieval metrics
            for metric in results["retrieval_metrics"]:
                if metric in retrieval_metrics:
                    results["retrieval_metrics"][metric] += retrieval_metrics[metric]

            # Update aggregated generation metrics
            for metric in results["generation_metrics"]:
                if metric in generation_metrics:
                    results["generation_metrics"][metric] += generation_metrics[metric]

            logger.info(
                f"Processed query {i + 1}/{total_queries}: {query_text[:50]}..."
            )
            logger.info(
                "  Precision: %.2f, Recall: %.2f, F1: %.2f, AQWV: %.2f",
                retrieval_metrics["precision"],
                retrieval_metrics["recall"],
                retrieval_metrics["f1"],
                retrieval_metrics["aqwv"],
            )
            logger.info(
                "  P@1: %.2f, P@5: %.2f, MRR: %.2f",
                retrieval_metrics["precision@1"],
                retrieval_metrics["precision@5"],
                retrieval_metrics["mrr"],
            )
            logger.info(f"  Response time: {response_time:.2f}s")

        # Calculate MAP across all queries (global corpus-level metric)
        map_score = RetrievalMetrics.mean_average_precision(
            retrieved_docs_per_query, relevant_docs_per_query
        )
        results["retrieval_metrics"]["map"] = map_score

        # Calculate averages for metrics
        if total_queries > 0:
            for metric in results["retrieval_metrics"]:
                if metric != "map":  # MAP is already calculated as global metric
                    results["retrieval_metrics"][metric] /= total_queries

            for metric in results["generation_metrics"]:
                results["generation_metrics"][metric] /= total_queries

            results["avg_response_time"] = sum(results["response_times"]) / len(
                results["response_times"]
            )

        return results
