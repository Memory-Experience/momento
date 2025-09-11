import asyncio
import logging
import time
import uuid
from typing import Any

import grpc
import pandas as pd
from dataset.generation_metrics import GenerationMetrics
from dataset.marco_dataset import MSMarcoDataset
from dataset.retrieval_metrics import RetrievalMetrics
from protos.generated.py import stt_pb2, stt_pb2_grpc
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# gRPC Server configuration
SERVER_ADDRESS = "localhost:50051"
LIMIT_DOCS = 1_000


class RAGEvaluationClient:
    """Client for evaluating RAG system performance using gRPC."""

    def __init__(self, server_address: str):
        """Initialize the evaluation client.

        Args:
            server_address: gRPC server address (host:port)
        """
        self.server_address = server_address
        self.channel = None
        self.stub = None
        self.doc_to_memory: dict[str, str] = {}
        self.memory_to_doc: dict[str, str] = {}

    async def connect(self):
        """Establish connection to the gRPC server."""
        self.channel = grpc.aio.insecure_channel(self.server_address)
        self.stub = stt_pb2_grpc.TranscriptionServiceStub(self.channel)
        logger.info(f"Connected to server at {self.server_address}")

    async def close(self):
        """Close the gRPC channel."""
        if self.channel:
            await self.channel.close()
            logger.info("Disconnected from server")

    async def stream_memories(self, dataset, max_docs: int | None = None) -> int:
        """
        Stream documents as memories AND build a mapping:
        dataset_doc_id -> saved_memory_id and saved_memory_id -> dataset_doc_id
        """
        docs_df = dataset.docs
        total_docs = len(docs_df) if max_docs is None else min(max_docs, len(docs_df))

        logger.info(f"Streaming {total_docs} documents as memories...")
        success_count = 0

        # reset mapping every run
        self.doc_to_memory.clear()
        self.memory_to_doc.clear()

        for i, (_, doc) in enumerate(tqdm(docs_df.iterrows(), total=total_docs)):
            if max_docs is not None and i >= max_docs:
                break

            session_id = str(uuid.uuid4())
            dataset_doc_id = str(doc["id"])
            content = str(doc["content"])

            try:
                # bind into closure defaults to avoid B023
                async def memory_stream(
                    session_id: str = session_id,
                    dataset_doc_id: str = dataset_doc_id,
                    content: str = content,
                ):
                    # First chunk with metadata + content
                    yield stt_pb2.MemoryChunk(
                        text_data=content,
                        metadata=stt_pb2.ChunkMetadata(
                            session_id=session_id,
                            memory_id=dataset_doc_id,  # we tag with the dataset doc id
                            type=stt_pb2.ChunkType.MEMORY,
                            is_final=False,
                        ),
                    )
                    # NEW: explicit final marker to trigger saving on the server
                    yield stt_pb2.MemoryChunk(
                        text_data="",
                        metadata=stt_pb2.ChunkMetadata(
                            session_id=session_id,
                            memory_id=dataset_doc_id,
                            type=stt_pb2.ChunkType.MEMORY,
                            is_final=True,
                        ),
                    )

                saved_memory_id = None

                # Collect server responses and capture the NEW saved memory id
                async for response in self.stub.Transcribe(memory_stream()):
                    # Server will echo TRANSCRIPT chunks; ignore those.
                    # The confirmation we want is a MEMORY chunk with is_final=True.
                    if (
                        response.metadata.type == stt_pb2.ChunkType.MEMORY
                        and response.metadata.is_final
                    ):
                        saved_memory_id = response.metadata.memory_id

                if saved_memory_id:
                    self.doc_to_memory[dataset_doc_id] = saved_memory_id
                    self.memory_to_doc[saved_memory_id] = dataset_doc_id
                    success_count += 1

                    if i % 100 == 0 or logger.level == logging.DEBUG:
                        logger.info(
                            f"Streamed doc {i + 1}/{total_docs} "
                            f"(dataset_doc_id: {dataset_doc_id} "
                            f"-> memory_id: {saved_memory_id})"
                        )
                else:
                    logger.error(
                        "Did not receive saved memory "
                        f"id for dataset_doc_id={dataset_doc_id}"
                    )

            except Exception as e:
                logger.error(f"Error streaming document {dataset_doc_id}: {str(e)}")

        logger.info(
            f"Successfully streamed {success_count}/{total_docs} documents; "
            f"built mapping for {len(self.doc_to_memory)} docs"
        )
        return success_count

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

            async for response in self.stub.Transcribe(query_stream()):
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
    ) -> dict:
        """Evaluate retrieval performance using standard IR metrics.

        Args:
            retrieved_doc_ids: List of document IDs retrieved by the system
                (in rank order)
            relevant_doc_ids: List of document IDs known to be relevant
            relevance_scores: Optional mapping of doc_id to relevance score

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

        recall_at_3 = RetrievalMetrics.recall_at_k(
            retrieved_doc_ids, relevant_doc_ids, 3
        )
        recall_at_5 = RetrievalMetrics.recall_at_k(
            retrieved_doc_ids, relevant_doc_ids, 5
        )

        mrr = RetrievalMetrics.mean_reciprocal_rank(retrieved_doc_ids, relevant_doc_ids)

        # For NDCG, we need relevance scores
        ndcg_at_3 = RetrievalMetrics.ndcg_at_k(retrieved_doc_ids, relevance_scores, 3)
        ndcg_at_5 = RetrievalMetrics.ndcg_at_k(retrieved_doc_ids, relevance_scores, 5)

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "precision@1": precision_at_1,
            "precision@3": precision_at_3,
            "precision@5": precision_at_5,
            "recall@3": recall_at_3,
            "recall@5": recall_at_5,
            "mrr": mrr,
            "ndcg@3": ndcg_at_3,
            "ndcg@5": ndcg_at_5,
            "retrieved_count": len(retrieved_set),
            "relevant_count": len(relevant_set),
            "true_positives": true_positives,
        }

    # evaluate generation
    def evaluate_generation(
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

        return {
            "exact_match": em,
            "f1": f1,
            "rouge_l_f1": rouge_l,
            "answer_relevance": ans_rel,
            "support_coverage": faith["support_coverage"],
            "support_density": faith["support_density"],
            "hallucination_rate": faith["hallucination_rate"],
            "answer_len_tokens": len(GenerationMetrics._tokens(answer)),
        }

    async def run_evaluation(
        self,
        dataset,
        max_docs: int | None = None,
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
        # Step 1: Stream memories
        docs_count = await self.stream_memories(dataset, max_docs)

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
                "recall@3": 0,
                "recall@5": 0,
                "mrr": 0,
                "ndcg@3": 0,
                "ndcg@5": 0,
                "retrieved_count": 0,
                "relevant_count": 0,
                "true_positives": 0,
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
            },
            "response_times": [],
            "total_docs_streamed": docs_count,
        }

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
                self.memory_to_doc.get(mem_id, mem_id)
                for mem_id in retrieved_doc_ids_ordered
            ]

            # Evaluate retrieval
            retrieval_metrics = self.evaluate_retrieval(
                retrieved_doc_ids_for_eval,  # <- mapped to dataset doc IDs
                relevant_doc_ids,
                relevance_scores,
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
            generation_metrics = self.evaluate_generation(
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
                "  Precision: %.2f, Recall: %.2f, F1: %.2f",
                retrieval_metrics["precision"],
                retrieval_metrics["recall"],
                retrieval_metrics["f1"],
            )
            logger.info(
                "  P@1: %.2f, P@5: %.2f, MRR: %.2f",
                retrieval_metrics["precision@1"],
                retrieval_metrics["precision@5"],
                retrieval_metrics["mrr"],
            )
            logger.info(f"  Response time: {response_time:.2f}s")

        # Calculate averages for metrics
        if total_queries > 0:
            for metric in results["retrieval_metrics"]:
                results["retrieval_metrics"][metric] /= total_queries

            for metric in results["generation_metrics"]:
                results["generation_metrics"][metric] /= total_queries

            results["avg_response_time"] = sum(results["response_times"]) / len(
                results["response_times"]
            )

        return results


async def main():
    """Main evaluation function."""
    try:
        # Create MS MARCO dataset
        dataset = MSMarcoDataset(limit=LIMIT_DOCS)
        logger.info(f"Loaded dataset: {dataset}")

        # Initialize and connect client
        client = RAGEvaluationClient(SERVER_ADDRESS)
        await client.connect()

        # Run evaluation
        results = await client.run_evaluation(
            dataset, max_docs=LIMIT_DOCS, max_queries=20
        )

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

        # Close client
        await client.close()

    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
