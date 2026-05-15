import json
import logging
from collections import defaultdict

from evaluation.rag_evaluation_client import RAGEvaluationClient
from evaluation.dataset.marco_dataset import MSMarcoDataset
from evaluation.dataset.timeline_qa_dataset import TimelineQADataset
from evaluation.dataset.open_lifelog_qa_dataset import OpenLifelogQADataset
from evaluation.metrics.retrieval_metrics import RetrievalMetrics

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def rrf(list1, list2, k=60):
    scores = defaultdict(float)
    for rank, doc_id in enumerate(list1):
        scores[doc_id] += 1.0 / (k + rank + 1)
    for rank, doc_id in enumerate(list2):
        scores[doc_id] += 1.0 / (k + rank + 1)

    # Sort by score descending
    sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, score in sorted_docs]


async def main():
    # Mock client to use evaluate_retrieval method easily
    # We pass None for dependencies since we only need the
    # metrics calculation functions.
    client = RAGEvaluationClient(qa_service=None, doc_to_memory={}, memory_to_doc={})

    pairs = [
        (
            "ms_marco_passage_dev",
            lambda: MSMarcoDataset(
                limit=10_000_000, dataset_name="msmarco-passage/dev/small"
            ),
            "runs/ms_marco_passage_dev_baseline.json",
            "runs/ms_marco_passage_dev_qwen3_0.6B.json",
            "runs/ms_marco_passage_dev_rrf.json",
        ),
        (
            "ms_marco_qna_full_dev",
            lambda: MSMarcoDataset(limit=10_000_000),
            "runs/ms_marco_qna_full_dev_baseline.json",
            "runs/ms_marco_qna_full_dev_qwen3_0.6B.json",
            "runs/ms_marco_qna_full_dev_rrf.json",
        ),
        (
            "openlifelog_qa",
            lambda: OpenLifelogQADataset(),
            "runs/openlifelog_qa_baseline.json",
            "runs/openlifelog_qa_qwen3_0.6B.json",
            "runs/openlifelog_qa_rrf.json",
        ),
        (
            "timeline_qa_sparse",
            lambda: TimelineQADataset.generate(category=0),
            "runs/timeline_qa_sparse_baseline.json",
            "runs/timeline_qa_sparse_qwen3_0.6B.json",
            "runs/timeline_qa_sparse_rrf.json",
        ),
        (
            "timeline_qa_medium",
            lambda: TimelineQADataset.generate(category=1),
            "runs/timeline_qa_medium_baseline.json",
            "runs/timeline_qa_medium_qwen3_0.6B.json",
            "runs/timeline_qa_medium_rrf.json",
        ),
        (
            "timeline_qa_dense",
            lambda: TimelineQADataset.generate(category=2),
            "runs/timeline_qa_dense_baseline.json",
            "runs/timeline_qa_dense_qwen3_0.6B.json",
            "runs/timeline_qa_dense_rrf.json",
        ),
    ]

    for name, dataset_fn, run1_file, run2_file, out_file in pairs:
        logger.info(f"Processing {name}...")
        try:
            with open(run1_file) as f:
                run1 = json.load(f)
            with open(run2_file) as f:
                run2 = json.load(f)
        except Exception as e:
            logger.warning(f"Could not load runs for {name}, skipping: {e}")
            continue

        logger.info("Loading dataset to extract qrels...")
        try:
            dataset = dataset_fn()
        except Exception as e:
            logger.warning(f"Could not load dataset for {name}, skipping: {e}")
            continue

        collection_size = len(dataset.docs)
        qrels_df = dataset.qrels

        # Build relevance scores per query mapping
        relevance_scores_db = {}
        if "score" in qrels_df.columns:
            for query_id, group in qrels_df.groupby("query_id"):
                query_id_str = str(query_id)
                # Map dataset doc ID strings to relevance score
                mapping = group.set_index("doc_id")["score"].to_dict()
                relevance_scores_db[query_id_str] = {
                    str(k): float(v) for k, v in mapping.items()
                }

        combined_results = {
            "queries": [],
            "retrieval_metrics": run1[
                "retrieval_metrics"
            ].copy(),  # Initialize empty structure
            "generation_metrics": run1.get("generation_metrics", {}).copy(),
            "avg_response_time": (run1["avg_response_time"] + run2["avg_response_time"])
            / 2,
            "total_docs_streamed": run1["total_docs_streamed"],
        }

        # Reset metric sums
        for k in combined_results["retrieval_metrics"]:
            combined_results["retrieval_metrics"][k] = 0.0

        run2_queries_dict = {str(q["query_id"]): q for q in run2["queries"]}

        retrieved_docs_per_query = []
        relevant_docs_per_query = []

        valid_query_count = 0

        for q1 in run1["queries"]:
            q_id = str(q1["query_id"])
            if q_id not in run2_queries_dict:
                continue

            q2 = run2_queries_dict[q_id]

            # Combine rankings using RRF
            list1 = q1["retrieved_doc_ids_for_eval"]
            list2 = q2["retrieved_doc_ids_for_eval"]

            combined_eval_docs = rrf(list1, list2)

            # Map back combined doc IDs to original system doc instances if needed
            # We'll just rely on metrics taking retrieved_doc_ids_for_eval

            # Get ground truth
            query_qrels = qrels_df[qrels_df["query_id"].astype(str) == q_id]
            relevant_doc_ids = query_qrels["doc_id"].astype(str).tolist()
            q_relevance_scores = relevance_scores_db.get(q_id, {})

            # Evaluate using RagEvaluationClient's method
            query_metrics = client.evaluate_retrieval(
                combined_eval_docs,
                relevant_doc_ids,
                q_relevance_scores,
                collection_size=collection_size,
            )

            # Save relevant subset back to query_result
            new_q = q1.copy()
            new_q["retrieved_doc_ids_for_eval"] = combined_eval_docs
            new_q["retrieval_metrics"] = query_metrics

            combined_results["queries"].append(new_q)
            valid_query_count += 1

            retrieved_docs_per_query.append(combined_eval_docs)
            relevant_docs_per_query.append(relevant_doc_ids)

            for m in query_metrics:
                if m in combined_results["retrieval_metrics"]:
                    combined_results["retrieval_metrics"][m] += query_metrics[m]

        # Average the metrics
        if valid_query_count > 0:
            for k in combined_results["retrieval_metrics"]:
                combined_results["retrieval_metrics"][k] /= valid_query_count

            combined_results["retrieval_metrics"]["map"] = (
                RetrievalMetrics.mean_average_precision(
                    retrieved_docs_per_query, relevant_docs_per_query
                )
            )

        with open(out_file, "w") as f:
            json.dump(combined_results, f, indent=4)

        logger.info(f"\n===== RRF RESULTS FOR {name} =====")
        logger.info(f"Queries evaluated: {valid_query_count}")
        logger.info("\nRetrieval Performance:")
        logger.info(
            f"  Precision: {combined_results['retrieval_metrics']['precision']:.4f}"
        )
        logger.info(f"  Recall: {combined_results['retrieval_metrics']['recall']:.4f}")
        logger.info(f"  F1 Score: {combined_results['retrieval_metrics']['f1']:.4f}")
        logger.info(
            f"  P@1: {combined_results['retrieval_metrics']['precision@1']:.4f}"
        )
        logger.info(
            f"  P@5: {combined_results['retrieval_metrics']['precision@5']:.4f}"
        )
        logger.info(f"  R@5: {combined_results['retrieval_metrics']['recall@5']:.4f}")
        logger.info(f"  MRR: {combined_results['retrieval_metrics']['mrr']:.4f}")
        logger.info(f"  NDCG@5: {combined_results['retrieval_metrics']['ndcg@5']:.4f}")
        logger.info(f"  MAP: {combined_results['retrieval_metrics']['map']:.4f}")
        logger.info(f"  AQWV: {combined_results['retrieval_metrics']['aqwv']:.4f}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
