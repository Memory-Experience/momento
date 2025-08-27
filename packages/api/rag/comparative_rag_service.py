"""
Comparative RAG Service: Run the simple RAG alongside MS MARCO evaluation
This gives the best of both worlds - production stability + evaluation insights
"""

import logging
import sys
from datetime import datetime
from pathlib import Path

# Add evaluation package to Python path
eval_path = Path(__file__).parent.parent.parent / "evaluation"
if str(eval_path) not in sys.path:
    sys.path.append(str(eval_path))

try:
    from evaluation.dataset_marco.metrics import precision_at_k, recall_at_k
    from evaluation.dataset_marco.prepare_ms_marco import MSMarcoDataset
    from evaluation.dataset_marco.run_marco_eval import convert_ms_marco_to_dataframes

    MARCO_AVAILABLE = True
    logging.info("MS MARCO evaluation modules loaded successfully")
except ImportError as e:
    logging.warning(f"MS MARCO evaluation not available: {e}")
    MARCO_AVAILABLE = False


class ComparativeRAGService:
    """
    RAG service that runs your simple algorithm alongside MS MARCO evaluation.
    Perfect for A/B testing and performance comparison.
    """

    def __init__(self, enable_marco_comparison: bool = True):
        # Your existing simple RAG (unchanged)
        self.memories = []

        # MS MARCO comparison (optional)
        self.marco_enabled = enable_marco_comparison and MARCO_AVAILABLE
        self.marco_data = None

        if self.marco_enabled:
            self._load_marco_data()

    def _load_marco_data(self):
        """Load MS MARCO data for comparison (non-blocking)."""
        try:
            logging.info("Loading MS MARCO dataset for comparison...")
            marco_dataset = MSMarcoDataset("msmarco-passage/dev/small")
            queries, docs, qrels = convert_ms_marco_to_dataframes(
                marco_dataset, limit=1000
            )

            self.marco_data = {
                "queries": queries,
                "passages": docs,  # docs contains 'id' and 'content' columns
                "qrels": qrels,
            }
            logging.info(f"Loaded {len(docs)} MS MARCO passages for comparison")
            logging.info(
                f"DataFrame columns - Passages: {list(docs.columns)}, "
                f"Queries: {list(queries.columns)}, QRELs: {list(qrels.columns)}"
            )

        except Exception as e:
            logging.warning(f"Could not load MS MARCO: {e}")
            self.marco_enabled = False

    def add_memory(self, text: str, audio_filename: str = None):
        """Add memory (unchanged from your original)."""
        memory = {
            "id": len(self.memories),
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "audio_file": audio_filename,
        }
        self.memories.append(memory)
        logging.info(f"Added memory: {memory['id']}")

    def search_memories(self, query: str, include_comparison: bool = False) -> str:
        """
        Search memories with optional MS MARCO comparison.

        Args:
            query: User's question
            include_comparison: If True, adds MS MARCO benchmark results

        Returns:
            Response with optional comparison metrics
        """
        # 1. Run your original simple search (unchanged)
        simple_result = self._search_simple_memories(query)

        # 2. Optionally add MS MARCO comparison
        if include_comparison and self.marco_enabled:
            comparison_data = self._get_marco_comparison(query)
            return self._format_comparative_response(simple_result, comparison_data)

        return simple_result

    def _search_simple_memories(self, query: str) -> str:
        """Your original simple keyword search (unchanged)."""
        query_words = set(query.lower().split())
        relevant_memories = []

        for memory in self.memories:
            memory_words = set(memory["text"].lower().split())
            overlap = len(query_words.intersection(memory_words))
            if overlap > 0:
                relevant_memories.append((memory, overlap))

        relevant_memories.sort(key=lambda x: x[1], reverse=True)

        if not relevant_memories:
            return (
                "I don't have any memories that match your question. "
                + "Try asking about something you've recorded before."
            )

        # Generate answer based on top relevant memories
        top_memories = relevant_memories[:3]
        answer_parts = ["Based on your memories, here's what I found:"]

        for i, (memory, _) in enumerate(top_memories, 1):
            timestamp = datetime.fromisoformat(memory["timestamp"])
            formatted_date = timestamp.strftime("%B %d, %Y at %I:%M %p")
            answer_parts.append(f"{i}. From {formatted_date}: {memory['text']}")

        return "\n\n".join(answer_parts)

    def _get_marco_comparison(self, query: str) -> dict:
        """Get MS MARCO benchmark results for comparison."""
        if not self.marco_data:
            return {}

        # Search MS MARCO passages with same algorithm
        query_words = set(query.lower().split())
        marco_results = []

        for _, passage in self.marco_data["passages"].iterrows():
            passage_words = set(
                passage["text"].lower().split()
            )  # Documents have 'text' column
            overlap = len(query_words.intersection(passage_words))
            if overlap > 0:
                score = overlap / len(query_words.union(passage_words))
                marco_results.append(
                    {
                        "docid": passage["id"],  # Use 'id'
                        "text": passage["text"],  # Use 'text' for documents
                        "score": score,
                    }
                )

        marco_results.sort(key=lambda x: x["score"], reverse=True)

        # Find similar queries in MS MARCO for evaluation
        similar_query = self._find_most_similar_marco_query(query)

        evaluation_metrics = {}
        if similar_query and marco_results:
            evaluation_metrics = self._calculate_retrieval_metrics(
                similar_query["id"], marco_results[:10]
            )

        return {
            "marco_results": marco_results[:5],
            "similar_query": similar_query,
            "metrics": evaluation_metrics,
            "algorithm_performance": self._analyze_algorithm_performance(marco_results),
        }

    def _find_most_similar_marco_query(self, user_query: str) -> dict | None:
        """Find the most similar query in MS MARCO dataset."""
        user_words = set(user_query.lower().split())
        best_match = None
        best_similarity = 0.0

        for _, query_row in self.marco_data["queries"].iterrows():
            query_words = set(
                query_row["content"].lower().split()
            )  # Queries have 'content' column
            if len(user_words) > 0 and len(query_words) > 0:
                intersection = len(user_words.intersection(query_words))
                union = len(user_words.union(query_words))
                similarity = intersection / union if union > 0 else 0.0

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = {
                        "id": query_row["id"],  # Use 'id'
                        "text": query_row["content"],  # Use 'content' for queries
                        "similarity": similarity,
                    }

        return best_match if best_similarity > 0.3 else None

    def _calculate_retrieval_metrics(
        self, query_id: str, retrieved_results: list[dict]
    ) -> dict:
        """Calculate standard IR metrics against MS MARCO ground truth."""
        # Get ground truth relevant documents
        relevant_docs = set(
            self.marco_data["qrels"][self.marco_data["qrels"]["query_id"] == query_id][
                "doc_id"
            ].astype(str)  # Use 'doc_id' instead of 'docid'
        )

        retrieved_docs = [r["docid"] for r in retrieved_results]

        metrics = {}
        for k in [1, 3, 5, 10]:
            if len(retrieved_docs) >= k:
                metrics[f"precision@{k}"] = precision_at_k(
                    retrieved_docs, relevant_docs, k
                )
                metrics[f"recall@{k}"] = recall_at_k(retrieved_docs, relevant_docs, k)

        return metrics

    def _analyze_algorithm_performance(self, results: list[dict]) -> dict:
        """Analyze how well your simple algorithm performs."""
        if not results:
            return {"status": "no_results"}

        scores = [r["score"] for r in results]

        return {
            "total_results": len(results),
            "avg_score": sum(scores) / len(scores),
            "max_score": max(scores),
            "min_score": min(scores),
            "score_distribution": {
                "high_quality": len([s for s in scores if s > 0.5]),
                "medium_quality": len([s for s in scores if 0.2 <= s <= 0.5]),
                "low_quality": len([s for s in scores if s < 0.2]),
            },
        }

    def _format_comparative_response(
        self, simple_result: str, comparison_data: dict
    ) -> str:
        """Format response with comparison insights."""
        response_parts = [simple_result]

        if comparison_data.get("metrics"):
            response_parts.append("\n" + "=" * 50)
            response_parts.append("PERFORMANCE ANALYSIS vs MS MARCO BENCHMARK:")

            metrics = comparison_data["metrics"]
            for metric, value in metrics.items():
                response_parts.append(f"   {metric}: {value:.3f}")

            perf = comparison_data.get("algorithm_performance", {})
            if perf.get("total_results"):
                response_parts.append("\nAlgorithm Analysis:")
                response_parts.append(
                    f"   Found {perf['total_results']} relevant passages"
                )
                response_parts.append(
                    f"   Average relevance score: {perf['avg_score']:.3f}"
                )

                dist = perf.get("score_distribution", {})
                response_parts.append(
                    f"   Quality distribution: {dist.get('high_quality', 0)} high, "
                    f"{dist.get('medium_quality', 0)} medium, "
                    f"{dist.get('low_quality', 0)} low"
                )

        if comparison_data.get("similar_query"):
            sim_query = comparison_data["similar_query"]
            response_parts.append(
                f'\nSimilar MS MARCO query: "{sim_query["text"]}" '
                f"(similarity: {sim_query['similarity']:.2f})"
            )

        return "\n".join(response_parts)

    def get_performance_summary(self) -> dict:
        """Get overall performance summary for analysis."""
        return {
            "total_memories": len(self.memories),
            "marco_enabled": self.marco_enabled,
            "marco_dataset_size": len(self.marco_data["passages"])
            if self.marco_data
            else 0,
            "evaluation_ready": self.marco_enabled and self.marco_data is not None,
        }


# Example usage:
if __name__ == "__main__":
    # Initialize with MS MARCO comparison
    rag_service = ComparativeRAGService(enable_marco_comparison=True)

    # Add some memories (your normal workflow)
    rag_service.add_memory("I learned about machine learning algorithms today")
    rag_service.add_memory("Python is great for data science projects")

    # Search with comparison (optional)
    result = rag_service.search_memories(
        "what is machine learning", include_comparison=True
    )
    print(result)
