"""
Integration Bridge: Connect MS MARCO evaluation to running RAG system
This would replace or enhance the SimpleRAGService with MS MARCO capabilities
"""

import sys
from datetime import datetime
from pathlib import Path

# Add evaluation package to path
sys.path.append(str(Path(__file__).parent.parent / "evaluation"))

from dataset_marco.metrics import precision_at_k
from dataset_marco.prepare_ms_marco import MSMarcoDataset
from dataset_marco.run_marco_eval import convert_ms_marco_to_dataframes


class EnhancedRAGService:
    """RAG service that combines simple keyword search with MS MARCO evaluation."""

    def __init__(self):
        self.memories = []  # Keep existing functionality
        self.load_marco_data()

    def load_marco_data(self):
        """Load MS MARCO dataset for enhanced retrieval."""
        try:
            # Load MS MARCO dataset
            marco_dataset = MSMarcoDataset("msmarco-passage/dev/small")
            self.marco_queries, self.marco_passages, self.marco_qrels = (
                convert_ms_marco_to_dataframes(marco_dataset, limit=1000)
            )
            print(f"✓ Loaded {len(self.marco_passages)} MS MARCO passages")
        except Exception as e:
            print(f"⚠️  Could not load MS MARCO: {e}")
            self.marco_passages = None

    def add_memory(self, text: str, audio_filename: str = None):
        """Add memory (existing functionality)."""
        memory = {
            "id": len(self.memories),
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "audio_file": audio_filename,
        }
        self.memories.append(memory)

    def search_memories(self, query: str) -> str:
        """Enhanced search using both memories and MS MARCO."""
        # 1. Search existing memories (current functionality)
        memory_results = self._search_simple_memories(query)

        # 2. Search MS MARCO passages (new functionality)
        marco_results = (
            self._search_marco_passages(query)
            if self.marco_passages is not None
            else []
        )

        # 3. Combine and rank results
        combined_results = self._combine_results(memory_results, marco_results)

        # 4. Calculate evaluation metrics
        metrics = self._calculate_metrics(query, combined_results)

        return self._format_response(combined_results, metrics)

    def _search_simple_memories(self, query: str):
        """Original simple keyword search."""
        query_words = set(query.lower().split())
        results = []

        for memory in self.memories:
            memory_words = set(memory["text"].lower().split())
            overlap = len(query_words.intersection(memory_words))
            if overlap > 0:
                score = overlap / len(query_words.union(memory_words))
                results.append(
                    {
                        "text": memory["text"],
                        "score": score,
                        "source": "memory",
                        "id": memory["id"],
                    }
                )

        return sorted(results, key=lambda x: x["score"], reverse=True)

    def _search_marco_passages(self, query: str):
        """Search MS MARCO passages."""
        query_words = set(query.lower().split())
        results = []

        for _, passage in self.marco_passages.iterrows():
            passage_words = set(passage["text"].lower().split())
            overlap = len(query_words.intersection(passage_words))
            if overlap > 0:
                score = overlap / len(query_words.union(passage_words))
                results.append(
                    {
                        "text": passage["text"],
                        "score": score,
                        "source": "marco",
                        "id": passage["docid"],
                    }
                )

        return sorted(results, key=lambda x: x["score"], reverse=True)[:5]

    def _combine_results(self, memory_results, marco_results):
        """Combine and rank all results."""
        all_results = memory_results + marco_results
        return sorted(all_results, key=lambda x: x["score"], reverse=True)[:10]

    def _calculate_metrics(self, query, results):
        """Calculate evaluation metrics if possible."""
        if not self.marco_passages is not None:
            return {}

        # Find similar queries in MS MARCO
        similar_queries = self._find_similar_marco_queries(query)
        if not similar_queries:
            return {}

        # Calculate precision@k for best matching query
        best_query_id = similar_queries[0][0]
        relevant_docs = set(
            self.marco_qrels[self.marco_qrels["query_id"] == best_query_id][
                "docid"
            ].astype(str)
        )
        retrieved_docs = [r["id"] for r in results if r["source"] == "marco"]

        metrics = {}
        for k in [1, 3, 5]:
            metrics[f"precision@{k}"] = precision_at_k(retrieved_docs, relevant_docs, k)

        return metrics

    def _find_similar_marco_queries(self, query, top_k=1):
        """Find similar queries in MS MARCO dataset."""
        query_words = set(query.lower().split())
        similarities = []

        for _, row in self.marco_queries.iterrows():
            row_words = set(row["text"].lower().split())
            if len(query_words) > 0 and len(row_words) > 0:
                intersection = len(query_words.intersection(row_words))
                union = len(query_words.union(row_words))
                similarity = intersection / union if union > 0 else 0.0
                similarities.append((row["query_id"], row["text"], similarity))

        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]

    def _format_response(self, results, metrics):
        """Format the final response."""
        if not results:
            return "No relevant information found."

        response_parts = []

        # Add top result
        top_result = results[0]
        response_parts.append(f"Most relevant: {top_result['text'][:200]}...")

        # Add metrics if available
        if metrics:
            response_parts.append(f"\nEvaluation metrics: {metrics}")

        # Add sources summary
        memory_count = sum(1 for r in results if r["source"] == "memory")
        marco_count = sum(1 for r in results if r["source"] == "marco")
        response_parts.append(
            f"\nSources: {memory_count} memories, {marco_count} MS MARCO passages"
        )

        return "\n".join(response_parts)


# To integrate this into your running system, you would:
# 1. Replace SimpleRAGService with EnhancedRAGService in main.py
# 2. Update imports to include evaluation packages
# 3. Install additional dependencies (ir_datasets, pandas, etc.)
