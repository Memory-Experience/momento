"""
End-to-End Pipeline: Recording → Transcription → RAG → MS MARCO Evaluation → Comparison

This module orchestrates the complete pipeline from gRPC recordings through transcription,
RAG retrieval, MS MARCO evaluation, and performance comparison.
"""

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd

# Add necessary paths for imports
api_path = str(Path(__file__).parent.parent / "api")
sys.path.insert(0, api_path)

# Import RAG and domain classes
from api.rag.rag_service import SimpleRAGService
from api.domain.memory_context import MemoryContext
from api.domain.memory_request import MemoryRequest

from dataset_marco.run_marco_eval import convert_ms_marco_to_dataframes, evaluate_retrieval_metrics
from dataset_marco.metrics import precision_at_k, recall_at_k, ndcg_at_k, mrr_score


class EndToEndPipeline:
    """Complete pipeline orchestrator for Recording → RAG → MARCO evaluation."""
    
    def __init__(self, recordings_dir: str, marco_variant: str = "msmarco-passage/dev/small"):
        """
        Initialize the end-to-end pipeline.
        
        Args:
            recordings_dir: Path to the recordings directory
            marco_variant: MS MARCO dataset variant to use
        """
        self.recordings_dir = Path(recordings_dir)
        self.marco_variant = marco_variant
        self.rag_service = SimpleRAGService()
        
        # Load MS MARCO data
        print(f"Loading MS MARCO dataset: {marco_variant}")
        self.marco_queries, self.marco_passages, self.marco_qrels = convert_ms_marco_to_dataframes(marco_variant)
        print(f"Loaded {len(self.marco_queries)} queries, {len(self.marco_passages)} passages")
        
    def load_recording_data(self, recording_id: str) -> Dict[str, Any]:
        """Load recording metadata and transcription text."""
        json_file = self.recordings_dir / f"{recording_id}.json"
        txt_file = self.recordings_dir / f"{recording_id}.txt"
        
        # Load metadata
        with open(json_file, 'r') as f:
            metadata = json.load(f)
            
        # Load transcription text if available
        transcription = ""
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                transcription = f.read().strip()
                
        return {
            "metadata": metadata,
            "transcription": transcription,
            "has_transcription": bool(transcription)
        }
    
    def populate_rag_with_marco_passages(self, max_passages: int = 1000):
        """Populate RAG service with MS MARCO passages as memories."""
        print(f"Populating RAG service with {min(max_passages, len(self.marco_passages))} MS MARCO passages...")
        
        for idx, row in self.marco_passages.head(max_passages).iterrows():
            memory_context = MemoryContext(
                content=row['text'],
                metadata={
                    'source': 'ms_marco',
                    'passage_id': row['docid'],
                    'idx': idx
                }
            )
            self.rag_service.add_memory(memory_context)
            
        print(f"RAG service now contains {len(self.rag_service.memories)} memories")
    
    def evaluate_recording_against_marco(self, recording_id: str, k_values: List[int] = [1, 3, 5, 10]) -> Dict[str, Any]:
        """
        Evaluate a recording's transcription against MS MARCO using RAG retrieval.
        
        Args:
            recording_id: ID of the recording to evaluate
            k_values: List of k values for evaluation metrics
            
        Returns:
            Dictionary containing evaluation results and comparisons
        """
        # Load recording data
        recording_data = self.load_recording_data(recording_id)
        
        if not recording_data["has_transcription"]:
            return {"error": f"No transcription found for recording {recording_id}"}
        
        transcription = recording_data["transcription"]
        print(f"Evaluating recording {recording_id}: '{transcription[:100]}...'")
        
        # Use transcription as query to RAG service
        memory_request = MemoryRequest(
            query=transcription,
            max_results=max(k_values),
            threshold=0.0  # Include all results for evaluation
        )
        
        # Get RAG retrieval results
        rag_results = self.rag_service.search_memories(memory_request)
        
        # Extract passage IDs and scores from RAG results
        rag_passage_ids = []
        rag_scores = []
        for memory in rag_results:
            if 'passage_id' in memory.metadata:
                rag_passage_ids.append(memory.metadata['passage_id'])
                rag_scores.append(memory.metadata.get('score', 0.0))
        
        # Find similar MS MARCO queries for comparison
        marco_similar_queries = self.find_similar_marco_queries(transcription, top_k=5)
        
        # Evaluate RAG performance against MS MARCO
        evaluation_results = {
            "recording_id": recording_id,
            "transcription": transcription,
            "transcription_length": len(transcription),
            "rag_results_count": len(rag_results),
            "rag_passage_ids": rag_passage_ids[:10],  # Top 10 for display
            "rag_scores": rag_scores[:10],
            "similar_marco_queries": marco_similar_queries,
            "metrics": {}
        }
        
        # Calculate metrics for each similar MARCO query
        for i, (query_id, query_text, similarity) in enumerate(marco_similar_queries):
            query_metrics = self.evaluate_rag_against_marco_query(
                query_id, rag_passage_ids, rag_scores, k_values
            )
            evaluation_results["metrics"][f"marco_query_{i+1}"] = {
                "query_id": query_id,
                "query_text": query_text,
                "similarity_to_transcription": similarity,
                **query_metrics
            }
        
        return evaluation_results
    
    def find_similar_marco_queries(self, transcription: str, top_k: int = 5) -> List[Tuple[str, str, float]]:
        """Find MS MARCO queries similar to the transcription."""
        similarities = []
        
        transcription_words = set(transcription.lower().split())
        
        for _, row in self.marco_queries.iterrows():
            query_words = set(row['text'].lower().split())
            
            # Calculate Jaccard similarity
            if len(transcription_words) > 0 and len(query_words) > 0:
                intersection = len(transcription_words.intersection(query_words))
                union = len(transcription_words.union(query_words))
                similarity = intersection / union if union > 0 else 0.0
            else:
                similarity = 0.0
            
            similarities.append((row['query_id'], row['text'], similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]
    
    def evaluate_rag_against_marco_query(self, query_id: str, rag_passage_ids: List[str], 
                                       rag_scores: List[float], k_values: List[int]) -> Dict[str, float]:
        """Evaluate RAG results against a specific MS MARCO query."""
        # Get relevant passages for this query from MS MARCO qrels
        relevant_passages = set()
        query_qrels = self.marco_qrels[self.marco_qrels['query_id'] == query_id]
        
        if len(query_qrels) == 0:
            # No ground truth for this query
            return {f"precision@{k}": 0.0 for k in k_values}
        
        relevant_passages = set(query_qrels['docid'].astype(str).tolist())
        
        # Convert RAG passage IDs to strings for comparison
        rag_passage_ids_str = [str(pid) for pid in rag_passage_ids]
        
        # Calculate metrics
        metrics = {}
        for k in k_values:
            top_k_passages = rag_passage_ids_str[:k]
            
            # Precision@k
            metrics[f"precision@{k}"] = precision_at_k(top_k_passages, relevant_passages)
            
            # Recall@k
            metrics[f"recall@{k}"] = recall_at_k(top_k_passages, relevant_passages)
            
            # NDCG@k (using binary relevance: 1 if relevant, 0 if not)
            relevance_scores = [1.0 if str(pid) in relevant_passages else 0.0 for pid in top_k_passages]
            metrics[f"ndcg@{k}"] = ndcg_at_k(relevance_scores, k)
        
        # MRR
        metrics["mrr"] = mrr_score([rag_passage_ids_str], [relevant_passages])
        
        return metrics
    
    def run_comprehensive_evaluation(self, recording_ids: List[str] = None, 
                                   max_passages: int = 1000) -> pd.DataFrame:
        """
        Run comprehensive evaluation across multiple recordings.
        
        Args:
            recording_ids: List of recording IDs to evaluate. If None, evaluates all available recordings.
            max_passages: Maximum number of MS MARCO passages to load into RAG
            
        Returns:
            DataFrame with evaluation results
        """
        # Populate RAG service with MS MARCO passages
        self.populate_rag_with_marco_passages(max_passages)
        
        # Get recording IDs if not provided
        if recording_ids is None:
            recording_ids = []
            for json_file in self.recordings_dir.glob("*.json"):
                recording_id = json_file.stem
                if (self.recordings_dir / f"{recording_id}.txt").exists():
                    recording_ids.append(recording_id)
        
        print(f"Evaluating {len(recording_ids)} recordings...")
        
        # Evaluate each recording
        all_results = []
        for recording_id in recording_ids:
            try:
                result = self.evaluate_recording_against_marco(recording_id)
                if "error" not in result:
                    # Flatten metrics for DataFrame
                    for query_key, query_metrics in result["metrics"].items():
                        row = {
                            "recording_id": recording_id,
                            "transcription_length": result["transcription_length"],
                            "rag_results_count": result["rag_results_count"],
                            "marco_query_id": query_metrics["query_id"],
                            "marco_query_text": query_metrics["query_text"][:100],  # Truncate for display
                            "similarity_to_transcription": query_metrics["similarity_to_transcription"],
                            **{k: v for k, v in query_metrics.items() if k.startswith(('precision@', 'recall@', 'ndcg@', 'mrr'))}
                        }
                        all_results.append(row)
                        
            except Exception as e:
                print(f"Error evaluating recording {recording_id}: {e}")
                continue
        
        return pd.DataFrame(all_results)
    
    def compare_rag_vs_marco_baseline(self, evaluation_df: pd.DataFrame) -> Dict[str, Any]:
        """Compare RAG performance against MS MARCO baseline."""
        if len(evaluation_df) == 0:
            return {"error": "No evaluation data available"}
        
        # Calculate average metrics across all evaluations
        metric_columns = [col for col in evaluation_df.columns if any(metric in col for metric in ['precision@', 'recall@', 'ndcg@', 'mrr'])]
        
        avg_metrics = {}
        for col in metric_columns:
            avg_metrics[f"avg_{col}"] = evaluation_df[col].mean()
            avg_metrics[f"std_{col}"] = evaluation_df[col].std()
        
        # Summary statistics
        summary = {
            "total_evaluations": len(evaluation_df),
            "unique_recordings": evaluation_df['recording_id'].nunique(),
            "avg_transcription_length": evaluation_df['transcription_length'].mean(),
            "avg_rag_results": evaluation_df['rag_results_count'].mean(),
            "avg_similarity_to_marco": evaluation_df['similarity_to_transcription'].mean(),
            "metrics": avg_metrics
        }
        
        # Performance insights
        best_performing = evaluation_df.loc[evaluation_df['precision@5'].idxmax()] if 'precision@5' in evaluation_df.columns else None
        worst_performing = evaluation_df.loc[evaluation_df['precision@5'].idxmin()] if 'precision@5' in evaluation_df.columns else None
        
        if best_performing is not None:
            summary["best_performing"] = {
                "recording_id": best_performing['recording_id'],
                "precision@5": best_performing.get('precision@5', 0),
                "similarity": best_performing['similarity_to_transcription']
            }
        
        if worst_performing is not None:
            summary["worst_performing"] = {
                "recording_id": worst_performing['recording_id'],
                "precision@5": worst_performing.get('precision@5', 0),
                "similarity": worst_performing['similarity_to_transcription']
            }
        
        return summary


def main():
    """Main function to run the end-to-end pipeline evaluation."""
    # Set up paths
    project_root = Path(__file__).parent.parent.parent
    recordings_dir = project_root / "packages" / "api" / "recordings"
    
    print("Starting End-to-End Pipeline Evaluation")
    print("=" * 50)
    
    # Initialize pipeline
    pipeline = EndToEndPipeline(str(recordings_dir))
    
    # Run evaluation on a few sample recordings
    sample_recordings = ["20250821_162616", "20250822_144105"]  # Add more as needed
    available_recordings = [r for r in sample_recordings if (recordings_dir / f"{r}.json").exists()]
    
    print(f"Available recordings for evaluation: {available_recordings}")
    
    if not available_recordings:
        print("No valid recordings found. Please check the recordings directory.")
        return
    
    # Run comprehensive evaluation
    evaluation_df = pipeline.run_comprehensive_evaluation(
        recording_ids=available_recordings,
        max_passages=500  # Limit for faster evaluation
    )
    
    print(f"\nEvaluation completed. Results shape: {evaluation_df.shape}")
    
    if len(evaluation_df) > 0:
        print("\nSample Results:")
        print(evaluation_df[['recording_id', 'marco_query_text', 'precision@5', 'recall@5', 'similarity_to_transcription']].head(10))
        
        # Generate comparison report
        comparison = pipeline.compare_rag_vs_marco_baseline(evaluation_df)
        
        print("\nRAG vs MS MARCO Comparison:")
        print("=" * 30)
        print(f"Total evaluations: {comparison['total_evaluations']}")
        print(f"Average Precision@5: {comparison['metrics'].get('avg_precision@5', 0):.4f}")
        print(f"Average Recall@5: {comparison['metrics'].get('avg_recall@5', 0):.4f}")
        print(f"Average NDCG@5: {comparison['metrics'].get('avg_ndcg@5', 0):.4f}")
        print(f"Average similarity to MARCO queries: {comparison['avg_similarity_to_marco']:.4f}")
        
        # Save results
        output_file = Path(__file__).parent / "end_to_end_results.csv"
        evaluation_df.to_csv(output_file, index=False)
        print(f"\nResults saved to: {output_file}")
        
    else:
        print("No evaluation results generated. Please check the data and try again.")


if __name__ == "__main__":
    main()
