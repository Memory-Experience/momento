"""
Simplified End-to-End Pipeline Demo

This demonstrates the complete pipeline integration concept:
Recording → Transcription → RAG → MS MARCO Evaluation → Comparison
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd

# Simple mock classes to simulate RAG functionality for demo
class MockMemoryContext:
    def __init__(self, content: str, metadata: dict):
        self.content = content
        self.metadata = metadata

class MockRAGService:
    def __init__(self):
        self.memories = []
    
    def add_memory(self, content: str, metadata: dict):
        self.memories.append(MockMemoryContext(content, metadata))
    
    def search_memories(self, query: str, max_results: int = 10):
        # Simple keyword-based search simulation
        query_words = set(query.lower().split())
        scored_memories = []
        
        for memory in self.memories:
            memory_words = set(memory.content.lower().split())
            # Jaccard similarity
            intersection = len(query_words.intersection(memory_words))
            union = len(query_words.union(memory_words))
            score = intersection / union if union > 0 else 0.0
            
            if score > 0:
                memory.metadata['score'] = score
                scored_memories.append((score, memory))
        
        # Sort by score and return top results
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:max_results]]

# Import evaluation functions
from dataset_marco.run_marco_eval import convert_ms_marco_to_dataframes
from dataset_marco.metrics import precision_at_k, recall_at_k, ndcg_at_k, mrr_score


class SimplifiedPipeline:
    """Simplified pipeline demonstrating the integration concept."""
    
    def __init__(self, recordings_dir: str):
        self.recordings_dir = Path(recordings_dir)
        self.rag_service = MockRAGService()
        
        # Load MS MARCO data (small subset for demo)
        print("Loading MS MARCO dataset...")
        self.marco_queries, self.marco_passages, self.marco_qrels = convert_ms_marco_to_dataframes("msmarco-passage/dev/small")
        print(f"Loaded {len(self.marco_queries)} queries, {len(self.marco_passages)} passages")
    
    def load_recording_transcription(self, recording_id: str) -> str:
        """Load transcription text from recording."""
        txt_file = self.recordings_dir / f"{recording_id}.txt"
        
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                return f.read().strip()
        return ""
    
    def populate_rag_with_passages(self, max_passages: int = 100):
        """Populate RAG with MS MARCO passages."""
        print(f"Populating RAG with {min(max_passages, len(self.marco_passages))} passages...")
        
        for idx, row in self.marco_passages.head(max_passages).iterrows():
            self.rag_service.add_memory(
                content=row['text'],
                metadata={
                    'passage_id': row['docid'],
                    'source': 'ms_marco'
                }
            )
        
        print(f"RAG now contains {len(self.rag_service.memories)} passages")
    
    def evaluate_recording(self, recording_id: str) -> Dict[str, Any]:
        """Complete pipeline: Recording → Transcription → RAG → Evaluation."""
        
        # Step 1: Load transcription (simulating gRPC → transcription)
        transcription = self.load_recording_transcription(recording_id)
        if not transcription:
            return {"error": f"No transcription for {recording_id}"}
        
        print(f"\nProcessing recording {recording_id}")
        print(f"Transcription: '{transcription}'")
        
        # Step 2: Query RAG service
        rag_results = self.rag_service.search_memories(transcription, max_results=10)
        print(f"RAG returned {len(rag_results)} results")
        
        # Step 3: Find similar MS MARCO queries
        similar_queries = self.find_similar_marco_queries(transcription, top_k=3)
        
        # Step 4: Evaluate performance
        evaluation = {
            "recording_id": recording_id,
            "transcription": transcription,
            "rag_results_count": len(rag_results),
            "similar_marco_queries": [],
            "performance_metrics": {}
        }
        
        # Evaluate against each similar MARCO query
        for query_id, query_text, similarity in similar_queries:
            print(f"Evaluating against MARCO query: '{query_text}' (similarity: {similarity:.3f})")
            
            # Get ground truth for this query
            query_qrels = self.marco_qrels[self.marco_qrels['query_id'] == query_id]
            if len(query_qrels) == 0:
                continue
            
            relevant_docs = set(query_qrels['docid'].astype(str))
            rag_doc_ids = [r.metadata['passage_id'] for r in rag_results if 'passage_id' in r.metadata]
            
            # Calculate metrics
            metrics = {}
            for k in [1, 3, 5]:
                top_k_docs = rag_doc_ids[:k]
                metrics[f"precision@{k}"] = precision_at_k(top_k_docs, relevant_docs)
                metrics[f"recall@{k}"] = recall_at_k(top_k_docs, relevant_docs)
                
                # NDCG with binary relevance
                relevance_scores = [1.0 if str(doc) in relevant_docs else 0.0 for doc in top_k_docs]
                metrics[f"ndcg@{k}"] = ndcg_at_k(relevance_scores, k)
            
            # MRR
            metrics["mrr"] = mrr_score([rag_doc_ids], [relevant_docs])
            
            evaluation["similar_marco_queries"].append({
                "query_id": query_id,
                "query_text": query_text,
                "similarity": similarity,
                "metrics": metrics
            })
        
        return evaluation
    
    def find_similar_marco_queries(self, text: str, top_k: int = 3) -> List[Tuple[str, str, float]]:
        """Find MS MARCO queries similar to the input text."""
        text_words = set(text.lower().split())
        similarities = []
        
        for _, row in self.marco_queries.iterrows():
            query_words = set(row['text'].lower().split())
            
            # Jaccard similarity
            if len(text_words) > 0 and len(query_words) > 0:
                intersection = len(text_words.intersection(query_words))
                union = len(text_words.union(query_words))
                similarity = intersection / union if union > 0 else 0.0
            else:
                similarity = 0.0
            
            similarities.append((row['query_id'], row['text'], similarity))
        
        similarities.sort(key=lambda x: x[2], reverse=True)
        return similarities[:top_k]
    
    def run_demo(self):
        """Run the complete pipeline demo."""
        print("=" * 60)
        print("End-to-End Pipeline Demo: Recording → RAG → MARCO Evaluation")
        print("=" * 60)
        
        # Populate RAG
        self.populate_rag_with_passages(max_passages=200)
        
        # Find available recordings
        recordings = []
        for txt_file in self.recordings_dir.glob("*.txt"):
            recording_id = txt_file.stem
            if (self.recordings_dir / f"{recording_id}.json").exists():
                recordings.append(recording_id)
        
        print(f"\nFound {len(recordings)} recordings with transcriptions")
        
        # Evaluate first few recordings
        results = []
        for recording_id in recordings[:3]:  # Limit to first 3 for demo
            try:
                result = self.evaluate_recording(recording_id)
                if "error" not in result:
                    results.append(result)
            except Exception as e:
                print(f"Error processing {recording_id}: {e}")
        
        # Summary
        print("\n" + "=" * 40)
        print("PIPELINE EVALUATION SUMMARY")
        print("=" * 40)
        
        if results:
            avg_precision_5 = []
            avg_mrr = []
            
            for result in results:
                print(f"\nRecording: {result['recording_id']}")
                print(f"Transcription: {result['transcription'][:100]}...")
                print(f"RAG Results: {result['rag_results_count']}")
                
                for query_eval in result['similar_marco_queries']:
                    metrics = query_eval['metrics']
                    print(f"  vs MARCO query (sim={query_eval['similarity']:.3f}): "
                          f"P@5={metrics.get('precision@5', 0):.3f}, "
                          f"MRR={metrics.get('mrr', 0):.3f}")
                    
                    avg_precision_5.append(metrics.get('precision@5', 0))
                    avg_mrr.append(metrics.get('mrr', 0))
            
            if avg_precision_5:
                print(f"\nOverall Performance:")
                print(f"Average Precision@5: {sum(avg_precision_5)/len(avg_precision_5):.4f}")
                print(f"Average MRR: {sum(avg_mrr)/len(avg_mrr):.4f}")
                
                # Analysis
                print(f"\nAnalysis:")
                print(f"- Evaluated {len(results)} recordings")
                print(f"- Total query comparisons: {len(avg_precision_5)}")
                print(f"- The scores show how well your recordings' transcriptions")
                print(f"  perform when used as queries against MS MARCO passages")
                print(f"- Low scores are expected since recordings are conversational,")
                print(f"  while MS MARCO queries are search-oriented")
        else:
            print("No results to analyze. Check recordings and try again.")


def main():
    """Run the simplified pipeline demo."""
    project_root = Path(__file__).parent.parent.parent
    recordings_dir = project_root / "packages" / "api" / "recordings"
    
    if not recordings_dir.exists():
        print(f"Recordings directory not found: {recordings_dir}")
        return
    
    pipeline = SimplifiedPipeline(str(recordings_dir))
    pipeline.run_demo()


if __name__ == "__main__":
    main()
