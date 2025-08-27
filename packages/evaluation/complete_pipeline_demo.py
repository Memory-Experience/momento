"""
Working End-to-End Pipeline Demo

Complete pipeline: Recording → Transcription → RAG → MS MARCO Evaluation → Comparison
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import pandas as pd

# Try to import ir_datasets, fallback to mock data if not available
try:
    import ir_datasets
    HAS_IR_DATASETS = True
except ImportError:
    HAS_IR_DATASETS = False
    print("ir_datasets not available, using mock data for demo")

# Simple mock classes to simulate RAG functionality
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
            union = len(query_words.union(query_words))
            score = intersection / union if union > 0 else 0.0
            
            if score > 0:
                memory.metadata['score'] = score
                scored_memories.append((score, memory))
        
        # Sort by score and return top results
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        return [memory for _, memory in scored_memories[:max_results]]

# Import our evaluation metrics
from dataset_marco.metrics import precision_at_k, recall_at_k, ndcg_at_k, mrr_score


def load_mock_marco_data():
    """Create mock MS MARCO data for demonstration."""
    # Mock queries
    queries = pd.DataFrame({
        'query_id': ['q1', 'q2', 'q3', 'q4', 'q5'],
        'text': [
            'what is my name',
            'who am I',
            'name introduction',
            'personal identification',
            'hello world greeting'
        ]
    })
    
    # Mock passages
    passages = pd.DataFrame({
        'docid': ['d1', 'd2', 'd3', 'd4', 'd5', 'd6'],
        'text': [
            'My name is Ramo Shadi and I am a person',
            'Hello, I am introducing myself to you today',
            'Personal identification includes your name and information',
            'A greeting is a way to say hello to someone',
            'Names are important for personal identity',
            'World greetings vary across different cultures'
        ]
    })
    
    # Mock relevance judgments
    qrels = pd.DataFrame({
        'query_id': ['q1', 'q1', 'q2', 'q2', 'q3', 'q4', 'q5'],
        'docid': ['d1', 'd5', 'd1', 'd2', 'd3', 'd4', 'd6'],
        'relevance': [1, 1, 1, 1, 1, 1, 1]
    })
    
    return queries, passages, qrels


def load_real_marco_data(dataset_name: str = "msmarco-passage/dev/small", limit: int = 100):
    """Load real MS MARCO data if available."""
    if not HAS_IR_DATASETS:
        return load_mock_marco_data()
    
    try:
        dataset = ir_datasets.load(dataset_name)
        
        # Load documents
        docs_data = []
        for i, doc in enumerate(dataset.docs_iter()):
            if i >= limit:
                break
            docs_data.append({
                'docid': doc.doc_id,
                'text': doc.text
            })
        
        # Load queries  
        queries_data = []
        for i, query in enumerate(dataset.queries_iter()):
            if i >= limit:
                break
            queries_data.append({
                'query_id': query.query_id,
                'text': query.text
            })
        
        # Load qrels
        qrels_data = []
        for i, qrel in enumerate(dataset.qrels_iter()):
            if i >= limit * 5:  # More qrels than queries
                break
            qrels_data.append({
                'query_id': qrel.query_id,
                'docid': qrel.doc_id,
                'relevance': qrel.relevance
            })
        
        queries_df = pd.DataFrame(queries_data)
        passages_df = pd.DataFrame(docs_data)
        qrels_df = pd.DataFrame(qrels_data)
        
        return queries_df, passages_df, qrels_df
        
    except Exception as e:
        print(f"Error loading real MS MARCO data: {e}")
        print("Falling back to mock data")
        return load_mock_marco_data()


class WorkingPipeline:
    """Complete working pipeline demonstration."""
    
    def __init__(self, recordings_dir: str):
        self.recordings_dir = Path(recordings_dir)
        self.rag_service = MockRAGService()
        
        # Load MS MARCO data
        print("Loading MS MARCO dataset...")
        self.marco_queries, self.marco_passages, self.marco_qrels = load_real_marco_data(limit=50)
        print(f"Loaded {len(self.marco_queries)} queries, {len(self.marco_passages)} passages, {len(self.marco_qrels)} qrels")
    
    def load_recording_transcription(self, recording_id: str) -> str:
        """Load transcription text from recording."""
        txt_file = self.recordings_dir / f"{recording_id}.txt"
        
        if txt_file.exists():
            with open(txt_file, 'r') as f:
                return f.read().strip()
        return ""
    
    def populate_rag_with_passages(self, max_passages: int = 50):
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
        
        print(f"\n{'='*50}")
        print(f"Processing recording: {recording_id}")
        print(f"Transcription: '{transcription}'")
        print(f"{'='*50}")
        
        # Step 2: Query RAG service
        rag_results = self.rag_service.search_memories(transcription, max_results=10)
        print(f"RAG retrieved {len(rag_results)} passages")
        
        if rag_results:
            print("Top RAG results:")
            for i, result in enumerate(rag_results[:3]):
                score = result.metadata.get('score', 0)
                content = result.content[:100] + "..." if len(result.content) > 100 else result.content
                print(f"  {i+1}. (score: {score:.3f}) {content}")
        
        # Step 3: Find similar MS MARCO queries
        similar_queries = self.find_similar_marco_queries(transcription, top_k=3)
        print(f"\nFound {len(similar_queries)} similar MS MARCO queries:")
        
        # Step 4: Evaluate performance
        evaluation = {
            "recording_id": recording_id,
            "transcription": transcription,
            "rag_results_count": len(rag_results),
            "evaluations": []
        }
        
        # Evaluate against each similar MARCO query
        for query_id, query_text, similarity in similar_queries:
            print(f"\nEvaluating against MARCO query:")
            print(f"  Query: '{query_text}'")
            print(f"  Similarity to transcription: {similarity:.3f}")
            
            # Get ground truth for this query
            query_qrels = self.marco_qrels[self.marco_qrels['query_id'] == query_id]
            if len(query_qrels) == 0:
                print("  No ground truth available, skipping...")
                continue
            
            relevant_docs = set(query_qrels['docid'].astype(str))
            rag_doc_ids = [r.metadata['passage_id'] for r in rag_results if 'passage_id' in r.metadata]
            rag_doc_ids_str = [str(doc_id) for doc_id in rag_doc_ids]
            
            print(f"  Relevant docs: {len(relevant_docs)}")
            print(f"  RAG retrieved: {len(rag_doc_ids_str)}")
            
            # Calculate metrics
            metrics = {}
            for k in [1, 3, 5]:
                if len(rag_doc_ids_str) >= k:
                    top_k_docs = rag_doc_ids_str[:k]
                    metrics[f"precision@{k}"] = precision_at_k(top_k_docs, relevant_docs)
                    metrics[f"recall@{k}"] = recall_at_k(top_k_docs, relevant_docs)
                    
                    # NDCG with binary relevance
                    relevance_scores = [1.0 if str(doc) in relevant_docs else 0.0 for doc in top_k_docs]
                    metrics[f"ndcg@{k}"] = ndcg_at_k(relevance_scores, k)
                else:
                    metrics[f"precision@{k}"] = 0.0
                    metrics[f"recall@{k}"] = 0.0
                    metrics[f"ndcg@{k}"] = 0.0
            
            # MRR
            if rag_doc_ids_str:
                metrics["mrr"] = mrr_score([rag_doc_ids_str], [relevant_docs])
            else:
                metrics["mrr"] = 0.0
            
            print(f"  Metrics: P@5={metrics.get('precision@5', 0):.3f}, R@5={metrics.get('recall@5', 0):.3f}, MRR={metrics.get('mrr', 0):.3f}")
            
            evaluation["evaluations"].append({
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
    
    def run_complete_demo(self):
        """Run the complete pipeline demo."""
        print("=" * 80)
        print("COMPLETE END-TO-END PIPELINE DEMONSTRATION")
        print("Recording → Transcription → RAG → MS MARCO Evaluation → Comparison")
        print("=" * 80)
        
        # Populate RAG
        self.populate_rag_with_passages(max_passages=40)
        
        # Find available recordings
        recordings = []
        for txt_file in self.recordings_dir.glob("*.txt"):
            recording_id = txt_file.stem
            if (self.recordings_dir / f"{recording_id}.json").exists():
                recordings.append(recording_id)
        
        print(f"\nFound {len(recordings)} recordings with transcriptions")
        
        # Evaluate recordings
        all_results = []
        for recording_id in recordings[:2]:  # Limit for demo
            try:
                result = self.evaluate_recording(recording_id)
                if "error" not in result:
                    all_results.append(result)
            except Exception as e:
                print(f"Error processing {recording_id}: {e}")
        
        # Generate comprehensive summary
        self.generate_summary(all_results)
    
    def generate_summary(self, results: List[Dict[str, Any]]):
        """Generate comprehensive summary of pipeline results."""
        print("\n" + "=" * 80)
        print("PIPELINE EVALUATION SUMMARY")
        print("=" * 80)
        
        if not results:
            print("No results to summarize.")
            return
        
        all_metrics = []
        total_evaluations = 0
        
        for result in results:
            print(f"\nRecording: {result['recording_id']}")
            print(f"Transcription: '{result['transcription']}'")
            print(f"RAG Results: {result['rag_results_count']}")
            print(f"Evaluated against {len(result['evaluations'])} MS MARCO queries")
            
            for eval_data in result['evaluations']:
                metrics = eval_data['metrics']
                all_metrics.append(metrics)
                total_evaluations += 1
                
                print(f"  vs '{eval_data['query_text'][:50]}...' (sim={eval_data['similarity']:.3f}):")
                print(f"    P@5={metrics.get('precision@5', 0):.3f}, "
                      f"R@5={metrics.get('recall@5', 0):.3f}, "
                      f"NDCG@5={metrics.get('ndcg@5', 0):.3f}, "
                      f"MRR={metrics.get('mrr', 0):.3f}")
        
        # Calculate averages
        if all_metrics:
            avg_precision_1 = sum(m.get('precision@1', 0) for m in all_metrics) / len(all_metrics)
            avg_precision_5 = sum(m.get('precision@5', 0) for m in all_metrics) / len(all_metrics)
            avg_recall_5 = sum(m.get('recall@5', 0) for m in all_metrics) / len(all_metrics)
            avg_ndcg_5 = sum(m.get('ndcg@5', 0) for m in all_metrics) / len(all_metrics)
            avg_mrr = sum(m.get('mrr', 0) for m in all_metrics) / len(all_metrics)
            
            print(f"\n" + "=" * 50)
            print("OVERALL PERFORMANCE METRICS")
            print("=" * 50)
            print(f"Total recordings evaluated: {len(results)}")
            print(f"Total query comparisons: {total_evaluations}")
            print(f"Average Precision@1: {avg_precision_1:.4f}")
            print(f"Average Precision@5: {avg_precision_5:.4f}")
            print(f"Average Recall@5: {avg_recall_5:.4f}")
            print(f"Average NDCG@5: {avg_ndcg_5:.4f}")
            print(f"Average MRR: {avg_mrr:.4f}")
            
            print(f"\n" + "=" * 50)
            print("ANALYSIS & INSIGHTS")
            print("=" * 50)
            print("• This pipeline demonstrates the complete flow from recorded audio")
            print("  through transcription, RAG retrieval, and MS MARCO evaluation")
            print("• The scores compare how well your recordings work as search queries")
            print("  against a standard information retrieval benchmark")
            print("• Low scores are expected since recordings are conversational")
            print("  while MS MARCO is optimized for search queries")
            print("• This framework can be extended to evaluate real RAG systems")
            print("  and compare different retrieval approaches")


def main():
    """Run the complete working pipeline demo."""
    project_root = Path(__file__).parent.parent.parent
    recordings_dir = project_root / "packages" / "api" / "recordings"
    
    if not recordings_dir.exists():
        print(f"Recordings directory not found: {recordings_dir}")
        return
    
    pipeline = WorkingPipeline(str(recordings_dir))
    pipeline.run_complete_demo()


if __name__ == "__main__":
    main()
