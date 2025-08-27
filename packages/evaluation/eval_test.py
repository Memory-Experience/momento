"""
Simple RAG evaluation test script.

This connects your evaluation framework to your actual RAG service.
"""

import sys
from pathlib import Path
from rag_client import RAGEvaluationClient
from packages.evaluation.dataset_marco.dataframe_dataset import DataFrameDataset
import pandas as pd

def create_test_dataset():
    """Create a small test dataset for evaluation."""
    # Test queries
    queries_data = [
        {'id': 'q1', 'text': 'What is machine learning?'},
        {'id': 'q2', 'text': 'Tell me about Python programming'},
        {'id': 'q3', 'text': 'How do neural networks work?'},
        {'id': 'q4', 'text': 'What is deep learning?'},
        {'id': 'q5', 'text': 'Explain natural language processing'}
    ]
    
    # Test documents (these will be populated in RAG service)
    docs_data = []
    for i in range(10):
        docs_data.append({
            'id': f'doc_{i}',
            'text': f'Test document {i}',
            'title': f'Document {i}'
        })
    
    # Test relevance judgments
    qrels_data = [
        {'query_id': 'q1', 'doc_id': 'doc_0', 'relevance': 1},
        {'query_id': 'q2', 'doc_id': 'doc_1', 'relevance': 1},
        {'query_id': 'q3', 'doc_id': 'doc_2', 'relevance': 1},
        {'query_id': 'q4', 'doc_id': 'doc_3', 'relevance': 1},
        {'query_id': 'q5', 'doc_id': 'doc_4', 'relevance': 1}
    ]
    
    return DataFrameDataset(
        pd.DataFrame(docs_data),
        pd.DataFrame(queries_data),
        pd.DataFrame(qrels_data)
    )

def evaluate_rag_system():
    """Run evaluation of your RAG system."""
    print("🚀 Starting RAG System Evaluation")
    print("=" * 50)
    
    # Initialize RAG client
    rag_client = RAGEvaluationClient(service_type="local")
    
    # Create test dataset
    dataset = create_test_dataset()
    
    print(f"📊 Dataset: {len(dataset.docs)} docs, {len(dataset.queries)} queries")
    print()
    
    # Run evaluation
    results = []
    total_retrieval_time = 0
    total_generation_time = 0
    
    for _, query_row in dataset.queries.iterrows():
        query_id = query_row['id']
        query_text = query_row['text']
        
        print(f"🔍 Evaluating: {query_text}")
        
        # Query RAG system
        rag_result = rag_client.query(query_text, top_k=5)
        
        # Collect timing
        timing = rag_result['timing']
        total_retrieval_time += timing['retrieval_ms']
        total_generation_time += timing['generation_ms']
        
        # Show results
        print(f"   📝 Answer: {rag_result['answer'][:100]}...")
        print(f"   ⏱️  Timing: {timing['total_ms']:.1f}ms")
        print(f"   📄 Retrieved {len(rag_result['retrieved_docs'])} docs")
        print()
        
        results.append({
            'query_id': query_id,
            'query': query_text,
            'answer': rag_result['answer'],
            'timing': timing,
            'retrieved_docs': rag_result['retrieved_docs']
        })
    
    # Print summary
    print("📈 EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Total queries processed: {len(results)}")
    print(f"Average retrieval time: {total_retrieval_time/len(results):.1f}ms")
    print(f"Average generation time: {total_generation_time/len(results):.1f}ms")
    print(f"Average total time: {(total_retrieval_time + total_generation_time)/len(results):.1f}ms")
    
    return results

def main():
    """Main evaluation entry point."""
    try:
        results = evaluate_rag_system()
        print("\n✅ Evaluation completed successfully!")
        print(f"📁 Processed {len(results)} queries")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
