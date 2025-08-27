"""
Complete Generic Evaluation Demo

This demo shows how to use the generic evaluation framework to evaluate
RAG systems against MS MARCO dataset. It's designed to be:

1. Easy to run
2. Clear results
3. Extensible to other datasets
4. Production-ready

Usage:
    python complete_evaluation_demo.py
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    """Run complete evaluation demo."""
    print(" GENERIC RAG EVALUATION FRAMEWORK")
    print("=" * 60)
    
    # Import components
    try:
        from generic_evaluation_framework import GenericEvaluationEngine
        from dataset_adapters.marco_adapter import create_ms_marco_adapter
        from dataset_adapters.rag_adapter import create_rag_adapter
    except ImportError as e:
        print(f" Import failed: {e}")
        print(" Make sure you're running from the evaluation package directory")
        return
    
    # Step 1: Create dataset adapter (MS MARCO)
    print("\n STEP 1: Loading MS MARCO Dataset")
    print("-" * 40)
    
    dataset = create_ms_marco_adapter(
        dataset_name="msmarco-passage/dev/small", 
        limit=100  # Small limit for demo
    )
    
    if not dataset:
        print(" Failed to load MS MARCO dataset")
        return
    
    print(f" Dataset loaded: {dataset.get_name()}")
    print(f"    Queries: {len(dataset.get_queries())}")
    print(f"    Documents: {len(dataset.get_documents())}")
    print(f"    Relevance judgments: {len(dataset.get_relevance_judgments())}")
    
    # Show sample query
    sample = dataset.get_sample_query()
    if sample:
        print(f"\n Sample Query:")
        print(f"   ID: {sample['id']}")
        print(f"   Text: {sample['text'][:100]}...")
        print(f"   Relevant docs: {len(sample['relevant_docs'])}")
    
    # Step 2: Create RAG system adapter
    print(f"\n STEP 2: Loading RAG System")
    print("-" * 40)
    
    # Try different RAG adapters in order of preference
    rag_adapters = [
        ("comparative", {"enable_marco_comparison": False}),
        ("simple", {}),
        ("mock", {"documents_df": dataset.get_documents()})
    ]
    
    rag_system = None
    for adapter_type, kwargs in rag_adapters:
        print(f"   Trying {adapter_type} adapter...")
        rag_system = create_rag_adapter(adapter_type, **kwargs)
        if rag_system:
            print(f" {adapter_type.title()} RAG adapter loaded successfully")
            break
        else:
            print(f" {adapter_type.title()} adapter not available")
    
    if not rag_system:
        print(" No RAG system available for evaluation")
        return
    
    # Step 3: Run evaluation
    print(f"\n🔬 STEP 3: Running Evaluation")
    print("-" * 40)
    
    evaluation_engine = GenericEvaluationEngine(dataset, rag_system)
    
    results = evaluation_engine.evaluate(
        sample_size=20,  # Small sample for demo
        k_values=[1, 3, 5, 10]
    )
    
    # Step 4: Display results
    print(f"\n STEP 4: Results")
    print("-" * 40)
    
    evaluation_engine.print_results(results)
    
    # Step 5: Show individual query examples
    print(f"\n STEP 5: Query Examples")
    print("-" * 40)
    
    if results['query_results']:
        for i, query_result in enumerate(results['query_results'][:3]):  # Show first 3
            print(f"\nQuery {i+1}:")
            print(f"  ID: {query_result['query_id']}")
            print(f"  Text: {query_result['query_text'][:80]}...")
            print(f"  Retrieved: {query_result['retrieved_count']} docs")
            print(f"  Relevant: {query_result['relevant_count']} docs")
            print(f"  Precision@1: {query_result['metrics']['precision@1']:.3f}")
            print(f"  Recall@5: {query_result['metrics']['recall@5']:.3f}")
            print(f"  MRR: {query_result['metrics']['mrr']:.3f}")
    
    print(f"\n EVALUATION COMPLETE!")
    print("=" * 60)
    print(" To extend this framework:")
    print("   1. Create new dataset adapters for personal memories")
    print("   2. Improve RAG systems based on these metrics")
    print("   3. Compare different RAG approaches")
    print("   4. Add custom metrics for specific use cases")

if __name__ == "__main__":
    main()
