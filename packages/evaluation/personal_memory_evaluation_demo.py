"""
Personal Memory Evaluation Demo

This demonstrates how to use the generic evaluation framework to evaluate
RAG systems against personal memory datasets. This represents a real-world use case
for evaluating how well systems retrieve personal memories.

Usage:
    python personal_memory_evaluation_demo.py
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
    """Run personal memory evaluation demo."""
    print(" PERSONAL MEMORY RAG EVALUATION")
    print("=" * 60)
    print("Use Case: Evaluate how well RAG systems retrieve personal memories")
    print("Example: User asks 'Tell me about my vacation in France'")
    print("Goal: System should find the RIGHT personal memories from user's database")
    
    # Import components
    try:
        from generic_evaluation_framework import GenericEvaluationEngine
        from dataset_adapters.personal_memory_adapter import (
            create_personal_memory_adapter, 
            SyntheticPersonalMemoryDataset
        )
        from dataset_adapters.rag_adapter import create_rag_adapter
    except ImportError as e:
        print(f" Import failed: {e}")
        return
    
    # Step 1: Create or load personal memory dataset
    print("\n STEP 1: Setting up Personal Memory Dataset")
    print("-" * 50)
    
    dataset_dir = Path("./sample_personal_memory_dataset")
    
    # Check if dataset already exists, if not create it
    if not dataset_dir.exists() or not (dataset_dir / "memories.csv").exists():
        print("  Creating synthetic personal memory dataset...")
        SyntheticPersonalMemoryDataset.create_sample_dataset(
            output_dir=dataset_dir,
            num_memories=50,
            num_queries=15
        )
    else:
        print(" Using existing personal memory dataset")
    
    # Load the dataset
    dataset = create_personal_memory_adapter(dataset_dir)
    if not dataset:
        print(" Failed to load personal memory dataset")
        return
    
    print(f" Dataset loaded: {dataset.get_name()}")
    print(f"    Memories: {len(dataset.get_documents())}")
    print(f"    Queries: {len(dataset.get_queries())}")
    print(f"    Relevance judgments: {len(dataset.get_relevance_judgments())}")
    
    # Show sample personal memory scenario
    sample_queries = dataset.get_queries().head(3)
    print(f"\n Sample Personal Memory Queries:")
    for _, query in sample_queries.iterrows():
        print(f"   * {query['text']}")
    
    # Step 2: Create RAG system adapter
    print(f"\n STEP 2: Setting up RAG System for Personal Memory Retrieval")
    print("-" * 60)
    
    # For personal memory evaluation, we'll use a mock adapter that simulates
    # a memory retrieval system. In practice, this would connect to an
    # actual vector database with user's recorded memories.
    
    rag_system = create_rag_adapter("mock", documents_df=dataset.get_documents())
    if not rag_system:
        print(" Failed to create RAG system")
        return
    
    print(" Mock RAG system created (simulates personal memory retrieval)")
    print(" In production, this would connect to the user's vector database")
    
    # Step 3: Run evaluation
    print(f"\n STEP 3: Evaluating Personal Memory Retrieval")
    print("-" * 50)
    
    evaluation_engine = GenericEvaluationEngine(dataset, rag_system)
    
    results = evaluation_engine.evaluate(
        sample_size=10,  # Evaluate 10 personal memory queries
        k_values=[1, 3, 5]  # Check top-1, top-3, top-5 retrieved memories
    )
    
    # Step 4: Analyze results for personal memory context
    print(f"\n STEP 4: Personal Memory Retrieval Results")
    print("-" * 50)
    
    evaluation_engine.print_results(results)
    
    # Step 5: Interpret results for personal memory use case
    print(f"\n STEP 5: What These Metrics Mean for Personal Memory")
    print("-" * 60)
    
    if results['query_results']:
        avg_precision_at_1 = results['avg_metrics']['precision@1']
        avg_recall_at_5 = results['avg_metrics']['recall@5']
        avg_mrr = results['avg_metrics']['mrr']
        
        print(f" Key Insights:")
        print(f"   Precision@1: {avg_precision_at_1:.1%}")
        print(f"   → How often the TOP memory retrieved is actually relevant")
        print(f"   → User satisfaction: Is the first result what they wanted?")
        print()
        print(f"   Recall@5: {avg_recall_at_5:.1%}")
        print(f"   → How many relevant memories are found in top 5 results")
        print(f"   → Coverage: Does the system find all related memories?")
        print()
        print(f"   MRR: {avg_mrr:.3f}")
        print(f"   → Average position of first relevant memory")
        print(f"   → User experience: How quickly do they find what they want?")
        
        # Show specific examples
        print(f"\n Example Personal Memory Retrievals:")
        for i, query_result in enumerate(results['query_results'][:3]):
            print(f"\nScenario {i+1}:")
            print(f"  User Query: '{query_result['query_text']}'")
            print(f"  Relevant Memories Available: {query_result['relevant_count']}")
            print(f"  System Retrieved: {query_result['retrieved_count']} memories")
            print(f"  Success Metrics:")
            print(f"    - Found right memory immediately: {query_result['metrics']['precision@1']:.0%}")
            print(f"    - Found most relevant memories: {query_result['metrics']['recall@5']:.0%}")
            
            if query_result['metrics']['precision@1'] == 1.0:
                print(f"     Perfect! User got exactly what they wanted")
            elif query_result['metrics']['recall@5'] > 0.5:
                print(f"     Good! User can find their memories with some browsing")
            else:
                print(f"     Poor! User might not find their memories")
    
    print(f"\n IMPROVEMENT RECOMMENDATIONS")
    print("-" * 40)
    print("Based on these results, one can:")
    print("1.  Improve embedding models for better semantic understanding")
    print("2.  Tune similarity thresholds for personal context")
    print("3.  Add temporal weighting (recent memories more important)")
    print("4.  Include metadata (location, people, activities) in retrieval")
    print("5.  Personalize ranking based on user's query patterns")
    
    print(f"\n NEXT STEPS")
    print("-" * 20)
    print("To evaluate a REAL personal memory system:")
    print("1. Export user memories from the vector database")
    print("2. Create realistic queries (or use real user queries)")
    print("3. Manually label which memories are relevant (gold standard)")
    print("4. Replace MockRAGAdapter with the actual RAG service")
    print("5. Run evaluation and iterate on improvements")
    
    print(f"\n PERSONAL MEMORY EVALUATION COMPLETE!")

if __name__ == "__main__":
    main()
