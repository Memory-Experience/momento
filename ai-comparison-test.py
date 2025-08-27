#!/usr/bin/env python3
"""
MS MARCO Comparison: "What is artificial intelligence"
Using your implementation to compare Simple RAG vs MS MARCO
"""

import sys
from pathlib import Path

# Add your packages to path
project_root = Path(__file__).parent
api_path = project_root / "packages" / "api"
eval_path = project_root / "packages" / "evaluation"
sys.path.extend([str(api_path), str(eval_path)])

try:
    from rag.comparative_rag_service import ComparativeRAGService
    from rag.rag_service import SimpleRAGService
    print("✅ Successfully imported your RAG services")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def run_ai_comparison():
    """Compare Simple RAG vs MS MARCO on AI question."""
    
    print("🤖 AI QUESTION COMPARISON TEST")
    print("=" * 50)
    
    query = "what is artificial intelligence"
    
    # Initialize both services
    print("📝 Initializing Simple RAG...")
    simple_rag = SimpleRAGService()
    
    print("🔬 Initializing MS MARCO Comparative RAG...")
    try:
        comparative_rag = ComparativeRAGService(enable_marco_comparison=True)
        marco_available = True
    except Exception as e:
        print(f"⚠️ MS MARCO not available: {e}")
        marco_available = False
        return
    
    # Add your actual AI memory
    print("\n📚 Adding your actual AI memory to both services...")
    your_ai_memory = "artificial intelligence. Intelligence is an intelligence. demonstrated by machines. It is different than the entire intelligence of humans and animals. especially distinguished by the data."
    
    simple_rag.add_memory(your_ai_memory)
    comparative_rag.add_memory(your_ai_memory)
    
    print(f"Added your AI memory: '{your_ai_memory[:50]}...'")
    
    # Test the query
    print(f"\n🔍 TESTING QUERY: '{query}'")
    print("=" * 60)
    
    # Simple RAG result
    print("\n📝 SIMPLE RAG RESULT:")
    print("-" * 30)
    simple_result = simple_rag.search_memories(query)
    print(simple_result)
    
    # MS MARCO Comparative result
    print("\n🔬 MS MARCO COMPARISON RESULT:")
    print("-" * 30)
    if marco_available:
        try:
            comparative_result = comparative_rag.search_memories(query, include_comparison=True)
            print(comparative_result)
            
            # Get actual MS MARCO metrics
            print("\n📈 MS MARCO METRICS:")
            print("-" * 25)
            if hasattr(comparative_rag, 'marco_service'):
                marco_service = comparative_rag.marco_service
                # Get related queries from MS MARCO
                related_queries = marco_service.get_related_queries(query, limit=3)
                print(f"📋 Related MS MARCO queries ({len(related_queries)}):")
                for i, q in enumerate(related_queries[:3], 1):
                    print(f"   {i}. {q}")
                
                # Get document matches
                doc_matches = marco_service.get_matching_documents(query, limit=5)
                print(f"\n📄 MS MARCO document matches ({len(doc_matches)}):")
                for i, doc in enumerate(doc_matches[:3], 1):
                    content = doc['content'][:100] + "..." if len(doc['content']) > 100 else doc['content']
                    print(f"   {i}. {content}")
                    
        except Exception as e:
            print(f"❌ MS MARCO comparison failed: {e}")
            # Fallback to simple comparison
            print("\n⚠️ Falling back to simple mode:")
            fallback_result = comparative_rag.search_memories(query, include_comparison=False)
            print(fallback_result)
    
    # Performance summary
    if marco_available and hasattr(comparative_rag, 'get_performance_summary'):
        print("\n📊 PERFORMANCE SUMMARY:")
        print("-" * 30)
        summary = comparative_rag.get_performance_summary()
        for key, value in summary.items():
            print(f"   {key}: {value}")
    
    print("\n🎯 COMPARISON ANALYSIS:")
    print("-" * 30)
    print("• Your Memory: 'artificial intelligence. Intelligence is an intelligence...'")
    print("• Simple RAG: Uses only your recorded memory")
    print("• MS MARCO: Compares your memory against 1000+ passages")
    print("• Quality Check: How does your AI definition compare to dataset?")
    print("• Similar Queries: MS MARCO finds related AI questions")
    
    # Show detailed comparison
    print("\n🔍 DETAILED MEMORY VS MARCO COMPARISON:")
    print("-" * 45)
    print(f"📝 Your AI Memory:")
    print(f"   '{your_ai_memory}'")
    print(f"\n🔬 MS MARCO Dataset Context:")
    print(f"   - Contains {summary.get('marco_dataset_size', 'N/A')} passages")
    print(f"   - Professional AI definitions and explanations")
    print(f"   - Industry-standard benchmark for AI questions")

if __name__ == "__main__":
    try:
        run_ai_comparison()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
