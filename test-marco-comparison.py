#!/usr/bin/env python3
"""
Quick MS MARCO Comparison Test
Run this while your system is running to see the difference!
"""

import sys
from pathlib import Path

# Add the api package to Python path
api_path = Path(__file__).parent / "packages" / "api"
eval_path = Path(__file__).parent / "packages" / "evaluation"
sys.path.extend([str(api_path), str(eval_path)])

try:
    from rag.comparative_rag_service import ComparativeRAGService
    from rag.rag_service import SimpleRAGService
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Make sure you're running this from the project root!")
    sys.exit(1)

def test_comparison():
    print("🔬 MS MARCO Comparison Test")
    print("=" * 50)
    
    # Test queries
    test_queries = [
        "What is machine learning?",
        "How does Python work?",
        "Explain artificial intelligence",
        "What are neural networks?"
    ]
    
    # Initialize both services
    print("📝 Initializing Simple RAG...")
    simple_rag = SimpleRAGService()
    
    print("🔬 Initializing Comparative RAG with MS MARCO...")
    comparative_rag = ComparativeRAGService(enable_marco_comparison=True)
    
    # Add some test memories
    test_memories = [
        "Machine learning is a method where computers learn from data automatically",
        "Python is a programming language popular for data science and AI",
        "Artificial intelligence involves creating smart computer systems",
        "Neural networks are computing systems inspired by biological brains"
    ]
    
    print("\n📚 Adding test memories to both services...")
    for memory in test_memories:
        simple_rag.add_memory(memory)
        comparative_rag.add_memory(memory)
    
    # Test each query
    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"🔍 TEST {i}: {query}")
        print('='*60)
        
        print(f"\n📝 SIMPLE RAG RESULT:")
        print("-" * 30)
        simple_result = simple_rag.search_memories(query)
        print(simple_result)
        
        print(f"\n🔬 MS MARCO COMPARISON RESULT:")
        print("-" * 30)
        comparative_result = comparative_rag.search_memories(query, include_comparison=True)
        print(comparative_result)
        
        print("\n" + "="*60)
        input("Press Enter to continue to next test...")

if __name__ == "__main__":
    try:
        test_comparison()
    except KeyboardInterrupt:
        print("\n\n👋 Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
