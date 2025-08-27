"""
Connect your real RAG service to the evaluation framework.

This bridges your existing RAG system with the MS MARCO evaluation.
"""

import sys
from pathlib import Path

# Add API package to path
api_path = Path(__file__).parent.parent.parent / "api"
sys.path.insert(0, str(api_path))

# Import your real RAG service
from api.rag.rag_service import SimpleRAGService

class RealRAGEvaluationClient:
    """Client that uses your actual RAG service for evaluation."""
    
    def __init__(self):
        """Initialize with your real RAG service."""
        self.rag_service = SimpleRAGService()
        self._populate_from_recordings()
    
    def _populate_from_recordings(self):
        """Load your actual recorded memories into RAG service."""
        # Load from your recordings directory
        recordings_dir = Path(__file__).parent.parent.parent / "api" / "recordings"
        
        if recordings_dir.exists():
            json_files = list(recordings_dir.glob("*.json"))
            print(f"📁 Loading {len(json_files)} recorded memories...")
            
            for json_file in json_files[:50]:  # Limit for evaluation speed
                try:
                    import json
                    with open(json_file, 'r') as f:
                        recording_data = json.load(f)
                    
                    # Extract text content
                    if 'transcription' in recording_data:
                        text = recording_data['transcription']
                        audio_file = json_file.stem + '.wav'
                        self.rag_service.add_memory(text, audio_file)
                        
                except Exception as e:
                    print(f"⚠️  Error loading {json_file}: {e}")
            
            print(f"✅ Loaded {len(self.rag_service.memories)} memories from recordings")
        else:
            print("⚠️  No recordings directory found, using test data")
            self._add_test_memories()
    
    def _add_test_memories(self):
        """Add test memories if no recordings available."""
        test_memories = [
            "Machine learning algorithms can learn patterns from data without explicit programming.",
            "Python is widely used for data science and artificial intelligence applications.", 
            "Neural networks are inspired by the structure of biological brain networks.",
            "Deep learning models require large amounts of training data to perform well.",
            "Natural language processing enables computers to understand human language.",
        ]
        
        for i, memory in enumerate(test_memories):
            self.rag_service.add_memory(memory, f"test_audio_{i}.wav")
    
    def retrieve_and_rank(self, query: str, top_k: int = 10):
        """
        Use your real RAG service to retrieve and rank documents.
        
        Args:
            query: Search query
            top_k: Number of top results to return
            
        Returns:
            list: Document IDs ranked by relevance
        """
        # Use your RAG service's search logic
        query_words = set(query.lower().split())
        scored_docs = []
        
        for memory in self.rag_service.memories:
            memory_words = set(memory["text"].lower().split())
            overlap = len(query_words.intersection(memory_words))
            
            if overlap > 0:
                # Calculate relevance score (same as your RAG service)
                score = overlap / len(query_words) if query_words else 0
                scored_docs.append({
                    'id': f"memory_{memory['id']}",
                    'score': score,
                    'text': memory['text']
                })
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return [doc['id'] for doc in scored_docs[:top_k]]
    
    def generate_answer(self, query: str) -> str:
        """Use your real RAG service to generate an answer."""
        return self.rag_service.search_memories(query)

# Integration function for your existing evaluation
def use_real_rag_in_evaluation():
    """
    Example of how to integrate your real RAG into the MS MARCO evaluation.
    """
    print("🔗 Connecting Real RAG Service to Evaluation")
    print("=" * 50)
    
    # Initialize your real RAG client
    rag_client = RealRAGEvaluationClient()
    
    # Test with sample queries
    test_queries = [
        "What is machine learning?",
        "How does Python work for AI?", 
        "Tell me about neural networks"
    ]
    
    print("🧪 Testing Real RAG Service:")
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        
        # Get ranked results
        ranked_docs = rag_client.retrieve_and_rank(query, top_k=5)
        print(f"   📄 Retrieved: {len(ranked_docs)} documents")
        
        # Generate answer
        answer = rag_client.generate_answer(query)
        print(f"   💬 Answer: {answer[:100]}...")
    
    return rag_client

if __name__ == "__main__":
    use_real_rag_in_evaluation()
