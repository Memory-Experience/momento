"""
RAG service client for evaluation.

This module provides a client to connect the evaluation framework
to your actual RAG service.
"""

import grpc
import time
from typing import Dict, List, Any
import sys
from pathlib import Path

# Add project root for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import your RAG service
from packages.api.rag.rag_service import SimpleRAGService


class RAGEvaluationClient:
    """Client to interface with RAG service for evaluation."""
    
    def __init__(self, service_type: str = "local"):
        """Initialize RAG client.
        
        Args:
            service_type: "local" for direct service, "grpc" for remote service
        """
        self.service_type = service_type
        
        if service_type == "local":
            # Direct connection to your RAG service
            self.rag_service = SimpleRAGService()
            # Pre-populate with some test memories
            self._populate_test_data()
        elif service_type == "grpc":
            # TODO: Add gRPC client when you have the service running
            self.grpc_channel = None
            self.grpc_stub = None
    
    def _populate_test_data(self):
        """Add some test memories to the RAG service."""
        test_memories = [
            "Machine learning is a subset of artificial intelligence that enables computers to learn without being explicitly programmed.",
            "Python is a high-level programming language known for its simplicity and readability.",
            "Neural networks are computing systems inspired by biological neural networks.",
            "Deep learning uses neural networks with multiple layers to model complex patterns.",
            "Natural language processing helps computers understand and generate human language.",
            "Information retrieval systems help users find relevant documents from large collections.",
            "Vector databases store and search high-dimensional embeddings efficiently.",
            "Transformer models have revolutionized natural language processing tasks.",
            "RAG systems combine retrieval and generation for better question answering.",
            "Evaluation metrics like NDCG measure ranking quality in search systems."
        ]
        
        for i, memory in enumerate(test_memories):
            self.rag_service.add_memory(memory, f"test_audio_{i}.wav")
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """Query the RAG service and return structured results.
        
        Args:
            question: The input question
            top_k: Number of documents to retrieve
            
        Returns:
            Dictionary with query results, timing, and metadata
        """
        start_time = time.time()
        
        if self.service_type == "local":
            # Time the retrieval phase
            retrieval_start = time.time()
            # For SimpleRAGService, we'll simulate retrieval
            answer = self.rag_service.search_memories(question)
            retrieval_time = (time.time() - retrieval_start) * 1000
            
            # Simulate generation time
            generation_start = time.time()
            # The search_memories already includes generation
            generation_time = (time.time() - generation_start) * 1000
            
            # Extract retrieved documents (simulate this for evaluation)
            retrieved_docs = self._extract_retrieved_docs(question, top_k)
            
        elif self.service_type == "grpc":
            # TODO: Implement gRPC calls
            retrieval_time = 50.0  # Placeholder
            generation_time = 200.0  # Placeholder
            answer = "gRPC not implemented yet"
            retrieved_docs = []
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            'query': question,
            'answer': answer,
            'retrieved_docs': retrieved_docs,
            'timing': {
                'retrieval_ms': retrieval_time,
                'generation_ms': generation_time,
                'total_ms': total_time
            }
        }
    
    def _extract_retrieved_docs(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Extract retrieved documents for evaluation.
        
        This simulates the retrieval results for evaluation purposes.
        """
        query_words = set(query.lower().split())
        scored_docs = []
        
        for memory in self.rag_service.memories:
            memory_words = set(memory["text"].lower().split())
            overlap = len(query_words.intersection(memory_words))
            
            if overlap > 0:
                score = overlap / len(query_words)  # Simple relevance score
                scored_docs.append({
                    'id': f"doc_{memory['id']}",
                    'score': score,
                    'text': memory['text'][:200] + "..." if len(memory['text']) > 200 else memory['text']
                })
        
        # Sort by score and return top_k
        scored_docs.sort(key=lambda x: x['score'], reverse=True)
        return scored_docs[:top_k]
