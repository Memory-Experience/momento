"""
RAG System Adapter for Generic Evaluation Framework

This adapter connects the existing RAG service to the generic evaluation framework,
allowing to evaluate it against any dataset (MS MARCO, personal memories, etc.).
"""

import sys
import logging
from pathlib import Path
from typing import List, Optional

# Add api package to path
api_path = Path(__file__).parent.parent.parent / "api"
sys.path.insert(0, str(api_path))

# Import generic framework
sys.path.insert(0, str(Path(__file__).parent.parent))
from generic_evaluation_framework import RetrievalSystemInterface

class SimpleRAGAdapter(RetrievalSystemInterface):
    """
    Adapter to use SimpleRAGService with the generic evaluation framework.
    """
    
    def __init__(self):
        """Initialize the RAG adapter with existing service."""
        try:
            # Import from api package
            from rag.rag_service import SimpleRAGService  # type: ignore
            self.rag_service = SimpleRAGService()
            logging.info(" SimpleRAGService initialized for evaluation")
        except ImportError as e:
            logging.error(f" Failed to import SimpleRAGService: {e}")
            self.rag_service = None
    
    def search(self, query: str, top_k: int = 10) -> List[str]:
        """
        Search using RAG service.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of document IDs ranked by relevance
        """
        if self.rag_service is None:
            logging.warning(" RAG service not available, returning empty results")
            return []
            
        try:
            # Use existing search method
            results = self.rag_service.search_memories(query, limit=top_k)
            
            # Extract document IDs from results
            # Assuming results have an 'id' field - adapt as needed
            doc_ids = []
            for result in results:
                if hasattr(result, 'id'):
                    doc_ids.append(str(result.id))
                elif isinstance(result, dict) and 'id' in result:
                    doc_ids.append(str(result['id']))
                else:
                    # Fallback: use index as ID
                    doc_ids.append(str(len(doc_ids)))
            
            return doc_ids[:top_k]
            
        except Exception as e:
            logging.error(f" Search failed: {e}")
            return []

class ComparativeRAGAdapter(RetrievalSystemInterface):
    """
    Adapter to use ComparativeRAGService with the generic evaluation framework.
    """
    
    def __init__(self, enable_marco_comparison: bool = False):
        """Initialize the comparative RAG adapter."""
        try:
            # Import from api package
            from rag.comparative_rag_service import ComparativeRAGService  # type: ignore
            self.rag_service = ComparativeRAGService(enable_marco_comparison=enable_marco_comparison)
            logging.info(" ComparativeRAGService initialized for evaluation")
        except ImportError as e:
            logging.error(f" Failed to import ComparativeRAGService: {e}")
            self.rag_service = None
    
    def search(self, query: str, top_k: int = 10) -> List[str]:
        """
        Search using comparative RAG service.
        
        Args:
            query: The search query
            top_k: Number of results to return
            
        Returns:
            List of document IDs ranked by relevance
        """
        if self.rag_service is None:
            logging.warning(" Comparative RAG service not available, returning empty results")
            return []
            
        try:
            # Use existing search method
            results = self.rag_service.search_memories(query, include_comparison=False)
            
            # Extract document IDs from results
            doc_ids = []
            for result in results:
                if hasattr(result, 'id'):
                    doc_ids.append(str(result.id))
                elif isinstance(result, dict) and 'id' in result:
                    doc_ids.append(str(result['id']))
                else:
                    # Fallback: use index as ID
                    doc_ids.append(str(len(doc_ids)))
            
            return doc_ids[:top_k]
            
        except Exception as e:
            logging.error(f" Search failed: {e}")
            return []

class MockRAGAdapter(RetrievalSystemInterface):
    """
    Mock RAG adapter for testing the evaluation framework.
    Returns random/simple results for testing purposes.
    """
    
    def __init__(self, documents_df=None):
        """Initialize with mock data."""
        self.documents_df = documents_df
        logging.info(" MockRAGAdapter initialized for testing")
    
    def search(self, query: str, top_k: int = 10) -> List[str]:
        """
        Mock search that returns random documents.
        In a real scenario, this would use embeddings, BM25, etc.
        """
        if self.documents_df is not None and not self.documents_df.empty:
            # Return first top_k document IDs (simple mock)
            available_ids = self.documents_df['id'].tolist()
            return available_ids[:top_k]
        else:
            # Return mock IDs
            return [f"doc_{i}" for i in range(top_k)]

def create_rag_adapter(adapter_type: str = "simple", **kwargs) -> Optional[RetrievalSystemInterface]:
    """
    Factory function to create RAG adapters with error handling.
    
    Args:
        adapter_type: Type of adapter ("simple", "comparative", "mock")
        **kwargs: Additional arguments for the adapter
        
    Returns:
        RetrievalSystemInterface instance or None if failed
    """
    try:
        if adapter_type == "simple":
            return SimpleRAGAdapter()
        elif adapter_type == "comparative":
            return ComparativeRAGAdapter(**kwargs)
        elif adapter_type == "mock":
            return MockRAGAdapter(**kwargs)
        else:
            raise ValueError(f"Unknown adapter type: {adapter_type}")
    except Exception as e:
        logging.error(f" Failed to create RAG adapter ({adapter_type}): {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Test the adapters
    logging.basicConfig(level=logging.INFO)
    
    print(" Testing RAG Adapters...")
    
    # Test mock adapter
    mock_adapter = create_rag_adapter("mock")
    if mock_adapter:
        results = mock_adapter.search("test query", top_k=3)
        print(f" Mock adapter: {results}")
    
    # Test simple adapter (may fail if not available)
    simple_adapter = create_rag_adapter("simple")
    if simple_adapter:
        print(" Simple adapter created successfully")
    else:
        print(" Simple adapter not available")
    
    # Test comparative adapter
    comp_adapter = create_rag_adapter("comparative", enable_marco_comparison=False)
    if comp_adapter:
        print(" Comparative adapter created successfully")
    else:
        print(" Comparative adapter not available")
