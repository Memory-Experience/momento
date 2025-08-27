"""
MS MARCO Dataset Adapter for Generic Evaluation Framework

This adapter integrates MS MARCO dataset with the generic evaluation framework,
making it easy to evaluate any retrieval system against MS MARCO benchmarks.
"""

import sys
import logging
import pandas as pd
from pathlib import Path
from typing import Optional

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

# Import generic framework
sys.path.insert(0, str(current_dir.parent))
from generic_evaluation_framework import DatasetInterface

try:
    from dataset_marco.prepare_ms_marco import MSMarcoDataset
    from dataset_marco.run_marco_eval import convert_ms_marco_to_dataframes
    MARCO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"MS MARCO not available: {e}")
    MARCO_AVAILABLE = False

class MSMarcoDatasetAdapter(DatasetInterface):
    """
    Adapter to use MS MARCO dataset with the generic evaluation framework.
    """
    
    def __init__(self, dataset_name: str = "msmarco-passage/dev/small", limit: int = 1000):
        """
        Initialize MS MARCO dataset adapter.
        
        Args:
            dataset_name: MS MARCO dataset variant to use
            limit: Maximum number of items to load (for performance)
        """
        if not MARCO_AVAILABLE:
            raise RuntimeError("MS MARCO dataset dependencies not available")
            
        self.dataset_name = dataset_name
        self.limit = limit
        
        # Load data
        logging.info(f"🔄 Loading MS MARCO dataset: {dataset_name} (limit: {limit})")
        try:
            self.marco_dataset = MSMarcoDataset(dataset_name)
            self.docs_df, self.queries_df, self.qrels_df = convert_ms_marco_to_dataframes(
                self.marco_dataset, limit=limit
            )
            logging.info(f" MS MARCO loaded: {len(self.docs_df)} docs, {len(self.queries_df)} queries")
        except Exception as e:
            logging.error(f" Failed to load MS MARCO: {e}")
            raise
    
    def get_queries(self) -> pd.DataFrame:
        """Return queries DataFrame with columns: ['id', 'text']"""
        return self.queries_df
    
    def get_documents(self) -> pd.DataFrame:
        """Return documents DataFrame with columns: ['id', 'content']"""
        return self.docs_df
    
    def get_relevance_judgments(self) -> pd.DataFrame:
        """Return relevance judgments DataFrame with columns: ['query_id', 'doc_id', 'relevance']"""
        return self.qrels_df
    
    def get_name(self) -> str:
        """Return dataset name for logging/reporting"""
        return f"MS MARCO ({self.dataset_name}, limit={self.limit})"
    
    def get_sample_query(self) -> Optional[dict]:
        """Get a sample query for testing."""
        if self.queries_df.empty:
            return None
        
        sample_row = self.queries_df.iloc[0]
        query_id = sample_row['id']
        
        # Get relevant documents for this query
        relevant_qrels = self.qrels_df[self.qrels_df['query_id'] == query_id]
        
        return {
            'id': query_id,
            'text': sample_row['text'],
            'relevant_docs': relevant_qrels['doc_id'].tolist(),
            'relevance_scores': dict(zip(relevant_qrels['doc_id'], relevant_qrels['relevance']))
        }

def create_ms_marco_adapter(dataset_name: str = "msmarco-passage/dev/small", limit: int = 1000) -> Optional[MSMarcoDatasetAdapter]:
    """
    Factory function to create MS MARCO adapter with error handling.
    
    Args:
        dataset_name: MS MARCO dataset variant
        limit: Maximum items to load
        
    Returns:
        MSMarcoDatasetAdapter instance or None if failed
    """
    try:
        return MSMarcoDatasetAdapter(dataset_name, limit)
    except Exception as e:
        logging.error(f" Failed to create MS MARCO adapter: {e}")
        return None

# Example usage
if __name__ == "__main__":
    # Test the adapter
    adapter = create_ms_marco_adapter(limit=100)
    
    if adapter:
        print(f" Adapter created: {adapter.get_name()}")
        print(f" Queries: {len(adapter.get_queries())}")
        print(f" Documents: {len(adapter.get_documents())}")
        print(f" Relevance judgments: {len(adapter.get_relevance_judgments())}")
        
        # Show sample
        sample = adapter.get_sample_query()
        if sample:
            print(f"\n Sample Query:")
            print(f"  ID: {sample['id']}")
            print(f"  Text: {sample['text']}")
            print(f"  Relevant docs: {len(sample['relevant_docs'])}")
    else:
        print(" Failed to create adapter")
