"""
MS MARCO Dataset Adapter for Generic Evaluation Framework

This adapter integrates MS MARCO dataset with the generic evaluation framework,
making it easy to evaluate any retrieval system against MS MARCO benchmarks.
"""

import logging
import pandas as pd
from typing import Optional
import ir_datasets
from .dataset import DataFrameDataset


def _convert_ms_marco_to_dataframes(ms_marco_dataset, limit: int = 1000):
    """Convert MS MARCO dataset to pandas DataFrames with proper matching.
    
    Args:
        ms_marco_dataset: The MS MARCO dataset instance
        limit: Maximum number of items to process (for speed)
    
    Returns:
        tuple: (docs_df, queries_df, qrels_df)
    """
    print(f"Converting MS MARCO to DataFrames (limit: {limit})...")
    
    # STEP 1: Get qrels first to know which docs/queries we need
    qrels_data = []
    needed_doc_ids = set()
    needed_query_ids = set()
    
    for i, qrel in enumerate(ms_marco_dataset.qrels_iter()):
        if i >= limit:
            break
        qrels_data.append({
            'query_id': qrel.query_id,
            'doc_id': qrel.doc_id,
            'relevance': qrel.relevance
        })
        needed_doc_ids.add(qrel.doc_id)
        needed_query_ids.add(qrel.query_id)
    
    qrels_df = pd.DataFrame(qrels_data)
    print(f"  Loaded {len(qrels_df)} qrels")
    print(f"  Need {len(needed_doc_ids)} docs and {len(needed_query_ids)} queries")
    
    # STEP 2: Get documents that are actually needed
    docs_data = []
    found_doc_ids = set()
    
    for doc in ms_marco_dataset.docs_iter():
        if doc.doc_id in needed_doc_ids:
            docs_data.append({
                'id': doc.doc_id,
                'content': doc.text
            })
            found_doc_ids.add(doc.doc_id)
            
        # Stop if we found all needed docs
        if len(found_doc_ids) >= len(needed_doc_ids):
            break
    
    docs_df = pd.DataFrame(docs_data)
    print(f"  Found {len(docs_df)} out of {len(needed_doc_ids)} needed documents")
    
    # STEP 3: Get queries that are actually needed
    queries_data = []
    found_query_ids = set()
    
    for query in ms_marco_dataset.queries_iter():
        if query.query_id in needed_query_ids:
            queries_data.append({
                'id': query.query_id,
                'text': query.text
            })
            found_query_ids.add(query.query_id)
            
        # Stop if we found all needed queries
        if len(found_query_ids) >= len(needed_query_ids):
            break
    
    queries_df = pd.DataFrame(queries_data)
    print(f"  Found {len(queries_df)} out of {len(needed_query_ids)} needed queries")
    
    # STEP 4: Filter qrels to only include found docs and queries
    valid_qrels = qrels_df[
        qrels_df['doc_id'].isin(found_doc_ids) & 
        qrels_df['query_id'].isin(found_query_ids)
    ]
    
    print(f"  Final valid qrels: {len(valid_qrels)} (filtered from {len(qrels_df)})")
    print(f"  Final dataset: {len(docs_df)} docs, {len(queries_df)} queries, {len(valid_qrels)} qrels")
    
    return docs_df, queries_df, valid_qrels


class MSMarcoDataset(DataFrameDataset):
    """
    Adapter to use MS MARCO dataset with the generic evaluation framework.
    """
    
    def get_name(self) -> str:
        """Return the dataset name."""
        return f"MS MARCO ({self.subset_name}, limit={self.limit})"

    
    def __init__(self, dataset_name: str = "msmarco-passage/dev/small", limit: int = 1000):
        """
        Initialize MS MARCO dataset adapter.
        
        Args:
            dataset_name: MS MARCO dataset variant to use
            limit: Maximum number of items to load (for performance)
        """
            
        self.dataset_name = dataset_name
        self.limit = limit
        
        # Load data
        logging.info(f"Loading MS MARCO dataset: {dataset_name} (limit: {limit})")
        try:
            self.marco_dataset = ir_datasets.load(dataset_name)
            docs_df, queries_df, qrels_df = _convert_ms_marco_to_dataframes(
                self.marco_dataset, limit=limit
            )
            super().__init__(docs_df, queries_df, qrels_df)
            
            logging.info(f" MS MARCO loaded: {len(self._docs_df)} docs, {len(self._queries_df)} queries")
        except Exception as e:
            logging.error(f" Failed to load MS MARCO: {e}")
            raise
    
    def get_name(self) -> str:
        """Return dataset name for logging/reporting"""
        return f"MS MARCO ({self.dataset_name}, limit={self.limit})"
    
    def get_sample_query(self) -> Optional[dict]:
        """Get a sample query for testing."""
        if self.queries.empty:
            return None
        
        sample_row = self.queries.iloc[0]
        query_id = sample_row['id']
        
        # Get relevant documents for this query
        relevant_qrels = self.qrels[self.qrels['query_id'] == query_id]

        return {
            'id': query_id,
            'text': sample_row['text'],
            'relevant_docs': relevant_qrels['doc_id'].tolist(),
            'relevance_scores': dict(zip(relevant_qrels['doc_id'], relevant_qrels['relevance']))
        }

    @staticmethod
    def create(dataset_name: str = "msmarco-passage/dev/small", limit: int = 1000) -> Optional['MSMarcoDataset']:
        """
        Factory function to create MS MARCO adapter with error handling.
        
        Args:
            dataset_name: MS MARCO dataset variant
            limit: Maximum items to load
            
        Returns:
            MSMarcoDatasetAdapter instance or None if failed
        """
        try:
            return MSMarcoDataset(dataset_name, limit)
        except Exception as e:
            logging.error(f" Failed to create MS MARCO adapter: {e}")
        return None
