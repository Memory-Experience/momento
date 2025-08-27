from typing import Iterator, Optional
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from evaluation.dataset_marco import Dataset

class DataFrameDataset(Dataset):
    """Dataset implementation using pandas DataFrames."""

    def __init__(self, 
                 docs_df: Optional[pd.DataFrame] = None, 
                 queries_df: Optional[pd.DataFrame] = None, 
                 qrels_df: Optional[pd.DataFrame] = None):
        """Initialize DataFrameDataset with pandas DataFrames.
        
        Args:
            docs_df: DataFrame with columns ['id', 'content']
            queries_df: DataFrame with columns ['id', 'text']
            qrels_df: DataFrame with columns ['query_id', 'doc_id', 'relevance']
        """
        self._validate_dataframes(docs_df, queries_df, qrels_df)
        
        # Initialize with empty DataFrames if None
        self._docs_df = docs_df if docs_df is not None else pd.DataFrame(columns=['id', 'content'])
        self._queries_df = queries_df if queries_df is not None else pd.DataFrame(columns=['id', 'text'])
        self._qrels_df = qrels_df if qrels_df is not None else pd.DataFrame(columns=['query_id', 'doc_id', 'relevance'])
    
    def _validate_dataframes(self, docs_df: Optional[pd.DataFrame], 
                           queries_df: Optional[pd.DataFrame], 
                           qrels_df: Optional[pd.DataFrame]) -> None:
        """Validate DataFrame schemas."""
        if docs_df is not None and not all(col in docs_df.columns for col in ['id', 'content']):
            raise ValueError("docs_df must contain 'id' and 'content' columns")
            
        if queries_df is not None and not all(col in queries_df.columns for col in ['id', 'text']):
            raise ValueError("queries_df must contain 'id' and 'text' columns")
            
        if qrels_df is not None and not all(col in qrels_df.columns for col in ['query_id', 'doc_id', 'relevance']):
            raise ValueError("qrels_df must contain 'query_id', 'doc_id', and 'relevance' columns")

    def docs_iter(self) -> Iterator[pd.DataFrame]:
        """Iterator over documents in the dataset."""
        for i in range(len(self._docs_df)):
            yield self._docs_df.iloc[[i]]

    def queries_iter(self) -> Iterator[pd.DataFrame]:
        """Iterator over queries in the dataset."""
        for i in range(len(self._queries_df)):
            yield self._queries_df.iloc[[i]]

    def qrels_iter(self) -> Iterator[pd.DataFrame]:
        """Iterator over relevance judgments in the dataset."""
        for i in range(len(self._qrels_df)):
            yield self._qrels_df.iloc[[i]]

    @property
    def docs(self) -> pd.DataFrame:
        """Get the documents DataFrame."""
        return self._docs_df

    @property
    def queries(self) -> pd.DataFrame:
        """Get the queries DataFrame."""
        return self._queries_df

    @property
    def qrels(self) -> pd.DataFrame:
        """Get the relevance judgments DataFrame."""
        return self._qrels_df