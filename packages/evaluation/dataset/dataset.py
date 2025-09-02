
import pandas as pd


class DataFrameDataset:
    """Dataset implementation using pandas DataFrames for evaluation tasks."""

    def __init__(self, 
                 docs_df: pd.DataFrame | None = None, 
                 queries_df: pd.DataFrame | None = None, 
                 qrels_df: pd.DataFrame | None = None):
        """Initialize dataset with pandas DataFrames.
        
        Args:
            docs_df: DataFrame with columns ['id', 'content']
            queries_df: DataFrame with columns ['id', 'text']
            qrels_df: DataFrame with columns ['query_id', 'doc_id', 'relevance']
        """
        self._validate_dataframes(docs_df, queries_df, qrels_df)
        
        self._docs_df = (
            docs_df
            if docs_df is not None
            else pd.DataFrame(columns=['id', 'content'])
        )
        self._queries_df = (
            queries_df
            if queries_df is not None
            else pd.DataFrame(columns=['id', 'text'])
        )
        self._qrels_df = (
            qrels_df
            if qrels_df is not None
            else pd.DataFrame(columns=['query_id', 'doc_id', 'relevance'])
        )

    def _validate_dataframes(self, docs_df: pd.DataFrame | None, 
                           queries_df: pd.DataFrame | None, 
                           qrels_df: pd.DataFrame | None) -> None:
        """Validate DataFrame schemas."""
        if docs_df is not None and len(docs_df) > 0:
            required_cols = ['id', 'content']
            if not all(col in docs_df.columns for col in required_cols):
                raise ValueError(
                    f"docs_df must contain {required_cols} columns, got "
                    f"{list(docs_df.columns)}"
                )
            
        if queries_df is not None and len(queries_df) > 0:
            required_cols = ['id', 'text']
            if not all(col in queries_df.columns for col in required_cols):
                raise ValueError(
                    f"queries_df must contain {required_cols} columns, got "
                    f"{list(queries_df.columns)}"
                )
            
        if qrels_df is not None and len(qrels_df) > 0:
            required_cols = ['query_id', 'doc_id', 'relevance']
            if not all(col in qrels_df.columns for col in required_cols):
                raise ValueError(
                    f"qrels_df must contain {required_cols} columns, got "
                    f"{list(qrels_df.columns)}"
                )

    def get_name(self) -> str:
        """Return dataset name."""
        return getattr(self, 'name', 'DataFrameDataset')
    
    def get_sample_query(self) -> dict | None:
        """Get a sample query with its relevant documents.
        
        Returns:
            Dictionary with query info and relevant documents, or None if unavailable
        """
        if self.queries.empty or self.qrels.empty:
            return None
        
        for _, query_row in self.queries.iterrows():
            query_id = str(query_row['id'])
            query_text = str(query_row['text'])
            
            relevant_qrels = self.qrels[self.qrels['query_id'].astype(str) == query_id]
            
            if not relevant_qrels.empty:
                relevant_docs = relevant_qrels['doc_id'].astype(str).tolist()
                relevance_scores = dict(zip(
                    relevant_qrels['doc_id'].astype(str), 
                    relevant_qrels['relevance'], strict=False
                ))
                
                return {
                    'id': query_id,
                    'text': query_text,
                    'relevant_docs': relevant_docs,
                    'relevance_scores': relevance_scores
                }
        
        return None
    
    def __len__(self) -> int:
        """Return number of documents."""
        return len(self.docs)
    
    def __str__(self) -> str:
        """String representation of the dataset."""
        name = self.get_name()
        return (
            f"{name}: {len(self.docs)} docs, {len(self.queries)} queries, "
            f"{len(self.qrels)} qrels"
        )

    @property
    def docs(self) -> pd.DataFrame:
        """Documents DataFrame."""
        return self._docs_df

    @property
    def queries(self) -> pd.DataFrame:
        """Queries DataFrame."""
        return self._queries_df

    @property
    def qrels(self) -> pd.DataFrame:
        """Relevance judgments DataFrame."""
        return self._qrels_df
