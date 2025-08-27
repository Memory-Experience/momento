from abc import ABC, abstractmethod
from typing import Iterator
import pandas as pd

class Dataset(ABC):
    """Abstract base class for IR datasets following MS MARCO format."""
    
    @abstractmethod
    def docs_iter(self) -> Iterator[pd.DataFrame]:
        """Iterator over documents in the dataset.
        Returns:
            Iterator yielding DataFrames with columns ['id', 'content']
        """
        pass

    @abstractmethod
    def queries_iter(self) -> Iterator[pd.DataFrame]:
        """Iterator over queries in the dataset.
        Returns:
            Iterator yielding DataFrames with columns ['id', 'text']
        """
        pass

    @abstractmethod
    def qrels_iter(self) -> Iterator[pd.DataFrame]:
        """Iterator over relevance judgments in the dataset.
        Returns:
            Iterator yielding DataFrames with columns ['query_id', 'doc_id', 'relevance']
        """
        pass
