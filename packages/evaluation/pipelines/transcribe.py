from typing import Iterator, Dict, Any
from common.dataset import Dataset

class TranscribedDataset(Dataset):
    """Dataset wrapper for transcribed diaries (speech → text)."""
    
    def __init__(self, docs, queries, qrels):
        self._docs = docs
        self._queries = queries
        self._qrels = qrels
    
    def docs_iter(self) -> Iterator[Dict[str, Any]]:
        for doc in self._docs:
            yield doc   # expects {"id": ..., "content": ...}
    
    def queries_iter(self) -> Iterator[Dict[str, Any]]:
        for q in self._queries:
            yield q     # expects {"id": ..., "text": ...}
    
    def qrels_iter(self) -> Iterator[Dict[str, Any]]:
        for qrel in self._qrels:
            yield qrel  # expects {"query_id": ..., "doc_id": ..., "relevance": ...}
