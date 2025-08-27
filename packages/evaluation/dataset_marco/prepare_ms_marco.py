import ir_datasets
from typing import Iterator, Dict, Any
import sys
from pathlib import Path

# Add parent directory to path for imports
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from common.dataset import Dataset

class MSMarcoDataset(Dataset):
    """Implementation of Dataset for MS MARCO using ir_datasets."""
    
    def __init__(self, variant: str = "msmarco-passage/dev/small"):
        """Initialize the MS MARCO dataset using ir_datasets.
        
        Args:
            variant: MS MARCO variant to load. Options:
                - "msmarco-passage/dev/small" (6,980 queries, ~1M docs) - RECOMMENDED
                - "msmarco-passage/train/small" (502,939 queries, ~8.8M docs) - Large
                - "msmarco-passage" (Full dataset - 8.8M docs) - Huge
                - "msmarco-passage/eval/small" (6,837 queries, ~1M docs) - Test set
        """
        print(f"Loading MS MARCO variant: {variant}")
        self.dataset = ir_datasets.load(variant)
    
    def docs_iter(self) -> Iterator[Dict[str, Any]]:
        """Iterator over documents in MS MARCO.
        
        Returns:
            Iterator yielding dicts with 'id' and 'content' keys
        """
        for doc in self.dataset.docs_iter():
            yield {
                'id': doc.doc_id,
                'content': doc.text
            }
    
    #TODO: Primitive Obsession codesmell
    def queries_iter(self) -> Iterator[Dict[str, Any]]:
        """Iterator over queries in MS MARCO.
        
        Returns:
            Iterator yielding dicts with 'id' and 'text' keys
        """
        for query in self.dataset.queries_iter():
            yield {
                'id': query.query_id,
                'text': query.text
            }
    
    def qrels_iter(self) -> Iterator[Dict[str, Any]]:
        """Iterator over relevance judgments in MS MARCO.
        
        Returns:
            Iterator yielding dicts with 'query_id', 'doc_id', and 'relevance' keys
        """
        for qrel in self.dataset.qrels_iter():
            yield {
                'query_id': qrel.query_id,
                'doc_id': qrel.doc_id,
                'relevance': qrel.relevance
            }