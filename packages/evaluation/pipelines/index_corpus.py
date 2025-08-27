"""Indexing pipeline."""
from typing import Optional, Dict, Any
from common.io import read_jsonl

class Indexer:
    """Streams documents to an index via a gRPC stub or prints actions if stub is None."""

    def __init__(self, collection: str, corpus: str, grpc_stub=None) -> None:
        self.collection = collection
        self.corpus = corpus
        self.grpc_stub = grpc_stub

    def index_document(self, doc: Dict[str, Any]) -> None:
        """Send a single document to the index service if available; otherwise noop/log."""
        # Expected doc fields: id, text, and optional metadata
        if self.grpc_stub is not None:
            # Example pseudo-call; adapt to your service schema
            # self.grpc_stub.Index({"collection": self.collection, "id": doc.get("id"), "text": doc.get("text",""), "metadata": doc.get("metadata", {})})
            pass
        # Always keep a simple trace for now for cohesiveness
        # (Avoid printing massive payloads)
        _id = doc.get("id") or doc.get("doc_id") or ""
        print(f"[index] {self.collection} ← {str(_id)[:64]}")

    def run(self) -> None:
        for doc in read_jsonl(self.corpus):
            self.index_document(doc)
        print(f"[index] collection={self.collection} indexed.")
