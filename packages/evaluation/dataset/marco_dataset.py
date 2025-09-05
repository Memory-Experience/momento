import logging
from typing import Optional

import ir_datasets
import pandas as pd

from .dataset import DataFrameDataset


def _convert_ms_marco_to_dataframes(ms_marco_dataset, limit: int = 1000):
    """Convert MS MARCO dataset to pandas DataFrames.

    Args:
        ms_marco_dataset: MS MARCO dataset instance
        limit: Maximum number of items to process

    Returns:
        tuple: (docs_df, queries_df, qrels_df)
    """
    print(f"Converting MS MARCO to DataFrames (limit: {limit})...")

    # Step 1: Load qrels to identify needed documents and queries
    qrels_data = []
    needed_doc_ids = set()
    needed_query_ids = set()

    for i, qrel in enumerate(ms_marco_dataset.qrels_iter()):
        if i >= limit:
            break
        qrels_data.append({
            "query_id": qrel.query_id,
            "doc_id": qrel.doc_id,
            "relevance": qrel.relevance,
        })
        needed_doc_ids.add(qrel.doc_id)
        needed_query_ids.add(qrel.query_id)

    qrels_df = pd.DataFrame(qrels_data)
    print(f"  Loaded {len(qrels_df)} qrels")
    print(f"  Need {len(needed_doc_ids)} docs and {len(needed_query_ids)} queries")

    # Step 2: Load needed documents
    docs_data = []
    found_doc_ids = set()

    for doc in ms_marco_dataset.docs_iter():
        if doc.doc_id in needed_doc_ids:
            docs_data.append({"id": doc.doc_id, "content": doc.text})
            found_doc_ids.add(doc.doc_id)

        if len(found_doc_ids) >= len(needed_doc_ids):
            break

    docs_df = pd.DataFrame(docs_data)
    print(f"  Found {len(docs_df)} out of {len(needed_doc_ids)} needed documents")

    # Step 3: Load needed queries
    queries_data = []
    found_query_ids = set()

    for query in ms_marco_dataset.queries_iter():
        if query.query_id in needed_query_ids:
            queries_data.append({"id": query.query_id, "text": query.text})
            found_query_ids.add(query.query_id)

        if len(found_query_ids) >= len(needed_query_ids):
            break

    queries_df = pd.DataFrame(queries_data)
    print(f"  Found {len(queries_df)} out of {len(needed_query_ids)} needed queries")

    # Step 4: Filter qrels to valid documents and queries
    valid_qrels = qrels_df[
        qrels_df["doc_id"].isin(found_doc_ids)
        & qrels_df["query_id"].isin(found_query_ids)
    ]

    logging.info(
        f"  Final valid qrels: {len(valid_qrels)} (filtered from {len(qrels_df)})"
    )
    logging.info(
        f"  Final dataset: {len(docs_df)} docs, {len(queries_df)} queries, "
        f"{len(valid_qrels)} qrels"
    )

    return docs_df, queries_df, valid_qrels


"""
MS MARCO Dataset Adapter

Adapter for MS MARCO dataset integration with the evaluation framework.
"""

class MSMarcoDataset(DataFrameDataset):
    """MS MARCO dataset adapter for evaluation framework."""

    def __init__(
        self,
        dataset_name: str = "msmarco-passage/dev/small",
        limit: int = 1000,
    ):
        """Initialize MS MARCO dataset adapter.

        Args:
            dataset_name: MS MARCO dataset variant to use
            limit: Maximum number of items to load
        """
        self.dataset_name = dataset_name
        self.limit = limit

        logging.info(f"Loading MS MARCO dataset: {dataset_name} (limit: {limit})")
        try:
            self.marco_dataset = ir_datasets.load(dataset_name)
            docs_df, queries_df, qrels_df = _convert_ms_marco_to_dataframes(
                self.marco_dataset, limit=limit
            )
            super().__init__(docs_df, queries_df, qrels_df)

            logging.info(
                f"MS MARCO loaded: {len(self._docs_df)} docs, "
                f"{len(self._queries_df)} queries"
            )
        except Exception as e:
            logging.error(f"Failed to load MS MARCO: {e}")
            raise

    def get_name(self) -> str:
        """Return dataset name for reporting."""
        return f"MS MARCO ({self.dataset_name}, limit={self.limit})"

    def get_sample_query(self) -> dict | None:
        """Get a sample query for testing."""
        if self.queries.empty:
            return None

        sample_row = self.queries.iloc[0]
        query_id = sample_row["id"]

        relevant_qrels = self.qrels[self.qrels["query_id"] == query_id]

        return {
            "id": query_id,
            "text": sample_row["text"],
            "relevant_docs": relevant_qrels["doc_id"].tolist(),
            "relevance_scores": dict(
                zip(
                    relevant_qrels["doc_id"],
                    relevant_qrels["relevance"],
                    strict=False,
                )
            ),
        }

    @staticmethod
    def create(
        dataset_name: str = "msmarco-passage/dev/small",
        limit: int = 1000,
    ) -> Optional["MSMarcoDataset"]:
        """Factory function to create MS MARCO adapter with error handling.

        Args:
            dataset_name: MS MARCO dataset variant
            limit: Maximum items to load

        Returns:
            MSMarcoDataset instance or None if failed
        """
        try:
            return MSMarcoDataset(dataset_name, limit)
        except Exception as e:
            logging.error(f"Failed to create MS MARCO adapter: {e}")
        return None
