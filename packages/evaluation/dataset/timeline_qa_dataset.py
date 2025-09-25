import json
import hashlib
import logging
from typing import Any
from collections.abc import Iterable

import pandas as pd

from .dataset import DataFrameDataset
from .timeline_qa import generateDB


def _safe_id(s: str, prefix: str = "") -> str:
    """Create a short, stable id from an arbitrary string (used as fallback)."""
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:12]
    return f"{prefix}{h}" if prefix else h


def _iter_lifelog_events(llqa: dict[str, Any]) -> Iterable[dict[str, Any]]:
    """
    Yield flattened LifelogQA events with keys:
      - date (str)
      - key (str)      # e.g., "birth_info", "chat0"
      - eid (str)
      - text (str)     # 'date: text_template_based'
      - atomic_qa_pairs (list[[question, answer], ...])
    """
    for date, items in llqa.items():
        if not isinstance(items, dict):
            continue
        for key, payload in items.items():
            if not isinstance(payload, dict):
                continue
            eid = payload.get("eid") or _safe_id(f"{date}|{key}", prefix="e_")
            text = payload.get("text_template_based") or ""
            aq = payload.get("atomic_qa_pairs") or []
            yield {
                "date": date,
                "key": key,
                "eid": str(eid),
                "text": f"{date}: {str(text)}",
                "atomic_qa_pairs": aq,
            }


def _convert_lifelogqa_to_dataframes(
    llqa: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Convert a LifelogQA json dict to MS MARCO-like DataFrames:
      - docs_df:    ['id', 'content']
      - queries_df: ['id', 'text']
      - qrels_df:   ['query_id', 'doc_id', 'relevance']
    """
    # 1) Documents: one per event
    docs_rows = []
    event_index: dict[str, dict[str, Any]] = {}  # eid -> event info
    for ev in _iter_lifelog_events(llqa):
        doc_id = ev["eid"]
        docs_rows.append({"id": doc_id, "content": ev["text"]})
        event_index[doc_id] = ev

    docs_df = pd.DataFrame(docs_rows, columns=["id", "content"])

    # 2) Queries: one per atomic QA question; 3) Qrels:
    #   link to source doc with relevance=1
    queries_rows = []
    qrels_rows = []

    for doc_id, ev in event_index.items():
        qa_pairs = ev.get("atomic_qa_pairs") or []
        for idx, pair in enumerate(qa_pairs):
            if not isinstance(pair, (list, tuple)) or len(pair) < 1:
                continue
            question = str(pair[0]).strip()
            if not question:
                continue

            query_id = f"q_{doc_id}_{idx}"
            queries_rows.append({"id": query_id, "text": question})
            qrels_rows.append({"query_id": query_id, "doc_id": doc_id, "relevance": 1})

    queries_df = pd.DataFrame(queries_rows, columns=["id", "text"]).drop_duplicates(
        subset=["id"], keep="first"
    )
    qrels_df = pd.DataFrame(qrels_rows, columns=["query_id", "doc_id", "relevance"])

    # Defensive filtering to valid ids
    if not docs_df.empty and not queries_df.empty and not qrels_df.empty:
        qrels_df = qrels_df[
            qrels_df["doc_id"].isin(docs_df["id"])
            & qrels_df["query_id"].isin(queries_df["id"])
        ].reset_index(drop=True)

    return (
        docs_df.reset_index(drop=True),
        queries_df.reset_index(drop=True),
        qrels_df.reset_index(drop=True),
    )


class LifelogQADataset(DataFrameDataset):
    """
    LifelogQA dataset adapter in MS MARCO-like DataFrames.

    Usage:
        # In-memory dict
        ds = LifelogQADataset(data_dict)

        # From a JSON file path
        ds = LifelogQADataset.from_file("lifelogqa.json")

        # Generate a synthetic dataset
        ds = LifelogQADataset.generate(seed=42, ...)
    """

    def __init__(self, data: dict[str, Any], name: str = "LifelogQA"):
        """
        Initialize the LifelogQA dataset adapter.

        Args:
            data: Parsed LifelogQA dict.
            name: Dataset name for reporting.
        """
        self.name = name
        try:
            docs_df, queries_df, qrels_df = _convert_lifelogqa_to_dataframes(data)
            logging.info(
                f"LifelogQA loaded: {len(docs_df)} docs, "
                f"{len(queries_df)} queries, {len(qrels_df)} qrels"
            )
            super().__init__(docs_df=docs_df, queries_df=queries_df, qrels_df=qrels_df)
        except Exception as e:
            logging.error(f"Failed to load LifelogQA: {e}")
            raise

    @classmethod
    def from_file(cls, path: str, **_: Any) -> "LifelogQADataset":
        """Convenience constructor from a JSON file path."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return cls(data=data)

    @classmethod
    def generate(
        cls,
        templatefilename: str = None,
        seed: int = 12345,
        final_year: int = 2025,
        current_age: int = 18,
        verbose: bool = False,
        category: int = 1,
        output_directory: str = ".",
    ) -> "LifelogQADataset":
        """
        Generate a synthetic LifelogQA dataset using the internal generator.

        Args:
            templatefilename: Path to the template file.
            seed: Random seed for reproducibility.
            final_year: The final year for the generated data.
            current_age: The current age of the persona.
            verbose: Whether to enable verbose logging.
            category: Category of the dataset to generate.
            output_directory: Directory to save the generated dataset.

        Returns:
            An instance of LifelogQADataset with the generated data.
        """
        _persona, episodic_db = generateDB.generateDb(
            templatefilename=templatefilename,
            seed=seed,
            final_year=final_year,
            current_age=current_age,
            verbose=verbose,
            category=category,
            output_directory=output_directory,
        )
        return cls(data=episodic_db)

    def get_name(self) -> str:
        return self.name or "LifelogQA"

    def get_sample_query(self) -> dict | None:
        """Return a sample query with relevant documents."""
        if self.queries.empty or self.qrels.empty:
            return None

        sample_row = self.queries.iloc[0]
        query_id = str(sample_row["id"])
        relevant_qrels = self.qrels[self.qrels["query_id"].astype(str) == query_id]

        return {
            "id": query_id,
            "text": str(sample_row["text"]),
            "relevant_docs": relevant_qrels["doc_id"].astype(str).tolist(),
            "relevance_scores": dict(
                zip(
                    relevant_qrels["doc_id"].astype(str),
                    relevant_qrels["relevance"],
                    strict=False,
                )
            ),
        }
