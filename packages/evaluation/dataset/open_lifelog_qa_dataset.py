import os
import ast
import pandas as pd
from evaluation.dataset.dataset import DataFrameDataset


class OpenLifelogQADataset(DataFrameDataset):
    """
    Adapter for OpenLifelogQA dataset in MS MARCO-like DataFrames.

    Usage:
        ds = OpenLifelogQADataset(split="test", data_dir="open_lifelog_qa")
    """

    def __init__(self, split: str = "test", data_dir: str = "open_lifelog_qa"):
        """
        Args:
            split: Which split to use ("train", "val", "test")
            data_dir: Directory containing the dataset CSV files
        """
        assert split in {"train", "val", "test"}, "Invalid split name"
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), data_dir))
        data_file = os.path.join(data_dir, f"{split}_data.csv")
        events_file = os.path.join(data_dir, "event_description.csv")

        # Load event descriptions as docs
        docs_df = pd.read_csv(events_file)
        docs_df = docs_df.rename(columns={"ImageID": "id", "context": "content"})
        docs_df = docs_df[["id", "content"]].drop_duplicates(subset=["id"])

        # Load queries
        queries_df = pd.read_csv(data_file)
        queries_df = queries_df.rename(columns={"question": "text"})
        queries_df = queries_df[["id", "text", "answer", "ImageID"]].drop_duplicates(
            subset=["id"]
        )

        # Build qrels: each query links to all ImageIDs in its row
        qrels_rows = []
        for _, row in queries_df.iterrows():
            query_id = str(row["id"])
            image_ids = ast.literal_eval(
                row.get("ImageID", "[]")
            )  # ImageID is a stringified list
            for img_id in image_ids:
                qrels_rows.append({
                    "query_id": query_id,
                    "doc_id": str(img_id),
                    "relevance": 1,
                })

        qrels_df = pd.DataFrame(qrels_rows, columns=["query_id", "doc_id", "relevance"])

        docs_df["id"] = docs_df["id"].astype(str).str.strip()
        queries_df["id"] = queries_df["id"].astype(str).str.strip()
        qrels_df["doc_id"] = qrels_df["doc_id"].astype(str).str.strip()
        qrels_df["query_id"] = qrels_df["query_id"].astype(str).str.strip()

        # Defensive filtering to valid ids
        if not docs_df.empty and not queries_df.empty and not qrels_df.empty:
            qrels_df = qrels_df[
                qrels_df["doc_id"].isin(docs_df["id"])
                & qrels_df["query_id"].isin(queries_df["id"])
            ].reset_index(drop=True)

        super().__init__(
            docs_df=docs_df.reset_index(drop=True),
            queries_df=queries_df[["id", "text", "answer"]].reset_index(drop=True),
            qrels_df=qrels_df.reset_index(drop=True),
        )
        self.name = f"OpenLifelogQA ({split})"

    def get_name(self) -> str:
        return self.name

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
