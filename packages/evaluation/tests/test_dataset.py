"""
Extensive tests for marco_dataset_mock.dataset.DataFrameDataset

Purpose
- Verify that DataFrameDataset constructs and validates the three core tables:
  - docs:      columns ["id", "content"]
  - queries:   columns ["id", "text"]
  - qrels:     columns ["query_id", "doc_id", "relevance"]
- Confirm behavior when initialized empty vs. with data.
- Confirm schema validation rules: required columns must be present when a
  non-empty DataFrame is provided; empty frames are allowed as “no data yet”.
- Verify helper methods such as:
  - get_name(): returns a default, but can be overridden by setting .name
  - get_sample_query(): returns a representative query with its relevant docs
  - __len__(): returns number of documents
  - __str__(): includes dataset name and basic counts
- Ensure properties docs, queries, qrels return pandas DataFrames.

Why these tests matter
- The evaluation pipeline depends on consistent dataset schemas.
- Early schema and helper-method failures should be surfaced clearly via tests.
- These tests act as a contract for any future changes to DataFrameDataset.
"""

import pandas as pd
import pytest
from dataset.dataset import DataFrameDataset


@pytest.fixture
def dataset():
    """
    Provide a fresh, empty DataFrameDataset for tests that need a baseline.

    Rationale:
    - Centralizes creation logic and avoids repeating boilerplate.
    - Ensures consistent starting state across tests.
    """
    return DataFrameDataset()


def test_init_with_none_creates_empty_dataframes(dataset):
    """
    DataFrameDataset() with no args should produce empty tables with canonical schemas.

    Verifies:
    - Column names match the expected schema.
    - All three DataFrames are empty by default.
    """
    assert list(dataset.docs.columns) == ["id", "content"]
    assert list(dataset.queries.columns) == ["id", "text"]
    assert list(dataset.qrels.columns) == ["query_id", "doc_id", "relevance"]
    assert len(dataset.docs) == 0
    assert len(dataset.queries) == 0
    assert len(dataset.qrels) == 0


@pytest.mark.parametrize(
    "docs_df,should_raise",
    [
        # Valid: all required columns present
        (pd.DataFrame([{"id": "d1", "content": "text"}]), False),
        # Invalid: missing "content"
        (pd.DataFrame([{"id": "d1"}]), True),
        # Invalid: missing "id"
        (pd.DataFrame([{"content": "text"}]), True),
        # Allowed: empty DF with wrong columns is treated as "no data yet"
        (pd.DataFrame(columns=["wrong"]), False),
    ],
)
def test_validate_docs_schema(docs_df, should_raise):
    """
    Supplying docs_df triggers schema validation only when non-empty.

    Rule:
    - Non-empty -> must contain columns ["id", "content"], else ValueError.
    - Empty     -> accepted as no-op placeholder.
    """
    if should_raise:
        with pytest.raises(ValueError):
            DataFrameDataset(docs_df=docs_df)
    else:
        DataFrameDataset(docs_df=docs_df)  # should not raise


@pytest.mark.parametrize(
    "queries_df,should_raise",
    [
        # Valid
        (pd.DataFrame([{"id": "q1", "text": "what is ai"}]), False),
        # Invalid: missing "text"
        (pd.DataFrame([{"id": "q1"}]), True),
        # Invalid: missing "id"
        (pd.DataFrame([{"text": "what is ai"}]), True),
        # Allowed: empty placeholder
        (pd.DataFrame(columns=["wrong"]), False),
    ],
)
def test_validate_queries_schema(queries_df, should_raise):
    """
    Schema rule for queries_df:
    - Non-empty requires ["id", "text"].
    - Empty DataFrame is allowed as a placeholder.
    """
    if should_raise:
        with pytest.raises(ValueError):
            DataFrameDataset(queries_df=queries_df)
    else:
        DataFrameDataset(queries_df=queries_df)  # should not raise


@pytest.mark.parametrize(
    "qrels_df,should_raise",
    [
        # Valid
        (pd.DataFrame([{"query_id": "q1", "doc_id": "d1", "relevance": 1}]), False),
        # Invalid: missing "relevance"
        (pd.DataFrame([{"query_id": "q1", "doc_id": "d1"}]), True),
        # Invalid: missing "doc_id"
        (pd.DataFrame([{"query_id": "q1", "relevance": 1}]), True),
        # Invalid: missing "query_id"
        (pd.DataFrame([{"doc_id": "d1", "relevance": 1}]), True),
        # Allowed: empty placeholder
        (pd.DataFrame(columns=["wrong"]), False),
    ],
)
def test_validate_qrels_schema(qrels_df, should_raise):
    """
    Schema rule for qrels_df:
    - Non-empty requires ["query_id", "doc_id", "relevance"].
    - Empty DataFrame is allowed as a placeholder.
    """
    if should_raise:
        with pytest.raises(ValueError):
            DataFrameDataset(qrels_df=qrels_df)
    else:
        DataFrameDataset(qrels_df=qrels_df)  # should not raise


def test_get_name_default_and_custom():
    """
    get_name() returns a sensible default and respects an optional .name override.

    Why:
    - Useful for logging/reporting when multiple datasets are involved.
    """
    ds = DataFrameDataset()
    assert ds.get_name() == "DataFrameDataset"
    # Simulate a user-defined display name
    ds.name = "CustomDataset"
    assert ds.get_name() == "CustomDataset"


def test_get_sample_query_none_when_empty():
    """
    get_sample_query() should return None when there are no queries with qrels.

    Setup:
    - One doc, but queries and qrels are empty.
    Expectation:
    - No query with relevance info -> returns None.
    """
    ds = DataFrameDataset(
        docs_df=pd.DataFrame([{"id": "d1", "content": "x"}]),
        queries_df=pd.DataFrame(columns=["id", "text"]),
        qrels_df=pd.DataFrame(columns=["query_id", "doc_id", "relevance"]),
    )
    assert ds.get_sample_query() is None


def test_get_sample_query_returns_expected_structure_with_type_mixing():
    """
    get_sample_query() should:
    - Normalize ID types (e.g., int->str) to match qrels.
    - Return a dict with keys: id, text, relevant_docs, relevance_scores.
    - Prefer the first query that has qrels.

    Scenario:
    - queries contain int IDs [1, 2], qrels use string IDs "1" and "999".
    - Ensures internal logic casts/aligns types to produce matches.
    """
    # Documents (string IDs)
    docs = pd.DataFrame([
        {"id": "d1", "content": "alpha"},
        {"id": "d2", "content": "beta"},
        {"id": "d3", "content": "gamma"},
    ])
    # Queries include int IDs to test normalization
    queries = pd.DataFrame([
        {"id": 1, "text": "alpha query"},
        {"id": 2, "text": "beta query"},
    ])
    # Qrels use string IDs; only query "1" has matching relevances
    qrels = pd.DataFrame([
        {"query_id": "1", "doc_id": "d1", "relevance": 2},
        {"query_id": "1", "doc_id": "d3", "relevance": 1},
        {"query_id": "999", "doc_id": "d2", "relevance": 1},
    ])
    ds = DataFrameDataset(docs_df=docs, queries_df=queries, qrels_df=qrels)

    sample = ds.get_sample_query()
    assert sample is not None
    # Structure contract: these keys must be present
    assert set(sample.keys()) == {"id", "text", "relevant_docs", "relevance_scores"}
    # The first query with qrels is ID "1"
    assert sample["id"] == "1"
    assert sample["text"] == "alpha query"
    # Relevant docs must preserve the association and ordering used by implementation
    assert sample["relevant_docs"] == ["d1", "d3"]
    assert sample["relevance_scores"] == {"d1": 2, "d3": 1}


def test_len_returns_number_of_docs():
    """
    __len__ should reflect the number of documents, not queries or qrels.

    Rationale:
    - Length semantics are used by callers to size retrieval indexes.
    """
    docs = pd.DataFrame([{"id": "d1", "content": "x"}, {"id": "d2", "content": "y"}])
    ds = DataFrameDataset(docs_df=docs)
    assert len(ds) == 2


def test_str_contains_name_and_counts():
    """
    __str__ should include dataset name and basic counts for quick diagnostics.

    Expectation:
    - String contains name and the counts of docs, queries, and qrels.
    """
    docs = pd.DataFrame([{"id": "d1", "content": "x"}])
    queries = pd.DataFrame([{"id": "q1", "text": "x?"}])
    qrels = pd.DataFrame([{"query_id": "q1", "doc_id": "d1", "relevance": 1}])
    ds = DataFrameDataset(docs_df=docs, queries_df=queries, qrels_df=qrels)
    s = str(ds)
    assert "DataFrameDataset" in s
    assert "1 docs" in s
    assert "1 queries" in s
    assert "1 qrels" in s


def test_properties_return_dataframes():
    """
    Properties docs, queries, and qrels must be pandas DataFrames.

    Reason:
    - Callers expect DataFrame APIs and behaviors (filtering, merging, etc.).
    """
    ds = DataFrameDataset()
    assert isinstance(ds.docs, pd.DataFrame)
    assert isinstance(ds.queries, pd.DataFrame)
    assert isinstance(ds.qrels, pd.DataFrame)
