import pandas as pd
import pytest
from types import SimpleNamespace

from dataset.dataset import DataFrameDataset
from dataset.evaluator import MarcoTopKEvaluator


@pytest.fixture
def dataset() -> DataFrameDataset:
    # Minimal, deterministic corpus and qrels
    docs = pd.DataFrame(
        [
            {"id": "d1", "content": "alpha alpha beta"},
            {"id": "d2", "content": "beta gamma"},
            {"id": "d3", "content": "delta"},
        ]
    )
    queries = pd.DataFrame(
        [
            {"id": "q1", "text": "alpha"},
            {"id": "q2", "text": "gamma"},
        ]
    )
    qrels = pd.DataFrame(
        [
            {"query_id": "q1", "doc_id": "d1", "relevance": 1},
            {"query_id": "q2", "doc_id": "d2", "relevance": 1},
        ]
    )
    return DataFrameDataset(docs_df=docs, queries_df=queries, qrels_df=qrels)


def make_system(mapping: dict[str, list[str]]):
    # Retrieval stub without classes; exposes a retrieve(query, top_k) method
    return SimpleNamespace(
        retrieve=lambda query, top_k=10, _m=mapping: _m.get(query, [])[:top_k]
    )


def test_evaluate_query_top_k_metrics(dataset: DataFrameDataset):
    # q1 relevant: d1; q2 relevant: d2
    system = make_system(
        {
            "alpha": ["d1", "d2", "d3"],  # relevant at rank 1
            "gamma": ["d3", "d2", "d1"],  # relevant at rank 2
        }
    )
    ev = MarcoTopKEvaluator(dataset, k_values=[1, 3])

    r1 = ev.evaluate_query_top_k("q1", "alpha", system)["metrics"]
    assert r1["precision@1"] == 1.0
    assert r1["recall@1"] == 1.0
    assert r1["ndcg@1"] == 1.0
    assert r1["precision@3"] == pytest.approx(1 / 3)
    assert r1["recall@3"] == 1.0
    assert r1["ndcg@3"] == pytest.approx(1.0)
    assert r1["mrr"] == 1.0

    r2 = ev.evaluate_query_top_k("q2", "gamma", system)["metrics"]
    assert r2["precision@1"] == 0.0
    assert r2["recall@1"] == 0.0
    assert r2["ndcg@1"] == 0.0
    assert r2["precision@3"] == pytest.approx(1 / 3)
    assert r2["recall@3"] == 1.0
    # relevant at rank 2 → DCG = 1/log2(3)
    assert r2["ndcg@3"] == pytest.approx(1 / 1.58496, rel=1e-3)
    assert r2["mrr"] == pytest.approx(0.5)


def test_evaluate_system_top_k_aggregate(dataset: DataFrameDataset):
    system = make_system({"alpha": ["d1", "d2", "d3"], "gamma": ["d3", "d2", "d1"]})
    ev = MarcoTopKEvaluator(dataset, k_values=[1, 3])
    results = ev.evaluate_system_top_k(system, max_queries=2, verbose=False)
    agg = results["aggregate_metrics"]

    assert agg["precision@1"] == pytest.approx(0.5)
    assert agg["recall@1"] == pytest.approx(0.5)
    assert agg["precision@3"] == pytest.approx(1 / 3)
    assert agg["recall@3"] == pytest.approx(1.0)
    assert agg["ndcg@1"] == pytest.approx(0.5)
    assert agg["ndcg@3"] == pytest.approx((1.0 + (1 / 1.58496)) / 2, rel=1e-3)
    assert agg["mrr"] == pytest.approx((1.0 + 0.5) / 2)


def test_compare_systems_top_k_returns_dataframe(dataset: DataFrameDataset):
    good = make_system({"alpha": ["d1", "d2"], "gamma": ["d2", "d1"]})  # relevant at rank1
    bad = make_system({"alpha": ["d2", "d1"], "gamma": ["d1", "d2"]})   # relevant at rank2
    ev = MarcoTopKEvaluator(dataset, k_values=[1, 3])

    df = ev.compare_systems_top_k({"good": good, "bad": bad}, max_queries=2, verbose=False)
    assert set(df["system"]) == {"good", "bad"}
    assert "precision@1" in df.columns
    p1 = dict(zip(df["system"], df["precision@1"]))
    assert p1["good"] > p1["bad"]