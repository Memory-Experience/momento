import pandas as pd
import pytest

from dataset.dataset import DataFrameDataset
from dataset import evaluation_runner as er


@pytest.fixture
def dataset() -> DataFrameDataset:
    # Small, deterministic corpus and labels
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


# MarcoRetrievalSystem tests
def test_build_index_and_properties(dataset: DataFrameDataset):
    # Build index
    system = er.MarcoRetrievalSystem(docs_df=dataset.docs, method="tfidf")
    # One entry per doc id
    assert set(system.doc_index.keys()) == {"d1", "d2", "d3"}
    assert system.doc_index["d1"] == "alpha alpha beta"
    # Term postings created
    for term in ("alpha", "beta", "gamma", "delta"):
        assert term in system.term_index

    # Term frequencies
    assert system.term_index["alpha"]["d1"] == 2
    assert system.term_index["beta"]["d1"] == 1
    assert system.term_index["beta"]["d2"] == 1
    # Tokenized length
    assert system.doc_lengths["d1"] == 3
    # Normalized preprocessing
    assert system._preprocess_text("Alpha  BeTa") == ["alpha", "beta"]


def test_tfidf_retrieval_ranks_matching_docs_first(dataset: DataFrameDataset):
    system = er.MarcoRetrievalSystem(docs_df=dataset.docs, method="tfidf")
    result = system.retrieve("alpha", top_k=5)
    assert result[:1] == ["d1"]
    assert set(result).issubset({"d1", "d2", "d3"})
    assert system.retrieve("unknown_token", top_k=5) == []


def test_bm25_retrieval_prefers_matching_docs(dataset: DataFrameDataset):
    system = er.MarcoRetrievalSystem(docs_df=dataset.docs, method="bm25")
    result = system.retrieve("gamma", top_k=5)
    assert result[:1] == ["d2"]
    assert set(result).issubset({"d1", "d2", "d3"})


def test_random_retrieval_is_bounded_and_unique(monkeypatch, dataset: DataFrameDataset):
    system = er.MarcoRetrievalSystem(docs_df=dataset.docs, method="random")
    def fake_sample(pop, k):
        return list(pop)[:k]
    monkeypatch.setattr(er.random, "sample", fake_sample)
    result = system.retrieve("anything", top_k=2)
    assert result == ["d1", "d2"]
    assert len(result) == 2
    assert len(set(result)) == 2


# evaluate_multiple_queries (monkeypatch)
def test_evaluate_multiple_queries_with_dataset(monkeypatch, dataset: DataFrameDataset):
    # Use the fixture dataset for loader
    monkeypatch.setattr(er.MSMarcoDataset, "create", lambda *a, **k: dataset)
    system_results, valid_queries = er.evaluate_multiple_queries()
    assert isinstance(valid_queries, list)
    assert len(valid_queries) == 2
    assert set(valid_queries[0].keys()) == {"id", "text", "relevant_docs", "relevance_scores"}
    assert set(system_results.keys()) == {"TF-IDF", "BM25", "Random"}
    for metrics in system_results.values():
        assert set(metrics.keys()) == {
            "precision@1", "precision@3", "precision@5",
            "recall@1", "recall@3", "recall@5",
            "ndcg@1", "ndcg@3", "ndcg@5",
            "mrr",
        }
        for v in metrics.values():
            assert 0.0 <= v <= 1.0


# evaluate_with_marco_evaluator (monkeypatch)
def test_evaluate_with_marco_evaluator_monkeypatched(monkeypatch, dataset: DataFrameDataset):
    monkeypatch.setattr(er.MSMarcoDataset, "create", lambda *a, **k: dataset)

    class FakeEvaluator:
        def __init__(self, dataset, k_values):
            self.dataset = dataset
            self.k_values = k_values

        def evaluate_system_top_k(self, system, max_queries=50, verbose=False):
            return {
                "aggregate_metrics": {
                    "precision@1": 1.0,
                    "precision@3": 1.0,
                    "precision@5": 1.0,
                    "recall@1": 0.5,
                    "recall@3": 0.5,
                    "recall@5": 0.5,
                    "ndcg@1": 1.0,
                    "ndcg@3": 1.0,
                    "ndcg@5": 1.0,
                    "mrr": 1.0,
                },
                "per_query": [],
            }

    monkeypatch.setattr(er, "MarcoTopKEvaluator", FakeEvaluator)

    tfidf_results, bm25_results = er.evaluate_with_marco_evaluator()
    for results in (tfidf_results, bm25_results):
        am = results["aggregate_metrics"]
        assert am["precision@1"] == 1.0
        assert am["mrr"] == 1.0


# main
def test_main_calls_all_sections(monkeypatch, capsys, dataset: DataFrameDataset):
    # Orchestration smoke test; uses simple stubs
    called = {"multi": False, "eval": False, "spec": False}

    def fake_multi():
        called["multi"] = True
        return ({"TF-IDF": {}}, [])

    def fake_eval():
        called["eval"] = True
        return ({}, {})

    def fake_spec():
        called["spec"] = True

    monkeypatch.setattr(er, "evaluate_multiple_queries", fake_multi)
    monkeypatch.setattr(er, "evaluate_with_marco_evaluator", fake_eval)
    monkeypatch.setattr(er, "test_specific_query", fake_spec)

    er.main()
    assert called == {"multi": True, "eval": True, "spec": True}

    out = capsys.readouterr().out
    assert "MS MARCO Retrieval System Evaluation" in out
