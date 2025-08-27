"""Retrieval evaluation pipeline."""
from typing import List, Dict, Set
from common.io import read_jsonl, write_csv
from packages.evaluation.dataset_marco.metrics import precision_at_k, recall_at_k, reciprocal_rank, ndcg_at_k

class RetrievalEvaluator:
    def __init__(self, gold_path: str, runs_path: str, out_csv: str, k_list: List[int]):
        self.gold_path = gold_path  # reserved for future use if needed
        self.runs_path = runs_path
        self.out_csv = out_csv
        self.k_list = k_list or [5, 10]

    def compute(self) -> Dict[str, float]:
        rows = list(read_jsonl(self.runs_path))
        metrics: Dict[str, float] = {}
        for k in self.k_list:
            p = []; r = []; nd = []; rr = []
            for row in rows:
                ranked = row.get("ranked_ids", [])
                gold: Set[str] = set(row.get("gold_ids", []))
                p.append(precision_at_k(ranked, gold, k))
                r.append(recall_at_k(ranked, gold, k))
                nd.append(ndcg_at_k(ranked, gold, k))
                rr.append(reciprocal_rank(ranked, gold))
            metrics[f"precision@{k}"] = sum(p)/len(p) if p else 0.0
            metrics[f"recall@{k}"]    = sum(r)/len(r) if r else 0.0
            metrics[f"ndcg@{k}"]      = sum(nd)/len(nd) if nd else 0.0
            # mrr is independent of k (per-row), so keep a single scalar; average rr over rows.
            metrics["mrr"]            = sum(rr)/len(rr) if rr else 0.0
        return metrics

    def run(self) -> Dict[str, float]:
        metrics = self.compute()
        write_csv(self.out_csv, [metrics])
        print(f"[eval] retrieval → {self.out_csv} :: {metrics}")
        return metrics
