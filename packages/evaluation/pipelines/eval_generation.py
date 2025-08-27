"""Generation evaluation pipeline."""
from typing import Dict, List
from common.io import read_jsonl, write_csv

def _normalize(s: str) -> str:
    return " ".join(s.lower().split())

def exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0

def f1(pred: str, gold: str) -> float:
    p = _normalize(pred).split()
    g = _normalize(gold).split()
    if not p and not g: return 1.0
    if not p or not g: return 0.0
    common = 0
    g_counts = {}
    for w in g: g_counts[w] = g_counts.get(w, 0) + 1
    for w in p:
        if g_counts.get(w, 0) > 0:
            common += 1
            g_counts[w] -= 1
    if common == 0: return 0.0
    precision = common / len(p)
    recall = common / len(g)
    return 2 * precision * recall / (precision + recall)

class GenerationEvaluator:
    def __init__(self, gold_path: str, answers_path: str, out_csv: str):
        self.gold_path = gold_path  # reserved for future extension
        self.answers_path = answers_path
        self.out_csv = out_csv

    def compute(self) -> Dict[str, float]:
        rows = list(read_jsonl(self.answers_path))
        ems: List[float] = []
        f1s: List[float] = []
        for r in rows:
            system = r.get("system_answer", "")
            gold = r.get("gold_answer", "")
            ems.append(exact_match(system, gold))
            f1s.append(f1(system, gold))
        return {
            "exact_match": sum(ems)/len(ems) if ems else 0.0,
            "f1":          sum(f1s)/len(f1s) if f1s else 0.0,
        }

    def run(self) -> Dict[str, float]:
        metrics = self.compute()
        write_csv(self.out_csv, [metrics])
        print(f"[eval] generation → {self.out_csv} :: {metrics}")
        return metrics

