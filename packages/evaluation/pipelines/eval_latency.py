import argparse
import numpy as np
from common.io import read_jsonl, write_csv

def pct(vals, p): return float(np.percentile(vals, p)) if vals else 0.0

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True)   # carries timing for retrieval/gen/total
    ap.add_argument("--retrieval", required=False)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    totals, ret, gen = [], [], []
    for r in read_jsonl(args.answers):
        t = r.get("timing", {})
        totals.append(t.get("total_ms", 0))
        ret.append(t.get("retrieval_ms", 0))
        gen.append(t.get("gen_ms", 0))
    out = [{
        "retrieval_p50_ms": pct(ret, 50), "retrieval_p90_ms": pct(ret, 90), "retrieval_p95_ms": pct(ret, 95),
        "gen_p50_ms": pct(gen, 50), "gen_p90_ms": pct(gen, 90), "gen_p95_ms": pct(gen, 95),
        "e2e_p50_ms": pct(totals, 50), "e2e_p90_ms": pct(totals, 90), "e2e_p95_ms": pct(totals, 95)
    }]
    write_csv(args.out, out)
    print(f"[eval] latency → {args.out} :: {out[0]}")
