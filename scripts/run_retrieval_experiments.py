#!/usr/bin/env python3
"""Porównanie E0-E3 na train/tune bez kontaktu z LLM."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Pozwala uruchamiać: python scripts/nazwa_skryptu.py z katalogu projektu.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import statistics
import time

from scifact_rag.data import gold_document_ids, load_documents, load_jsonl, save_jsonl
from scifact_rag.metrics import retrieval_score
from scifact_rag.retrieval import Retriever, VARIANTS, DEFAULT_EMBEDDING, DEFAULT_RERANKER


SPLITS = {
    "train_fit": "data/processed/claims_train_fit.jsonl",
    "tune": "data/processed/claims_tune.jsonl",
    "final_eval": "data/processed/claims_final_eval.jsonl",
}


def main():
    parser = argparse.ArgumentParser(description="SciFact E0-E3: retrieval benchmark")
    parser.add_argument("--split", choices=SPLITS, default="tune")
    parser.add_argument("--confirm-final", action="store_true",
                        help="Ostateczny zbiór otwórz dopiero po wyborze wariantu!")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--limit", type=int, default=None, help="Smoke test; do raportu używaj pełnego zbioru")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--rrf-depth", type=int, default=100)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING)
    parser.add_argument("--reranker-model", default=DEFAULT_RERANKER)
    parser.add_argument("--output-dir", type=Path, default=Path("results/retrieval"))
    args = parser.parse_args()
    if args.split == "final_eval" and not args.confirm_final:
        parser.error("Final Eval jest zarezerwowany na milestone 3; potrzebna flaga --confirm-final")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit musi być dodatni")
    documents = load_documents()
    claims = load_jsonl(SPLITS[args.split])
    if args.limit:
        claims = claims[:args.limit]
    retriever = Retriever(documents, embedding_model=args.embedding_model,
                          reranker_model=args.reranker_model, rrf_k=args.rrf_k,
                          rrf_depth=args.rrf_depth, candidate_k=args.candidate_k)
    run_name = f"{args.split}{'_preview' if args.limit else ''}"
    out_dir = args.output_dir / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, prediction_rows = [], []
    for variant in args.variants:
        print(f"Przygotowanie wariantu {variant} (inicjalizacja poza pomiarem czasu)...")
        retriever.prepare(variant, warmup_claim=claims[0]["claim"])
        per_claim, latencies, details = [], [], []
        for n, claim in enumerate(claims, 1):
            t0 = time.perf_counter()
            retrieved = retriever.search(claim["claim"], variant, top_k=args.top_k)
            latency_ms = (time.perf_counter() - t0) * 1000
            ids = [r.doc_id for r in retrieved]
            metric = retrieval_score(claim, ids, args.top_k)
            if metric is not None:
                per_claim.append(metric)
            latencies.append(latency_ms)
            details.append({"id": claim["id"], "claim": claim["claim"],
                            "gold_doc_ids": sorted(gold_document_ids(claim)),
                            "retrieved_doc_ids": ids, "latency_ms": round(latency_ms, 3),
                            "retrieval_metrics": metric})
            if n % 25 == 0 or n == len(claims):
                print(f"{variant}: {n}/{len(claims)} twierdzeń")
        def avg(key):
            return round(statistics.mean(d[key] for d in per_claim), 6) if per_claim else None
        row = {"variant": variant, "split": args.split,
               "preview": bool(args.limit), "queries": len(claims),
               "positive_claims": len(per_claim),
               "no_gold_claims_excluded": len(claims) - len(per_claim),
               f"recall_at_{args.top_k}": avg("recall_at_k"),
               f"hit_at_{args.top_k}": avg("hit_at_k"),
               f"mrr_at_{args.top_k}": avg("mrr_at_k"),
               f"ndcg_at_{args.top_k}": avg("ndcg_at_k"),
               "avg_latency_ms": round(statistics.mean(latencies), 2),
               "median_latency_ms": round(statistics.median(latencies), 2)}
        rows.append(row)
        save_jsonl(out_dir / f"{variant}_per_claim.jsonl", details)
        print(json.dumps(row, ensure_ascii=False, indent=2))
    report = {"generated_utc": datetime.now(timezone.utc).isoformat(),
              "config": {"embedding_model": args.embedding_model,
                         "reranker_model": args.reranker_model,
                         "top_k": args.top_k, "rrf_k": args.rrf_k,
                         "rrf_depth": args.rrf_depth, "candidate_k": args.candidate_k,
                         "split": args.split, "limit": args.limit},
              "metrics": rows,
              "methodology": "Recall/MRR/nDCG średnia po claims z >=1 gold evidence; NOINFO wyłączone. Latency mierzona po inicjalizacji modeli i cache, pierwsze zapytanie może zawierać koszt wczytania modelu."}
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    import csv
    with (out_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nZapisano: {out_dir / 'summary.json'} i {out_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
