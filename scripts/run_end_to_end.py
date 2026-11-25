#!/usr/bin/env python3
"""Uruchamia LLM na twierdzeniach Tune i zapisuje metryki dowodów SciFact.

WYMAGA Ollama (lokalnie) lub API. Może być czasochłonne / płatne.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

# Pozwala uruchamiać: python scripts/nazwa_skryptu.py z katalogu projektu.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import time

from scifact_rag.data import load_documents, load_jsonl, save_jsonl
from scifact_rag.llm import build_prediction
from scifact_rag.metrics import evaluate_evidence, macro_f1_for_verified_pairs
from scifact_rag.retrieval import Retriever, VARIANTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--limit", type=int, default=10,
                        help="Domyślnie 10 dla sprawdzenia; --limit 0 oznacza pełne Tune")
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/end_to_end"))
    args = parser.parse_args()
    if args.limit < 0:
        parser.error("--limit >= 0")
    docs = load_documents()
    claims = load_jsonl("data/processed/claims_tune.jsonl")
    if args.limit:
        claims = claims[:args.limit]
    ranker = Retriever(docs)
    out_dir = args.output_dir / ("preview" if args.limit else "full_tune")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {"generated_utc": datetime.now(timezone.utc).isoformat(),
               "model": args.model, "backend": args.backend,
               "claim_count": len(claims), "preview": bool(args.limit), "variants": {}}
    for variant in args.variants:
        predictions = []
        for n, claim in enumerate(claims, 1):
            results = ranker.search(claim["claim"], variant, 5)
            prediction = build_prediction(claim, results, ranker.by_id,
                                          args.backend, args.model, args.base_url)
            predictions.append(prediction)
            # zapis po każdym przykładzie — można zachować wyniki przy przerwaniu
            save_jsonl(out_dir / f"{variant}_predictions.jsonl", predictions)
            print(f"{variant}: {n}/{len(claims)} ({prediction['verdict']})")
        scores = evaluate_evidence(claims, predictions)
        scores["macro_f1_verified_pairs"] = macro_f1_for_verified_pairs(claims, predictions)
        summary["variants"][variant] = scores
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("Zapisano do", out_dir)


if __name__ == "__main__":
    main()
