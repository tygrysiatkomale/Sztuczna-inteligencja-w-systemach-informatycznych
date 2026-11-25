#!/usr/bin/env python3
"""Interaktywny prototyp: wpisz twierdzenie i wyświetl cytowane zdania."""
import argparse
import json
from pathlib import Path
import sys

# Pozwala uruchamiać: python scripts/nazwa_skryptu.py z katalogu projektu.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scifact_rag.data import load_documents
from scifact_rag.llm import build_prediction
from scifact_rag.retrieval import Retriever, VARIANTS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim", required=True, help="Twierdzenie naukowe po angielsku")
    parser.add_argument("--variant", choices=VARIANTS, default="E0")
    parser.add_argument("--backend", choices=["ollama", "openai"], default="ollama")
    parser.add_argument("--model", default="qwen2.5:3b")
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    docs = load_documents()
    ranker = Retriever(docs)
    retrieved = ranker.search(args.claim, args.variant, args.top_k)
    result = build_prediction({"id": None, "claim": args.claim}, retrieved,
                              ranker.by_id, args.backend, args.model, args.base_url)
    print(f"\nTwierdzenie: {args.claim}\nOcena: {result['verdict']}")
    if not result["sources"]:
        print("Brak wystarczających dowodów w odnalezionych abstraktach.")
    for source in result["sources"]:
        print(f"\n{source['label']} | {source['title']} (doc_id={source['doc_id']})")
        for sentence in source["sentences"]:
            print(f"  [{sentence['id']}] {sentence['text']}")
        print("  Uzasadnienie:", source["explanation"])
    print("\nUwaga: obecność cytatu nie gwarantuje poprawności logicznej oceny.")
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print("Zapisano:", args.save)


if __name__ == "__main__":
    main()
