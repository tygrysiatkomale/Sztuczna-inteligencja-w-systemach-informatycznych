#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Brakuje pliku: {path}")
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def save_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")


def claim_label(claim: dict) -> str:
    labels = {
        rationale["label"]
        for rationales in claim.get("evidence", {}).values()
        for rationale in rationales
    }
    if not labels:
        return "NOINFO"
    if len(labels) > 1:
        return "MIXED"
    return labels.pop()


def stratified_split(claims: list[dict], seed: int, train_fraction=0.8):
    rng = random.Random(seed)
    by_label = defaultdict(list)
    for claim in claims:
        by_label[claim_label(claim)].append(claim)
    fit, tune = [], []
    for label in sorted(by_label):
        items = sorted(by_label[label], key=lambda c: c["id"])
        rng.shuffle(items)
        n_fit = round(len(items) * train_fraction)
        fit.extend(items[:n_fit])
        tune.extend(items[n_fit:])
    rng.shuffle(fit)
    rng.shuffle(tune)
    return fit, tune


def validate(documents: list[dict], split_to_claims: dict) -> None:
    document_map = {doc["doc_id"]: doc for doc in documents}
    if len(document_map) != len(documents):
        raise ValueError("Powtarzające się doc_id w korpusie!")

    ids_by_split = {}
    for split, claims in split_to_claims.items():
        ids = {claim["id"] for claim in claims}
        if len(ids) != len(claims):
            raise ValueError(f"Powtarzające się id w {split}!")
        ids_by_split[split] = ids

        for claim in claims:
            if not isinstance(claim.get("claim"), str):
                raise ValueError(f"Nieprawidłowe twierdzenie id={claim['id']}")
            for doc_id, rationales in claim.get("evidence", {}).items():
                doc_id_int = int(doc_id)
                if doc_id_int not in document_map:
                    raise ValueError(f"Brak doc_id={doc_id} dla claim_id={claim['id']}")
                sentences = document_map[doc_id_int]["abstract"]
                for rationale in rationales:
                    if rationale["label"] not in {"SUPPORT", "CONTRADICT"}:
                        raise ValueError(f"Nieznana etykieta: {rationale['label']}")
                    for sentence_id in rationale["sentences"]:
                        if not isinstance(sentence_id, int) or not 0 <= sentence_id < len(sentences):
                            raise ValueError(
                                f"Błędny sentence_id={sentence_id} dla claim_id={claim['id']}, doc_id={doc_id}"
                            )

    if ids_by_split["train"] & ids_by_split["dev"]:
        raise ValueError("Przeciek: identyczne claim_id w train i dev!")


def summary_for(claims: list[dict]) -> dict:
    return {
        "claims": len(claims),
        "labels": dict(sorted(Counter(claim_label(claim) for claim in claims).items())),
        "claims_with_gold_evidence": sum(bool(claim.get("evidence")) for claim in claims),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Kontrola i przygotowanie SciFact")
    parser.add_argument("--raw-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    corpus = load_jsonl(args.raw_dir / "corpus.jsonl")
    train = load_jsonl(args.raw_dir / "claims_train.jsonl")
    dev = load_jsonl(args.raw_dir / "claims_dev.jsonl")
    validate(corpus, {"train": train, "dev": dev})
    fit, tune = stratified_split(train, args.seed)

    docs_processed = []
    for doc in corpus:
        abstract = doc["abstract"]
        docs_processed.append({
            "doc_id": doc["doc_id"],
            "title": doc["title"],
            "abstract": abstract,
            "structured": doc.get("structured", False),
            "retrieval_text": doc["title"] + "\n" + " ".join(abstract),
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.output_dir / "documents.jsonl", docs_processed)
    save_jsonl(args.output_dir / "claims_train_fit.jsonl", fit)
    save_jsonl(args.output_dir / "claims_tune.jsonl", tune)
    save_jsonl(args.output_dir / "claims_final_eval.jsonl", dev)

    sentence_counts = [len(doc["abstract"]) for doc in corpus]
    report = {
        "source": "https://github.com/allenai/scifact",
        "seed": args.seed,
        "corpus": {
            "documents": len(corpus),
            "mean_sentences_per_abstract": round(mean(sentence_counts), 2) if sentence_counts else None,
            "median_sentences_per_abstract": median(sentence_counts) if sentence_counts else None,
        },
        "splits": {
            "original_train": summary_for(train),
            "train_fit": summary_for(fit),
            "tune": summary_for(tune),
            "final_eval_original_dev": summary_for(dev),
        },
    }
    with (args.output_dir / "dataset_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nPrzetworzone dane zapisano w: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
