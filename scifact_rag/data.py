"""Ładowanie danych SciFact bez wykorzystywania etykiet w retrievalu."""
from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Nie znaleziono {path}. Uruchom scripts/prepare_data.py")
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def save_jsonl(path: str | Path, records: list[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_documents(path: str | Path = "data/processed/documents.jsonl") -> list[dict]:
    documents = load_jsonl(path)
    for doc in documents:
        if not isinstance(doc.get("doc_id"), int):
            raise ValueError("Nieprawidłowy doc_id")
        if "retrieval_text" not in doc:
            doc["retrieval_text"] = doc["title"] + "\n" + " ".join(doc["abstract"])
    if len({doc["doc_id"] for doc in documents}) != len(documents):
        raise ValueError("Zduplikowane doc_id")
    return documents


def gold_document_ids(claim: dict) -> set[int]:
    """Tylko evidence; cited_doc_ids NIE SĄ dowodami pozytywnymi."""
    return {int(doc_id) for doc_id in claim.get("evidence", {})}


def verified_pairs(claim: dict) -> dict[int, str]:
    """Pary z evidence oraz cited_doc_ids; negatives z cited bez evidence.

    Dla innego (niecytowanego) dokumentu etykiety NIE znamy.
    """
    pairs = {int(doc_id): "NOINFO" for doc_id in claim.get("cited_doc_ids", [])}
    for doc_id, rationales in claim.get("evidence", {}).items():
        labels = {r["label"] for r in rationales}
        if len(labels) != 1:
            continue  # brak jednoznacznej etykiety dla pary
        pairs[int(doc_id)] = labels.pop()
    return pairs
