"""Metryki retrievalu i uzupełniająca ewaluacja SciFact.

Wyniki label+rationale liczone wg definicji SciFact na poziomie abstraktów:
- TP: dokument i etykieta zgodne oraz przynajmniej jeden cały gold rationale
  znajduje się wśród trzech pierwszych wskazanych zdań.
"""
from __future__ import annotations

from collections import Counter
import math

from scifact_rag.data import gold_document_ids, verified_pairs


def retrieval_score(claim: dict, ranked_ids: list[int], k: int = 5) -> dict | None:
    """None dla twierdzeń bez gold evidence: Recall jest dla nich niezdefiniowany."""
    gold = gold_document_ids(claim)
    if not gold:
        return None
    selected = ranked_ids[:k]
    matches = [i for i, doc_id in enumerate(selected, start=1) if doc_id in gold]
    ideal_dcg = sum(1 / math.log2(i + 1) for i in range(1, min(len(gold), k) + 1))
    dcg = sum(1 / math.log2(i + 1) for i in matches)
    return {
        "hit_at_k": int(bool(matches)),
        "recall_at_k": len(matches) / len(gold),
        "mrr_at_k": 1.0 / matches[0] if matches else 0.0,
        "ndcg_at_k": dcg / ideal_dcg,
        "hits": len(matches),
        "gold_count": len(gold),
    }


def prf(tp: int, predicted: int, gold: int) -> dict:
    precision = tp / predicted if predicted else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1,
            "true_positive": tp, "predicted": predicted, "gold": gold}


def macro_f1_for_verified_pairs(claims: list[dict], predictions: list[dict]) -> dict:
    """Tylko jawnie ocenione pary evidence+cited_doc_ids.

    Nieuzyskany z retrievalu pozytywny dokument jest przewidywany jako NOINFO;
    nie są przypisywane etykiety pozostałym 5 tysiącom dokumentów.
    """
    pred_by_id = {int(row["id"]): row for row in predictions}
    confusion = Counter()
    for claim in claims:
        row = pred_by_id.get(int(claim["id"]))
        if row is None:
            continue
        predicted = {int(k): v["label"] for k, v in row.get("evidence", {}).items()}
        for doc_id, gold_label in verified_pairs(claim).items():
            confusion[gold_label, predicted.get(doc_id, "NOINFO")] += 1
    labels = ["SUPPORT", "CONTRADICT", "NOINFO"]
    total = sum(confusion.values())
    by_label = {}
    for label in labels:
        tp = confusion[label, label]
        fp = sum(confusion[g, label] for g in labels if g != label)
        fn = sum(confusion[label, p] for p in labels if p != label)
        by_label[label] = prf(tp, tp + fp, tp + fn)
        by_label[label]["support"] = sum(confusion[label, p] for p in labels)
    return {"pair_count": total, "macro_f1": sum(v["f1"] for v in by_label.values()) / 3,
            "by_label": by_label, "accuracy": sum(confusion[x, x] for x in labels) / total if total else 0.0,
            "scope": "tylko evidence i cited_doc_ids, nie wszystkie pary korpusu"}


def evaluate_evidence(claims: list[dict], predictions: list[dict]) -> dict:
    """Mikro-F1 na relacjach claim-doc i rationale; pred może być pusty."""
    gold_by_id = {int(c["id"]): c for c in claims}
    label_tp = rationale_tp = sent_tp = 0
    n_gold_docs = n_pred_docs = 0
    n_gold_sent = n_pred_sent = 0
    docs_by_claim = set()

    for row in predictions:
        claim_id = int(row["id"])
        if claim_id not in gold_by_id:
            continue
        docs_by_claim.add(claim_id)
        gold_evidence = gold_by_id[claim_id].get("evidence", {})
        predicted_evidence = row.get("evidence", {})
        n_gold_docs += len(gold_evidence)
        n_pred_docs += len(predicted_evidence)
        for rationales in gold_evidence.values():
            n_gold_sent += len({i for rationale in rationales for i in rationale["sentences"]})
        for doc_id, pred in predicted_evidence.items():
            selected = [int(i) for i in pred.get("sentences", [])]
            n_pred_sent += len(set(selected))
            selected_set = set(selected)
            gold_rationales = gold_evidence.get(str(doc_id), [])
            matching = [r for r in gold_rationales if r["label"] == pred["label"]]
            if matching:
                label_tp += 1
                if any(set(r["sentences"]).issubset(set(selected[:3])) for r in matching):
                    rationale_tp += 1
                complete_sets = [set(r["sentences"]) for r in matching
                                 if set(r["sentences"]).issubset(selected_set)]
                if complete_sets:
                    sent_tp += len(selected_set & set().union(*complete_sets))
    return {
        "abstract_label": prf(label_tp, n_pred_docs, n_gold_docs),
        "abstract_label_rationale": prf(rationale_tp, n_pred_docs, n_gold_docs),
        "sentence_level_approx": prf(sent_tp, n_pred_sent, n_gold_sent),
        "evaluated_claims": len(docs_by_claim),
        "note": "Sentence-level to własna wersja przybliżona; do oficjalnego benchmarku użyj scifact-evaluator.",
    }
