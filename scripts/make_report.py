#!/usr/bin/env python3
"""Wygeneruj tabelę wyników z własnych, zmierzonych eksperymentów na Tune."""
import argparse
import json
from pathlib import Path
import sys

# Pozwala uruchamiać: python scripts/nazwa_skryptu.py z katalogu projektu.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fmt(x, percent=False):
    if x is None:
        return "—"
    if percent:
        return f"{100*x:.2f}%"
    return f"{x:.2f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=Path, default=Path("results/retrieval/tune/summary.json"))
    parser.add_argument("--end-to-end", type=Path, default=Path("results/end_to_end/full_tune/summary.json"))
    parser.add_argument("--output", type=Path, default=Path("results/RAPORT_MILESTONE_2.md"))
    args = parser.parse_args()
    if not args.retrieval.is_file():
        parser.error("Brak wyników retrievalu. Najpierw uruchom run_retrieval_experiments.py")
    retr = json.loads(args.retrieval.read_text(encoding="utf-8"))
    config = retr["config"]
    if config.get("split") != "tune" or config.get("limit") is not None:
        parser.error("Raport główny wymaga PEŁNEGO zbioru tune, bez --limit")
    rows = retr["metrics"]
    k = config["top_k"]
    m_key = f"recall_at_{k}"
    best = max(rows, key=lambda r: (r[m_key] or -1, r[f"ndcg_at_{k}"] or -1, -r["avg_latency_ms"]))
    doc = ["# Kamień milowy 2 — wyniki eksperymentów SciFact", "",
           "> Wyniki zostały wygenerowane automatycznie z pliku `summary.json`, nie są wartościami przykładowymi.", "",
           f"**Dane:** `claims_tune.jsonl`, {rows[0]['queries']} zapytań; "
           f"{rows[0]['positive_claims']} z dowodami (mierzalne dla retrievalu).", "",
           "| Wariant | Recall@5 | Hit@5 | MRR@5 | nDCG@5 | Czas średni [ms] |", "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        doc.append(f"| {r['variant']} | {fmt(r.get('recall_at_5'), True)} | "
                   f"{fmt(r.get('hit_at_5'), True)} | {fmt(r.get('mrr_at_5'))} | "
                   f"{fmt(r.get('ndcg_at_5'))} | {fmt(r['avg_latency_ms'])} |")
    doc += ["", f"**Wariant z najlepszym Recall@{k} na Tune:** {best['variant']} "
            "(przy remisie decyduje nDCG, a następnie czas).",
            "", "To jest wybór wstępny, wymagający późniejszego potwierdzenia na odłożonym Final Eval.",
            "", "**Uwaga:** metryki retrievalu ignorują twierdzenia bez oznaczonych dowodów, "
            "ponieważ Recall@5 jest dla nich niezdefiniowany.", ""]
    if args.end_to_end.is_file():
        e2e = json.loads(args.end_to_end.read_text(encoding="utf-8"))
        if not e2e.get("preview"):
            doc += ["## Ocena end-to-end (LLM)", "", "| Wariant | Label F1 | Label + Rationale F1 | Macro-F1 (zweryfikowane pary) |", "|---|---:|---:|---:|"]
            for v, entry in e2e["variants"].items():
                doc.append(f"| {v} | {fmt(entry['abstract_label']['f1'])} | "
                           f"{fmt(entry['abstract_label_rationale']['f1'])} | "
                           f"{fmt(entry['macro_f1_verified_pairs']['macro_f1'])} |")
            doc.append("\nMacierz dowodów i kompletne predykcje zapisano w wynikach `end_to_end`.\n")
        else:
            doc.append("Wyniki end-to-end pochodzą z podzbioru PREVIEW, dlatego nie porównano ich jako pełnego eksperymentu.\n")
    else:
        doc.append("## Ocena end-to-end (LLM)\n\nDo uzupełnienia po uruchomieniu `run_end_to_end.py --limit 0`.\n")
    doc += ["## Interpretacja", "",
            "Na podstawie tego raportu opisz czy H1/H2/H3 uzyskały wsparcie w danych, "
            "zwracając uwagę na istotne błędy, koszt obliczeń i mały rozmiar Tune.", "",
            "**Ważne:** wyniki z Tune nie są ostatecznym oszacowaniem generalizacji. "
            "Zbiór Final Eval pozostaje odłożony do kamienia milowego 3.\n"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(doc), encoding="utf-8")
    print(f"Raport: {args.output}")


if __name__ == "__main__":
    main()
