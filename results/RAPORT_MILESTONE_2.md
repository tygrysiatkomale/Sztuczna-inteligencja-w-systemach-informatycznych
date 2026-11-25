# Kamień milowy 2 — wyniki eksperymentów SciFact

> Wyniki zostały wygenerowane automatycznie z pliku `summary.json`, nie są wartościami przykładowymi.

**Dane:** `claims_tune.jsonl`, 162 zapytań; 101 z dowodami (mierzalne dla retrievalu).

| Wariant | Recall@5 | Hit@5 | MRR@5 | nDCG@5 | Czas średni [ms] |
|---|---:|---:|---:|---:|---:|
| E0 | 85.40% | 88.12% | 0.79 | 0.80 | 7.72 |
| E1 | 89.93% | 91.09% | 0.80 | 0.82 | 14.60 |
| E2 | 86.22% | 88.12% | 0.81 | 0.82 | 25.41 |
| E3 | 88.37% | 90.10% | 0.81 | 0.83 | 96.52 |

**Wariant z najlepszym Recall@5 na Tune:** E1 (przy remisie decyduje nDCG, a następnie czas).

To jest wybór wstępny, wymagający późniejszego potwierdzenia na odłożonym Final Eval.

**Uwaga:** metryki retrievalu ignorują twierdzenia bez oznaczonych dowodów, ponieważ Recall@5 jest dla nich niezdefiniowany.

## Ocena end-to-end (LLM)

Do uzupełnienia po uruchomieniu `run_end_to_end.py --limit 0`.

## Interpretacja

Na podstawie tego raportu opisz czy H1/H2/H3 uzyskały wsparcie w danych, zwracając uwagę na istotne błędy, koszt obliczeń i mały rozmiar Tune.

**Ważne:** wyniki z Tune nie są ostatecznym oszacowaniem generalizacji. Zbiór Final Eval pozostaje odłożony do kamienia milowego 3.
