# Wyjaśnialna weryfikacja twierdzeń naukowych z wykorzystaniem RAG

**Projekt:** Sztuczna inteligencja w systemach informatycznych  
**Kamień milowy 1:** Definicja problemu, dobór danych i projekt eksperymentu  

## 1. Cel projektu i definicja problemu

Celem projektu jest stworzenie systemu wykorzystującego Retrieval-Augmented Generation (RAG) do weryfikowania twierdzeń naukowych na podstawie treści publikacji. Użytkownik podaje twierdzenie w języku angielskim, a system:

1. wyszukuje publikacje mogące zawierać odpowiednie dowody;
2. wskazuje konkretne zdania z tych publikacji, które odnoszą się do twierdzenia;
3. ocenia relację między twierdzeniem a znalezionymi dowodami;
4. generujevuzasadnienie oparte na wskazanych zdaniach.

Przewidziane wyniki oceny to:

- SUPPORT - wskazane źródło potwierdza twierdzenie;
- CONTRADICT - wskazane źródło podważa twierdzenie;
- NOT_ENOUGH_INFO - w dostępnych, odnalezionych źródłach brak wystarczających dowodów do rozstrzygnięcia.

### Główne pytanie badawcze

**W jakim stopniu metoda wyszukiwania dokumentów w systemie RAG wpływa na poprawność weryfikacji twierdzeń naukowych oraz trafność wskazywanych zdań uzasadniających decyzję?**

Projekt nie ogranicza się zatem do klasyfikacji. Wymaganiem funkcjonalnym jest wyjaśnialność każdej decyzji: dla oceny `SUPPORT` lub `CONTRADICT` system musi podać identyfikator publikacji, numery wykorzystanych zdań oraz opis powiązania tych zdań z twierdzeniem.

## 2. Zbiór danych: SciFact

**Źródło:** [allenai/scifact](https://github.com/allenai/scifact)  
**Publikacja:** [Wadden i in., *Fact or Fiction: Verifying Scientific Claims* (EMNLP 2020)](https://arxiv.org/abs/2004.14974)  
**Dokumentacja struktury:** [SciFact - `doc/data.md`](https://github.com/allenai/scifact/blob/master/doc/data.md)

SciFact zawiera **5 183 abstrakty artykułów naukowych i 1 409 twierdzeń** wraz z oznaczeniami relacji do publikacji oraz **rationales** - zestawami zdań uzasadniających ocenę. Dotyczy głównie zagadnień biomedycznych. Dane są w języku angielskim.

### 2.1. Oficjalny podział danych

| Zbiór | Liczba twierdzeń | SUPPORT | CONTRADICT | NOINFO |
|---|---:|---:|---:|---:|
| `claims_train.jsonl` | 809 | 332 | 173 | 304 |
| `claims_dev.jsonl` | 300 | 124 | 64 | 112 |
| `claims_test.jsonl` | 300 | 100* | 100* | 100* |
| **Łącznie** | **1 409** | **556** | **337** | **516** |


### 2.2. Pliki i struktura

| Plik | Zawartość | Rola w projekcie |
|---|---|---|
| `corpus.jsonl` | 5 183 publikacje (`doc_id`, `title`, lista zdań `abstract`, `structured`) | Baza wiedzy systemu RAG |
| `claims_train.jsonl` | Twierdzenia z oznaczonymi dokumentami, relacjami i zdaniami dowodowymi | Projektowanie i strojenie rozwiązania |
| `claims_dev.jsonl` | Oznaczone twierdzenia walidacyjne | Końcowa, odłożona ewaluacja |
| `claims_test.jsonl` | Twierdzenia bez publicznych ocen referencyjnych | Opcjonalna demonstracja, nie lokalna ocena metryk |

Najważniejsze pola:

- `claim` - tekst twierdzenia;
- `evidence` - mapa identyfikatorów dokumentów na oznaczone zestawy dowodowe;
- `label` - `SUPPORT` albo `CONTRADICT` dla danego dowodu;
- `sentences` - indeksy zdań w tablicy `abstract` (od 0);
- `cited_doc_ids` - prace cytowane w zdaniu źródłowym, niekoniecznie będące dowodami.

Jedno twierdzenie może mieć więcej niż jeden dokument dowodowy, a jeden dokument więcej niż jeden alternatywny zestaw zdań.


## 3. Przygotowanie danych (etap 1)


Plan:

1. Pobranie oryginalnych plików JSONL ze źródła SciFact.
2. Sprawdzenie poprawności schematu, braku duplikatów `doc_id` i `claim.id` oraz referencje do dokumentów.
3. Zachowanie `doc_id`, tytułu i tekstu każdego zdania wraz z jego indeksem.
4. Utworzenie tekstu do indeksowania jako `title + abstract`, bez usuwania treści merytorycznej.
5. Podzielenie zbioru `claims_train` (stratyfikowanie, `seed=42`) na część rozwojową 80% i część strojenia 20%; `claims_dev`.
6. Zweryfikowanie, że `evidence` i indeksy zdań wskazują rzeczywiste rekordy.
7. Zapisanie plików przetworzonych oraz statystyk zbioru (liczebności klas, odsetek przypadków bez dowodów, długości abstraktów itp.).

### Przygotowanie danych

```bash
wget https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz
tar -xzf data.tar.gz

python scripts/prepare_data.py --raw-dir data --output-dir data/processed --seed 42
```

Skrypt przygotowuje:

```text
data/
├── corpus.jsonl
├── claims_train.jsonl
├── claims_dev.jsonl
├── claims_test.jsonl
└── processed/
    ├── documents.jsonl
    ├── claims_train_fit.jsonl
    ├── claims_tune.jsonl
    ├── claims_final_eval.jsonl
    └── dataset_summary.json
```

## 4. Proponowana architektura RAG

```text
Twierdzenie użytkownika
         |
         v
Wyszukiwanie abstraktów (BM25 / Dense / Hybrid)
         |
         v
Reranking znalezionych abstraktów
         |
         v
Top-5 dokumentów z numerowanymi zdaniami
         |
         v
LLM: ocena relacji dla każdego dokumentu
     SUPPORT / CONTRADICT / NOINFO
         |
         v
Wybór zdań dowodowych + krótka argumentacja
         |
         v
Walidacja cytatów i identyfikatorów w korpusie
         |
         v
Wynik: ocena + dokumenty + zdania + uzasadnienie (XAI)
```

**Zasada wyjaśnialności:** dla każdego rozstrzygnięcia model przedstawia fakty/przesłanki z publikacji. Dostaje źródłowe zdania z oznaczeniami `(doc_id, sentence_id)`. Program weryfikuje, czy wskazane zdania rzeczywiście znajdują się w dokumencie. To zapewnia sprawdzalność odwołań, ale samo w sobie nie gwarantuje poprawnego rozumowania - dlatego trafność uzasadnień będzie osobno oceniana.

Format odpowiedzi:

```json
{
  "claim": "Przykładowe twierdzenie",
  "verdict": "CONTRADICT",
  "evidence": [
    {
      "doc_id": 11328820,
      "sentence_ids": [7, 9],
      "relation": "CONTRADICT"
    }
  ],
  "explanation": "Wskazane zdania w publikacji opisują wyniki sprzeczne z treścią twierdzenia."
}
```

## 5. Design of Experiment (DoE)

### 5.1. Hipotezy

- **H1:** Dense Retrieval lub Hybrid Retrieval zapewni wyższy `Recall@5` właściwych abstraktów niż klasyczny BM25.
- **H2:** Dodatkowy reranking poprawi trafność wskazywanych dokumentów i zdań dowodowych, ale zwiększy czas odpowiedzi.
- **H3:** Wyższa jakość wyszukiwania przełoży się na wyższy wynik łącznej oceny etykiety i uzasadnienia, a nie tylko na większą liczbę znalezionych dokumentów.

### 5.2. Baseline i warianty

| ID | Metoda odnajdywania dokumentów | Wybór kontekstu dla LLM | Rola |
|---|---|---|---|
| **E0** | **BM25** (dopasowanie leksykalne) | Top-5 abstraktów | **Baseline** |
| **E1** | Dense Retrieval (embeddingi) | Top-5 abstraktów | Wpływ wyszukiwania semantycznego |
| **E2** | Hybrid: BM25 + Dense, połączenie rankingów RRF | Top-5 abstraktów | Wpływ łączenia metod |
| **E3** | Hybrid + Cross-Encoder Reranker | Przesortowanie kandydatów, Top-5 | Wpływ dodatkowego rerankingu |

**Wspólne warunki porównania:** ten sam korpus, identyczne twierdzenia, jeden model LLM, ten sam prompt, ten sam limit kontekstu i końcowe `Top-5`. Dla E3 reranker otrzymuje np. 20 kandydatów, a przekazuje 5 najlepszych.

### 5.3. Metryki

| Poziom | Metryka | Co pozwala ocenić? |
|---|---|---|
| Wyszukiwanie | **Recall@5** | Czy w pięciu dokumentach znalazły się oznaczone abstrakty dowodowe? |
| Wyszukiwanie | **MRR@5**, **nDCG@5** | Jak wysoko w rankingu występują właściwe abstrakty? |
| Ocena relacji | **Macro-F1** dla SUPPORT / CONTRADICT / NOINFO | Czy system poprawnie określa relację twierdzenie-źródło? |
| Uzasadnienia | **Sentence-level Precision / Recall / F1** | Czy wskazane zdania pokrywają się z oznaczonymi zestawami dowodowymi? |
| Wynik końcowy | **Abstract Label+Rationale F1** (metryka SciFact) | Czy jednocześnie poprawne są dokument, relacja i wystarczający zestaw zdań? |
| Wiarygodność cytatów | **Odsetek poprawnych odwołań** | Czy każde wskazane `(doc_id, sentence_id)` istnieje w przekazanym kontekście? |
| Wydajność | **Latency (ms lub s / zapytanie)** | Ile kosztuje czasowo zastosowanie bardziej złożonej metody? |
