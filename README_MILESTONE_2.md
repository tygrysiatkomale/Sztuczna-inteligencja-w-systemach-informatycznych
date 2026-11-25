# Kamień milowy 2 — implementacja RAG i eksperymenty na SciFact

**Projekt:** Wyjaśnialna weryfikacja twierdzeń naukowych z wykorzystaniem RAG  
**Przedmiot:** Sztuczna inteligencja w systemach informatycznych  
**Kontynuacja kamienia milowego 1:** istniejące `README.md`, `scripts/prepare_data.py` oraz `data/processed/*` pozostają bez zmian.

## 1. Cel etapu

Implementacja działającego prototypu weryfikacji twierdzeń naukowych, który:

1. na podstawie twierdzenia wyszukuje pięć abstraktów z korpusu SciFact;
2. za pomocą modelu LLM określa relację **każdego** dokumentu do twierdzenia: `SUPPORT`, `CONTRADICT` lub `NOINFO`;
3. dla `SUPPORT`/`CONTRADICT` zwraca identyfikatory autentycznych zdań z abstraktu i krótkie uzasadnienie;
4. porównuje cztery metody retrievalu pod względem jakości i szybkości;
5. zapisuje wyniki w postaci JSONL, JSON i CSV, umożliwiając niezależną analizę eksperymentów.

**Ważne:** potwierdzamy albo podważamy twierdzenie **w świetle konkretnych publikacji**, a nie stwierdzamy jego absolutnej prawdziwości. Jeśli źródła są sprzeczne, końcowy werdykt może być `MIXED`. Brak dowodów (`NOINFO`) nie jest dowodem fałszu.

## 2. Cztery warianty eksperymentalne

| Wariant | Metoda | Konfiguracja | Cel eksperymentu |
|---|---|---|---|
| E0 | **BM25** | `k1=1.5`, `b=0.75`, Top-5 | Bazowa skuteczność dopasowania słów (baseline) |
| E1 | **Dense Retrieval** | `all-MiniLM-L6-v2`, podobieństwo kosinusowe, Top-5 | Wpływ dopasowania semantycznego |
| E2 | **Hybrid Retrieval** | BM25 + Dense, Reciprocal Rank Fusion (RRF k=60), rankingi Top-100, Top-5 | Wpływ połączenia obu podejść |
| E3 | **Hybrid + Reranker** | E2: kandydaci Top-20 → `cross-encoder/ms-marco-MiniLM-L-6-v2` → Top-5 | Wpływ dokładniejszego ponownego uszeregowania |

Korpus zawiera **5183 abstrakty**. Jednostką wyszukiwania jest cały abstrakt z tytułem, z zachowaniem oryginalnego podziału na zdania do wskazywania dowodów. BM25 jest zaimplementowany samodzielnie. Wyszukiwanie semantyczne stosuje gotowy model embeddingowy, a reranker — gotowy cross-encoder. Nie prowadzimy fine-tuningu LLM.

Wszystkie warianty otrzymują ten sam korpus, te same twierdzenia i `Top-5`. W części LLM używamy tego samego modelu i promptu, żeby badać **wpływ zmiany retrievalu**, a nie różnice między LLM.

Parametry eksperymentalne można kontrolować z wiersza poleceń: `--rrf-k`, `--rrf-depth`, `--candidate-k`, `--embedding-model`, `--reranker-model`. Zmiany wymagają nowej, jawnej serii pomiarowej — nie wolno dobierać parametrów na zbiorze końcowym.

## 3. Dane i zapobieganie wyciekowi

| Plik | Przeznaczenie |
|---|---|
| `data/processed/documents.jsonl` | baza wiedzy; tekst i identyfikatory zdań, bez etykiet |
| `data/processed/claims_train_fit.jsonl` | analiza błędów, rozwijanie promptu, eksperymenty developerskie |
| `data/processed/claims_tune.jsonl` | **162 twierdzenia do porównania wariantów w kamieniu milowym 2** |
| `data/processed/claims_final_eval.jsonl` | **300 twierdzeń odkładamy do kamienia milowego 3** |

Na Tune jest **101 twierdzeń z gold evidence i 61 bez oznaczonego dowodu**. W metrykach retrievalu `Recall@5`, `MRR@5` i `nDCG@5` bierzemy pod uwagę wyłącznie 101 twierdzeń, bo dla pozostałych mianownik Recall wynosi zero. Liczymy średnią metryk po twierdzeniach z dowodami; nie traktujemy `NOINFO` jako porażki retrievalu.

**Nie korzystamy z `evidence` ani z `cited_doc_ids` przy wyszukiwaniu dokumentów i generowaniu odpowiedzi.** Pola te są wykorzystywane wyłącznie przez evaluator. W rzeczywistych danych etykiety mogą być niekompletne: nie należy uznawać każdego dokumentu bez wskazanego gold evidence za zweryfikowany negatyw.

## 4. Struktura kodu

```text
scifact_rag/
    data.py          # JSONL, odczyt danych, dostęp do etykiet tylko dla ewaluatora
    retrieval.py     # BM25, embeddingi, RRF, cross-encoder
    llm.py           # prompt, Ollama/OpenAI API, walidacja odwołań
    metrics.py       # Recall@5, MRR@5, nDCG@5, Label+Rationale F1, Macro-F1
scripts/
    prepare_data.py               # Twój istniejący skrypt z etapu 1
    run_retrieval_experiments.py  # eksperymenty E0-E3, bez LLM
    verify_claim.py               # pojedyncza weryfikacja i uzasadnienie
    run_end_to_end.py             # eksperymenty E0-E3 + LLM
    make_report.py                # automatyczny raport wyników z Tune
tests/
    test_scifact_rag.py           # testy jednostkowe / kontrolowane przykłady
requirements-milestone2.txt
README_MILESTONE_2.md
```

Pliki `data/`, istniejący `README.md` oraz `scripts/prepare_data.py` **nie są nadpisywane** przez paczkę drugiego etapu.

## 5. Uruchomienie w Ubuntu / WSL

Uruchom w **głównym katalogu swojego projektu** (tam, gdzie `data/`, `README.md`, `scripts/`). Po rozpakowaniu paczki od razu do tego katalogu:

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-milestone2.txt

# Testy nie pobierają modeli i nie wymagają Ollama:
python -m unittest discover -s tests -v
```

Pierwsze uruchomienie Dense i E3 **pobiera wagi modeli z Hugging Face**, a ich lokalne ładowanie może zająć dłuższy czas; wymagane jest połączenie z Internetem. Modele są używane na CPU, jeśli nie ma GPU. Embeddingi korpusu zapisujemy w `data/cache/` i wykorzystujemy ponownie przy kolejnych uruchomieniach.

### 5.1. Pierwszy test BM25 bez pobierania modeli

```bash
python scripts/run_retrieval_experiments.py --variants E0 --limit 10
```

Wyniki testowe (PREVIEW) zapiszą się w `results/retrieval/tune_preview/`. **Nie używaj wyników PREVIEW jako finalnych pomiarów.**

### 5.2. Pełny eksperyment E0–E3 (najważniejszy dla kamienia milowego 2)

```bash
python scripts/run_retrieval_experiments.py --split tune
```

Skrypt wykona 4 warianty na wszystkich **162 twierdzeniach** i zapisze m.in.:

```text
results/retrieval/tune/
    summary.json          # zbiorcze metryki i konfiguracja
    comparison.csv        # porównanie E0-E3 do tabeli/wykresu
    E0_per_claim.jsonl    # wyszukane dokumenty dla każdego twierdzenia
    E1_per_claim.jsonl
    E2_per_claim.jsonl
    E3_per_claim.jsonl
```

Latencja jest mierzona **po inicjalizacji i rozgrzaniu modeli**, aby nie mylić czasu pobierania/indeksowania z kosztem pojedynczego zapytania. Pomiar na CPU i GPU może się różnić. Nie obejmuje opóźnienia LLM.

### 5.3. Opcjonalna konfiguracja i porównanie parametrów

```bash
# Porównaj RRF k=10 zamiast 60 — na Tune, bez naruszania Final Eval.
python scripts/run_retrieval_experiments.py --variants E2 --rrf-k 10 \
  --output-dir results/retrieval_rrf10

# Oddzielny test rerankingu na 40 kandydatach.
python scripts/run_retrieval_experiments.py --variants E3 --candidate-k 40 \
  --output-dir results/retrieval_candidates40
```

Nie zastępuj pomiarów jedną „najładniejszą” próbką — przechowuj komplet konfiguracji i wyników.

## 6. Prototyp z LLM i wyjaśnieniami

**Retriever mówi, który dokument jest podobny. LLM sprawdza, czy zdania z tego dokumentu rzeczywiście popierają czy podważają twierdzenie.** RAG nie rozstrzyga tego na podstawie samych embeddingów.

LLM otrzymuje twierdzenie i **pięć rzeczywiście znalezionych abstraktów**, każdy z `doc_id` i ponumerowanymi zdaniami. Zwraca JSON ze stosunkiem każdego dokumentu do twierdzenia i maksymalnie 3 `sentence_ids` na dokument. System odrzuca nieistniejące identyfikatory, duplikaty, brak wymaganych zdań, nadmiar zdań oraz odwołania spoza przekazanego kontekstu.

**Dwie możliwości uruchomienia LLM:**

**A. Lokalnie przez Ollama** (bez klucza API; potrzebny lokalny model i odpowiedni RAM; GPU przyspiesza): zainstaluj Ollama wg [ollama.com](https://ollama.com), a następnie:

```bash
ollama pull qwen2.5:3b
# Gdy serwer nie jest uruchomiony, w osobnym terminalu: ollama serve
python scripts/verify_claim.py \
  --claim "Antiretroviral therapy increases rates of tuberculosis across a broad range of CD4 strata." \
  --variant E0 --backend ollama --model qwen2.5:3b \
  --save results/demo.json
```

**B. Przez kompatybilne API** (wymaga klucza, zwykle płatne):

```bash
export OPENAI_API_KEY="TWÓJ_KLUCZ"
python scripts/verify_claim.py \
  --claim "Antiretroviral therapy increases rates of tuberculosis across a broad range of CD4 strata." \
  --variant E2 --backend openai --model gpt-4o-mini
```

Klucza nie dodawaj do repozytorium. W razie korzystania z API sprawdź obowiązujące ceny i uprawnienia modelu. W `run_end_to_end.py` podaj identyczny `--backend` i `--model` dla wszystkich wariantów.

### 6.1. Eksperyment end-to-end z LLM

```bash
# Krótki test działania i formatu odpowiedzi — 10 pierwszych twierdzeń.
python scripts/run_end_to_end.py --variants E0 E1 E2 E3 --limit 10 \
  --backend ollama --model qwen2.5:3b

# PEŁNA ewaluacja na Tune (może wymagać wielu wywołań LLM).
python scripts/run_end_to_end.py --variants E0 E1 E2 E3 --limit 0 \
  --backend ollama --model qwen2.5:3b
```

Wyniki otrzymasz w `results/end_to_end/preview` lub `results/end_to_end/full_tune` (`*_predictions.jsonl` oraz `summary.json`). W predykcjach są zarówno końcowa etykieta, jak i dokładne cytowane zdania z publikacji.

W przypadku kilku publikacji o różnych etykietach wynik całego twierdzenia może mieć `MIXED` — jest to **etykieta interfejsu**, a nie czwarta klasa SciFact. Ewaluacja SciFact odbywa się przede wszystkim na poziomie **pary (twierdzenie, abstrakt)**, nie na podstawie jednego werdyktu zbiorczego.

## 7. Metryki i ich interpretacja

### Retrieval (główna część eksperymentu)

- **Recall@5** — średnia część oznaczonych publikacji odnaleziona w pierwszych 5 wynikach; metryka priorytetowa.
- **Hit@5** — odsetek twierdzeń, dla których znaleziono choć jeden oznaczony dokument.
- **MRR@5** — jak wysoko w rankingu pojawia się pierwszy właściwy dokument.
- **nDCG@5** — ocena kolejności właściwych dokumentów (trafność binarna).
- **Średni i medianowy czas retrievalu [ms]** — koszt wyszukania publikacji po przygotowaniu indeksów.

### End-to-end z LLM

- **Abstract Label F1** — czy model wskazał właściwą publikację z właściwą etykietą `SUPPORT`/`CONTRADICT`.
- **Abstract Label+Rationale F1** — czy do właściwej publikacji i etykiety wskazał **pełny referencyjny zestaw zdań** (w pierwszych 3 predykcjach); najważniejsze dla wyjaśnialności.
- **Sentence F1** — własny pomocniczy wariant oceny fragmentów, z uwzględnieniem pełnych zestawów zdań. **Nie należy przedstawiać go jako certyfikowanego wyniku oficjalnego benchmarku.**
- **Macro-F1 (zweryfikowane pary)** — klasyfikacja trójklasowa tylko w zbiorze par oznaczonych `evidence` lub występujących w `cited_doc_ids` bez dowodów. Nie traktujemy wszystkich 5183 nieoznaczonych par jako `NOINFO`.

Zgodność samych identyfikatorów źródeł gwarantuje **sprawdzalność cytatów**, nie poprawność uzasadnienia. Dodatkowa ręczna analiza trafności argumentacji na próbie błędów i wyjaśnień należy do kamienia milowego 3.

### Raport automatyczny

Po pełnym retrievalu:

```bash
python scripts/make_report.py
cat results/RAPORT_MILESTONE_2.md
```

Raport pobiera **rzeczywiste** metryki z `results/retrieval/tune/summary.json` i, jeśli przeprowadzono pełny eksperyment LLM, także z `results/end_to_end/full_tune/summary.json`. Nie wpisuje fikcyjnych liczb ani nie wyświetla eksperymentu PREVIEW jako końcowego.

## 8. Hipotezy i wybór wariantu

| Hipoteza | Jak testujemy |
|---|---|
| **H1:** Dense/Hybrid poprawi Recall@5 względem BM25 | E0 vs E1, E2 na Tune |
| **H2:** reranking poprawi kolejność/trafność, ale zwiększy czas | E2 vs E3, nDCG@5, MRR@5, latency |
| **H3:** lepszy retrieval poprawi decyzję i wskazanie dowodów | E0–E3 z tym samym LLM, Abstract Label+Rationale F1 |

**Dobór rozwiązania:** priorytetem jest **Abstract Label+Rationale F1**, jeśli pełne wyniki LLM są dostępne. W przypadku ich braku raportujemy wyłącznie wstępnie najlepszy wariant retrievalu na podstawie `Recall@5` (remis: nDCG@5, następnie czas). Wybór końcowy i generalizacja muszą zostać później potwierdzone na `claims_final_eval.jsonl`.

## 9. Ograniczenia

- Prace są w języku angielskim i dotyczą przeważnie biomedycyny.
- Zbiór ma etykiety tylko dla wybranych dokumentów; brak etykiety nie gwarantuje braku dowodu.
- `all-MiniLM-L6-v2` ma ograniczoną długość wejścia, więc niektóre dłuższe abstrakty mogą być ucięte przy embeddingowaniu; BM25 indeksuje całość.
- Cross-Encoder także może obciąć bardzo długi tekst do limitu 512 tokenów.
- Mały LLM lokalny może błędnie interpretować relacje i format JSON; zweryfikowane cytaty nie eliminują błędów semantycznych.
- Metryki liczone na zbiorze Tune służą do wyboru konfiguracji, nie do końcowej niezależnej oceny.
- Pomiar czasu zależy od konkretnego komputera, wersji bibliotek i tego, czy model pracuje na CPU czy GPU.

## 10. Co musi znaleźć się w oddanym kamieniu milowym 2?

1. Kod działającego retrievera E0–E3 i prototypu LLM, testy oraz instrukcja uruchomienia.
2. Plik `results/retrieval/tune/comparison.csv` wygenerowany na pełnym Tune.
3. Przykład rzeczywistej odpowiedzi z `results/demo.json` pokazujący konkretne zdania dowodowe.
4. Jeśli sprzęt pozwoli: `results/end_to_end/full_tune/summary.json` z wynikami LLM dla czterech wariantów.
5. `results/RAPORT_MILESTONE_2.md` z tabelą wyników, porównaniem E0–E3 i wnioskami H1–H3; bez twierdzeń wykraczających poza wykonane pomiary.

**Stan przekazanej paczki:** kod i testy są przygotowane. Wyniki liczbowych eksperymentów na prawdziwym SciFact powstaną **dopiero po uruchomieniu skryptów na Twoim lokalnym zbiorze**. Nie są dołączone jako dane „zmierzone”, ponieważ nie było dostępu do plików z Twojego WSL.

## Źródła

- [SciFact — repozytorium](https://github.com/allenai/scifact)
- [SciFact — format danych](https://github.com/allenai/scifact/blob/master/doc/data.md)
- [SciFact — szczegóły ewaluacji](https://github.com/allenai/scifact/blob/master/doc/evaluation.md)
- [Wadden i in. (EMNLP 2020)](https://aclanthology.org/2020.emnlp-main.609/)
