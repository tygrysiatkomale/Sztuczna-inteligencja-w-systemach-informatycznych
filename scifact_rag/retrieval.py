"""E0: BM25, E1: embeddingi, E2: RRF, E3: RRF + cross-encoder.

Nie ładuje żadnych etykiet SciFact. Indeksuje wyłącznie tekst dokumentów.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
import math
from pathlib import Path
import re
from typing import Any


TOKEN_RE = re.compile(r"\b\w+\b", re.UNICODE)
DEFAULT_EMBEDDING = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_RERANKER = "cross-encoder/ms-marco-MiniLM-L-6-v2"
VARIANTS = ("E0", "E1", "E2", "E3")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


@dataclass(frozen=True)
class Result:
    doc_id: int
    score: float
    rank: int


class BM25:
    """Prosty, dokładny BM25 (wariant Robertson, dodatnie IDF)."""

    def __init__(self, texts: list[str], k1: float = 1.5, b: float = 0.75):
        if not texts:
            raise ValueError("Korpus jest pusty")
        self.k1 = k1
        self.b = b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_lengths: list[int] = []
        for i, text in enumerate(texts):
            frequencies = Counter(tokenize(text))
            self.doc_lengths.append(sum(frequencies.values()))
            for token, freq in frequencies.items():
                self.postings[token].append((i, freq))
        self.n = len(texts)
        self.avgdl = max(sum(self.doc_lengths) / self.n, 1.0)
        self.idf = {
            word: math.log(1 + (self.n - len(occurrences) + 0.5) / (len(occurrences) + 0.5))
            for word, occurrences in self.postings.items()
        }

    def scores(self, query: str) -> list[float]:
        scores = [0.0] * self.n
        for term in set(tokenize(query)):
            for i, tf in self.postings.get(term, ()):
                normalizer = tf + self.k1 * (1 - self.b + self.b * self.doc_lengths[i] / self.avgdl)
                scores[i] += self.idf[term] * tf * (self.k1 + 1) / normalizer
        return scores


class Retriever:
    def __init__(
        self,
        documents: list[dict],
        cache_dir: str | Path = "data/cache",
        embedding_model: str = DEFAULT_EMBEDDING,
        reranker_model: str = DEFAULT_RERANKER,
        rrf_k: int = 60,
        rrf_depth: int = 100,
        candidate_k: int = 20,
    ):
        if rrf_k <= 0 or rrf_depth <= 0 or candidate_k < 5:
            raise ValueError("Niepoprawne parametry RRF/rerankingu")
        self.docs = documents
        self.by_id = {d["doc_id"]: d for d in documents}
        self.texts = [d["retrieval_text"] for d in documents]
        self.bm25 = BM25(self.texts)
        self.cache_dir = Path(cache_dir)
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.rrf_k = rrf_k
        self.rrf_depth = rrf_depth
        self.candidate_k = candidate_k
        self._embedding_model: Any = None
        self._doc_vectors: Any = None
        self._reranker: Any = None

    def _dense_model(self):
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def _dense_vectors(self):
        import numpy as np
        if self._doc_vectors is not None:
            return self._doc_vectors
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        hash_source = self.embedding_model_name + "\n" + "\n".join(
            f"{doc['doc_id']}:{doc['retrieval_text']}" for doc in self.docs
        )
        digest = sha256(hash_source.encode("utf-8")).hexdigest()[:20]
        cache_path = self.cache_dir / f"dense_{digest}.npy"
        if cache_path.is_file():
            self._doc_vectors = np.load(cache_path)
        else:
            vectors = self._dense_model().encode(
                self.texts, batch_size=32, show_progress_bar=True,
                normalize_embeddings=True, convert_to_numpy=True,
            )
            self._doc_vectors = np.asarray(vectors, dtype=np.float32)
            np.save(cache_path, self._doc_vectors)
        if len(self._doc_vectors) != len(self.docs):
            raise ValueError("Niezgodny rozmiar pamięci embeddingów")
        return self._doc_vectors

    def _dense_scores(self, query: str):
        import numpy as np
        document_vectors = self._dense_vectors()
        query_vector = self._dense_model().encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )[0]
        return np.dot(document_vectors, query_vector).tolist()

    def _sorted_doc_indices(self, scores: list[float]) -> list[int]:
        return sorted(range(len(scores)), key=lambda i: (-float(scores[i]), self.docs[i]["doc_id"]))

    def _rrf_scores(self, bm_ranking: list[int], dense_ranking: list[int]) -> dict[int, float]:
        fusion = defaultdict(float)
        for ranking in (bm_ranking[:self.rrf_depth], dense_ranking[:self.rrf_depth]):
            for rank, index in enumerate(ranking, start=1):
                fusion[index] += 1.0 / (self.rrf_k + rank)
        return dict(fusion)

    def prepare(self, variant: str, warmup_claim: str | None = None) -> None:
        """Załaduj modele i rozgrzej je PRZED pomiarem latency."""
        if variant in {"E1", "E2", "E3"}:
            self._dense_vectors()
        if variant == "E3" and self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(self.reranker_model_name, max_length=512)
        if warmup_claim:
            self.search(warmup_claim, variant, top_k=5)

    def search(self, claim: str, variant: str = "E0", top_k: int = 5) -> list[Result]:
        if variant not in VARIANTS:
            raise ValueError(f"Nieznany wariant: {variant}")
        if not 1 <= top_k <= len(self.docs):
            raise ValueError("top_k poza zakresem korpusu")
        if variant == "E0":
            scores = self.bm25.scores(claim)
            ranked = self._sorted_doc_indices(scores)[:top_k]
            final_scores = {i: float(scores[i]) for i in ranked}
        elif variant == "E1":
            scores = self._dense_scores(claim)
            ranked = self._sorted_doc_indices(scores)[:top_k]
            final_scores = {i: float(scores[i]) for i in ranked}
        else:
            bm_ranks = self._sorted_doc_indices(self.bm25.scores(claim))
            dense_ranks = self._sorted_doc_indices(self._dense_scores(claim))
            fused = self._rrf_scores(bm_ranks, dense_ranks)
            candidate_idxs = sorted(fused, key=lambda i: (-fused[i], self.docs[i]["doc_id"]))
            if variant == "E2":
                ranked = candidate_idxs[:top_k]
                final_scores = fused
            else:
                if self._reranker is None:
                    from sentence_transformers import CrossEncoder
                    self._reranker = CrossEncoder(self.reranker_model_name, max_length=512)
                candidates = candidate_idxs[:max(top_k, self.candidate_k)]
                pairs = [(claim, self.texts[i]) for i in candidates]
                raw_scores = self._reranker.predict(pairs, batch_size=16, show_progress_bar=False)
                rerank_scores = {i: float(score) for i, score in zip(candidates, raw_scores)}
                ranked = sorted(candidates, key=lambda i: (-rerank_scores[i], self.docs[i]["doc_id"]))[:top_k]
                final_scores = rerank_scores
        return [Result(self.docs[i]["doc_id"], float(final_scores[i]), rank)
                for rank, i in enumerate(ranked, 1)]
