import json
import math
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from scifact_rag.data import gold_document_ids, verified_pairs
from scifact_rag.llm import validated_decisions, build_prediction
from scifact_rag.metrics import retrieval_score, evaluate_evidence, macro_f1_for_verified_pairs
from scifact_rag.retrieval import BM25, Retriever, Result

DOCS = [
    {"doc_id": 10, "title": "Treatment results", "abstract": [
        "Researchers investigated drug treatment.",
        "The treatment decreased disease incidence."],
     "retrieval_text": "Treatment results Researchers investigated drug treatment. The treatment decreased disease incidence."},
    {"doc_id": 11, "title": "A software engineering study", "abstract": [
        "A program improves compilation time."],
     "retrieval_text": "A software engineering study A program improves compilation time."},
    {"doc_id": 12, "title": "A different analysis", "abstract": [
        "Several authors compared antibiotic outcomes."],
     "retrieval_text": "A different analysis Several authors compared antibiotic outcomes."},
    {"doc_id": 13, "title": "Another science paper", "abstract": [
        "Researchers studied new cells."],
     "retrieval_text": "Another science paper Researchers studied new cells."},
    {"doc_id": 14, "title": "Physics experiment", "abstract": [
        "The magnetic field varied with temperature."],
     "retrieval_text": "Physics experiment The magnetic field varied with temperature."},
]
CLAIM = {"id": 1, "claim": "The treatment reduces disease incidence.",
         "evidence": {"10": [{"label": "SUPPORT", "sentences": [1]}]},
         "cited_doc_ids": [10, 11]}


class DataAndRetrievalTests(unittest.TestCase):
    def test_bm25_finds_obvious_doc(self):
        result = Retriever(DOCS).search("treatment decreased disease incidence", "E0", 5)
        self.assertEqual(result[0].doc_id, 10)

    def test_dense_hybrid_rerank_controlled(self):
        import numpy as np
        ranker = Retriever(DOCS, candidate_k=5)
        # Deterministyczny ranking semantyczny bez pobierania wag modeli.
        with patch.object(ranker, "_dense_scores", return_value=[0.9, 0.5, 0.4, 0.2, 0.1]):
            self.assertEqual(ranker.search("treatment decreased incidence", "E1")[0].doc_id, 10)
            self.assertEqual(ranker.search("treatment decreased incidence", "E2")[0].doc_id, 10)
            ranker._reranker = SimpleNamespace(predict=lambda pairs, **kwargs:
                np.array([0.9 if "Treatment results" in d else 0.0 for _, d in pairs]))
            self.assertEqual(ranker.search("treatment decreased incidence", "E3")[0].doc_id, 10)

    def test_gold_only_evidence(self):
        self.assertEqual(gold_document_ids(CLAIM), {10})
        self.assertEqual(verified_pairs(CLAIM), {10: "SUPPORT", 11: "NOINFO"})

    def test_retrieval_metrics(self):
        result = retrieval_score(CLAIM, [11, 10, 13, 12, 14], 5)
        self.assertEqual(result["recall_at_k"], 1.0)
        self.assertEqual(result["mrr_at_k"], 0.5)
        self.assertAlmostEqual(result["ndcg_at_k"], 1.0 / math.log2(3))
        self.assertIsNone(retrieval_score({"evidence": {}}, [10, 11]))


class LLMTests(unittest.TestCase):
    def test_validates_sentences_and_labels(self):
        valid = json.dumps({"decisions": [
            {"doc_id": 10, "label": "SUPPORT", "sentence_ids": [1], "explanation": "Lower incidence."},
            {"doc_id": 11, "label": "NOINFO", "sentence_ids": [], "explanation": "Unrelated."},
        ]})
        self.assertEqual(len(validated_decisions(valid, DOCS[:2])), 2)
        with self.assertRaises(ValueError):
            validated_decisions(valid.replace('"sentence_ids": [1]', '"sentence_ids": [55]'), DOCS[:2])

    def test_end_to_end_with_mock_llm(self):
        answer = {"decisions": [{"doc_id": 10, "label": "SUPPORT", "sentence_ids": [1],
                                 "explanation": "Decreased incidence is evidence."}]}
        with patch("scifact_rag.llm.ask_llm", return_value=json.dumps(answer)):
            output = build_prediction(CLAIM, [Result(10, 3.0, 1)], {10: DOCS[0]},
                                      "ollama", "mock")
        self.assertEqual(output["verdict"], "SUPPORT")
        self.assertEqual(output["sources"][0]["sentences"][0]["id"], 1)
        scores = evaluate_evidence([CLAIM], [output])
        self.assertEqual(scores["abstract_label_rationale"]["f1"], 1.0)
        pairs = macro_f1_for_verified_pairs([CLAIM], [output])
        self.assertEqual(pairs["pair_count"], 2)
        self.assertEqual(pairs["macro_f1"], 2 / 3)  # brak klasy CONTRADICT w próbce

    def test_multi_sentence_gold_rationale_must_be_complete(self):
        claim = {"id": 2, "evidence": {"10": [{"label": "SUPPORT", "sentences": [0, 1]}]}}
        result = {"id": 2, "evidence": {"10": {"label": "SUPPORT", "sentences": [1]}}}
        scores = evaluate_evidence([claim], [result])
        self.assertEqual(scores["abstract_label"]["f1"], 1)
        self.assertEqual(scores["abstract_label_rationale"]["f1"], 0)


if __name__ == "__main__":
    unittest.main()
