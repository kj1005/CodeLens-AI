import unittest
from unittest.mock import patch

from reranking_service import (
    DEFAULT_RERANKER_DEVICE,
    RERANKER_MODEL_NAME,
    RerankingService,
)
from rrf_service import RRFResult


class FakeCrossEncoder:
    def __init__(self, scores):
        self.scores = scores
        self.pairs = []

    def predict(self, pairs):
        self.pairs.append(pairs)
        return self.scores[: len(pairs)]


def make_candidate(
    chunk_id,
    code,
    *,
    rrf_score=None,
    dense_rank=None,
    bm25_rank=None,
    repository_name="sample-repo",
    file_path=None,
):
    document = {
        "chunk_id": chunk_id,
        "document": code,
        "file_path": file_path or f"src/{chunk_id}.py",
        "language": "Python",
        "start_line": 2,
        "end_line": 8,
        "repository_name": repository_name,
    }
    if rrf_score is None and dense_rank is None and bm25_rank is None:
        return document
    return RRFResult(
        chunk_id=chunk_id,
        rrf_score=rrf_score or 0.0,
        dense_rank=dense_rank,
        bm25_rank=bm25_rank,
            dense_result=document if dense_rank is not None else None,
        bm25_result=document if bm25_rank is not None else None,
    )


class RerankingServiceTests(unittest.TestCase):
    def test_model_loads_lazily_once_per_service_instance(self):
        model = FakeCrossEncoder([0.25, 0.5])
        with patch("reranking_service.CrossEncoder", return_value=model) as cross_encoder:
            service = RerankingService()

            self.assertEqual(cross_encoder.call_count, 0)
            service.rerank(
                "find users",
                [make_candidate("one", "get_user_by_id"), make_candidate("two", "authenticate_user")],
                top_k=2,
            )
            service.rerank("find users", [make_candidate("one", "get_user_by_id")], top_k=1)

        cross_encoder.assert_called_once_with(
            RERANKER_MODEL_NAME,
            device=DEFAULT_RERANKER_DEVICE,
        )
        self.assertEqual(len(model.pairs), 2)

    def test_model_scores_query_document_pairs(self):
        model = FakeCrossEncoder([0.75])
        service = RerankingService()
        service._model = model
        candidate = make_candidate("auth", "def authenticate_user(token): pass")

        service.rerank("where is authentication handled", [candidate], top_k=1)

        self.assertEqual(
            model.pairs,
            [[("where is authentication handled", "def authenticate_user(token): pass")]],
        )

    def test_candidates_are_sorted_by_raw_score_descending(self):
        model = FakeCrossEncoder([0.1, 4.25, -0.3])
        service = RerankingService()
        service._model = model
        candidates = [
            make_candidate("low", "low score"),
            make_candidate("high", "high score"),
            make_candidate("negative", "negative score"),
        ]

        results = service.rerank("query", candidates, top_k=3)

        self.assertEqual([result.chunk_id for result in results], ["high", "low", "negative"])
        self.assertEqual([result.rerank_score for result in results], [4.25, 0.1, -0.3])

    def test_top_k_limits_results_and_larger_than_candidates_returns_all(self):
        model = FakeCrossEncoder([0.1, 0.9])
        service = RerankingService()
        service._model = model
        candidates = [make_candidate("one", "one"), make_candidate("two", "two")]

        self.assertEqual(len(service.rerank("query", candidates, top_k=1)), 1)
        self.assertEqual(len(service.rerank("query", candidates, top_k=5)), 2)

    def test_empty_candidates_return_empty_without_loading_model(self):
        service = RerankingService()
        with patch("reranking_service.CrossEncoder") as cross_encoder:
            self.assertEqual(service.rerank("query", [], top_k=5), [])

        cross_encoder.assert_not_called()

    def test_non_positive_top_k_is_rejected(self):
        service = RerankingService()

        with self.assertRaises(ValueError):
            service.rerank("query", [], top_k=0)
        with self.assertRaises(ValueError):
            service.rerank("query", [], top_k=-1)

    def test_candidate_metadata_and_retrieval_scores_are_preserved(self):
        model = FakeCrossEncoder([1.75])
        service = RerankingService()
        service._model = model
        candidate = make_candidate(
            "src/auth.py:2-8",
            "def authenticate_user(token): pass",
            file_path="src/auth.py",
            rrf_score=0.041,
            dense_rank=2,
            bm25_rank=1,
        )

        result = service.rerank("authentication", [candidate], top_k=1)[0]

        self.assertIs(result.original_result, candidate)
        self.assertEqual(result.chunk_id, candidate.chunk_id)
        self.assertEqual(result.code, candidate.result["document"])
        self.assertEqual(result.document, candidate.result["document"])
        self.assertEqual(result.file_path, "src/auth.py")
        self.assertEqual(result.language, "Python")
        self.assertEqual(result.start_line, 2)
        self.assertEqual(result.end_line, 8)
        self.assertEqual(result.repository_name, "sample-repo")
        self.assertEqual(result.rrf_score, 0.041)
        self.assertEqual(result.dense_rank, 2)
        self.assertEqual(result.bm25_rank, 1)
        self.assertEqual(result.rerank_score, 1.75)

    def test_single_candidate_is_returned_with_its_raw_score(self):
        service = RerankingService()
        service._model = FakeCrossEncoder([-2.5])

        results = service.rerank(
            "query",
            [make_candidate("only", "one document")],
            top_k=5,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].rerank_score, -2.5)


if __name__ == "__main__":
    unittest.main()