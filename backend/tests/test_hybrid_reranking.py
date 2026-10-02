import unittest
from unittest.mock import Mock, patch

from hybrid_retrieval_service import HybridRetrievalService
from reranking_service import RerankedResult
from rrf_service import RRFResult


def make_rrf_result(chunk_id, rrf_score=0.01, repository_name="repo-a"):
    source_result = {
        "chunk_id": chunk_id,
        "document": f"code for {chunk_id}",
        "file_path": f"src/{chunk_id}.py",
        "language": "Python",
        "start_line": 1,
        "end_line": 3,
        "repository_name": repository_name,
    }
    return RRFResult(
        chunk_id=chunk_id,
        rrf_score=rrf_score,
        dense_rank=1,
        bm25_rank=2,
        dense_result=source_result,
        bm25_result=source_result,
    )


def make_reranked_result(candidate, rerank_score):
    source = candidate.result
    return RerankedResult(
        chunk_id=candidate.chunk_id,
        code=source["document"],
        document=source["document"],
        file_path=source["file_path"],
        language=source["language"],
        start_line=source["start_line"],
        end_line=source["end_line"],
        repository_name=source["repository_name"],
        rrf_score=candidate.rrf_score,
        dense_rank=candidate.dense_rank,
        bm25_rank=candidate.bm25_rank,
        rerank_score=rerank_score,
        original_result=candidate,
    )


class FakeBM25Index:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeBM25Registry:
    repository_names = ("repo-a",)

    def __init__(self, index):
        self.index = index

    def get_repository_index(self, repository_name):
        return self.index if repository_name == "repo-a" else None


class FakeReranker:
    def __init__(self, results=None):
        self.results = results
        self.calls = []

    def rerank(self, query, candidates, top_k):
        self.calls.append((query, candidates, top_k))
        if self.results is not None:
            return sorted(self.results, key=lambda result: result.rerank_score, reverse=True)[:top_k]
        return [make_reranked_result(candidate, float(index)) for index, candidate in enumerate(candidates[:top_k])]


class HybridRerankingTests(unittest.TestCase):
    def setUp(self):
        self.dense_results = [make_rrf_result(f"chunk-{index}") for index in range(20)]
        self.bm25_results = [make_rrf_result(f"chunk-{index}") for index in range(20, 40)]
        self.candidates = [make_rrf_result(f"candidate-{index}") for index in range(20)]
        self.dense_retriever = Mock(return_value=self.dense_results)
        self.bm25_index = FakeBM25Index(self.bm25_results)
        self.fusion = Mock(side_effect=lambda _dense, _bm25, top_k, **_kwargs: self.candidates[:top_k])
        self.reranker = FakeReranker()
        self.service = HybridRetrievalService(
            dense_retriever=self.dense_retriever,
            bm25_registry=FakeBM25Registry(self.bm25_index),
            fusion_function=self.fusion,
            reranker=self.reranker,
            reranking_enabled=True,
            candidate_k=20,
        )

    def test_rrf_candidate_k_is_expanded_before_final_top_k_rerank(self):
        results = self.service.search("user authentication", top_k=5)

        self.dense_retriever.assert_called_once_with(
            "user authentication",
            top_k=20,
            repository_name=None,
        )
        self.assertEqual(self.bm25_index.calls, [("user authentication", 20)])
        self.assertEqual(self.fusion.call_args.kwargs["top_k"], 20)
        self.assertEqual(len(self.reranker.calls[0][1]), 20)
        self.assertEqual(self.reranker.calls[0][2], 5)
        self.assertEqual(len(results), 5)

    def test_rrf_candidates_are_passed_to_reranker_after_fusion(self):
        results = self.service.search("query", top_k=3)

        query, candidates, top_k = self.reranker.calls[0]
        self.assertEqual(query, "query")
        self.assertEqual(candidates[0], self.candidates[0])
        self.assertEqual(top_k, 3)
        self.assertEqual([result.chunk_id for result in results], ["candidate-0", "candidate-1", "candidate-2"])

    def test_final_results_follow_rerank_score_and_preserve_rrf_score(self):
        candidate_a = make_rrf_result("candidate-a", rrf_score=0.031)
        candidate_b = make_rrf_result("candidate-b", rrf_score=0.028)
        reranked = [
            make_reranked_result(candidate_a, 0.1),
            make_reranked_result(candidate_b, 2.75),
        ]
        reranker = FakeReranker(reranked)
        service = HybridRetrievalService(
            dense_retriever=Mock(return_value=[]),
            bm25_registry=FakeBM25Registry(FakeBM25Index([])),
            fusion_function=Mock(return_value=[candidate_a, candidate_b]),
            reranker=reranker,
            reranking_enabled=True,
        )

        results = service.search("query", top_k=2)

        self.assertEqual([result.chunk_id for result in results], ["candidate-b", "candidate-a"])
        self.assertEqual([result.rerank_score for result in results], [2.75, 0.1])
        self.assertEqual([result.rrf_score for result in results], [0.028, 0.031])

    def test_disabling_reranking_returns_rrf_candidates_without_constructing_model(self):
        with patch("hybrid_retrieval_service.RerankingService") as reranking_service_class:
            service = HybridRetrievalService(
                dense_retriever=Mock(return_value=[]),
                bm25_registry=FakeBM25Registry(FakeBM25Index([])),
                fusion_function=Mock(return_value=self.candidates[:3]),
                reranking_enabled=False,
            )

            results = service.search("query", top_k=3)

        reranking_service_class.assert_not_called()
        self.assertEqual(results, self.candidates[:3])
        self.assertFalse(hasattr(results[0], "rerank_score"))

    def test_empty_candidates_do_not_construct_or_call_reranker(self):
        with patch("hybrid_retrieval_service.RerankingService") as reranking_service_class:
            service = HybridRetrievalService(
                dense_retriever=Mock(return_value=[]),
                bm25_registry=FakeBM25Registry(FakeBM25Index([])),
                fusion_function=Mock(return_value=[]),
                reranking_enabled=True,
            )

            results = service.search("query", top_k=5)

        reranking_service_class.assert_not_called()
        self.assertEqual(results, [])

    def test_repository_filter_is_forwarded_and_kept_on_reranked_results(self):
        results = self.service.search("query", top_k=2, repository_name="repo-a")

        self.dense_retriever.assert_called_once_with(
            "query",
            top_k=20,
            repository_name="repo-a",
        )
        self.assertEqual(self.bm25_index.calls, [("query", 20)])
        self.assertTrue(all(result.original_result.result["repository_name"] == "repo-a" for result in results))


if __name__ == "__main__":
    unittest.main()