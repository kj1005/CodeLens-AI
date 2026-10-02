import unittest
from dataclasses import dataclass
from unittest.mock import Mock

from bm25_service import BM25IndexRegistry
from hybrid_retrieval_service import HybridRetrievalService
from rrf_service import RRFResult, reciprocal_rank_fusion


@dataclass
class FakeResult:
    chunk_id: str
    document: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    repository_name: str


def make_result(chunk_id, repository_name="repo-a", document=None):
    return FakeResult(
        chunk_id=chunk_id,
        document=document or f"code for {chunk_id}",
        file_path=f"src/{chunk_id}.py",
        language="Python",
        start_line=3,
        end_line=8,
        repository_name=repository_name,
    )


class FakeBM25Index:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def search(self, query, top_k):
        self.calls.append((query, top_k))
        return self.results[:top_k]


class FakeBM25Registry:
    def __init__(self, indexes):
        self.indexes = indexes

    @property
    def repository_names(self):
        return tuple(sorted(self.indexes))

    def get_repository_index(self, repository_name):
        return self.indexes.get(repository_name)


class HybridRetrievalServiceTests(unittest.TestCase):
    def setUp(self):
        self.dense_results = [make_result("shared"), make_result("dense-only")]
        self.bm25_results = [make_result("shared"), make_result("bm25-only")]
        self.dense_retriever = Mock(return_value=self.dense_results)
        self.bm25_index = FakeBM25Index(self.bm25_results)
        self.bm25_registry = FakeBM25Registry({"repo-a": self.bm25_index})
        self.fusion = Mock(side_effect=reciprocal_rank_fusion)
        self.service = HybridRetrievalService(
            dense_retriever=self.dense_retriever,
            bm25_registry=self.bm25_registry,
            fusion_function=self.fusion,
        )

    def test_dense_and_bm25_are_requested_and_both_lists_go_to_rrf(self):
        results = self.service.search("find shared", top_k=4, repository_name="repo-a")

        self.dense_retriever.assert_called_once_with(
            "find shared",
            top_k=4,
            repository_name="repo-a",
        )
        self.assertEqual(self.bm25_index.calls, [("find shared", 4)])
        self.fusion.assert_called_once()
        dense_input, bm25_input = self.fusion.call_args.args
        self.assertIs(dense_input[0].original_result, self.dense_results[0])
        self.assertIs(bm25_input[0].original_result, self.bm25_results[0])
        self.assertEqual(self.fusion.call_args.kwargs, {"top_k": 4, "k": 60})
        self.assertIsInstance(results[0], RRFResult)

    def test_shared_and_single_source_results_are_preserved(self):
        results = self.service.search("find shared", top_k=5, repository_name="repo-a")
        result_by_id = {result.chunk_id: result for result in results}

        self.assertEqual(result_by_id["shared"].dense_rank, 1)
        self.assertEqual(result_by_id["shared"].bm25_rank, 1)
        self.assertAlmostEqual(result_by_id["shared"].rrf_score, 2 / 61)
        self.assertIs(result_by_id["shared"].dense_result, self.dense_results[0])
        self.assertIs(result_by_id["shared"].bm25_result, self.bm25_results[0])
        self.assertIs(result_by_id["dense-only"].dense_result, self.dense_results[1])
        self.assertIsNone(result_by_id["dense-only"].bm25_result)
        self.assertIsNone(result_by_id["bm25-only"].dense_result)
        self.assertIs(result_by_id["bm25-only"].bm25_result, self.bm25_results[1])

    def test_top_k_limits_final_results(self):
        results = self.service.search("find shared", top_k=2, repository_name="repo-a")

        self.assertEqual(len(results), 2)

    def test_repository_filter_selects_only_requested_bm25_index(self):
        repo_a_index = FakeBM25Index([make_result("a-result", "repo-a")])
        repo_b_index = FakeBM25Index([make_result("b-result", "repo-b")])
        registry = FakeBM25Registry({"repo-a": repo_a_index, "repo-b": repo_b_index})
        dense_retriever = Mock(return_value=[])
        service = HybridRetrievalService(
            dense_retriever=dense_retriever,
            bm25_registry=registry,
        )

        results = service.search("query", repository_name="repo-a")

        self.assertEqual(repo_a_index.calls, [("query", 5)])
        self.assertEqual(repo_b_index.calls, [])
        self.assertEqual([result.chunk_id for result in results], ["a-result"])
        self.assertEqual(results[0].result.repository_name, "repo-a")
        dense_retriever.assert_called_once_with(
            "query",
            top_k=5,
            repository_name="repo-a",
        )

    def test_unfiltered_bm25_results_merge_repository_lists_by_local_rank(self):
        repo_a_index = FakeBM25Index(
            [make_result("a-rank-one", "repo-a"), make_result("a-rank-two", "repo-a")]
        )
        repo_b_index = FakeBM25Index(
            [make_result("b-rank-one", "repo-b"), make_result("b-rank-two", "repo-b")]
        )
        registry = FakeBM25Registry({"repo-b": repo_b_index, "repo-a": repo_a_index})
        fusion = Mock(side_effect=reciprocal_rank_fusion)
        service = HybridRetrievalService(
            dense_retriever=Mock(return_value=[]),
            bm25_registry=registry,
            fusion_function=fusion,
        )

        service.search("query", top_k=2)
        bm25_input = fusion.call_args.args[1]

        self.assertEqual(
            [result.original_result.chunk_id for result in bm25_input],
            ["a-rank-one", "b-rank-one", "a-rank-two", "b-rank-two"],
        )
        self.assertEqual(repo_a_index.calls, [("query", 2)])
        self.assertEqual(repo_b_index.calls, [("query", 2)])

    def test_unfiltered_repositories_with_same_chunk_id_remain_distinct(self):
        repo_a_result = make_result("src/main.py:1-2", "repo-a", "JWT_SECRET = 1")
        repo_b_result = make_result("src/main.py:1-2", "repo-b", "DATABASE_URL = 1")
        registry = FakeBM25Registry(
            {
                "repo-a": FakeBM25Index([repo_a_result]),
                "repo-b": FakeBM25Index([repo_b_result]),
            }
        )
        service = HybridRetrievalService(
            dense_retriever=Mock(return_value=[]),
            bm25_registry=registry,
        )

        results = service.search("query", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(
            {result.result.repository_name for result in results},
            {"repo-a", "repo-b"},
        )
        self.assertEqual({result.chunk_id for result in results}, {"src/main.py:1-2"})

    def test_metadata_is_preserved_from_original_result(self):
        dense_result = make_result(
            "metadata-chunk",
            document="def load_user(): pass",
        )
        dense_retriever = Mock(return_value=[dense_result])
        service = HybridRetrievalService(
            dense_retriever=dense_retriever,
            bm25_registry=FakeBM25Registry({}),
        )

        fused = service.search("load user", top_k=1)[0]

        self.assertIs(fused.result, dense_result)
        self.assertEqual(fused.chunk_id, "metadata-chunk")
        self.assertEqual(fused.result.document, "def load_user(): pass")
        self.assertEqual(fused.result.file_path, dense_result.file_path)
        self.assertEqual(fused.result.language, "Python")
        self.assertEqual(fused.result.start_line, 3)
        self.assertEqual(fused.result.end_line, 8)
        self.assertEqual(fused.result.repository_name, "repo-a")

    def test_empty_dense_bm25_or_both_sources(self):
        dense_empty_service = HybridRetrievalService(
            dense_retriever=Mock(return_value=[]),
            bm25_registry=FakeBM25Registry({
                "repo-a": FakeBM25Index([make_result("bm25-only")]),
            }),
        )
        bm25_empty_service = HybridRetrievalService(
            dense_retriever=Mock(return_value=[make_result("dense-only")]),
            bm25_registry=FakeBM25Registry({}),
        )
        both_empty_service = HybridRetrievalService(
            dense_retriever=Mock(return_value=[]),
            bm25_registry=FakeBM25Registry({}),
        )

        self.assertEqual(
            [result.chunk_id for result in dense_empty_service.search("query", repository_name="repo-a")],
            ["bm25-only"],
        )
        self.assertEqual(
            [result.chunk_id for result in bm25_empty_service.search("query")],
            ["dense-only"],
        )
        self.assertEqual(both_empty_service.search("query"), [])


if __name__ == "__main__":
    unittest.main()