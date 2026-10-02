import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

from main import app
from retrieval_service import RetrievedChunk
from rrf_service import RRFResult


class HybridSearchApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def make_result(self, chunk_id, dense_result=None, bm25_result=None, dense_rank=None, bm25_rank=None):
        source_result = dense_result if dense_result is not None else bm25_result
        return RRFResult(
            chunk_id=chunk_id,
            rrf_score=1 / 61,
            dense_rank=dense_rank,
            bm25_rank=bm25_rank,
            dense_result=dense_result,
            bm25_result=bm25_result,
        )

    def test_valid_request_returns_hybrid_response_and_forwards_options(self):
        dense_result = RetrievedChunk(
            chunk_id="src/auth.py:1-3",
            document="def authenticate_user(): pass",
            file_path="src/auth.py",
            language="Python",
            start_line=1,
            end_line=3,
            repository_name="auth-repo",
            distance=0.1,
        )
        bm25_result = SimpleNamespace(
            chunk_id="src/token.py:1-2",
            code="JWT_SECRET = 'secret'",
            file_path="src/token.py",
            language="Python",
            start_line=1,
            end_line=2,
            repository_name="auth-repo",
        )
        fused_results = [
            self.make_result(
                dense_result.chunk_id,
                dense_result=dense_result,
                bm25_result=SimpleNamespace(**dense_result.__dict__),
                dense_rank=1,
                bm25_rank=2,
            ),
            self.make_result(
                bm25_result.chunk_id,
                bm25_result=bm25_result,
                bm25_rank=1,
            ),
        ]

        with patch("hybrid_api.hybrid_retrieval_service.search", return_value=fused_results) as hybrid_search:
            response = self.client.post(
                "/repository/hybrid-search",
                json={
                    "query": "Where is authentication handled?",
                    "top_k": 2,
                    "repository_name": "auth-repo",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query"], "Where is authentication handled?")
        self.assertEqual(payload["top_k"], 2)
        self.assertEqual(payload["result_count"], 2)
        self.assertEqual(payload["results"][0]["chunk_id"], "src/auth.py:1-3")
        self.assertEqual(payload["results"][0]["document"], "def authenticate_user(): pass")
        self.assertEqual(payload["results"][0]["file_path"], "src/auth.py")
        self.assertEqual(payload["results"][0]["language"], "Python")
        self.assertEqual(payload["results"][0]["start_line"], 1)
        self.assertEqual(payload["results"][0]["end_line"], 3)
        self.assertEqual(payload["results"][0]["repository_name"], "auth-repo")
        self.assertEqual(payload["results"][0]["rrf_score"], 1 / 61)
        self.assertEqual(payload["results"][0]["dense_rank"], 1)
        self.assertEqual(payload["results"][0]["bm25_rank"], 2)
        self.assertEqual(payload["results"][1]["document"], "JWT_SECRET = 'secret'")
        self.assertIsNone(payload["results"][1]["dense_rank"])
        hybrid_search.assert_called_once_with(
            "Where is authentication handled?",
            top_k=2,
            repository_name="auth-repo",
        )

    def test_empty_retrieval_returns_empty_results(self):
        with patch("hybrid_api.hybrid_retrieval_service.search", return_value=[]):
            response = self.client.post(
                "/repository/hybrid-search",
                json={"query": "Unknown feature"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["top_k"], 5)
        self.assertEqual(response.json()["result_count"], 0)
        self.assertEqual(response.json()["results"], [])

    def test_blank_query_and_invalid_top_k_are_rejected(self):
        blank_query_response = self.client.post(
            "/repository/hybrid-search",
            json={"query": "   "},
        )
        zero_top_k_response = self.client.post(
            "/repository/hybrid-search",
            json={"query": "search", "top_k": 0},
        )
        excessive_top_k_response = self.client.post(
            "/repository/hybrid-search",
            json={"query": "search", "top_k": 51},
        )

        self.assertEqual(blank_query_response.status_code, 422)
        self.assertEqual(zero_top_k_response.status_code, 422)
        self.assertEqual(excessive_top_k_response.status_code, 422)

    def test_dense_search_endpoint_remains_the_existing_baseline(self):
        dense_result = RetrievedChunk(
            chunk_id="src/auth.py:1-3",
            document="def authenticate_user(): pass",
            file_path="src/auth.py",
            language="Python",
            start_line=1,
            end_line=3,
            repository_name="auth-repo",
            distance=0.25,
        )

        with patch("retrieval_service.search_code_chunks", return_value=[dense_result]) as dense_search:
            response = self.client.post(
                "/repository/search",
                json={"query": "Where is authentication handled?", "top_k": 1},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["result_count"], 1)
        self.assertEqual(payload["results"][0]["document"], dense_result.document)
        self.assertEqual(payload["results"][0]["distance"], 0.25)
        dense_search.assert_called_once_with(
            "Where is authentication handled?",
            top_k=1,
            repository_name=None,
        )


if __name__ == "__main__":
    unittest.main()