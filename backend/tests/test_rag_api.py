import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import rag_api
from gemini_service import ExplanationLevel
from gemini_service import EMPTY_CONTEXT_ANSWER, GeminiGenerationError
from main import app
from reranking_service import RerankedResult
from retrieval_service import RetrievedChunk
from rrf_service import RRFResult


class RAGApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def make_reranked_result(self):
        source = RetrievedChunk(
            chunk_id="src/auth.py:10-12",
            document="def authenticate_user(token): return verify(token)",
            file_path="src/auth.py",
            language="Python",
            start_line=10,
            end_line=12,
            repository_name="sample-repo",
            distance=0.25,
        )
        candidate = RRFResult(
            chunk_id=source.chunk_id,
            rrf_score=0.03,
            dense_rank=1,
            bm25_rank=2,
            dense_result=source,
            bm25_result=source,
        )
        return RerankedResult(
            chunk_id=source.chunk_id,
            code=source.document,
            document=source.document,
            file_path=source.file_path,
            language=source.language,
            start_line=source.start_line,
            end_line=source.end_line,
            repository_name=source.repository_name,
            rrf_score=0.03,
            dense_rank=1,
            bm25_rank=2,
            rerank_score=1.75,
            original_result=candidate,
        )

    def test_successful_rag_request_returns_answer_and_source_metadata(self):
        result = self.make_reranked_result()
        question = "Where is authentication checked?"

        with (
            patch("rag_api.hybrid_retrieval_service.search", return_value=[result]) as hybrid_search,
            patch.object(
                rag_api.gemini_generation_service,
                "generate_answer",
                return_value="Authentication is checked in the auth module.",
            ) as generate_answer,
        ):
            response = self.client.post(
                "/repository/ask",
                json={"query": question, "top_k": 4, "repository_name": "sample-repo"},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["query"], question)
        self.assertEqual(payload["explanation_level"], "intermediate")
        self.assertEqual(payload["answer"], "Authentication is checked in the auth module.")
        self.assertEqual(payload["result_count"], 1)
        self.assertEqual(payload["results"][0]["chunk_id"], result.chunk_id)
        self.assertEqual(payload["results"][0]["document"], result.document)
        self.assertEqual(payload["results"][0]["file_path"], "src/auth.py")
        self.assertEqual(payload["results"][0]["start_line"], 10)
        self.assertEqual(payload["results"][0]["end_line"], 12)
        self.assertEqual(payload["results"][0]["language"], "Python")
        self.assertEqual(payload["results"][0]["rrf_score"], 0.03)
        self.assertEqual(payload["results"][0]["rerank_score"], 1.75)
        hybrid_search.assert_called_once_with(
            question,
            top_k=4,
            repository_name="sample-repo",
        )
        generate_answer.assert_called_once_with(
            question,
            generate_answer.call_args.args[1],
            ExplanationLevel.INTERMEDIATE,
        )
        self.assertIn("File: src/auth.py | Lines: 10-12 | Language: Python", generate_answer.call_args.args[1])

    def test_missing_and_empty_query_are_rejected(self):
        missing_query = self.client.post("/repository/ask", json={})
        empty_query = self.client.post("/repository/ask", json={"query": "   "})

        self.assertEqual(missing_query.status_code, 422)
        self.assertEqual(empty_query.status_code, 422)

    def test_repository_filter_is_forwarded(self):
        with (
            patch("rag_api.hybrid_retrieval_service.search", return_value=[]) as hybrid_search,
            patch.object(
                rag_api.gemini_generation_service,
                "generate_answer",
                return_value=EMPTY_CONTEXT_ANSWER,
            ),
        ):
            response = self.client.post(
                "/repository/ask",
                json={"query": "Question", "repository_name": "only-this-repo"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["explanation_level"], "intermediate")
        hybrid_search.assert_called_once_with(
            "Question",
            top_k=5,
            repository_name="only-this-repo",
        )

    def test_no_retrieved_results_returns_insufficient_context_answer(self):
        with patch("rag_api.hybrid_retrieval_service.search", return_value=[]):
            response = self.client.post(
                "/repository/ask",
                json={"query": "Question with no matches"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], EMPTY_CONTEXT_ANSWER)
        self.assertEqual(response.json()["result_count"], 0)
        self.assertEqual(response.json()["results"], [])

    def test_gemini_service_error_returns_gateway_error(self):
        with (
            patch("rag_api.hybrid_retrieval_service.search", return_value=[self.make_reranked_result()]),
            patch.object(
                rag_api.gemini_generation_service,
                "generate_answer",
                side_effect=GeminiGenerationError("provider failure"),
            ),
        ):
            response = self.client.post(
                "/repository/ask",
                json={"query": "Question"},
            )

        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "Gemini answer generation failed.")

    def test_each_explanation_level_is_forwarded(self):
        for level in ExplanationLevel:
            with self.subTest(level=level.value):
                with (
                    patch("rag_api.hybrid_retrieval_service.search", return_value=[]),
                    patch.object(
                        rag_api.gemini_generation_service,
                        "generate_answer",
                        return_value="answer",
                    ) as generate_answer,
                ):
                    response = self.client.post(
                        "/repository/ask",
                        json={"query": "Question", "explanation_level": level.value},
                    )

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["explanation_level"], level.value)
                self.assertEqual(generate_answer.call_args.args[2], level)

    def test_invalid_explanation_level_is_rejected(self):
        with patch.object(rag_api.gemini_generation_service, "generate_answer") as generate_answer:
            response = self.client.post(
                "/repository/ask",
                json={"query": "Question", "explanation_level": "advanced"},
            )

        self.assertEqual(response.status_code, 422)
        generate_answer.assert_not_called()

    def test_existing_dense_and_hybrid_endpoints_still_work(self):
        dense_result = RetrievedChunk(
            chunk_id="src/auth.py:10-12",
            document="def authenticate_user(token): return verify(token)",
            file_path="src/auth.py",
            language="Python",
            start_line=10,
            end_line=12,
            repository_name="sample-repo",
            distance=0.25,
        )
        hybrid_source = self.make_reranked_result().original_result

        with (
            patch("retrieval_service.search_code_chunks", return_value=[dense_result]),
            patch("hybrid_api.hybrid_retrieval_service.search", return_value=[hybrid_source]),
        ):
            dense_response = self.client.post("/repository/search", json={"query": "auth"})
            hybrid_response = self.client.post("/repository/hybrid-search", json={"query": "auth"})

        self.assertEqual(dense_response.status_code, 200)
        self.assertEqual(dense_response.json()["results"][0]["distance"], 0.25)
        self.assertEqual(hybrid_response.status_code, 200)
        self.assertEqual(hybrid_response.json()["results"][0]["rrf_score"], 0.03)


if __name__ == "__main__":
    unittest.main()