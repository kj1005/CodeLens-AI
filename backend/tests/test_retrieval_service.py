import unittest
import gc
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import ValidationError

from chroma_service import create_persistent_client, get_code_collection, upsert_embedded_chunks
from embedding_service import MODEL_NAME, EmbeddedCodeChunk
from retrieval_service import (
    DEFAULT_TOP_K,
    EMBEDDING_DIMENSION,
    MAX_TOP_K,
    QUERY_INSTRUCTION,
    RepositorySearchRequest,
    search_code_chunks,
    search_repository,
)


class FakeQueryModel:
    def __init__(self, vector):
        self.vector = vector
        self.inputs = None

    def encode(self, inputs, **_kwargs):
        self.inputs = inputs
        return [self.vector]


def make_embedded_chunk(chunk_id, code, vector, repository_name="sample-repository"):
    return EmbeddedCodeChunk(
        chunk_id=chunk_id,
        file_path=f"src/{chunk_id.split(':')[0]}.py",
        language="Python",
        start_line=1,
        end_line=2,
        code=code,
        embedding=vector,
    )


class RepositorySearchTests(unittest.TestCase):
    def setUp(self):
        self.database_directory = TemporaryDirectory()
        self.client = create_persistent_client(self.database_directory.name)
        self.collection = get_code_collection(self.client)

    def tearDown(self):
        self.collection = None
        if hasattr(self.client, "_system"):
            self.client._system.stop()
        self.client = None
        gc.collect()
        self.database_directory.cleanup()

    def vector(self, first_value, second_value=0.0):
        return [first_value, second_value] + [0.0] * (EMBEDDING_DIMENSION - 2)

    def test_query_embedding_uses_the_shared_bge_model(self):
        chunk = make_embedded_chunk(
            "load_user.py:1-2",
            "def load_user(): return user",
            self.vector(1.0),
        )
        upsert_embedded_chunks(self.collection, "sample-repository", [chunk])
        fake_model = FakeQueryModel(self.vector(1.0))
        request = RepositorySearchRequest(query="Where is the user loaded?")

        with (
            patch("retrieval_service.get_code_collection", return_value=self.collection),
            patch("retrieval_service.get_embedding_model", return_value=fake_model),
        ):
            response = search_repository(request)

        self.assertEqual(MODEL_NAME, "BAAI/bge-small-en-v1.5")
        self.assertEqual(fake_model.inputs, [f"{QUERY_INSTRUCTION}{request.query}"])
        self.assertEqual(response.result_count, 1)

    def test_top_k_returns_nearest_source_chunks_and_native_distances(self):
        chunks = [
            make_embedded_chunk(
                "load_user.py:1-2",
                "def load_user(): return user",
                self.vector(0.0),
            ),
            make_embedded_chunk(
                "load_config.py:1-2",
                "def load_config(): return config",
                self.vector(0.1),
            ),
            make_embedded_chunk(
                "sort_values.py:1-2",
                "def sort_values(values): return sorted(values)",
                self.vector(4.0),
            ),
        ]
        upsert_embedded_chunks(self.collection, "sample-repository", chunks)
        query_model = FakeQueryModel(self.vector(0.0))

        with (
            patch("retrieval_service.get_code_collection", return_value=self.collection),
            patch("retrieval_service.get_embedding_model", return_value=query_model),
        ):
            results = search_code_chunks("Where is the user loaded?", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual(
            [result.chunk_id for result in results],
            ["load_user.py:1-2", "load_config.py:1-2"],
        )
        self.assertEqual(results[0].document, "def load_user(): return user")
        self.assertIsInstance(results[0].distance, float)
        self.assertLess(results[0].distance, results[1].distance)

    def test_semantically_relevant_query_retrieves_expected_chunk(self):
        relevant_chunk = make_embedded_chunk(
            "user_repository.py:1-2",
            "def load_user(user_id): return repository.get(user_id)",
            self.vector(0.0),
        )
        unrelated_chunk = make_embedded_chunk(
            "number_sort.py:1-2",
            "def sort_numbers(values): return sorted(values)",
            self.vector(3.0),
        )
        upsert_embedded_chunks(
            self.collection,
            "sample-repository",
            [relevant_chunk, unrelated_chunk],
        )

        with patch(
            "retrieval_service.get_code_collection",
            return_value=self.collection,
        ):
            with patch(
                "retrieval_service.get_embedding_model",
                return_value=FakeQueryModel(self.vector(0.0)),
            ):
                response = search_repository(
                    RepositorySearchRequest(query="How are user records loaded?")
                )

        self.assertGreaterEqual(response.result_count, 1)
        self.assertEqual(response.results[0].chunk_id, relevant_chunk.chunk_id)
        self.assertEqual(response.results[0].document, relevant_chunk.code)

    def test_result_metadata_and_repository_filter_are_preserved(self):
        target_chunk = make_embedded_chunk(
            "get_user.py:1-2", "def get_user(): pass", self.vector(0.0)
        )
        other_repository_chunk = make_embedded_chunk(
            "other_get_user.py:1-2",
            "def get_user(): return None",
            self.vector(0.1),
        )
        upsert_embedded_chunks(self.collection, "sample-repository", [target_chunk])
        upsert_embedded_chunks(self.collection, "other-repository", [other_repository_chunk])

        with (
            patch("retrieval_service.get_code_collection", return_value=self.collection),
            patch(
                "retrieval_service.get_embedding_model",
                return_value=FakeQueryModel(self.vector(0.0)),
            ),
        ):
            response = search_repository(
                RepositorySearchRequest(
                    query="Find the user lookup",
                    repository_name="sample-repository",
                )
            )

        self.assertEqual(response.result_count, 1)
        result = response.results[0]
        self.assertEqual(result.chunk_id, target_chunk.chunk_id)
        self.assertEqual(result.file_path, target_chunk.file_path)
        self.assertEqual(result.language, target_chunk.language)
        self.assertEqual(result.start_line, target_chunk.start_line)
        self.assertEqual(result.end_line, target_chunk.end_line)
        self.assertEqual(result.repository_name, "sample-repository")
        self.assertIsInstance(result.distance, float)

    def test_empty_collection_returns_no_results_without_loading_model(self):
        request = RepositorySearchRequest(query="Where is authentication handled?")

        with (
            patch("retrieval_service.get_code_collection", return_value=self.collection),
            patch("retrieval_service.get_embedding_model") as get_model,
        ):
            response = search_repository(request)

        self.assertEqual(response.result_count, 0)
        self.assertEqual(response.results, [])
        get_model.assert_not_called()

    def test_invalid_top_k_is_rejected(self):
        with self.assertRaises(ValidationError):
            RepositorySearchRequest(query="Find a function", top_k=0)
        with self.assertRaises(ValidationError):
            RepositorySearchRequest(query="Find a function", top_k=MAX_TOP_K + 1)

    def test_default_top_k_is_five(self):
        request = RepositorySearchRequest(query="Find a function")

        self.assertEqual(request.top_k, 5)
        self.assertEqual(DEFAULT_TOP_K, 5)


if __name__ == "__main__":
    unittest.main()