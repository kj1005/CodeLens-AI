import unittest
from unittest.mock import patch

from fastapi import HTTPException
from pydantic import ValidationError

from code_chunking import CodeChunk
from embedding_service import (
    MODEL_NAME,
    RepositoryEmbeddingRequest,
    embed_repository,
    generate_embeddings,
    get_embedding_model,
)
from repository_ingestion import RepositoryFile


class FakeEmbeddingModel:
    def __init__(self, dimension=384):
        self.dimension = dimension
        self.encoded_texts = None

    def encode(self, texts, **_kwargs):
        self.encoded_texts = texts
        return [
            [float(row + column) for column in range(self.dimension)]
            for row, _text in enumerate(texts)
        ]


def make_chunk(chunk_id, code):
    return CodeChunk(
        chunk_id=chunk_id,
        file_path="src/main.py",
        language="Python",
        start_line=1,
        end_line=2,
        code=code,
    )


class EmbeddingServiceTests(unittest.TestCase):
    def tearDown(self):
        get_embedding_model.cache_clear()

    def test_model_is_initialized_once_and_cached(self):
        fake_model = FakeEmbeddingModel()
        with patch("embedding_service._load_sentence_transformer", return_value=fake_model) as load_model:
            get_embedding_model.cache_clear()

            first_model = get_embedding_model()
            second_model = get_embedding_model()

        self.assertIs(first_model, fake_model)
        self.assertIs(second_model, fake_model)
        load_model.assert_called_once_with()
        self.assertEqual(MODEL_NAME, "BAAI/bge-small-en-v1.5")

    def test_sentence_transformer_uses_configured_model_on_cpu(self):
        from embedding_service import _load_sentence_transformer

        fake_model = FakeEmbeddingModel()
        with patch("sentence_transformers.SentenceTransformer", return_value=fake_model) as sentence_transformer:
            loaded_model = _load_sentence_transformer()

        self.assertIs(loaded_model, fake_model)
        sentence_transformer.assert_called_once_with(MODEL_NAME, device="cpu")

    def test_generation_encodes_code_once_and_preserves_metadata(self):
        chunks = [make_chunk("src/main.py:1-2", "print('hello')\npass")]
        fake_model = FakeEmbeddingModel()
        with patch("embedding_service.get_embedding_model", return_value=fake_model):
            embeddings = generate_embeddings(chunks)

        self.assertEqual(fake_model.encoded_texts, ["print('hello')\npass"])
        self.assertEqual(len(embeddings), 1)
        self.assertEqual(embeddings[0].chunk_id, chunks[0].chunk_id)
        self.assertEqual(embeddings[0].file_path, chunks[0].file_path)
        self.assertEqual(embeddings[0].language, chunks[0].language)
        self.assertEqual(embeddings[0].start_line, chunks[0].start_line)
        self.assertEqual(embeddings[0].end_line, chunks[0].end_line)
        self.assertEqual(embeddings[0].code, chunks[0].code)

    def test_embeddings_are_numeric_and_have_consistent_dimensions(self):
        chunks = [make_chunk(f"src/{index}.py:1-2", f"value = {index}") for index in range(3)]
        with patch("embedding_service.get_embedding_model", return_value=FakeEmbeddingModel()):
            embeddings = generate_embeddings(chunks)

        self.assertEqual(len(embeddings), len(chunks))
        self.assertEqual({len(item.embedding) for item in embeddings}, {384})
        self.assertTrue(all(isinstance(value, float) for item in embeddings for value in item.embedding))

    def test_empty_input_returns_empty_without_loading_model(self):
        with patch("embedding_service.get_embedding_model") as get_model:
            self.assertEqual(generate_embeddings([]), [])

        get_model.assert_not_called()

    def test_repository_endpoint_limits_work_and_reports_full_totals(self):
        source = "".join(f"line {line_number}\n" for line_number in range(1, 121))
        repository_file = RepositoryFile(path="main.py", language="Python", content=source)
        request = RepositoryEmbeddingRequest(url="https://github.com/owner/repository", max_chunks=1)
        fake_model = FakeEmbeddingModel()

        with (
            patch("embedding_service.clone_and_scan_repository", return_value=("repository", [repository_file])),
            patch("embedding_service.get_embedding_model", return_value=fake_model),
        ):
            response = embed_repository(request)

        self.assertEqual(response.repository_name, "repository")
        self.assertEqual(response.total_files, 1)
        self.assertEqual(response.total_chunks, 3)
        self.assertEqual(response.embedding_dimension, 384)
        self.assertEqual(len(response.embeddings), 1)
        self.assertTrue(response.embeddings_truncated)

    def test_invalid_url_is_rejected(self):
        with self.assertRaises(ValidationError):
            RepositoryEmbeddingRequest(url="https://example.com/owner/repository")

    def test_empty_repository_and_empty_files_are_reported(self):
        request = RepositoryEmbeddingRequest(url="https://github.com/owner/repository")
        with patch("embedding_service.clone_and_scan_repository", return_value=("repository", [])):
            with self.assertRaises(HTTPException) as no_files_error:
                embed_repository(request)
        self.assertEqual(no_files_error.exception.status_code, 400)

        empty_file = RepositoryFile(path="empty.py", language="Python", content="")
        with patch(
            "embedding_service.clone_and_scan_repository",
            return_value=("repository", [empty_file]),
        ):
            with self.assertRaises(HTTPException) as empty_file_error:
                embed_repository(request)
        self.assertEqual(empty_file_error.exception.status_code, 400)
        self.assertIn("empty", empty_file_error.exception.detail)

    def test_model_errors_return_a_useful_api_error(self):
        repository_file = RepositoryFile(
            path="main.py",
            language="Python",
            content="print('hello')",
        )
        request = RepositoryEmbeddingRequest(url="https://github.com/owner/repository")
        with (
            patch("embedding_service.clone_and_scan_repository", return_value=("repository", [repository_file])),
            patch("embedding_service.generate_embeddings", side_effect=RuntimeError("model failed")),
        ):
            with self.assertRaises(HTTPException) as raised_error:
                embed_repository(request)

        self.assertEqual(raised_error.exception.status_code, 500)
        self.assertIn("Could not generate", raised_error.exception.detail)


if __name__ == "__main__":
    unittest.main()