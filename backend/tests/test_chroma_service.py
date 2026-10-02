import gc
import unittest
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException

from chroma_service import (
    COLLECTION_NAME,
    EMBEDDING_DIMENSION,
    RepositoryStorageResponse,
    create_persistent_client,
    get_code_collection,
    index_repository_to_chroma,
    upsert_embedded_chunks,
)
from embedding_service import EmbeddedCodeChunk, RepositoryEmbeddingRequest, RepositoryEmbeddingResponse


def make_embedded_chunk(chunk_id="src/main.py:1-2", code="print('hello')\n"):
    return EmbeddedCodeChunk(
        chunk_id=chunk_id,
        file_path="src/main.py",
        language="Python",
        start_line=1,
        end_line=2,
        code=code,
        embedding=[0.25] * EMBEDDING_DIMENSION,
    )


class ChromaServiceTests(unittest.TestCase):
    def setUp(self):
        self.database_directory = TemporaryDirectory()
        self.client = create_persistent_client(self.database_directory.name)
        self.collection = get_code_collection(self.client)

    def tearDown(self):
        self.collection = None
        if self.client is not None and hasattr(self.client, "_system"):
            self.client._system.stop()
        self.client = None
        self.database_directory.cleanup()

    def test_persistent_client_initializes_in_isolated_directory(self):
        self.assertTrue(self.database_directory.name)
        self.assertEqual(self.client.get_settings().is_persistent, True)

    def test_collection_is_created_and_retrieved_without_duplication(self):
        retrieved_collection = get_code_collection(self.client)

        self.assertEqual(self.collection.name, COLLECTION_NAME)
        self.assertEqual(retrieved_collection.id, self.collection.id)
        self.assertEqual(len(self.client.list_collections()), 1)

    def test_storing_chunk_preserves_document_vector_and_metadata(self):
        chunk = make_embedded_chunk()

        stored_count = upsert_embedded_chunks(self.collection, "sample-repository", [chunk])
        result = self.collection.get(
            ids=[chunk.chunk_id],
            include=["documents", "embeddings", "metadatas"],
        )

        self.assertEqual(stored_count, 1)
        self.assertEqual(result["ids"], [chunk.chunk_id])
        self.assertEqual(result["documents"], [chunk.code])
        self.assertEqual(len(result["embeddings"][0]), EMBEDDING_DIMENSION)
        self.assertEqual(
            result["metadatas"][0],
            {
                "file_path": chunk.file_path,
                "language": chunk.language,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "repository_name": "sample-repository",
            },
        )

    def test_vectors_survive_reopening_the_persistent_database(self):
        chunk = make_embedded_chunk()
        upsert_embedded_chunks(self.collection, "sample-repository", [chunk])
        self.client._system.stop()
        self.collection = None
        self.client = None
        gc.collect()

        verification_code = (
            "import sys; import chromadb; "
            "client = chromadb.PersistentClient(path=sys.argv[1]); "
            f"collection = client.get_collection(name={COLLECTION_NAME!r}); "
            "stored = collection.get(ids=[sys.argv[2]], include=['documents']); "
            "assert stored['documents'] == [sys.argv[3]]; "
            "print(collection.count())"
        )

        result = subprocess.run(
            [
                sys.executable,
                "-c",
                verification_code,
                self.database_directory.name,
                chunk.chunk_id,
                chunk.code,
            ],
            check=True,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.stdout.strip(), "1")

    def test_embedding_dimension_is_validated(self):
        chunk = make_embedded_chunk()
        chunk.embedding = [0.25] * (EMBEDDING_DIMENSION - 1)

        with self.assertRaises(ValueError):
            upsert_embedded_chunks(self.collection, "sample-repository", [chunk])

        self.assertEqual(self.collection.count(), 0)

    def test_upsert_updates_existing_chunk_instead_of_duplicating_it(self):
        original_chunk = make_embedded_chunk()
        updated_chunk = make_embedded_chunk(code="print('updated')\n")

        upsert_embedded_chunks(self.collection, "sample-repository", [original_chunk])
        upsert_embedded_chunks(self.collection, "sample-repository", [updated_chunk])
        result = self.collection.get(ids=[original_chunk.chunk_id], include=["documents"])

        self.assertEqual(self.collection.count(), 1)
        self.assertEqual(result["documents"], ["print('updated')\n"])

    def test_indexing_response_reports_operation_and_collection_statistics(self):
        chunk = make_embedded_chunk()
        embedding_result = RepositoryEmbeddingResponse(
            repository_name="sample-repository",
            total_files=2,
            total_chunks=1,
            embedding_dimension=EMBEDDING_DIMENSION,
            embeddings=[chunk],
            embeddings_truncated=False,
        )
        request = RepositoryEmbeddingRequest(url="https://github.com/owner/sample-repository")

        with (
            patch("chroma_service.embed_repository", return_value=embedding_result),
            patch("chroma_service.get_code_collection", return_value=self.collection),
        ):
            response = index_repository_to_chroma(request)

        self.assertEqual(
            response,
            RepositoryStorageResponse(
                repository_name="sample-repository",
                file_count=2,
                total_chunks=1,
                vectors_stored=1,
                collection_total_vectors=1,
                embedding_dimension=EMBEDDING_DIMENSION,
            ),
        )

    def test_chroma_storage_errors_return_a_useful_api_error(self):
        embedding_result = RepositoryEmbeddingResponse(
            repository_name="sample-repository",
            total_files=1,
            total_chunks=1,
            embedding_dimension=EMBEDDING_DIMENSION,
            embeddings=[make_embedded_chunk()],
            embeddings_truncated=False,
        )
        request = RepositoryEmbeddingRequest(url="https://github.com/owner/sample-repository")

        with (
            patch("chroma_service.embed_repository", return_value=embedding_result),
            patch("chroma_service.get_code_collection", side_effect=RuntimeError("storage failed")),
        ):
            with self.assertRaises(HTTPException) as raised_error:
                index_repository_to_chroma(request)

        self.assertEqual(raised_error.exception.status_code, 500)
        self.assertIn("ChromaDB", raised_error.exception.detail)


if __name__ == "__main__":
    unittest.main()