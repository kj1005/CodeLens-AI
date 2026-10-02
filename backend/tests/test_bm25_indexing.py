import gc
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from chroma_service import (
    EMBEDDING_DIMENSION,
    BM25_INDEX_REGISTRY,
    create_persistent_client,
    get_code_collection,
    index_repository_to_chroma,
)
from embedding_service import (
    EmbeddedCodeChunk,
    RepositoryEmbeddingRequest,
    RepositoryEmbeddingResponse,
)
from retrieval_service import search_code_chunks
from bm25_service import BM25IndexRegistry


def make_embedded_chunk(chunk_id, code, file_path):
    return EmbeddedCodeChunk(
        chunk_id=chunk_id,
        file_path=file_path,
        language="Python",
        start_line=1,
        end_line=code.count("\n") + 1,
        code=code,
        embedding=[0.1] * EMBEDDING_DIMENSION,
    )


class BM25IndexingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.database_directory = TemporaryDirectory()
        self.client = create_persistent_client(self.database_directory.name)
        self.collection = get_code_collection(self.client)
        self.bm25_registry = BM25IndexRegistry()

    def tearDown(self):
        self.collection = None
        if self.client is not None and hasattr(self.client, "_system"):
            self.client._system.stop()
        self.client = None
        gc.collect()
        self.database_directory.cleanup()

    def index_chunks(self, repository_name, chunks):
        embedding_result = RepositoryEmbeddingResponse(
            repository_name=repository_name,
            total_files=len({chunk.file_path for chunk in chunks}),
            total_chunks=len(chunks),
            embedding_dimension=EMBEDDING_DIMENSION,
            embeddings=chunks,
            embeddings_truncated=False,
        )
        request = RepositoryEmbeddingRequest(
            url=f"https://github.com/example/{repository_name}"
        )

        with (
            patch("chroma_service.embed_repository", return_value=embedding_result),
            patch("chroma_service.get_code_collection", return_value=self.collection),
            patch("chroma_service.BM25_INDEX_REGISTRY", self.bm25_registry),
        ):
            return index_repository_to_chroma(request)

    def test_indexing_writes_chroma_and_indexes_the_same_chunks_in_bm25(self):
        chunk = make_embedded_chunk(
            "src/auth.py:1-2",
            "JWT_SECRET = load_secret()\ndef authenticate_user(token):\n",
            "src/auth.py",
        )

        response = self.index_chunks("auth-repository", [chunk])
        chroma_record = self.collection.get(
            ids=[chunk.chunk_id],
            include=["documents", "embeddings", "metadatas"],
        )
        bm25_index = self.bm25_registry.get_repository_index("auth-repository")
        bm25_result = bm25_index.search("JWT_SECRET", top_k=1)[0]

        self.assertEqual(response.vectors_stored, 1)
        self.assertEqual(chroma_record["documents"], [chunk.code])
        self.assertEqual(len(chroma_record["embeddings"][0]), EMBEDDING_DIMENSION)
        self.assertEqual(chroma_record["metadatas"][0]["repository_name"], "auth-repository")
        self.assertEqual(bm25_result.chunk_id, chunk.chunk_id)
        self.assertEqual(bm25_result.code, chunk.code)
        self.assertEqual(bm25_result.file_path, chunk.file_path)
        self.assertEqual(bm25_result.repository_name, "auth-repository")

    def test_reindexing_replaces_repository_bm25_corpus_without_duplicates(self):
        original_chunk = make_embedded_chunk(
            "src/auth.py:1-2",
            "JWT_SECRET = load_secret()\ndef authenticate_user(token):\n",
            "src/auth.py",
        )
        updated_chunk = make_embedded_chunk(
            original_chunk.chunk_id,
            "SESSION_KEY = load_key()\ndef create_session(user):\n",
            original_chunk.file_path,
        )

        self.index_chunks("auth-repository", [original_chunk])
        self.index_chunks("auth-repository", [updated_chunk])

        bm25_index = self.bm25_registry.get_repository_index("auth-repository")
        bm25_results = bm25_index.search("SESSION_KEY", top_k=10)
        chroma_records = self.collection.get(ids=[updated_chunk.chunk_id], include=["documents"])

        self.assertEqual(len(bm25_results), 1)
        self.assertEqual(bm25_results[0].code, updated_chunk.code)
        self.assertEqual(self.collection.count(), 1)
        self.assertEqual(chroma_records["documents"], [updated_chunk.code])

    def test_repository_indexes_are_kept_separate(self):
        first_repository_chunk = make_embedded_chunk(
            "src/main.py:1-2", "JWT_SECRET = load_secret()\n", "src/main.py"
        )
        second_repository_chunk = make_embedded_chunk(
            "src/main.py:1-2", "DATABASE_URL = load_database()\n", "src/main.py"
        )

        self.bm25_registry.index_repository("first-repository", [first_repository_chunk])
        self.bm25_registry.index_repository("second-repository", [second_repository_chunk])

        first_index = self.bm25_registry.get_repository_index("first-repository")
        second_index = self.bm25_registry.get_repository_index("second-repository")

        self.assertEqual(first_index.search("JWT_SECRET", top_k=1)[0].code, first_repository_chunk.code)
        self.assertEqual(first_index.search("DATABASE_URL", top_k=1)[0].repository_name, "first-repository")
        self.assertEqual(second_index.search("DATABASE_URL", top_k=1)[0].repository_name, "second-repository")

    def test_dense_search_still_reads_chroma_after_bm25_indexing(self):
        chunk = make_embedded_chunk(
            "src/auth.py:1-2",
            "def authenticate_user(token): return verify_token(token)\n",
            "src/auth.py",
        )
        self.index_chunks("auth-repository", [chunk])

        with (
            patch("retrieval_service.get_code_collection", return_value=self.collection),
            patch("retrieval_service.embed_query", return_value=chunk.embedding),
        ):
            dense_results = search_code_chunks("authenticate user", top_k=1)

        self.assertEqual(len(dense_results), 1)
        self.assertEqual(dense_results[0].chunk_id, chunk.chunk_id)
        self.assertEqual(dense_results[0].document, chunk.code)


if __name__ == "__main__":
    unittest.main()