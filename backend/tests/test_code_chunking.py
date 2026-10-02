import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException
from git import Repo
from pydantic import ValidationError

from code_chunking import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    RepositoryChunkRequest,
    chunk_repository,
    chunk_source_code,
)
from repository_ingestion import (
    RepositoryFile,
    RepositoryIndexRequest,
    clone_and_scan_repository,
    index_repository,
    scan_repository,
)


class ChunkSourceCodeTests(unittest.TestCase):
    def test_chunks_respect_configured_size_and_overlap(self):
        source_lines = [f"line {line_number}\n" for line_number in range(1, 121)]
        chunks = chunk_source_code("".join(source_lines), "src/app.py", "Python")

        self.assertEqual([len(chunk.code.splitlines()) for chunk in chunks], [50, 50, 40])
        self.assertEqual(DEFAULT_CHUNK_SIZE, 50)
        self.assertEqual(DEFAULT_CHUNK_OVERLAP, 10)
        self.assertEqual(chunks[0].code.splitlines()[-10:], chunks[1].code.splitlines()[:10])
        self.assertEqual(chunks[1].code.splitlines()[-10:], chunks[2].code.splitlines()[:10])

    def test_line_numbers_are_one_based_and_metadata_is_preserved(self):
        chunks = chunk_source_code("first\nsecond\n", "docs/readme.md", "Markdown")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "docs/readme.md:1-2")
        self.assertEqual(chunks[0].file_path, "docs/readme.md")
        self.assertEqual(chunks[0].language, "Markdown")
        self.assertEqual(chunks[0].start_line, 1)
        self.assertEqual(chunks[0].end_line, 2)
        self.assertEqual(chunks[0].code, "first\nsecond\n")

    def test_file_smaller_than_chunk_size_produces_one_chunk(self):
        source_code = "print('hello')"

        chunks = chunk_source_code(source_code, "main.py", "Python")

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].code, source_code)
        self.assertEqual((chunks[0].start_line, chunks[0].end_line), (1, 1))

    def test_multiple_chunks_have_original_line_ranges(self):
        source_code = "".join(f"line {line_number}\n" for line_number in range(1, 76))

        chunks = chunk_source_code(source_code, "main.js", "JavaScript")

        self.assertEqual(
            [(chunk.start_line, chunk.end_line) for chunk in chunks],
            [(1, 50), (41, 75)],
        )
        self.assertTrue(chunks[0].code.startswith("line 1\n"))
        self.assertTrue(chunks[1].code.startswith("line 41\n"))
        self.assertTrue(chunks[1].code.endswith("line 75\n"))

    def test_empty_file_produces_no_chunks(self):
        self.assertEqual(chunk_source_code("", "empty.ts", "TypeScript"), [])

    def test_invalid_chunk_configuration_is_rejected(self):
        with self.assertRaises(ValueError):
            chunk_source_code("text", "file.py", "Python", chunk_size=0)
        with self.assertRaises(ValueError):
            chunk_source_code("text", "file.py", "Python", chunk_size=10, overlap=10)


class RepositoryChunkEndpointTests(unittest.TestCase):
    def test_invalid_github_url_is_rejected(self):
        with self.assertRaises(ValidationError):
            RepositoryChunkRequest(url="https://example.com/owner/repository")

    def test_no_supported_files_returns_a_useful_error(self):
        with patch("code_chunking.clone_and_scan_repository", return_value=("repo", [])):
            with self.assertRaises(HTTPException) as raised_error:
                chunk_repository(RepositoryChunkRequest(url="https://github.com/owner/repo"))

        self.assertEqual(raised_error.exception.status_code, 400)
        self.assertIn("No supported source files", raised_error.exception.detail)

    def test_empty_supported_files_returns_a_useful_error(self):
        empty_file = RepositoryFile(path="empty.py", language="Python", content="")
        with patch(
            "code_chunking.clone_and_scan_repository",
            return_value=("repo", [empty_file]),
        ):
            with self.assertRaises(HTTPException) as raised_error:
                chunk_repository(RepositoryChunkRequest(url="https://github.com/owner/repo"))

        self.assertEqual(raised_error.exception.status_code, 400)
        self.assertIn("files are empty", raised_error.exception.detail)

    def test_response_chunk_limit_does_not_change_total(self):
        source_code = "".join(f"line {line_number}\n" for line_number in range(1, 121))
        repository_file = RepositoryFile(
            path="main.py",
            language="Python",
            content=source_code,
        )
        request = RepositoryChunkRequest(url="https://github.com/owner/repo", max_chunks=1)

        with patch(
            "code_chunking.clone_and_scan_repository",
            return_value=("repo", [repository_file]),
        ):
            response = chunk_repository(request)

        self.assertEqual(response.total_files, 1)
        self.assertEqual(response.total_chunks, 3)
        self.assertEqual(len(response.chunks), 1)
        self.assertTrue(response.chunks_truncated)

    def test_clone_failure_returns_a_useful_error(self):
        with patch("repository_ingestion.Repo.clone_from", side_effect=OSError("clone failed")):
            with self.assertRaises(HTTPException) as raised_error:
                chunk_repository(RepositoryChunkRequest(url="https://github.com/owner/repo"))

        self.assertEqual(raised_error.exception.status_code, 400)
        self.assertIn("Could not clone", raised_error.exception.detail)

    def test_repository_scan_preserves_original_line_endings(self):
        with TemporaryDirectory() as temporary_directory:
            source_file = Path(temporary_directory) / "main.py"
            source_file.write_bytes(b"first\r\nsecond\r\n")

            files = scan_repository(Path(temporary_directory))

        self.assertEqual(files[0].content, "first\r\nsecond\r\n")

    def test_existing_index_response_does_not_expose_source_content(self):
        repository_file = RepositoryFile(
            path="main.py",
            language="Python",
            content="print('hello')\n",
        )
        request = RepositoryIndexRequest(github_url="https://github.com/owner/repo")

        with patch(
            "repository_ingestion.clone_and_scan_repository",
            return_value=("repo", [repository_file]),
        ):
            response = index_repository(request)

        self.assertEqual(
            response.model_dump(),
            {
                "repository_name": "repo",
                "file_count": 1,
                "files": [{"path": "main.py", "language": "Python"}],
            },
        )

    def test_clone_pipeline_provides_content_for_chunking(self):
        with TemporaryDirectory() as temporary_directory:
            source_path = Path(temporary_directory) / "source"
            source_path.mkdir()
            source_file = source_path / "main.py"
            source_file.write_text("first\nsecond\n", encoding="utf-8")
            source_repo = Repo.init(source_path)
            source_repo.index.add(["main.py"])
            source_repo.index.commit("test source")
            source_repo.close()

            original_clone = Repo.clone_from

            def clone_fixture(_url, destination):
                cloned_repo = original_clone(source_path, destination)
                cloned_repo.close()

            with patch("repository_ingestion.Repo.clone_from", side_effect=clone_fixture):
                repository_name, files = clone_and_scan_repository(
                    "https://github.com/owner/sample"
                )

        self.assertEqual(repository_name, "sample")
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].content, "first\r\nsecond\r\n")
        chunks = chunk_source_code(files[0].content, files[0].path, files[0].language)
        self.assertEqual(chunks[0].code, "first\r\nsecond\r\n")


if __name__ == "__main__":
    unittest.main()