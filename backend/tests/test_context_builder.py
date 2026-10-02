import unittest
from dataclasses import dataclass

from context_builder import DEFAULT_MAX_CONTEXT_CHARS, RAGContextBuilder


@dataclass
class FakeRerankedResult:
    chunk_id: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    code: str
    document: str
    rrf_score: float
    rerank_score: float


def make_result(chunk_id, file_path, start_line, end_line, language, code):
    return FakeRerankedResult(
        chunk_id=chunk_id,
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        language=language,
        code=code,
        document=code,
        rrf_score=0.1,
        rerank_score=1.0,
    )


class RAGContextBuilderTests(unittest.TestCase):
    def test_empty_results_return_empty_context(self):
        self.assertEqual(RAGContextBuilder().build_context([]), "")

    def test_single_result_formats_metadata_and_preserves_code(self):
        source_code = "def authenticate_user(token):\n    return verify(token)\n"
        result = make_result(
            "auth.py:10-11",
            "src/auth.py",
            10,
            11,
            "Python",
            source_code,
        )

        context = RAGContextBuilder().build_context([result])

        self.assertEqual(
            context,
            "[File: src/auth.py | Lines: 10-11 | Language: Python]\n" + source_code,
        )

    def test_multiple_results_are_separated_and_keep_source_text(self):
        first_code = "def connect():\n    return database.connect()\n"
        second_code = "JWT_SECRET = load_secret()\n"
        results = [
            make_result("database.py:1-2", "src/database.py", 1, 2, "Python", first_code),
            make_result("config.py:4-4", "src/config.py", 4, 4, "Python", second_code),
        ]

        context = RAGContextBuilder().build_context(results)

        self.assertEqual(
            context,
            "[File: src/database.py | Lines: 1-2 | Language: Python]\n"
            + first_code
            + "\n\n"
            + "[File: src/config.py | Lines: 4-4 | Language: Python]\n"
            + second_code,
        )

    def test_document_field_is_used_when_code_field_is_unavailable(self):
        result = {
            "file_path": "README.md",
            "start_line": 1,
            "end_line": 2,
            "language": "Markdown",
            "document": "# Setup\nRun the service.\n",
        }

        context = RAGContextBuilder().build_context([result])

        self.assertIn("Language: Markdown", context)
        self.assertIn(result["document"], context)

    def test_configurable_limit_stops_before_next_complete_entry(self):
        first = make_result("one", "one.py", 1, 1, "Python", "one()")
        second = make_result("two", "two.py", 1, 1, "Python", "two()")
        builder = RAGContextBuilder(max_context_chars=55)

        context = builder.build_context([first, second])

        self.assertEqual(
            context,
            "[File: one.py | Lines: 1-1 | Language: Python]\none()",
        )
        self.assertLessEqual(len(context), 55)

    def test_oversized_single_entry_is_rejected(self):
        result = make_result("large", "large.py", 1, 1, "Python", "x" * 100)

        with self.assertRaises(ValueError):
            RAGContextBuilder(max_context_chars=20).build_context([result])

    def test_invalid_context_limit_is_rejected(self):
        with self.assertRaises(ValueError):
            RAGContextBuilder(max_context_chars=0)
        self.assertEqual(DEFAULT_MAX_CONTEXT_CHARS, 16_000)


if __name__ == "__main__":
    unittest.main()