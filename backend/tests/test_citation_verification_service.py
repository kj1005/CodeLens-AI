import unittest
from types import SimpleNamespace

from citation_verification_service import CitationStatus, verify_citations


def make_result(file_path="src/auth.py", start_line=10, end_line=28):
    return SimpleNamespace(
        chunk_id="auth.py:10-28",
        file_path=file_path,
        start_line=start_line,
        end_line=end_line,
        language="Python",
        code="def authenticate_user(): pass",
    )


class CitationVerificationTests(unittest.TestCase):
    def test_valid_file_citation_is_verified(self):
        citations = verify_citations(
            "Authentication is implemented here [src/auth.py].",
            [make_result()],
        )

        self.assertEqual(len(citations), 1)
        self.assertEqual(citations[0].status, CitationStatus.VERIFIED)
        self.assertEqual(citations[0].file_path, "src/auth.py")
        self.assertIsNone(citations[0].start_line)
        self.assertIsNone(citations[0].end_line)

    def test_valid_file_and_line_citation_is_verified(self):
        citations = verify_citations(
            "See [src/auth.py:12-18].",
            [make_result()],
        )

        self.assertEqual(citations[0].status, CitationStatus.VERIFIED)
        self.assertEqual(citations[0].start_line, 12)
        self.assertEqual(citations[0].end_line, 18)

    def test_overlapping_line_range_is_verified(self):
        citations = verify_citations(
            "The relevant line is [src/auth.py:28-30].",
            [make_result()],
        )

        self.assertEqual(citations[0].status, CitationStatus.VERIFIED)

    def test_non_overlapping_line_range_is_unverified(self):
        citations = verify_citations(
            "See [src/auth.py:29-31].",
            [make_result()],
        )

        self.assertEqual(citations[0].status, CitationStatus.UNVERIFIED)

    def test_unknown_file_is_unverified(self):
        citations = verify_citations(
            "See [src/database.py:10-12].",
            [make_result()],
        )

        self.assertEqual(citations[0].status, CitationStatus.UNVERIFIED)

    def test_answer_without_citations_returns_empty_list(self):
        self.assertEqual(
            verify_citations("No file references are included.", [make_result()]),
            [],
        )

    def test_multiple_citations_are_verified_independently(self):
        answer = "[src/auth.py:10-14] and [src/database.py:4-6] and [missing.py]."
        results = [
            make_result(),
            make_result("src/database.py", 5, 9),
        ]

        citations = verify_citations(answer, results)

        self.assertEqual(
            [citation.status for citation in citations],
            [
                CitationStatus.VERIFIED,
                CitationStatus.VERIFIED,
                CitationStatus.UNVERIFIED,
            ],
        )
        self.assertEqual(
            [citation.file_path for citation in citations],
            ["src/auth.py", "src/database.py", "missing.py"],
        )


if __name__ == "__main__":
    unittest.main()