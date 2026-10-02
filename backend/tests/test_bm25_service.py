import unittest

from code_chunking import CodeChunk
from bm25_service import BM25Service, tokenize_source_code


def make_chunk(chunk_id, code, file_path=None, start_line=1):
    return CodeChunk(
        chunk_id=chunk_id,
        file_path=file_path or f"src/{chunk_id}.py",
        language="Python",
        start_line=start_line,
        end_line=start_line + code.count("\n"),
        code=code,
    )


class BM25ServiceTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            make_chunk(
                "auth.py:1-4",
                "JWT_SECRET = load_secret()\n"
                "def authenticate_user(token):\n"
                "    return verify_jwt(token, JWT_SECRET)\n",
                file_path="src/auth.py",
                start_line=1,
            ),
            make_chunk(
                "users.py:10-13",
                "class UserRepository:\n"
                "    def get_user_by_id(self, user_id):\n"
                "        return self.database.get(user_id)\n",
                file_path="src/users.py",
                start_line=10,
            ),
            make_chunk(
                "sort.py:1-3",
                "def sort_values(values):\n"
                "    return sorted(values)\n",
                file_path="src/sort.py",
                start_line=1,
            ),
        ]
        self.service = BM25Service()

    def test_index_creation_returns_number_of_chunks(self):
        indexed_count = self.service.build_index(self.chunks)

        self.assertEqual(indexed_count, 3)
        self.assertEqual(len(self.service.search("sort_values")), 3)

    def test_tokenizer_preserves_code_identifiers(self):
        tokens = tokenize_source_code(
            "JWT_SECRET authenticate_user get_user_by_id UserRepository"
        )

        self.assertEqual(
            tokens,
            ["jwt_secret", "authenticate_user", "get_user_by_id", "userrepository"],
        )

    def test_exact_identifier_search_finds_jwt_secret(self):
        self.service.build_index(self.chunks)

        results = self.service.search("JWT_SECRET", top_k=1)

        self.assertEqual(results[0].chunk_id, "auth.py:1-4")
        self.assertIn("JWT_SECRET", results[0].code)

    def test_function_name_search_finds_authenticate_user(self):
        self.service.build_index(self.chunks)

        results = self.service.search("authenticate_user", top_k=1)

        self.assertEqual(results[0].chunk_id, "auth.py:1-4")

    def test_natural_language_query_matches_lexical_terms_in_source(self):
        authentication_chunk = make_chunk(
            "authentication.py:1-3",
            "def handle_user_authentication(user_id):\n"
            "    # user authentication is handled here\n"
            "    return verify_user(user_id)\n",
        )
        self.service.build_index([*self.chunks, authentication_chunk])

        results = self.service.search("Where is user authentication handled?", top_k=3)

        self.assertEqual(results[0].chunk_id, "authentication.py:1-3")
        self.assertGreater(results[0].score, results[1].score)

    def test_non_matching_term_returns_stable_zero_score_top_k(self):
        self.service.build_index(self.chunks)

        results = self.service.search("quantum_flarnivore_not_in_corpus", top_k=2)

        self.assertEqual(len(results), 2)
        self.assertEqual([result.rank for result in results], [1, 2])
        self.assertEqual([result.score for result in results], [0.0, 0.0])

    def test_metadata_and_repository_name_are_preserved(self):
        self.service.build_index(self.chunks, repository_name="sample-repository")

        result = self.service.search("get_user_by_id", top_k=1)[0]

        self.assertEqual(result.chunk_id, "users.py:10-13")
        self.assertEqual(result.file_path, "src/users.py")
        self.assertEqual(result.language, "Python")
        self.assertEqual(result.start_line, 10)
        self.assertEqual(result.end_line, 13)
        self.assertEqual(result.code, self.chunks[1].code)
        self.assertEqual(result.repository_name, "sample-repository")

    def test_top_k_limits_results_and_large_top_k_returns_all(self):
        self.service.build_index(self.chunks)

        self.assertEqual(len(self.service.search("def", top_k=2)), 2)
        self.assertEqual(len(self.service.search("def", top_k=10)), 3)

    def test_search_before_index_and_empty_index_return_no_results(self):
        self.assertEqual(self.service.search("authenticate_user"), [])
        self.assertEqual(self.service.build_index([]), 0)
        self.assertEqual(self.service.search("authenticate_user"), [])

    def test_empty_query_returns_no_results(self):
        self.service.build_index(self.chunks)

        self.assertEqual(self.service.search("   "), [])
        self.assertEqual(self.service.search("***"), [])

    def test_non_positive_top_k_is_rejected(self):
        self.service.build_index(self.chunks)

        with self.assertRaises(ValueError):
            self.service.search("authenticate_user", top_k=0)
        with self.assertRaises(ValueError):
            self.service.search("authenticate_user", top_k=-1)

    def test_results_are_ranked_by_descending_bm25_score(self):
        self.service.build_index(self.chunks)

        results = self.service.search("authenticate_user JWT_SECRET", top_k=3)

        self.assertEqual(results[0].chunk_id, "auth.py:1-4")
        self.assertEqual([result.rank for result in results], [1, 2, 3])
        self.assertEqual(
            [result.score for result in results],
            sorted((result.score for result in results), reverse=True),
        )


if __name__ == "__main__":
    unittest.main()