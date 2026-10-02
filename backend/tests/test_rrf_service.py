import unittest
from dataclasses import dataclass

from rrf_service import DEFAULT_RRF_K, reciprocal_rank_fusion


@dataclass
class RankedChunk:
    chunk_id: str
    document: str
    metadata: dict
    source_score: float


def make_result(chunk_id, source_score=0.0):
    return RankedChunk(
        chunk_id=chunk_id,
        document=f"source for {chunk_id}",
        metadata={"file_path": f"{chunk_id}.py", "language": "Python"},
        source_score=source_score,
    )


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_shared_result_adds_both_rank_contributions(self):
        dense_result = make_result("shared", source_score=0.03)
        bm25_result = make_result("shared", source_score=1.4)

        results = reciprocal_rank_fusion(
            [dense_result],
            [make_result("other"), bm25_result],
        )

        shared = next(result for result in results if result.chunk_id == "shared")
        self.assertEqual(shared.dense_rank, 1)
        self.assertEqual(shared.bm25_rank, 2)
        self.assertAlmostEqual(
            shared.rrf_score,
            1 / (DEFAULT_RRF_K + 1) + 1 / (DEFAULT_RRF_K + 2),
        )
        self.assertIs(shared.dense_result, dense_result)
        self.assertIs(shared.bm25_result, bm25_result)

    def test_dense_only_result_is_included(self):
        dense_result = make_result("dense-only")

        results = reciprocal_rank_fusion([dense_result], [])

        self.assertEqual(len(results), 1)
        self.assertIs(results[0].result, dense_result)
        self.assertEqual(results[0].dense_rank, 1)
        self.assertIsNone(results[0].bm25_rank)
        self.assertAlmostEqual(results[0].rrf_score, 1 / (DEFAULT_RRF_K + 1))

    def test_bm25_only_result_is_included(self):
        bm25_result = make_result("bm25-only")

        results = reciprocal_rank_fusion([], [bm25_result])

        self.assertEqual(len(results), 1)
        self.assertIs(results[0].result, bm25_result)
        self.assertIsNone(results[0].dense_rank)
        self.assertEqual(results[0].bm25_rank, 1)
        self.assertAlmostEqual(results[0].rrf_score, 1 / (DEFAULT_RRF_K + 1))

    def test_rank_one_contributes_more_than_rank_two(self):
        results = reciprocal_rank_fusion(
            [make_result("rank-one"), make_result("rank-two")],
            [],
            top_k=2,
        )

        self.assertGreater(results[0].rrf_score, results[1].rrf_score)

    def test_results_are_sorted_by_descending_rrf_score(self):
        results = reciprocal_rank_fusion(
            [make_result("a"), make_result("b"), make_result("c")],
            [make_result("b"), make_result("c")],
        )

        scores = [result.rrf_score for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertEqual(results[0].chunk_id, "b")

    def test_top_k_limits_fused_results(self):
        results = reciprocal_rank_fusion(
            [make_result("a"), make_result("b"), make_result("c")],
            [make_result("d"), make_result("e")],
            top_k=3,
        )

        self.assertEqual(len(results), 3)

    def test_custom_k_changes_reciprocal_rank_score(self):
        result = make_result("a")

        default_score = reciprocal_rank_fusion([result], [])[0].rrf_score
        custom_score = reciprocal_rank_fusion([result], [], k=10)[0].rrf_score

        self.assertAlmostEqual(default_score, 1 / (DEFAULT_RRF_K + 1))
        self.assertAlmostEqual(custom_score, 1 / 11)
        self.assertGreater(custom_score, default_score)

    def test_duplicate_ids_within_a_source_are_counted_once(self):
        first_occurrence = make_result("duplicate", source_score=3.0)
        duplicate_occurrence = make_result("duplicate", source_score=99.0)
        other_result = make_result("other")

        results = reciprocal_rank_fusion(
            [first_occurrence, duplicate_occurrence, other_result],
            [],
        )

        duplicate = next(result for result in results if result.chunk_id == "duplicate")
        other = next(result for result in results if result.chunk_id == "other")
        self.assertEqual(duplicate.dense_rank, 1)
        self.assertIs(duplicate.dense_result, first_occurrence)
        self.assertAlmostEqual(duplicate.rrf_score, 1 / (DEFAULT_RRF_K + 1))
        self.assertEqual(other.dense_rank, 3)
        self.assertAlmostEqual(other.rrf_score, 1 / (DEFAULT_RRF_K + 3))

    def test_empty_inputs_return_empty_and_single_source_inputs_work(self):
        self.assertEqual(reciprocal_rank_fusion([], []), [])
        self.assertEqual(len(reciprocal_rank_fusion([make_result("dense")], [])), 1)
        self.assertEqual(len(reciprocal_rank_fusion([], [make_result("bm25")])), 1)

    def test_invalid_top_k_and_k_are_rejected(self):
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], [], top_k=0)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], [], top_k=-1)
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], [], k=0)

    def test_original_result_and_metadata_are_preserved(self):
        dense_result = {
            "chunk_id": "file.py:1-5",
            "document": "def handle_user(): pass",
            "metadata": {
                "file_path": "file.py",
                "language": "Python",
                "start_line": 1,
                "end_line": 5,
            },
            "distance": 0.25,
        }
        bm25_result = make_result("file.py:1-5", source_score=2.1)

        fused = reciprocal_rank_fusion([dense_result], [bm25_result])[0]

        self.assertEqual(fused.chunk_id, "file.py:1-5")
        self.assertIs(fused.dense_result, dense_result)
        self.assertIs(fused.bm25_result, bm25_result)
        self.assertEqual(fused.result["metadata"]["start_line"], 1)
        self.assertEqual(fused.result["metadata"]["end_line"], 5)
        self.assertEqual(fused.result["document"], "def handle_user(): pass")


if __name__ == "__main__":
    unittest.main()