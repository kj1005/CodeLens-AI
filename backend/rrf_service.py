from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

DEFAULT_RRF_K = 60
DEFAULT_TOP_K = 5


@dataclass(frozen=True)
class RRFResult:
    chunk_id: str
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None
    dense_result: Any | None
    bm25_result: Any | None

    @property
    def result(self) -> Any:
        """Return one original result while retaining both source results above."""
        return self.dense_result if self.dense_result is not None else self.bm25_result


@dataclass
class _RRFEntry:
    chunk_id: str
    rrf_score: float = 0.0
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_result: Any | None = None
    bm25_result: Any | None = None


def _get_chunk_id(result: Any) -> str:
    if isinstance(result, Mapping):
        chunk_id = result.get("chunk_id")
    else:
        chunk_id = getattr(result, "chunk_id", None)

    if not isinstance(chunk_id, str) or not chunk_id:
        raise ValueError("Every ranked result must have a non-empty chunk_id")
    return chunk_id


def reciprocal_rank_fusion(
    dense_results: Sequence[Any],
    bm25_results: Sequence[Any],
    top_k: int = DEFAULT_TOP_K,
    k: int = DEFAULT_RRF_K,
) -> list[RRFResult]:
    """Fuse two ranked lists using rank contributions only.

    Duplicate chunk IDs within one source use their first occurrence and rank.
    Ties retain insertion order: dense-list order first, then BM25-only results.
    """
    if top_k <= 0:
        raise ValueError("top_k must be greater than zero")
    if k <= 0:
        raise ValueError("k must be greater than zero")

    entries: dict[str, _RRFEntry] = {}

    for source_results, source in (
        (dense_results, "dense"),
        (bm25_results, "bm25"),
    ):
        seen_chunk_ids = set()
        for rank, result in enumerate(source_results, start=1):
            chunk_id = _get_chunk_id(result)
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)

            entry = entries.setdefault(chunk_id, _RRFEntry(chunk_id=chunk_id))
            entry.rrf_score += 1 / (k + rank)
            if source == "dense":
                entry.dense_rank = rank
                entry.dense_result = result
            else:
                entry.bm25_rank = rank
                entry.bm25_result = result

    ranked_entries = sorted(entries.values(), key=lambda entry: -entry.rrf_score)
    return [
        RRFResult(
            chunk_id=entry.chunk_id,
            rrf_score=entry.rrf_score,
            dense_rank=entry.dense_rank,
            bm25_rank=entry.bm25_rank,
            dense_result=entry.dense_result,
            bm25_result=entry.bm25_result,
        )
        for entry in ranked_entries[:top_k]
    ]