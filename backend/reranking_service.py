from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from sentence_transformers import CrossEncoder

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
DEFAULT_RERANKER_DEVICE = "cpu"


@dataclass(frozen=True)
class RerankedResult:
    chunk_id: str
    code: str
    document: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    repository_name: str | None
    rrf_score: float | None
    dense_rank: int | None
    bm25_rank: int | None
    rerank_score: float
    original_result: Any


def _get_field(result: Any, field_name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(field_name, default)
    return getattr(result, field_name, default)


class RerankingService:
    def __init__(
        self,
        model_name: str = RERANKER_MODEL_NAME,
        device: str = DEFAULT_RERANKER_DEVICE,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model: CrossEncoder | None = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            self._model = CrossEncoder(self.model_name, device=self.device)
        return self._model

    @staticmethod
    def _candidate_document(candidate: Any) -> tuple[Any, str]:
        source_result = _get_field(candidate, "result") or candidate
        code = _get_field(source_result, "code")
        if code is None:
            code = _get_field(source_result, "document")
        if not isinstance(code, str):
            raise ValueError("Every candidate must contain code or document text")
        return source_result, code

    def rerank(
        self,
        query: str,
        candidates: Sequence[Any],
        top_k: int,
    ) -> list[RerankedResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if not candidates:
            return []

        prepared_candidates = [
            (candidate, *self._candidate_document(candidate))
            for candidate in candidates
        ]
        query_document_pairs = [
            (query, document)
            for _candidate, _source_result, document in prepared_candidates
        ]
        scores = self._get_model().predict(query_document_pairs)
        if hasattr(scores, "tolist"):
            scores = scores.tolist()
        if not isinstance(scores, (list, tuple)):
            scores = [scores]
        if len(scores) != len(prepared_candidates):
            raise RuntimeError("The cross-encoder returned an unexpected number of scores.")

        reranked_results = []
        for original_index, ((candidate, source_result, code), score) in enumerate(
            zip(prepared_candidates, scores)
        ):
            chunk_id = _get_field(candidate, "chunk_id") or _get_field(source_result, "chunk_id")
            if not isinstance(chunk_id, str) or not chunk_id:
                raise ValueError("Every candidate must have a non-empty chunk_id")

            reranked_results.append(
                (
                    float(score),
                    original_index,
                    RerankedResult(
                        chunk_id=chunk_id,
                        code=code,
                        document=code,
                        file_path=_get_field(source_result, "file_path", ""),
                        language=_get_field(source_result, "language", ""),
                        start_line=_get_field(source_result, "start_line", 0),
                        end_line=_get_field(source_result, "end_line", 0),
                        repository_name=_get_field(source_result, "repository_name"),
                        rrf_score=_get_field(candidate, "rrf_score"),
                        dense_rank=_get_field(candidate, "dense_rank"),
                        bm25_rank=_get_field(candidate, "bm25_rank"),
                        rerank_score=float(score),
                        original_result=candidate,
                    ),
                )
            )

        reranked_results.sort(key=lambda item: (-item[0], item[1]))
        return [result for _score, _index, result in reranked_results[:top_k]]