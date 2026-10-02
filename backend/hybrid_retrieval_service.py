from collections.abc import Callable, Sequence
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from bm25_service import BM25IndexRegistry
from chroma_service import BM25_INDEX_REGISTRY
from reranking_service import RerankedResult, RerankingService
from retrieval_service import search_code_chunks
from rrf_service import DEFAULT_RRF_K, DEFAULT_TOP_K, RRFResult, reciprocal_rank_fusion

DEFAULT_CANDIDATE_K = 20


@dataclass(frozen=True)
class _RepositoryScopedResult:
    chunk_id: str
    original_result: Any


def _get_result_field(result: Any, field_name: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(field_name)
    return getattr(result, field_name, None)


def _scope_result(result: Any) -> Any:
    repository_name = _get_result_field(result, "repository_name")
    chunk_id = _get_result_field(result, "chunk_id")
    if not repository_name or not chunk_id:
        return result

    scoped_id = f"{len(repository_name)}:{repository_name}{chunk_id}"
    return _RepositoryScopedResult(chunk_id=scoped_id, original_result=result)


def _unscoped_result(result: Any) -> Any:
    if isinstance(result, _RepositoryScopedResult):
        return result.original_result
    return result


class HybridRetrievalService:
    def __init__(
        self,
        dense_retriever: Callable[..., Sequence[Any]] | None = None,
        bm25_registry: BM25IndexRegistry | None = None,
        fusion_function: Callable[..., list[RRFResult]] | None = None,
        reranker: RerankingService | None = None,
        reranking_enabled: bool = False,
        candidate_k: int = DEFAULT_CANDIDATE_K,
    ) -> None:
        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero")

        self._dense_retriever = dense_retriever or search_code_chunks
        self._bm25_registry = bm25_registry or BM25_INDEX_REGISTRY
        self._fusion_function = fusion_function or reciprocal_rank_fusion
        self._reranker = reranker
        self._reranking_enabled = reranking_enabled
        self._candidate_k = candidate_k

    def _get_reranker(self) -> RerankingService:
        if self._reranker is None:
            self._reranker = RerankingService()
        return self._reranker

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        repository_name: str | None = None,
        rrf_k: int = DEFAULT_RRF_K,
        candidate_k: int | None = None,
        reranking_enabled: bool | None = None,
    ) -> list[RRFResult] | list[RerankedResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero")

        use_reranking = (
            self._reranking_enabled
            if reranking_enabled is None
            else reranking_enabled
        )
        candidate_limit = self._candidate_k if candidate_k is None else candidate_k
        if candidate_limit <= 0:
            raise ValueError("candidate_k must be greater than zero")
        retrieval_limit = candidate_limit if use_reranking else top_k

        dense_results = self._dense_retriever(
            query,
            top_k=retrieval_limit,
            repository_name=repository_name,
        )
        bm25_results = self._search_bm25(query, retrieval_limit, repository_name)

        fused_results = self._fusion_function(
            [_scope_result(result) for result in dense_results],
            [_scope_result(result) for result in bm25_results],
            top_k=retrieval_limit,
            k=rrf_k,
        )
        restored_results = [
            RRFResult(
                chunk_id=_get_result_field(
                    _unscoped_result(result.dense_result)
                    if result.dense_result is not None
                    else _unscoped_result(result.bm25_result),
                    "chunk_id",
                ),
                rrf_score=result.rrf_score,
                dense_rank=result.dense_rank,
                bm25_rank=result.bm25_rank,
                dense_result=_unscoped_result(result.dense_result),
                bm25_result=_unscoped_result(result.bm25_result),
            )
            for result in fused_results
        ]
        if not use_reranking or not restored_results:
            return restored_results

        return self._get_reranker().rerank(query, restored_results, top_k=top_k)

    def _search_bm25(
        self,
        query: str,
        top_k: int,
        repository_name: str | None,
    ) -> list[Any]:
        if repository_name is not None:
            repository_index = self._bm25_registry.get_repository_index(repository_name)
            if repository_index is None:
                return []
            return repository_index.search(query, top_k=top_k)

        repository_results = [
            self._bm25_registry.get_repository_index(name).search(query, top_k=top_k)
            for name in self._bm25_registry.repository_names
        ]

        # Per-repository BM25 scores use different corpus statistics; merge by
        # local rank layers instead of comparing those raw scores.
        merged_results = []
        for rank_index in range(top_k):
            for results in repository_results:
                if rank_index < len(results):
                    merged_results.append(results[rank_index])
        return merged_results