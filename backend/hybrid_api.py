from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from hybrid_retrieval_service import HybridRetrievalService
from retrieval_service import RepositorySearchRequest

router = APIRouter()
hybrid_retrieval_service = HybridRetrievalService(
    reranking_enabled=True,
    candidate_k=20,
)


class HybridSearchResultResponse(BaseModel):
    chunk_id: str
    document: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    repository_name: str | None
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None
    rerank_score: float | None = None


class HybridSearchResponse(BaseModel):
    query: str
    top_k: int
    result_count: int
    results: list[HybridSearchResultResponse]


def _result_field(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _serialize_hybrid_result(result: Any) -> HybridSearchResultResponse:
    source_result = (
        result
        if _result_field(result, "rerank_score") is not None
        else result.result
    )
    document = _result_field(source_result, "document")
    if document is None:
        document = _result_field(source_result, "code", "")

    return HybridSearchResultResponse(
        chunk_id=result.chunk_id,
        document=document,
        file_path=_result_field(source_result, "file_path", ""),
        language=_result_field(source_result, "language", ""),
        start_line=_result_field(source_result, "start_line", 0),
        end_line=_result_field(source_result, "end_line", 0),
        repository_name=_result_field(source_result, "repository_name"),
        rrf_score=result.rrf_score,
        dense_rank=result.dense_rank,
        bm25_rank=result.bm25_rank,
        rerank_score=_result_field(result, "rerank_score"),
    )


@router.post("/repository/hybrid-search", response_model=HybridSearchResponse)
def hybrid_search(request: RepositorySearchRequest) -> HybridSearchResponse:
    try:
        results = hybrid_retrieval_service.search(
            request.query,
            top_k=request.top_k,
            repository_name=request.repository_name,
        )
        serialized_results = [_serialize_hybrid_result(result) for result in results]
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not complete hybrid code search.",
        ) from None

    return HybridSearchResponse(
        query=request.query,
        top_k=request.top_k,
        result_count=len(serialized_results),
        results=serialized_results,
    )