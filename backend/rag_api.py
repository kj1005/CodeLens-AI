from collections.abc import Mapping
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from context_builder import RAGContextBuilder
from gemini_service import (
    DEFAULT_EXPLANATION_LEVEL,
    ExplanationLevel,
    GeminiConfigurationError,
    GeminiGenerationError,
    GeminiGenerationService,
)
from hybrid_api import hybrid_retrieval_service
from retrieval_service import RepositorySearchRequest

router = APIRouter()
context_builder = RAGContextBuilder()
gemini_generation_service = GeminiGenerationService()


class RAGSourceResult(BaseModel):
    chunk_id: str
    document: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    repository_name: str | None
    rrf_score: float | None
    dense_rank: int | None
    bm25_rank: int | None
    rerank_score: float | None


class RAGAskRequest(RepositorySearchRequest):
    explanation_level: ExplanationLevel = DEFAULT_EXPLANATION_LEVEL


class RAGAnswerResponse(BaseModel):
    query: str
    explanation_level: ExplanationLevel
    answer: str
    results: list[RAGSourceResult]
    result_count: int


def _field(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _source_result(result: Any) -> Any:
    if _field(result, "rerank_score") is not None:
        return result
    return _field(result, "result", result)


def _serialize_source(result: Any) -> RAGSourceResult:
    source = _source_result(result)
    document = _field(source, "document")
    if document is None:
        document = _field(source, "code", "")

    return RAGSourceResult(
        chunk_id=_field(result, "chunk_id"),
        document=document,
        file_path=_field(source, "file_path", ""),
        language=_field(source, "language", ""),
        start_line=_field(source, "start_line", 0),
        end_line=_field(source, "end_line", 0),
        repository_name=_field(source, "repository_name"),
        rrf_score=_field(result, "rrf_score"),
        dense_rank=_field(result, "dense_rank"),
        bm25_rank=_field(result, "bm25_rank"),
        rerank_score=_field(result, "rerank_score"),
    )


@router.post("/repository/ask", response_model=RAGAnswerResponse)
def ask_repository(request: RAGAskRequest) -> RAGAnswerResponse:
    try:
        results = hybrid_retrieval_service.search(
            request.query,
            top_k=request.top_k,
            repository_name=request.repository_name,
        )
        context = context_builder.build_context(results)
        answer = gemini_generation_service.generate_answer(
            request.query,
            context,
            request.explanation_level,
        )
    except GeminiConfigurationError:
        raise HTTPException(
            status_code=503,
            detail="Gemini is not configured. Set GEMINI_API_KEY to enable answer generation.",
        ) from None
    except GeminiGenerationError:
        raise HTTPException(
            status_code=502,
            detail="Gemini answer generation failed.",
        ) from None
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not complete the repository question request.",
        ) from None

    return RAGAnswerResponse(
        query=request.query,
        explanation_level=request.explanation_level,
        answer=answer,
        results=[_serialize_source(result) for result in results],
        result_count=len(results),
    )