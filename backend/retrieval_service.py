from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from chroma_service import get_code_collection
from embedding_service import EMBEDDING_BATCH_SIZE, MODEL_NAME, get_embedding_model

DEFAULT_TOP_K = 5
MAX_TOP_K = 50
EMBEDDING_DIMENSION = 384
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

router = APIRouter()


class RepositorySearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=DEFAULT_TOP_K, gt=0, le=MAX_TOP_K)
    repository_name: str | None = Field(default=None, min_length=1, max_length=200)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        query = value.strip()
        if not query:
            raise ValueError("Query must not be blank")
        return query


class RetrievedChunk(BaseModel):
    chunk_id: str
    document: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    repository_name: str
    distance: float


class RepositorySearchResponse(BaseModel):
    query: str
    top_k: int
    result_count: int
    results: list[RetrievedChunk]


def embed_query(query: str) -> list[float]:
    model = get_embedding_model()
    vectors = model.encode(
        [f"{QUERY_INSTRUCTION}{query}"],
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if len(vectors) != 1:
        raise RuntimeError("The embedding model did not return exactly one query vector.")

    query_embedding = [float(value) for value in vectors[0]]
    if len(query_embedding) != EMBEDDING_DIMENSION:
        raise RuntimeError(
            f"Expected a {EMBEDDING_DIMENSION}-dimensional query embedding, "
            f"received {len(query_embedding)} dimensions."
        )
    return query_embedding


def search_code_chunks(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    repository_name: str | None = None,
) -> list[RetrievedChunk]:
    collection = get_code_collection()
    collection_size = collection.count()
    if collection_size == 0:
        return []

    query_embedding = embed_query(query)
    where = {"repository_name": repository_name} if repository_name else None
    search_response = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection_size),
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    ids = search_response["ids"][0]
    documents = search_response["documents"][0]
    metadatas = search_response["metadatas"][0]
    distances = search_response["distances"][0]
    for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
        results.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                document=document,
                file_path=metadata["file_path"],
                language=metadata["language"],
                start_line=metadata["start_line"],
                end_line=metadata["end_line"],
                repository_name=metadata["repository_name"],
                distance=float(distance),
            )
        )

    return results


@router.post("/repository/search", response_model=RepositorySearchResponse)
def search_repository(request: RepositorySearchRequest) -> RepositorySearchResponse:
    try:
        results = search_code_chunks(
            request.query,
            top_k=request.top_k,
            repository_name=request.repository_name,
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail=f"Could not search code embeddings with {MODEL_NAME}.",
        ) from None

    return RepositorySearchResponse(
        query=request.query,
        top_k=request.top_k,
        result_count=len(results),
        results=results,
    )