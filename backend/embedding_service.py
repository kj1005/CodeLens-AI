from functools import lru_cache
from typing import Sequence

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from code_chunking import CodeChunk, chunk_source_code
from repository_ingestion import clone_and_scan_repository, validate_github_url

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_MAX_EMBEDDING_CHUNKS = 100
MAX_EMBEDDING_CHUNKS = 500
EMBEDDING_BATCH_SIZE = 32

router = APIRouter()


class EmbeddedCodeChunk(CodeChunk):
    embedding: list[float]


class RepositoryEmbeddingRequest(BaseModel):
    url: str
    max_chunks: int = Field(default=DEFAULT_MAX_EMBEDDING_CHUNKS, gt=0, le=MAX_EMBEDDING_CHUNKS)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return validate_github_url(value)


class RepositoryEmbeddingResponse(BaseModel):
    repository_name: str
    total_files: int
    total_chunks: int
    embedding_dimension: int
    embeddings: list[EmbeddedCodeChunk]
    embeddings_truncated: bool


def _load_sentence_transformer():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(MODEL_NAME, device="cpu")


@lru_cache(maxsize=1)
def get_embedding_model():
    return _load_sentence_transformer()


def generate_embeddings(chunks: Sequence[CodeChunk]) -> list[EmbeddedCodeChunk]:
    if not chunks:
        return []

    model = get_embedding_model()
    vectors = model.encode(
        [chunk.code for chunk in chunks],
        batch_size=EMBEDDING_BATCH_SIZE,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    if len(vectors) != len(chunks):
        raise RuntimeError("The embedding model returned an unexpected number of vectors.")

    embedded_chunks = []
    for chunk, vector in zip(chunks, vectors):
        embedded_chunks.append(
            EmbeddedCodeChunk(
                **chunk.model_dump(),
                embedding=[float(value) for value in vector],
            )
        )

    return embedded_chunks


@router.post("/repository/embeddings", response_model=RepositoryEmbeddingResponse)
def embed_repository(request: RepositoryEmbeddingRequest) -> RepositoryEmbeddingResponse:
    repository_name, files = clone_and_scan_repository(request.url)
    if not files:
        raise HTTPException(
            status_code=400,
            detail="No supported source files were found in the repository.",
        )

    chunks = [
        chunk
        for file in files
        for chunk in chunk_source_code(file.content, file.path, file.language)
    ]
    if not chunks:
        raise HTTPException(
            status_code=400,
            detail="Supported source files are empty; no code chunks could be embedded.",
        )

    chunks_to_embed = chunks[: request.max_chunks]
    try:
        embeddings = generate_embeddings(chunks_to_embed)
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not generate code embeddings. Check model installation and model download access.",
        ) from None

    if not embeddings:
        raise HTTPException(status_code=400, detail="No code chunks were available to embed.")

    return RepositoryEmbeddingResponse(
        repository_name=repository_name,
        total_files=len(files),
        total_chunks=len(chunks),
        embedding_dimension=len(embeddings[0].embedding),
        embeddings=embeddings,
        embeddings_truncated=len(chunks) > len(embeddings),
    )