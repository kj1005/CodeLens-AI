from functools import lru_cache
from pathlib import Path
from typing import Sequence

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from bm25_service import BM25IndexRegistry
from embedding_service import (
    EmbeddedCodeChunk,
    RepositoryEmbeddingRequest,
    embed_repository,
)

COLLECTION_NAME = "codelens_code_chunks"
EMBEDDING_DIMENSION = 384
CHROMA_PERSIST_DIRECTORY = Path(__file__).resolve().parent / "chroma_data"
BM25_INDEX_REGISTRY = BM25IndexRegistry()

router = APIRouter()


class RepositoryStorageResponse(BaseModel):
    repository_name: str
    file_count: int
    total_chunks: int
    vectors_stored: int
    collection_total_vectors: int
    embedding_dimension: int


def create_persistent_client(persist_directory: str | Path):
    import chromadb

    return chromadb.PersistentClient(path=str(persist_directory))


@lru_cache(maxsize=1)
def get_chroma_client():
    return create_persistent_client(CHROMA_PERSIST_DIRECTORY)


def get_code_collection(client=None):
    active_client = client or get_chroma_client()
    return active_client.get_or_create_collection(name=COLLECTION_NAME)


def upsert_embedded_chunks(
    collection,
    repository_name: str,
    chunks: Sequence[EmbeddedCodeChunk],
) -> int:
    if not chunks:
        return 0

    for chunk in chunks:
        if len(chunk.embedding) != EMBEDDING_DIMENSION:
            raise ValueError(
                f"Expected {EMBEDDING_DIMENSION}-dimensional embeddings; "
                f"received {len(chunk.embedding)} for {chunk.chunk_id}."
            )

    collection.upsert(
        ids=[chunk.chunk_id for chunk in chunks],
        documents=[chunk.code for chunk in chunks],
        embeddings=[chunk.embedding for chunk in chunks],
        metadatas=[
            {
                "file_path": chunk.file_path,
                "language": chunk.language,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "repository_name": repository_name,
            }
            for chunk in chunks
        ],
    )
    return len(chunks)


@router.post("/repository/index-to-chroma", response_model=RepositoryStorageResponse)
def index_repository_to_chroma(
    request: RepositoryEmbeddingRequest,
) -> RepositoryStorageResponse:
    embedding_result = embed_repository(request)

    try:
        collection = get_code_collection()
        vectors_stored = upsert_embedded_chunks(
            collection,
            embedding_result.repository_name,
            embedding_result.embeddings,
        )
        collection_total_vectors = collection.count()
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not store code embeddings in ChromaDB.",
        ) from None

    try:
        BM25_INDEX_REGISTRY.index_repository(
            embedding_result.repository_name,
            embedding_result.embeddings,
        )
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="ChromaDB was updated, but the in-memory BM25 index could not be updated.",
        ) from None

    return RepositoryStorageResponse(
        repository_name=embedding_result.repository_name,
        file_count=embedding_result.total_files,
        total_chunks=embedding_result.total_chunks,
        vectors_stored=vectors_stored,
        collection_total_vectors=collection_total_vectors,
        embedding_dimension=embedding_result.embedding_dimension,
    )