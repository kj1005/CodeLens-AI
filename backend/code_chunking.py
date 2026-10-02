from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from repository_ingestion import clone_and_scan_repository, validate_github_url

DEFAULT_CHUNK_SIZE = 50
DEFAULT_CHUNK_OVERLAP = 10
DEFAULT_MAX_RESPONSE_CHUNKS = 100
MAX_RESPONSE_CHUNKS = 500

router = APIRouter()


class CodeChunk(BaseModel):
    chunk_id: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    code: str


class RepositoryChunkRequest(BaseModel):
    url: str
    max_chunks: int = Field(default=DEFAULT_MAX_RESPONSE_CHUNKS, gt=0, le=MAX_RESPONSE_CHUNKS)

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        return validate_github_url(value)


class RepositoryChunkResponse(BaseModel):
    repository_name: str
    total_files: int
    total_chunks: int
    chunks: list[CodeChunk]
    chunks_truncated: bool


def chunk_source_code(
    source_code: str,
    file_path: str,
    language: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[CodeChunk]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    lines = source_code.splitlines(keepends=True)
    chunks = []
    step = chunk_size - overlap

    for start_index in range(0, len(lines), step):
        end_index = min(start_index + chunk_size, len(lines))
        start_line = start_index + 1
        end_line = end_index
        chunks.append(
            CodeChunk(
                chunk_id=f"{file_path}:{start_line}-{end_line}",
                file_path=file_path,
                language=language,
                start_line=start_line,
                end_line=end_line,
                code="".join(lines[start_index:end_index]),
            )
        )
        if end_index == len(lines):
            break

    return chunks


@router.post("/repository/chunks", response_model=RepositoryChunkResponse)
def chunk_repository(request: RepositoryChunkRequest) -> RepositoryChunkResponse:
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
            detail="Supported source files are empty; no code chunks could be generated.",
        )

    return RepositoryChunkResponse(
        repository_name=repository_name,
        total_files=len(files),
        total_chunks=len(chunks),
        chunks=chunks[: request.max_chunks],
        chunks_truncated=len(chunks) > request.max_chunks,
    )