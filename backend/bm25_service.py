import re
from dataclasses import dataclass
from typing import Sequence

from rank_bm25 import BM25Okapi

from code_chunking import CodeChunk

_TOKEN_PATTERN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")


def tokenize_source_code(text: str) -> list[str]:
    return [token.casefold() for token in _TOKEN_PATTERN.findall(text)]


@dataclass(frozen=True)
class BM25SearchResult:
    chunk_id: str
    file_path: str
    language: str
    start_line: int
    end_line: int
    code: str
    repository_name: str | None
    score: float
    rank: int


class BM25Service:
    def __init__(self) -> None:
        self._chunks: list[CodeChunk] = []
        self._repository_name: str | None = None
        self._index: BM25Okapi | None = None

    def build_index(
        self,
        chunks: Sequence[CodeChunk],
        repository_name: str | None = None,
    ) -> int:
        self._chunks = list(chunks)
        self._repository_name = repository_name
        tokenized_documents = [tokenize_source_code(chunk.code) for chunk in self._chunks]
        self._index = BM25Okapi(tokenized_documents) if tokenized_documents else None
        return len(self._chunks)

    def search(self, query: str, top_k: int = 5) -> list[BM25SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        if not self._index or not self._chunks or not query.strip():
            return []

        query_tokens = tokenize_source_code(query)
        if not query_tokens:
            return []

        scores = self._index.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(self._chunks)),
            key=lambda index: (-float(scores[index]), index),
        )[:top_k]

        return [
            BM25SearchResult(
                chunk_id=self._chunks[index].chunk_id,
                file_path=self._chunks[index].file_path,
                language=self._chunks[index].language,
                start_line=self._chunks[index].start_line,
                end_line=self._chunks[index].end_line,
                code=self._chunks[index].code,
                repository_name=self._repository_name,
                score=float(scores[index]),
                rank=rank,
            )
            for rank, index in enumerate(ranked_indices, start=1)
        ]


class BM25IndexRegistry:
    def __init__(self) -> None:
        self._repository_indexes: dict[str, BM25Service] = {}

    def index_repository(
        self,
        repository_name: str,
        chunks: Sequence[CodeChunk],
    ) -> int:
        repository_index = BM25Service()
        indexed_count = repository_index.build_index(chunks, repository_name)
        self._repository_indexes[repository_name] = repository_index
        return indexed_count

    def get_repository_index(self, repository_name: str) -> BM25Service | None:
        return self._repository_indexes.get(repository_name)

    @property
    def repository_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._repository_indexes))