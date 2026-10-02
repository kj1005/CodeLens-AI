import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

_CITATION_PATTERN = re.compile(r"\[([^\[\]\r\n]+)\]")
_LINE_RANGE_PATTERN = re.compile(r"^(?P<file_path>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?$")


class CitationStatus(str, Enum):
    VERIFIED = "verified"
    UNVERIFIED = "unverified"


@dataclass(frozen=True)
class CitationVerification:
    citation: str
    file_path: str
    start_line: int | None
    end_line: int | None
    status: CitationStatus


def _get_field(result: Any, field_name: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(field_name)
    return getattr(result, field_name, None)


def _parse_citation(citation_text: str) -> tuple[str, int | None, int | None]:
    line_match = _LINE_RANGE_PATTERN.fullmatch(citation_text)
    if line_match is None:
        return citation_text, None, None

    start_line = int(line_match.group("start"))
    end_line = int(line_match.group("end") or start_line)
    return line_match.group("file_path"), start_line, end_line


def verify_citations(
    answer: str,
    retrieved_results: Sequence[Any],
) -> list[CitationVerification]:
    verified_results = []
    for match in _CITATION_PATTERN.finditer(answer):
        citation_text = match.group(1).strip()
        if not citation_text:
            continue

        file_path, cited_start, cited_end = _parse_citation(citation_text)
        matching_chunks = [
            result
            for result in retrieved_results
            if _get_field(result, "file_path") == file_path
        ]

        if cited_start is None:
            is_verified = bool(matching_chunks)
        elif cited_end < cited_start:
            is_verified = False
        else:
            is_verified = any(
                isinstance(chunk_start, int)
                and isinstance(chunk_end, int)
                and cited_start <= chunk_end
                and cited_end >= chunk_start
                for chunk_start, chunk_end in (
                    (
                        _get_field(result, "start_line"),
                        _get_field(result, "end_line"),
                    )
                    for result in matching_chunks
                )
            )

        verified_results.append(
            CitationVerification(
                citation=match.group(0),
                file_path=file_path,
                start_line=cited_start,
                end_line=cited_end,
                status=(
                    CitationStatus.VERIFIED
                    if is_verified
                    else CitationStatus.UNVERIFIED
                ),
            )
        )

    return verified_results