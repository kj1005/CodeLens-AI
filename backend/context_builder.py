from collections.abc import Mapping, Sequence
from typing import Any

DEFAULT_MAX_CONTEXT_CHARS = 16_000
_ENTRY_SEPARATOR = "\n\n"


def _get_field(result: Any, field_name: str) -> Any:
    if isinstance(result, Mapping):
        return result.get(field_name)
    return getattr(result, field_name, None)


class RAGContextBuilder:
    def __init__(self, max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS) -> None:
        if max_context_chars <= 0:
            raise ValueError("max_context_chars must be greater than zero")
        self.max_context_chars = max_context_chars

    def build_context(self, results: Sequence[Any]) -> str:
        entries = []
        for result in results:
            code = _get_field(result, "code")
            if code is None:
                code = _get_field(result, "document")

            file_path = _get_field(result, "file_path")
            start_line = _get_field(result, "start_line")
            end_line = _get_field(result, "end_line")
            language = _get_field(result, "language")
            if not isinstance(code, str):
                raise ValueError("Each result must contain code or document text")
            if not isinstance(file_path, str) or not isinstance(language, str):
                raise ValueError("Each result must include file_path and language metadata")
            if start_line is None or end_line is None:
                raise ValueError("Each result must include start_line and end_line metadata")

            entry = (
                f"[File: {file_path} | Lines: {start_line}-{end_line} | "
                f"Language: {language}]\n{code}"
            )
            proposed_context = _ENTRY_SEPARATOR.join([*entries, entry])
            if len(proposed_context) > self.max_context_chars:
                if not entries:
                    raise ValueError(
                        "A single context entry exceeds max_context_chars"
                    )
                break
            entries.append(entry)

        return _ENTRY_SEPARATOR.join(entries)