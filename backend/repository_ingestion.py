import codecs
import os
import re
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException
from git import Repo
from git.exc import GitError
from pydantic import BaseModel, field_validator

router = APIRouter()

SUPPORTED_LANGUAGES = {
    ".py": "Python",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".java": "Java",
    ".cpp": "C++",
    ".h": "C/C++",
    ".md": "Markdown",
}

IGNORED_DIRECTORIES = {
    ".git",
    "node_modules",
    "venv",
    ".venv",
    "__pycache__",
    "dist",
    "build",
}


class RepositoryIndexRequest(BaseModel):
    github_url: str

    @field_validator("github_url")
    @classmethod
    def validate_github_url(cls, value: str) -> str:
        return validate_github_url(value)


class IndexedFile(BaseModel):
    path: str
    language: str


class RepositoryFile(IndexedFile):
    content: str


class RepositoryIndexResponse(BaseModel):
    repository_name: str
    file_count: int
    files: list[IndexedFile]


def is_binary_file(file_path: Path) -> bool:
    decoder = codecs.getincrementaldecoder("utf-8")()
    try:
        with file_path.open("rb") as file:
            while chunk := file.read(8192):
                if b"\0" in chunk:
                    return True
                decoder.decode(chunk)
            decoder.decode(b"", final=True)
    except (OSError, UnicodeDecodeError):
        return True
    return False


def scan_repository(repository_path: Path) -> list[RepositoryFile]:
    indexed_files = []

    for current_directory, directory_names, file_names in os.walk(repository_path):
        directory_names[:] = [
            name for name in directory_names if name not in IGNORED_DIRECTORIES
        ]

        for file_name in file_names:
            file_path = Path(current_directory) / file_name
            language = SUPPORTED_LANGUAGES.get(file_path.suffix.lower())
            if language is None or is_binary_file(file_path):
                continue

            try:
                with file_path.open("r", encoding="utf-8", newline="") as source_file:
                    content = source_file.read()
            except (OSError, UnicodeDecodeError):
                continue

            indexed_files.append(
                RepositoryFile(
                    path=file_path.relative_to(repository_path).as_posix(),
                    language=language,
                    content=content,
                )
            )

    return sorted(indexed_files, key=lambda indexed_file: indexed_file.path)


def repository_name_from_url(github_url: str) -> str:
    repository_name = urlparse(github_url).path.rstrip("/").split("/")[-1]
    return re.sub(r"\.git$", "", repository_name, flags=re.IGNORECASE)


def validate_github_url(value: str) -> str:
    value = value.strip()
    parsed_url = urlparse(value)
    path_parts = parsed_url.path.strip("/").split("/")

    if (
        parsed_url.scheme not in {"http", "https"}
        or parsed_url.netloc.lower() != "github.com"
        or parsed_url.query
        or parsed_url.fragment
        or len(path_parts) != 2
    ):
        raise ValueError("Enter a GitHub repository URL such as https://github.com/owner/repository")

    owner, repository = path_parts
    repository = re.sub(r"\.git$", "", repository, flags=re.IGNORECASE)
    if (
        not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?", owner)
        or not re.fullmatch(r"[A-Za-z0-9_.-]+", repository)
        or repository in {".", ".."}
    ):
        raise ValueError("Enter a valid GitHub owner and repository name")

    return value.rstrip("/")


def clone_and_scan_repository(github_url: str) -> tuple[str, list[RepositoryFile]]:
    repository_name = repository_name_from_url(github_url)

    with TemporaryDirectory() as temporary_directory:
        clone_path = Path(temporary_directory) / "repository"
        try:
            Repo.clone_from(github_url, clone_path)
        except (GitError, OSError):
            raise HTTPException(
                status_code=400,
                detail="Could not clone the repository. Check that the GitHub URL is correct and the repository is accessible.",
            ) from None

        files = scan_repository(clone_path)

    return repository_name, files


@router.post("/repository/index", response_model=RepositoryIndexResponse)
def index_repository(request: RepositoryIndexRequest) -> RepositoryIndexResponse:
    repository_name, repository_files = clone_and_scan_repository(request.github_url)

    return RepositoryIndexResponse(
        repository_name=repository_name,
        file_count=len(repository_files),
        files=[IndexedFile(path=file.path, language=file.language) for file in repository_files],
    )