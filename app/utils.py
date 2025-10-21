"""Utility helpers shared across the EduPolicy agent backend."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from .config import settings

LOGGER = logging.getLogger(__name__)


_embedder = None


def get_embedder() -> SentenceTransformer:
    """Return a lazily instantiated sentence transformer model."""

    global _embedder
    if _embedder is None:
        LOGGER.info("Loading embedding model %s", settings.embedding_model)
        _embedder = SentenceTransformer(settings.embedding_model)
    return _embedder


@dataclass
class DocumentChunk:
    """Lightweight container representing a document chunk."""

    text: str
    metadata: dict


def chunk_text(text: str, *, chunk_size: int | None = None, chunk_overlap: int | None = None) -> List[DocumentChunk]:
    """Split a string into overlapping chunks suitable for embeddings."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
        length_function=len,
    )
    chunks = splitter.split_text(text)
    return [DocumentChunk(text=c, metadata={"source": "regulations.pdf", "chunk": idx}) for idx, c in enumerate(chunks)]


def embed_texts(texts: Sequence[str]) -> List[List[float]]:
    """Generate dense embeddings for provided texts."""

    embedder = get_embedder()
    return embedder.encode(list(texts), normalize_embeddings=True).tolist()


def load_pdf_text(pdf_path: Path) -> str:
    """Extract plain text from a PDF file."""

    from PyPDF2 import PdfReader

    reader = PdfReader(str(pdf_path))
    text_content: List[str] = []
    for page in reader.pages:
        text_content.append(page.extract_text() or "")
    return "\n".join(text_content)


class SessionMemory:
    """Persist conversation history to a JSON file.

    The structure stored on disk is a mapping from session identifier to a list of
    dictionaries with ``role`` and ``content`` keys.  This simple approach keeps
    the memory human readable whilst satisfying the project requirement of JSON
    based memory persistence.
    """

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or settings.session_memory_path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self._write({})

    def _read(self) -> dict:
        with self.path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write(self, payload: dict) -> None:
        with self.path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    def get_history(self, session_id: str) -> List[dict]:
        data = self._read()
        return data.get(session_id, [])

    def append(self, session_id: str, role: str, content: str) -> None:
        data = self._read()
        history = data.setdefault(session_id, [])
        history.append({"role": role, "content": content})
        self._write(data)

    def reset(self, session_id: str) -> None:
        data = self._read()
        if session_id in data:
            del data[session_id]
            self._write(data)


def build_tool_observation(tool_name: str, observation: str, *, max_length: int = 600) -> str:
    """Format a tool observation string for inclusion in reasoning trace.

    Parameters
    ----------
    tool_name:
        Name of the tool invoked by the agent.
    observation:
        Raw textual observation returned by the tool.
    max_length:
        Maximum number of characters to include in the formatted observation to
        prevent excessively long traces in the UI.
    """

    observation = observation.strip()
    if len(observation) > max_length:
        observation = observation[: max_length - 3] + "..."
    return f"[{tool_name}] {observation}"
