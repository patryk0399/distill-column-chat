from __future__ import annotations

"""Data loading utilities for data types layer.
   Loads documents from the local filesystem into simple
   """

from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class RawDocument:
    """Represents a document via a class. 
    Wrapper for retrieved documents.

    content: Full text content (for now)= of the document.
    path: Filesystem path of document.
    title: Short title taken from the filename. 
    #todo later make sure documents are named with title as file name
    """

    content: str
    path: Path
    title: str


def load_raw_text_documents(root: Path) -> List[RawDocument]:
    """Load all .md and .txt documents from given directory.

    Only files ending in .md or .txt are considered. Subdirectories are
    traversed recursively.

    Parameters
    ----------
    root:
        Directory under which documents are searched (e.g. data/docs/).

    Returns
    -------
    List[RawDocument]
        List of loaded documents. If the directory does not exist, an
        empty list is returned.
    """
    if not root.exists():
        print(f"[data] Root directory does not exist: {root}")
        return []

    docs: List[RawDocument] = []

    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {".md", ".txt"}:
            continue

        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Skip files that cannot be decoded as UTF-8.
            print(f"[data] Skipping non-text file: {path}")
            continue

        title = path.stem.replace("_", " ").replace("-", " ").strip()
        docs.append(RawDocument(content=content, path=path, title=title))

    print(f"[data] Loaded {len(docs)} documents from {root}")
    for doc in docs:
        print(f"[data] Doc: {doc.path.name}, {len(doc.content)} chars")
    return docs

def load_pdf_documents(root: Path) -> List[RawDocument]:
    raise NotImplementedError("PDF document loading is not implemented yet.")


def load_docx_documents(root: Path) -> List[RawDocument]:
    raise NotImplementedError("DOCX document loading is not implemented yet.")


def load_xlsx_documents(root: Path) -> List[RawDocument]:
    raise NotImplementedError("XLSX document loading is not implemented yet.")


def load_csv_documents(root: Path) -> List[RawDocument]:
    raise NotImplementedError("CSV document loading is not implemented yet.")


__all__ = [
    "RawDocument",
    "load_raw_text_documents",
    "load_pdf_documents",
    "load_docx_documents",
    "load_xlsx_documents",
    "load_csv_documents",
]
