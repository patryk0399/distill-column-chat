from __future__ import annotations

"""Data loading utilities for data types layer.
   Loads documents from the local filesystem into simple
   """

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
from pdf2image import convert_from_path
import easyocr


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


def _create_ocr_reader() -> easyocr.Reader:
    """Create and return an EasyOCR reader instance.
    """

    reader = easyocr.Reader(["de", "en"], gpu=False)
    return reader


def _ocr_pdf_to_text(pdf_path: Path, reader: easyocr.Reader) -> str:
    """Run OCR on all pages of a PDF file and return concatenated text.

    Each page is converted to an image and passed through EasyOCR reader. 
    Page texts are joined with blank lines between the pages.
    """

    try:
        images = convert_from_path(str(pdf_path))
    except Exception as exc:  # pragma: no cover - defensive logging branch
        print(f"[data] Error converting PDF to images: {pdf_path}: {exc}")
        return ""

    page_texts: list[str] = []

    for page_number, image in enumerate(images, start=1):
        print(f"[data] OCR page {page_number} of {pdf_path.name}")
        # Convert PIL image to NumPy array for EasyOCR.
        result = reader.readtext(np.array(image), detail=0)

        lines = [line.strip() for line in result if isinstance(line, str) and line.strip()]
        page_texts.append("\n".join(lines))

    return "\n\n".join(page_texts)


def load_scanned_pdf_documents(root: Path) -> List[RawDocument]:
    """Load all .pdf documents from a directory using OCR.

    For scanned PDFs uses EasyOCR on rendered page instead of 
    text extraction via parsing.

    Parameters
    ----------
    root:
        Directory under which PDF documents are searched.

    Returns
    -------
    List[RawDocument]
        List of loaded PDF documents with OCR text content. If the
        directory does not exist, an empty list is returned.
    """

    if not root.exists():
        print(f"[data] Root directory does not exist for PDFs: {root}")
        return []

    pdf_paths = sorted(root.rglob("*.pdf"))
    if not pdf_paths:
        print(f"[data] No PDF documents found under: {root}")
        return []

    reader = _create_ocr_reader()
    docs: List[RawDocument] = []

    for path in pdf_paths:
        print(f"[data] Loading PDF via OCR: {path}")
        content = _ocr_pdf_to_text(path, reader)

        if not content.strip():
            print(f"[data] No OCR text extracted for PDF, skipping: {path}")
            continue

        title = path.stem.replace("_", " ").replace("-", " ").strip()
        docs.append(RawDocument(content=content, path=path, title=title))

    print(f"[data] Loaded {len(docs)} PDF documents from {root}")
    for doc in docs:
        print(f"[data] PDF doc: {doc.path.name}, {len(doc.content)} chars")

    return docs


def load_docx_documents(root: Path) -> List[RawDocument]:
    raise NotImplementedError("DOCX document loading is not implemented yet.")


def load_xlsx_documents(root: Path) -> List[RawDocument]:
    raise NotImplementedError("XLSX document loading is not implemented yet.")


def load_csv_documents(root: Path) -> List[RawDocument]:
    raise NotImplementedError("CSV document loading is not implemented yet.")


__all__ = [
    "RawDocument",
    "load_raw_text_documents",
    "load_scanned_pdf_documents",
    "load_docx_documents",
    "load_xlsx_documents",
    "load_csv_documents",
]
