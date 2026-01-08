from __future__ import annotations

"""Index-building tools for context layer.

This builds a local vector index from documents (raw text and PDFs with OCR processing).
Note: belongs to the "context layer" in the framework: it turns data-layer RawDocument objects into
chunked and embedded representations stored in the vector store.
"""

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
# from transformers import AutoTokenizer, AutoModel
# from sentence_transformers import SentenceTransformer

from data.loading import RawDocument, load_raw_text_documents, load_scanned_pdf_documents
from src.config import AppConfig, load_config


def _build_text_chunks(
    docs: Sequence[RawDocument],
    chunk_size: int = 750, 
    chunk_overlap: int = 150,
) -> Tuple[List[str], List[Dict[str, str]]]:
    """Split documents into smaller text chunks with metadata.

    Parameters
    ----------
    docs:
        RawDocument instances loaded from the data layer.
    chunk_size:
        (Approximate ) maximum number of characters per chunk.
    chunk_overlap:
        Number of characters of overlap between consecutive chunks.

    Returns
    -------
    texts:
        List of chunk strings.
    metadatas:
        List of metadata dicts aligned with texts.
    """

    if not docs:
        print("[index] No documents provided for chunking.")
        return [], []

    cfg = load_config()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.index_chunk_size,
        chunk_overlap=cfg.index_chunk_overlap,
    )

    texts: List[str] = []
    metadatas: List[Dict[str, str]] = []

    for doc in docs:
        chunks = splitter.split_text(doc.content)
        if not chunks:
            continue

        for i, chunk in enumerate(chunks):
            texts.append(chunk)
            metadatas.append(
                {
                    "source": str(doc.path),
                    "title": doc.title,
                    "chunk_index": str(i),
                }
            )

    print(f"[index] Created {len(texts)} chunks from {len(docs)} documents.")
    return texts, metadatas


def build_index(cfg: AppConfig) -> None:
    """Build a FAISS(for now for testing) index from raw text and PDF documents.

    This function reads documents from data/docs/, splits them into chunks, 
    embeds them using a local HuggingFace sentence-transformer model (for now, may change later), and stores
    the resulting index under data/index/faiss/.
    """
    docs_root = Path(cfg.data_dir) / "docs"
    index_root = Path(cfg.data_dir) / "index" / "faiss"

    print("[index] Using docs root:", docs_root)
    print("[index] Using index root:", index_root)

    raw_text_docs = load_raw_text_documents(docs_root)
    pdf_docs = load_scanned_pdf_documents(docs_root)

    total_docs = len(raw_text_docs) + len(pdf_docs)
    if total_docs == 0:
        print("[index] No documents found. Nothing to index.")
        return

    print(f"[index] Loaded {len(raw_text_docs)} text documents and {len(pdf_docs)} PDF documents.")
    raw_docs = raw_text_docs + pdf_docs

    texts, metadatas = _build_text_chunks(raw_docs)
    if not texts:
        print("[index] No chunks produced from documents. Nothing to index.")
        return

    print("[index] Initialising embeddings with model:", cfg.embedding_model_name)
    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model_name)
    # NOTE: if the Embedding-Model itself DOES NOT normalise the embeddings automatically
    # (all-MiniLM-L12-v2 has a Normalise() module), we would need to do it here before storing
    print("[============embeddings============]", embeddings)

    print("[index] Building FAISS index from text chunks...")
    vector_store = FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas)
    # later replace from_texts with better implementation

    index_root.mkdir(parents=True, exist_ok=True)
    print("[index] Saving FAISS index to:", index_root)
    vector_store.save_local(str(index_root))

    print("[index] Index build completed successfully.")


def main() -> None:
    """CLI entrypoint for building the index (qucik debugging)."""
    cfg = load_config()
    print()
    # print(f"[============model_max_length===================] {AutoTokenizer.from_pretrained(cfg.embedding_model_name).model_max_length}")
    #model = SentenceTransformer(cfg.embedding_model_name)
    # print(f"[============max_seq_length=====================] {model.max_seq_length}")
    # print(f"[============max_position_embeddings============] {model[0].auto_model.config.max_position_embeddings}")
    build_index(cfg)
    


if __name__ == "__main__":
    main()