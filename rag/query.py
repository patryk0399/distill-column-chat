from __future__ import annotations

"""RAG query helpers.

Prodiving functions that can be reused by other layers.
This module belongs to the context and LLM layers:
- Context layer:
  - Loads FAISS index built in rag.build_index.
  - Retrieves relevant chunks for a given query.
- LLM layer:
  - Builds the RAG-prompt from retrieved chunks with user query.
  - Uses the configured local LLM backend via get_local_llm.
"""

from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import AppConfig, load_config
from src.llm_backend import get_local_llm


def _load_vector_store(cfg: AppConfig) -> FAISS:
    """Load the FAISS vector store for the current configuration.

    The index should to be under:
        data/index/faiss/ (faiss for now)#todo

    retrieve() and answer() can be used for debugging 
    (to test RAG only).
    """
    index_root = Path(cfg.data_dir) / "index" / "faiss"
    print("[query] Using index root:", index_root)

    embeddings = HuggingFaceEmbeddings(model_name=cfg.embedding_model_name)
    print("[query] Initialising embeddings with model:", cfg.embedding_model_name)

    vector_store = FAISS.load_local(
        folder_path=str(index_root),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )

    print("[query] Loaded FAISS vector store.")
    print("[query] Stored chunks:", len(vector_store.docstore._dict))  # type: ignore[attr-defined]

    return vector_store


def retrieve(
    query: str,
    k: int = 3,
    cfg: Optional[AppConfig] = None,
) -> List[Document]:
    """Retrieve the top-k documents for an user query.

    Parameters
    ----------
    query:
        User query String.
    k:
        Number of chunks to retrieve from the index.
    cfg:
        Optional AppConfig. If omitted, load_config() is used.

    Returns
    -------
    List[Document]
        Retrieved LangChain Document objects with metadata and content.
    """
    if cfg is None:
        cfg = load_config()
        print("[query] Loaded AppConfig from environment.")

    print(f"[query] Retrieving top-{k} chunks for query:")
    print("        ", repr(query))


    vector_store = _load_vector_store(cfg)
    docs = vector_store.similarity_search(query, k=k)

    print(f"[query] Retrieved {len(docs)} chunk(s).")
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "<unknown>")
        title = d.metadata.get("title", "<no title>")
        chunk_index = d.metadata.get("chunk_index", "?")
        print(f"[query]  [{i}] source={source} title={title} chunk_index={chunk_index}")

    return docs


def _build_rag_prompt(query: str, docs: List[Document]) -> str:
    """Build a RAG prompt from the user query and retrieved documents."""
    context_blocks: list[str] = []  
    for i, d in enumerate(docs, start=1): 
        source = d.metadata.get("source", "<unknown>")  
        chunk_index = d.metadata.get("chunk_index", "?")
        header = f"[{i}] Source: {source} | Chunk: {chunk_index}"
        context_blocks.append(f"{header}\n{d.page_content}")

    context_text = "\n\n".join(context_blocks)
    print("[query]------------------------------------\n", context_text)
    prompt = ( 
        "Use the context below to answer the question. If the answer is not clearly contained in the context, say that you do not know.\n\n"
        f"Context:\n{context_text}\n\n"  
        f"Question: {query}\n\n" 
        "Answer:" 
    )

    return prompt


def answer(
    query: str,
    k: int = 3,
    cfg: Optional[AppConfig] = None,
) -> str:
    """Answer user prompt using RAG with the configured LLM backend.

    - Loads configuration (if not provided).
    - Loads FAISS vector store.
    - Retrieves top-k relevant chunks for the query.
    - Builds a RAG prompt.
    - Invokes the configured local LLM backend with the prompt.
    - Returns the model's response as a string.

    Parameters
    ----------  
    query:
        User query string. 
    k:
        Number of chunks to retrieve and include as context.
    cfg:
        Optional AppConfig. If omitted, load_config() is used.

    Returns
    -------
    str
        The answer produced by the selected LLM backend.
    """
    if cfg is None:
        cfg = load_config()
        print("[query] Loaded AppConfig from environment.")

    print("[query] Using LLM backend:", cfg.llm_backend)
    docs = retrieve(query=query, k=k, cfg=cfg)

    if not docs:
        print("[query] No documents retrieved. Returning fallback message.")
        return "No relevant context found in the index. Unable to answer."

    prompt = _build_rag_prompt(query, docs)
    print("[query] Built RAG prompt. Prompt length (chars):", len(prompt))

    llm = get_local_llm(cfg)
    print("[query] Invoking LLM backend...")

    result = llm.invoke(prompt)

    # Convert result to string in case the backend returns a non-string object.
    answer_text = str(result)
    print("[query] LLM invocation completed. Answer length (chars):", len(answer_text))

    return answer_text


def main() -> None:
    """ CLI for manual RAG testing.   
    - python -m rag.query
      (then type the request when prompted)
    """
    import sys  

    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = input("Enter query: ").strip()
 
    if not user_query:
        print("[query] Empty query provided. Exiting.")
        return

    print("[query] Question:", repr(user_query))
    answer_text = answer(user_query)
    print("\n=== ANSWER ===")
    print(answer_text) 
 

if __name__ == "__main__":
    main() 


