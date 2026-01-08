from __future__ import annotations
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import Literal

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

    print("DOCS1:\n", docs)
    return docs

class Retrieve2Input(BaseModel):
    query: str = Field(
        default = None,
        description=
        """Input query for the retrival task. Must not be empty.
        Will be used for RAG functionality. 
        Based on this input query relevant documents are retrieved"""
                    )
    k: int = Field(
        default = 3,
        description = 
        """Number of chunks to retrieve from the index. Must be at least 3. At most 5.
        Range: 3 - 5.
        If the problem is simple, lower values for k suffice.
        If the problem is complex, higher values should be chosen for k.
        """
    )
    # Having to adjust this using natural language is tiring. The LLM does what it wants.
    # Needs to understand how much context is available to not overshoot the token
    # limit with just the retrieval.

@tool(args_schema=Retrieve2Input)
def retrieve2(
    query: str,
    k: int = 3,
) -> List[Document]:
    """Retrieve the top documents for an user query.

    Parameters
    ----------
    query:
        User query String.
    k:
        Number of chunks to retrieve from the index.
    Returns
    -------
    List[Document]
        Retrieved LangChain Document objects with metadata and content.
    """
    # if(query == None):
    #     return "No query given."
    
    cfg = load_config()
    print("[query] Loaded AppConfig from environment.")

    # failsafe if tool calling mechanism of LLM fails to give proper k-values.
    # Despite a schema and natural language description of values to set
    # most models will still fail to set them correctly
    if(k<cfg.retrieve_k_min):
        print(f"query] k is lower than config min of {cfg.retrieve_k_min}. k is {k}")
        k=cfg.retrieve_k_min
        print(f"query] k was set via config to {cfg.retrieve_k_min}")
    print(f"[query] Retrieving top-{k} chunks for query:")
    print("        ", repr(query))


    vector_store = _load_vector_store(cfg)
    docs = vector_store.similarity_search(query, k=k)

    context_blocks: list[str] = []  
    for i, d in enumerate(docs, start=1): 
        source = d.metadata.get("source", "<unknown>")  
        chunk_index = d.metadata.get("chunk_index", "?")
        header = f"[{i}] Source: {source} | Chunk: {chunk_index}"
        context_blocks.append(f"{header}\n{d.page_content}")

    context_text = "\n\n".join(context_blocks)
    context_text = f"Tool output: {context_text}" # Why does it work more reliable when including this ????????????
                                                  # May need to structure the responses more literally 

    print("DOCS2:\n", context_text)
    return context_text

    # ----------------
    # context_blocks: list[str] = []  
    # for i, d in enumerate(docs, start=1): 
    #     source = d.metadata.get("source", "<unknown>")  
    #     #chunk_index = d.metadata.get("chunk_index", "?")
    #     header = f"[{i}]" # Source: {source}"
    #     context_blocks.append(f"{header}\n{d.page_content}")

    # context_text = "\n\n".join(context_blocks)
    # print("---------------------- context text: ", context_text)
    # return context_text

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
    system_prompt = """\
        Du bist ein hilfreicher Assistent.
        Nutze den folgenden Kontext, um die Frage zu beantworten.
        Wenn die Antwort nicht klar im Kontext vorhanden ist, sag: "Ich weiß es nicht."
        Antworte ausschließlich auf Deutsch.
        """

    user_prompt = f"""\
        Kontext:
        {context_text}

        Frage:
        {query}
        """

    llm_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
        {system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
        {user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    return llm_prompt


def answer(
    query: str,
    k: int = 3,
    #cfg: Optional[AppConfig] = None,
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
    #if cfg is None:
    cfg = load_config()
    print("[query] Loaded AppConfig from environment.")

    print("[query] Using LLM backend:")#, cfg.llm_backend)
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

