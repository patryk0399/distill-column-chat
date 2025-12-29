from __future__ import annotations

import os
from typing import Annotated, Literal

from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from src.config import AppConfig, load_config
from src.llm_backend import get_local_llm
from agents.tools import search as search_function

# --- Debug helpers -----------------------------------------------------------
DEBUG_FLOW = os.getenv("DEBUG_FLOW", "1") == "1"


def _dbg(*parts):
    if DEBUG_FLOW:
        print("[dbg]", *parts)


def _msg_brief(msg: BaseMessage) -> str:
    t = type(msg).__name__
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        snippet = content.replace("\n", " ")
        if len(snippet) > 140:
            snippet = snippet[:140] + "…"
    else:
        snippet = str(content)
    tc = getattr(msg, "tool_calls", None)
    if tc:
        return f"{t}(tool_calls={len(tc)}) :: {snippet}"
    return f"{t} :: {snippet}"


def _dump_messages(messages: list[BaseMessage], label: str = "messages") -> None:
    _dbg(f"{label}: count={len(messages)}")
    if not DEBUG_FLOW:
        return
    for i, m in enumerate(messages):
        _dbg(f"  [{i}]", _msg_brief(m))
        tc = getattr(m, "tool_calls", None) or []
        for j, call in enumerate(tc):
            _dbg(
                f"     tool_call[{j}] name={call.get('name')} args={call.get('args')} id={call.get('id')}"
            )


# ---------------------------------------------------------------------------

# Example embeddings + FAISS store loader (adjust to your project paths)
embeddings = HuggingFaceEmbeddings(model_name=os.getenv("EMBEDDINGS_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
index_root = os.getenv("FAISS_INDEX_ROOT", "./faiss_index")


@tool
def get_vector_store(query: str, k: int = 4) -> str:
    """Search the local FAISS vector store and return the top-k documents (as a string)."""
    vector_store = FAISS.load_local(
        folder_path=str(index_root),
        embeddings=embeddings,
        allow_dangerous_deserialization=True,
    )
    docs = vector_store.similarity_search(query, k=k)
    return "\n\n".join([f"[doc{i}] {d.page_content}" for i, d in enumerate(docs)])


tools = [search_function, get_vector_store]
tool_node = ToolNode(tools)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _choose_forced_tool(last_user_text: str) -> str | None:
    """Router (concept), replace later.
    must force `tool_choice`.
    """
    t = (last_user_text or "").lower()

    if any(x in t for x in ["search"]):
        return "search"

    if any(x in t for x in ["document", "docs", "context"]):
        return "get_vector_store"

    return None


def chatbot_node(state: AgentState, llm) -> AgentState:
    """answer directly OR force a tool call (ChatLlamaCpp)."""
    _dbg("ENTER chatbot_node")
    _dump_messages(state["messages"], label="state.messages (incoming)")

    system = SystemMessage(
        content=(
            "You are a helpful assistant. "
            "If a tool is forced, call it using the model's native tool-calling mechanism. "
            "Otherwise answer normally."
        )
    )
    messages = [system, *state["messages"]]

    # Decide whether to force a tool call
    last_human = next((m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None)
    forced_tool = _choose_forced_tool(last_human.content if last_human else "")

    if forced_tool:
        _dbg("chatbot_node: forcing tool_choice =", forced_tool)
        # Force one tool because of ChatLlamaCpp limitations
        llm_with_tools = llm.bind_tools(
            tools,
            tool_choice={"type": "function", "function": {"name": forced_tool}},
        )
        response = llm_with_tools.invoke(messages)
    else:
        _dbg("chatbot_node: no tool forced; answering directly")
        response = llm.invoke(messages)

    _dbg("EXIT chatbot_node")
    _dbg("llm response type=", type(response).__name__)
    _dbg("llm response tool_calls=", getattr(response, "tool_calls", None))
    _dbg("llm response content_snippet=", (response.content or "").replace("\n", " ")[:200])
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    tc = getattr(last, "tool_calls", None)
    decision = "tools" if tc else "end"
    _dbg("should_continue:", decision, "| last=", _msg_brief(last))
    return decision


def build_graph(memory):
    cfg = load_config()
    llm = get_local_llm(cfg=cfg)

    _dbg("build_graph:", "model_path=", getattr(cfg, "llm_model_path", None))
    _dbg("build_graph:", "tools=", [getattr(t, "name", type(t).__name__) for t in tools])

    graph_builder = StateGraph(AgentState)

    from functools import partial
    graph_builder.add_node("chatbot", partial(chatbot_node, llm=llm))
    graph_builder.add_node("tools", tool_node)

    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_conditional_edges("chatbot", should_continue, {"tools": "tools", "end": END})
    graph_builder.add_edge("tools", "chatbot")

    return graph_builder.compile(checkpointer=memory)


def chat_with_memory(message: str, graph, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {"messages": [HumanMessage(content=message)]}

    _dbg("chat_with_memory:", f"thread_id={thread_id}")
    _dump_messages(initial_state["messages"], label="initial_state.messages")

    result = graph.invoke(initial_state, config)
    _dbg("chat_with_memory: graph.invoke returned")
    _dump_messages(result["messages"], label="result.messages")

    last = result["messages"][-1]
    print("AI:", last.content)


def main() -> None:
    memory = MemorySaver()
    graph = build_graph(memory)

    while True:
        user_input = input("User: ")
        if user_input.strip() == ":q":
            break
        chat_with_memory(user_input, graph, "thread-1")


if __name__ == "__main__":
    main()