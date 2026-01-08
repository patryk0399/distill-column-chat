from __future__ import annotations

import os
from typing import Annotated, Literal
from pprint import pformat
from datetime import datetime

from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
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
from rag.query import retrieve2 
from agents.prompts import GENERAL_SYSTEM_PROMPT
from agents.prompts import MANAGER_AGENT_SYSTEM_PROMPT

tools = [search_function, retrieve2]
tool_node = ToolNode(tools)


class AgentState(TypedDict):
    """Class that handels the list of messages 
    stored for the purpose of node navigation.
    Recommended: TypedDict format.
    add_messages is a reducer that automatically appends
    incoming messages and does not overwrite them."""
    messages: Annotated[list[BaseMessage], add_messages]
    #current_time: str(datetime.now())


# -----------------------------
# Debug helpers (messages -> dicts -> llama.cpp prompt)
# -----------------------------

def _preview(text: object, limit: int = 220) -> str:
    """Safe, single-line preview for logs."""
    try:
        s = "" if text is None else str(text)
    except Exception as e:
        return f"<unprintable: {e}>"
    s = s.replace("\n", "\\n")
    return (s[:limit] + "…") if len(s) > limit else s


def _dump_one_message(msg: BaseMessage, idx: int) -> None:
    """Print everything that matters for tool/debug flows."""
    cls = type(msg).__name__
    content = getattr(msg, "content", "")
    addk = getattr(msg, "additional_kwargs", {}) or {}
    rmeta = getattr(msg, "response_metadata", {}) or {}

    tool_calls = getattr(msg, "tool_calls", None)
    invalid_tool_calls = getattr(msg, "invalid_tool_calls", None)
    tool_call_id = getattr(msg, "tool_call_id", None)
    name = getattr(msg, "name", None)
    mid = getattr(msg, "id", None)

    print(
        f"[debug]  [{idx}] cls={cls} id={mid} name={name} "
        f"tool_call_id={tool_call_id} content_len={len(str(content))}"
    )
    print(f"[debug]      content_preview='{_preview(content)}'")

    if addk:
        print(f"[debug]      additional_kwargs={pformat(addk)}")
    if rmeta:
        print(f"[debug]      response_metadata={pformat(rmeta)}")
    if tool_calls:
        print(f"[debug]      tool_calls={pformat(tool_calls)}")
    if invalid_tool_calls:
        print(f"[debug]      invalid_tool_calls={pformat(invalid_tool_calls)}")

    # Show the exact dict that ChatLlamaCpp sends to llama-cpp-python
    try:
        from langchain_community.chat_models.llamacpp import _convert_message_to_dict  # type: ignore

        as_dict = _convert_message_to_dict(msg)
        print(f"[debug]      as_llamacpp_dict={pformat(as_dict)}")
    except Exception as e:
        print(f"[debug]      as_llamacpp_dict=<unavailable: {type(e).__name__}: {e}>")


def dump_messages(messages: list[BaseMessage], label: str, tail: int = 30) -> None:
    """Dump last N messages with tool-call linkage details."""
    total = len(messages)
    start = max(0, total - tail)
    print(
        f"\n[debug] ===== dump_messages label='{label}' total={total} tail={tail} showing=[{start}..{total-1}] ====="
    )
    for i, m in enumerate(messages[start:], start=start):
        _dump_one_message(m, i)
    print(f"[debug] ===== end dump_messages label='{label}' =====\n")

def _choose_forced_tool(last_user_text: str) -> str | None:
    """Router (concept), replace later.
    must force `tool_choice`.
    """
    
    print(f"[router] Entering forced tool selection with last_user_text='DUMMY'.")
    
    t = (last_user_text or "").lower()

    #------------------
    #todo make the whole thing into try catch and run until correct format is outputed. 
    #idea: if on first try the format is wrong then try generating again

    #todo implement here the management agent who decides what tools to call.
    # input: user query
    # output: list of tools to call based on the user query. can be size >= 0.
    # print("================================ TESTING TOOLS TO CALL ============================ ")
    # cfg = load_config()
    # llm = get_local_llm(cfg=cfg)
    # decide_tools = HumanMessage(content=MANAGER_AGENT_SYSTEM_PROMPT.format(user_query=last_user_text))
    # tools_to_call = llm.invoke([decide_tools])
    # print("================================ TESTING TOOLS TO CALL ============================ ", tools_to_call)
    # separators = set(",")
    # tool_content = tools_to_call.content
    # if(any(ch in separators for ch in tool_content)):
    #     list_of_tools_to_call= [tool.strip() for tool in tool_content.split(',')]
    #     print("================================ TESTING TOOLS TO CALL ============================ ", list_of_tools_to_call)
    
    
    #todo: check format of tools_to_call
    #------------------

    if any(x in t for x in ["search"]):
        print("[router] Forced tool selected: 'search'.")
        return "search"

    if any(x in t for x in ["document", "docs", "context"]):
        print("[router] Forced tool selected: 'retrieve2'.")
        return "retrieve2"

    print("[router] No forced tool selected.")
    return None

def chatbot_node(state: AgentState) -> AgentState:
    """answer directly OR force a tool call (ChatLlamaCpp)."""
    
    print(f"[chatbot] Entering chatbot_node with messages_in_state={len(state.get('messages', []))}.")
    system = SystemMessage(
        content=(GENERAL_SYSTEM_PROMPT)
        
    )

    messages = [system, *state["messages"]]
    dump_messages(messages, label="chatbot_node.input")

    last_msg = state["messages"][-1] if state.get("messages") else None
    print(f"[chatbot] Last message type='{type(last_msg).__name__}' content_preview='{getattr(last_msg, 'content', '')}'.")

    """
    Important considerations when coming back from ToolNode with tool results:
    Some chat templates do not trigger correct handling of ' role="tool" '.
    _convert_message_to_dict() may drop ToolMessage.name in this case.
    So the tool results must be parsed as plain text and augmented
    with an explicit structure (see HumanMessage(content="Tool ..." below)
    and tool results must be parsed as HumanMessage instead.
    """
    # Here am Error may occur if the chat_format breaks this code artefakt
    # This may inject "None" into the MessageList and this may break the LLM response via invoke
    if isinstance(last_msg, ToolMessage): 
        tool_name = getattr(last_msg, "name", None) or "tool"
        tool_payload = getattr(last_msg, "content", "")
        # It seems that not the BaseMessage type (ToolMessage) is what makes the model use the tool's content
        # but rather parsing tool's content as HumanMessage
        messages = [*messages[:-1], HumanMessage(content=f"Tool `{tool_name}` returned:\n{tool_payload}")]

    # Decide whether to force a tool call.
    # IMPORTANT: only force tool_choice on the first try (when the latest message is Human, avoids infinite call loopps).
    forced_tool: str | None = None
   
    if isinstance(last_msg, HumanMessage) or isinstance(last_msg, AIMessage):
        forced_tool = _choose_forced_tool(last_msg.content)
    print(f"[chatbot] forced_tool='{forced_tool}'.")

    print("[config] Loading app config.")
    cfg = load_config()
    print(f"[config] Loaded config. llm_model_path='{cfg.llm_model_path}'.")
    print(
        "[env] "
        f"LLM_CONTEXT_WINDOW={os.getenv('LLM_CONTEXT_WINDOW')} "
        f"LLM_N_GPU_LAYERS={os.getenv('LLM_N_GPU_LAYERS')} "
        f"LLM_N_THREADS={os.getenv('LLM_N_THREADS')} "
        f"LLM_N_BATCH={os.getenv('LLM_N_BATCH')}"
    )
    print(
        "[config.types] "
        f"llm_context_window={getattr(cfg, 'llm_context_window', None)}(type={type(getattr(cfg, 'llm_context_window', None)).__name__}) "
        f"llm_n_gpu_layers={getattr(cfg, 'llm_n_gpu_layers', None)}(type={type(getattr(cfg, 'llm_n_gpu_layers', None)).__name__}) "
        f"llm_n_threads={getattr(cfg, 'llm_n_threads', None)}(type={type(getattr(cfg, 'llm_n_threads', None)).__name__}) "
        f"llm_n_batch={getattr(cfg, 'llm_n_batch', None)}(type={type(getattr(cfg, 'llm_n_batch', None)).__name__})"
    )
    
    print(f"[llm] Initialising llama.cpp backend with model_path='{cfg.llm_model_path}'.")
    llm = get_local_llm(cfg=cfg)
    print(f"[llm] Backend initialised. llm_type='{type(llm).__name__}'.")
    mk = getattr(llm, "model_kwargs", None)
    if mk is not None:
        print(f"[llm] model_kwargs={pformat(mk)}")

    client = getattr(llm, "client", None)
    if client is not None:
        print(f"[llm] client_type={type(client).__name__}")
        if hasattr(client, "chat_format"):
            print(f"[llm] client.chat_format={getattr(client, 'chat_format')}")
        else:
            print("[llm] client.chat_format=<missing>")
    else:
        print("[llm] client=<missing>")

    print(f"[chatbot] ====================================== last_msg_type='{type(last_msg).__name__}'")
    
    # todo
    # for tool in forced_tools:
    if forced_tool and state:
        print(f"[chatbot] Forcing tool call via tool_choice name='{forced_tool}'.")
        # Force one tool because of ChatLlamaCpp limitations
        llm_with_tools = llm.bind_tools(
            tools,
            tool_choice={"type": "function", "function": {"name": forced_tool}},
        )
        dump_messages(messages, label=f"chatbot_node.before_invoke(forced_tool={forced_tool})")
        response = llm_with_tools.invoke(messages)
        # HERE the model can fail to parse the correct arguments even though a schema is provided in the tool description.
        # e.g it may pass 'k = 100' but ommits 'query' entirely, making retrieval not possible since there is no query to retrieve for.
        # This is a common occurence when using tools with LLMs.

        print(f"[chatbot] LLM returned after forced tool attempt. response_type='{type(response).__name__}'.")
        print("--------------")
        print(response)
        print("--------------")
    else:
        print(f"[chatbot] Invoking LLM without forced tool. messages_total={len(messages)}.")
        dump_messages(messages, label="chatbot_node.before_invoke(no_forced_tool)")

        # IMPORTANT:
        # When we are NOT trying to elicit a tool call, do not bind tools via llm.bind_tools().
        # For messages with prior tool calls the model does not need its tools to fulfill the user/prompt request.
        # Binding tools can re-inject tool schemas and keep the model in a tool-calling mode (endless loop).
        # This call is used both for normal answers and for responding after tool results.
        response = llm.invoke(messages)
        print(f"[chatbot] LLM invoke completed. response_type='{type(response).__name__}'.")

    print(f"[chatbot] Response received. type='{type(response).__name__}'.")
    dump_messages([response], label="chatbot_node.response", tail=1)
    if(type(response).__name__ == "AIMessage"):
        print("[GET FINISH REASON AFTER LLM.INVOKE] ", response.response_metadata["finish_reason"])# state["messages"][0][0])#[0]["finish_reason"])

    tc = getattr(response, "tool_calls", None)
    print(f"[chatbot] tool_calls_present={bool(tc)} tool_calls_count={(len(tc) if tc else 0)}.")
    
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last = state["messages"][-1]
    tc = getattr(last, "tool_calls", None)
    if tc:
        print(f"[router] tool_calls_detail={pformat(tc)}")
    addk = getattr(last, "additional_kwargs", {}) or {}
    if addk:
        print(f"[router] last.additional_kwargs={pformat(addk)}")
    decision = "tools" if (isinstance(last, AIMessage) and tc) else "end"
    
    print(f"[router] should_continue decision='{decision}' last_type='{type(last).__name__}' tool_calls_present={bool(tc)}.")
    return decision


def build_graph(memory):
    print(f"[graph] Building graph with checkpointer_type='{type(memory).__name__}'.")
    
    cfg = load_config()
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("chatbot", chatbot_node)
    print("[graph] Added node: 'chatbot'.")
    graph_builder.add_node("tools", tool_node)
    print(f"[graph] Added node: 'tools' with tools_count={len(tools)} tools={tools}.")

    graph_builder.add_edge(START, "chatbot")
    print("[graph] Added edge: START -> 'chatbot'.")
    graph_builder.add_conditional_edges("chatbot", should_continue, {"tools": "tools", "end": END})
    print("[graph] Added conditional edges from 'chatbot' -> ('tools' | END).")
    graph_builder.add_edge("tools", "chatbot")
    print("[graph] Added edge: 'tools' -> 'chatbot'.")

    graph = graph_builder.compile(checkpointer=memory)
    print(f"[graph] Graph compiled. graph_type='{type(graph).__name__}'.")
    return graph


def chat_with_memory(message: str, graph, thread_id: str):
    print(f"[memory] Entering chat_with_memory thread_id='{thread_id}' message_preview='{(message or '')[:120]}'.")
    
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": 10}
    print(f"[memory] Invoke config: thread_id='{thread_id}' recursion_limit={config.get('recursion_limit')}. ")
    initial_state = {"messages": [HumanMessage(content=message)]}

    print(f"[memory] Invoking graph. initial_state_messages={len(initial_state.get('messages', []))}.")
    result = graph.invoke(initial_state, config)
    print(f"[memory] Graph invoke finished. result_messages={len(result.get('messages', []))}.")
    dump_messages(result.get("messages", []), label="graph.result_state", tail=30)

    last = result["messages"][-1]
    print("[AI]:\n", last.content)

def main() -> None:
    memory = MemorySaver()
    graph = build_graph(memory)

    while True:
        print("[pipeline] Waiting for user input.")
        user_input = input("User: ")
        if user_input.strip() == ":q":
            break
        print("[pipeline] Giving message to chat_with_memory.")
        chat_with_memory(user_input, graph, "thread-1")

if __name__ == "__main__":
    main()