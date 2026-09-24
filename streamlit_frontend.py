import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import Command

from langgraph_backend import chatbot, get_pending_approval, retrieve_all_threads


# -----------------------------
# Helpers
# -----------------------------

def generate_thread_id() -> str:
    return str(uuid.uuid4())


def make_config(thread_id: str) -> dict:
    return {
        "configurable": {"thread_id": thread_id},
        "metadata": {"thread_id": thread_id},
        "run_name": "chat_turn",
    }


def load_conversation(thread_id: str) -> list[dict]:
    state = chatbot.get_state(make_config(thread_id))
    history = []
    for message in state.values.get("messages", []):
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage) and message.content:
            role = "assistant"  # skip empty AI messages that only hold tool calls
        else:
            continue
        history.append({"role": role, "content": message.content})
    return history


def reset_chat():
    st.session_state["thread_id"] = generate_thread_id()
    st.session_state["message_history"] = []


def stream_response(payload, config):
    """Yield AI text tokens. Payload is a new message or a Command(resume=...)."""
    for chunk, metadata in chatbot.stream(
        payload, config=config, stream_mode="messages"
    ):
        if (
            isinstance(chunk, AIMessage)
            and chunk.content
            and metadata.get("langgraph_node") == "chat_node"
        ):
            yield chunk.content


def stream_and_store(payload, config):
    with st.chat_message("assistant"):
        text = st.write_stream(stream_response(payload, config))
    if text:
        st.session_state["message_history"].append(
            {"role": "assistant", "content": text}
        )


# -----------------------------
# Session state
# -----------------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "Chat_Threads" not in st.session_state:
    st.session_state["Chat_Threads"] = retrieve_all_threads()


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("LangGraph Agent")

if st.sidebar.button("New Chat", key="new_chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

for tid in st.session_state["Chat_Threads"][::-1]:
    if st.sidebar.button(str(tid)[:8], key=f"thread_{tid}"):
        st.session_state["thread_id"] = tid
        st.session_state["message_history"] = load_conversation(tid)
        st.rerun()


# -----------------------------
# Chat history
# -----------------------------

for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# -----------------------------
# Approval gate (shown when the graph is paused on interrupt)
# -----------------------------

thread_id = st.session_state["thread_id"]
config = make_config(thread_id)
pending = get_pending_approval(thread_id)

if pending:
    gate = st.empty()
    with gate.container():
        with st.chat_message("assistant"):
            st.warning("The assistant wants to run these tools. Approve?")
            for tc in pending["tool_calls"]:
                st.code(f"{tc['name']}({tc['args']})", language="python")
            col1, col2 = st.columns(2)
            approve = col1.button(
                "Approve", type="primary", key=f"approve_{thread_id}"
            )
            reject = col2.button("Reject", key=f"reject_{thread_id}")

    if approve or reject:
        gate.empty()
        stream_and_store(Command(resume="yes" if approve else "no"), config)
        st.session_state["Chat_Threads"] = retrieve_all_threads()
        st.rerun()  # the resumed run may pause again on another tool call


# -----------------------------
# Chat input (disabled while an approval is pending)
# -----------------------------

user_input = st.chat_input("Type here...", disabled=pending is not None)

if user_input:
    st.session_state["message_history"].append(
        {"role": "user", "content": user_input}
    )
    with st.chat_message("user"):
        st.markdown(user_input)

    stream_and_store({"messages": [HumanMessage(content=user_input)]}, config)
    st.session_state["Chat_Threads"] = retrieve_all_threads()

    if get_pending_approval(thread_id):
        st.rerun()  # show the approve/reject buttons