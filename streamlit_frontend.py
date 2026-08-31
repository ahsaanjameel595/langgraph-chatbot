import streamlit as st
from langgraph_backend import chatbot, retrive_all_threads
from langchain_core.messages import HumanMessage, AIMessage
import uuid


# -----------------------------
# Generate Thread ID
# -----------------------------

def generate_thread_id():
    return str(uuid.uuid4())


# -----------------------------
# Load Conversation
# -----------------------------

def load_conversation(thread_id):

    state = chatbot.get_state(
        config={
            "configurable": {
                "thread_id": thread_id
            }
        }
    )

    return state.values.get("messages", [])


# -----------------------------
# Reset / New Chat
# -----------------------------

def reset_chat():

    thread_id = generate_thread_id()

    st.session_state["thread_id"] = thread_id
    st.session_state["message_history"] = []


# -----------------------------
# Session State
# -----------------------------

if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "Chat_Threads" not in st.session_state:
    st.session_state["Chat_Threads"] = retrive_all_threads()


# -----------------------------
# Sidebar
# -----------------------------

st.sidebar.title("LangGraph Chatbot")


if st.sidebar.button("New Chat", key="new_chat"):

    reset_chat()

    st.rerun()


st.sidebar.header("My Conversations")


for thread_id in st.session_state["Chat_Threads"][::-1]:

    if st.sidebar.button(
        str(thread_id),
        key=f"thread_{thread_id}"
    ):

        st.session_state["thread_id"] = thread_id

        messages = load_conversation(thread_id)

        temp_message = []

        for message in messages:

            if isinstance(message, HumanMessage):

                role = "user"

            elif isinstance(message, AIMessage):

                role = "assistant"

            else:

                continue

            temp_message.append({
                "role": role,
                "content": message.content
            })

        st.session_state["message_history"] = temp_message

        st.rerun()


# -----------------------------
# Display Previous Messages
# -----------------------------

for message in st.session_state["message_history"]:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# -----------------------------
# Streaming Function
# -----------------------------

def stream_response(user_input, config):

    full_response = ""

    for message_chunk, metadata in chatbot.stream(
        {
            "messages": [
                HumanMessage(content=user_input)
            ]
        },
        config=config,
        stream_mode="messages"
    ):

        # Only process AI messages
        if isinstance(message_chunk, AIMessage):

            content = message_chunk.content

            if content:

                full_response += content

                yield content

    # Debug information
    print("\n========== STREAM COMPLETE ==========")
    print("Total characters:", len(full_response))
    print("=====================================\n")


# -----------------------------
# Chat Input
# -----------------------------

user_input = st.chat_input("Type here...")


if user_input:

    # -------------------------
    # Current Thread
    # -------------------------

    thread_id = st.session_state["thread_id"]

    config = {
        "configurable": {
            "thread_id": thread_id
        },
        'metadata':{
            'thread_id':st.session_state['thread_id']
        },
        'run_name':'chat_turn'
    }


    # -------------------------
    # Display User Message
    # -------------------------

    st.session_state["message_history"].append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):

        st.markdown(user_input)


    # -------------------------
    # Generate Streaming AI Response
    # -------------------------

    with st.chat_message("assistant"):

        ai_message = st.write_stream(
            stream_response(
                user_input,
                config
            )
        )


    # -------------------------
    # Save Complete AI Response
    # -------------------------

    st.session_state["message_history"].append({
        "role": "assistant",
        "content": ai_message
    })


    # -------------------------
    # Refresh Thread List
    # -------------------------

    st.session_state["Chat_Threads"] = retrive_all_threads()
