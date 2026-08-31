import sqlite3
import os

from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.messages import BaseMessage, AIMessage
from langchain_groq import ChatGroq

from tools import tools  # <-- Calculator, get_stock_price, search_tool


load_dotenv()


# -------------------------
# Model
# -------------------------

model = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=os.getenv("GROQ_API_KEY"),
)

# Bind tools to the model so it can decide when to call them
llm_with_tools = model.bind_tools(tools)


# -------------------------
# State
# -------------------------

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# -------------------------
# SQLite
# -------------------------

conn = sqlite3.connect(
    "langgraph_chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn)


# -------------------------
# Graph
# -------------------------

graph = StateGraph(ChatState)


def ChatNode(state: ChatState):

    messages = state["messages"]

    try:
        response = llm_with_tools.invoke(messages)

    except Exception as e:
        # Groq occasionally sends malformed tool-call JSON mid-stream.
        # Retry once before giving up.
        try:
            response = llm_with_tools.invoke(messages)
        except Exception as e2:
            response = AIMessage(
                content=(
                    "Sorry, kuch technical issue aagaya tool call mein. "
                    "Dobara try karo ya sawal thoda different tarike se pucho."
                )
            )

    return {
        "messages": [response]
    }


tool_node = ToolNode(tools)

graph.add_node("ChatNode", ChatNode)
graph.add_node("tools", tool_node)

graph.add_edge(START, "ChatNode")

# If the model called a tool -> go to "tools", else -> END
graph.add_conditional_edges(
    "ChatNode",
    tools_condition
)

# After tool runs, go back to ChatNode so model can use the result
graph.add_edge("tools", "ChatNode")


chatbot = graph.compile(
    checkpointer=checkpointer
)


# -------------------------
# Get Threads
# -------------------------

def retrive_all_threads():

    all_threads = set()

    for checkpoint in checkpointer.list(None):

        thread_id = checkpoint.config["configurable"]["thread_id"]

        all_threads.add(thread_id)

    return list(all_threads)


if __name__ == "__main__":

    result = chatbot.invoke(
        {
            "messages": [
                ("user", "What's the current stock price of AAPL, and what is 45*3+12?")
            ]
        },
        config={
            "configurable": {
                "thread_id": "test-1"
            }
        }
    )

    print("\n========== FINAL RESULT ==========\n")
    print(result["messages"][-1].content)