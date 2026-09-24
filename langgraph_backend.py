"""LangGraph tool-calling agent with a human-in-the-loop approval gate.

Flow:
    START -> chat_node -> (tool call?) -> human_approval -> tools -> chat_node -> END
                                              |
                                              +-- rejected -> chat_node

Stack: LangGraph + Groq (gpt-oss-120b) + SQLite checkpointing.
"""

import os
import sqlite3
from typing import Annotated, Literal, Optional, TypedDict

import requests
from dotenv import load_dotenv
from simpleeval import simple_eval

from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_groq import ChatGroq

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.types import Command, interrupt


# ============================================================
# 1. CONFIG
# ============================================================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing in the .env file.")
if not ALPHA_VANTAGE_API_KEY:
    raise ValueError("ALPHA_VANTAGE_API_KEY is missing in the .env file.")

DB_PATH = os.getenv("CHAT_DB_PATH", "chatbot.db")
MAX_RETRIES = 2  # retry when the model produces a malformed tool call

SYSTEM_PROMPT = (
    "You are a helpful assistant. Use the calculator for math, "
    "get_stock_price for live stock prices, and the search tool for "
    "current information. Answer in the same language as the user."
)


# ============================================================
# 2. MODEL + TOOLS
# ============================================================

model = ChatGroq(
    model="openai/gpt-oss-120b",
    groq_api_key=GROQ_API_KEY,
    temperature=0,
)

search_tool = DuckDuckGoSearchRun(region="us-en")


@tool
def calculator(expression: str) -> str:
    """Evaluate a math expression, e.g. '45*2+12'."""
    try:
        return str(simple_eval(expression))  # safe evaluator, no arbitrary code
    except Exception as e:
        return f"Calculation error: {e}"


@tool
def get_stock_price(symbol: str) -> dict:
    """Get the latest stock quote for a ticker symbol (AAPL, TSLA, MSFT)."""
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={ALPHA_VANTAGE_API_KEY}"
    )
    try:
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        return {"error": f"API request failed: {e}"}


tools = [calculator, get_stock_price, search_tool]
llm_with_tools = model.bind_tools(tools)


# ============================================================
# 3. STATE
# ============================================================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# ============================================================
# 4. NODES
# ============================================================

def chat_node(state: ChatState):
    prompt = [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]

    for _ in range(MAX_RETRIES):
        try:
            return {"messages": [llm_with_tools.invoke(prompt)]}
        except Exception:
            continue  # malformed tool call / temporary API error -> retry

    return {
        "messages": [
            AIMessage(content="Sorry, something went wrong. Please try again.")
        ]
    }


def human_approval(state: ChatState) -> Command[Literal["tools", "chat_node"]]:
    """Pause the graph and ask the user before any tool runs.

    On resume this node re-runs from the top, so there must be no
    side effects before interrupt().
    """
    tool_calls = state["messages"][-1].tool_calls

    decision = interrupt(
        {
            "question": "Approve these tool calls?",
            "tool_calls": [
                {"name": tc["name"], "args": tc["args"]} for tc in tool_calls
            ],
        }
    )

    if str(decision).strip().lower() in ("y", "yes", "approve"):
        return Command(goto="tools")

    # Rejected: every tool_call_id needs a ToolMessage, otherwise the
    # message history becomes invalid for the model.
    rejections = [
        ToolMessage(
            content="The user rejected this tool call.",
            tool_call_id=tc["id"],
        )
        for tc in tool_calls
    ]
    return Command(goto="chat_node", update={"messages": rejections})


# ============================================================
# 5. GRAPH
# ============================================================

conn = sqlite3.connect(DB_PATH, check_same_thread=False)
checkpointer = SqliteSaver(conn)

graph = StateGraph(ChatState)
graph.add_node("chat_node", chat_node)
graph.add_node("human_approval", human_approval)
graph.add_node("tools", ToolNode(tools))

graph.add_edge(START, "chat_node")
graph.add_conditional_edges(
    "chat_node",
    tools_condition,
    {"tools": "human_approval", END: END},  # approval before every tool run
)
graph.add_edge("tools", "chat_node")
# human_approval routes itself through Command(goto=...)

chatbot = graph.compile(checkpointer=checkpointer)


# ============================================================
# 6. HELPERS (used by the frontend)
# ============================================================

def retrieve_all_threads() -> list[str]:
    """Thread ids, oldest -> newest."""
    ordered = dict.fromkeys(
        c.config["configurable"]["thread_id"] for c in checkpointer.list(None)
    )
    return list(ordered)[::-1]


def get_pending_approval(thread_id: str) -> Optional[dict]:
    """Return the interrupt payload if this thread is waiting for approval.

    Reads from the SQLite checkpoint, so a paused approval survives
    a page refresh or app restart.
    """
    state = chatbot.get_state({"configurable": {"thread_id": thread_id}})
    for task in state.tasks:
        if task.interrupts:
            return task.interrupts[0].value
    return None