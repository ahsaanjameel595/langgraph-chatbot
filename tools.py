# from typing import TypedDict, Annotated
# import os
# import requests

# from dotenv import load_dotenv

# from langchain_core.messages import HumanMessage, BaseMessage
# from langchain_core.tools import tool

# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

# from langchain_community.tools import DuckDuckGoSearchRun

# from langgraph.graph import StateGraph, START
# from langgraph.graph.message import add_messages
# from langgraph.prebuilt import ToolNode, tools_condition

from typing import TypedDict,Annotated
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage,BaseMessage
from langchain_core.tools import tool
from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph,START,END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode,tools_condition
import os
import requests

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

# load_dotenv()

# HF_TOKEN = os.getenv("HF_TOKEN")
# ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")


# # ============================================================
# # CHECK API KEYS
# # ============================================================

# if not HF_TOKEN:
#     raise ValueError("HF_TOKEN is missing from .env file")

# if not ALPHA_VANTAGE_API_KEY:
#     raise ValueError("ALPHA_VANTAGE_API_KEY is missing from .env file")


load_dotenv()
HF_TOKEN=os.getenv('HF_TOKEN')
ALPHA_VANTAGE_API_KEY=os.getenv('ALPHA_VANTAGE_API_KEY')
if not HF_TOKEN:
    raise ValueError('hf_token is missing from .env')

if not ALPHA_VANTAGE_API_KEY:
    raise ValueError('ALPHA_VANTAGE_API_KEY is missing in .env file')

# ============================================================
# LLM
# ============================================================

# llm = HuggingFaceEndpoint(
#     repo_id="openai/gpt-oss-120b",
#     task="text-generation",
#     huggingfacehub_api_token=HF_TOKEN,
#     max_new_tokens=4096,
# )
# model = ChatHuggingFace(llm=llm)

llm=HuggingFaceEndpoint(
    repo_id='openai/gpt-oss-120b',
    task='text-generation',
    huggingfacehub_api_token=HF_TOKEN,
    max_new_tokens=4096
)
model=ChatHuggingFace(llm=llm)

# ============================================================
# PRE-BUILT TOOL
# DuckDuckGo Search
# ============================================================

# search_tool = DuckDuckGoSearchRun(
#     region="us-en"
# )


# # ============================================================
# # CUSTOM CALCULATOR TOOL
# # ============================================================

# @tool
# def calculator(expression: str) -> str:
#     """
#     Calculate a mathematical expression.

#     Example:
#     25 * 4 + 10
#     """

#     try:
#         result = eval(expression)
#         return str(result)

#     except Exception as e:
#         return f"Calculation error: {str(e)}"


# # ============================================================
# # CUSTOM TOOL
# # Alpha Vantage Stock Price
# # ============================================================

# @tool
# def get_stock_price(symbol: str) -> dict:
#     """
#     Get the latest stock price for a stock symbol
#     using Alpha Vantage API.

#     Examples:
#     AAPL = Apple
#     TSLA = Tesla
#     MSFT = Microsoft
#     """

#     url = (
#         "https://www.alphavantage.co/query"
#         f"?function=GLOBAL_QUOTE"
#         f"&symbol={symbol}"
#         f"&apikey={ALPHA_VANTAGE_API_KEY}"
#     )

#     try:
#         response = requests.get(url, timeout=10)

#         response.raise_for_status()

#         data = response.json()

#         return data

#     except requests.RequestException as e:
#         return {
#             "error": f"API request failed: {str(e)}"
#         }


# # ============================================================
# # ALL TOOLS
# # ============================================================

# tools = [
#     search_tool,
#     calculator,
#     get_stock_price
# ]

search_tool=DuckDuckGoSearchRun(region='us-en')
@tool
def Calculator(expression:str)->str:
    """Solve the mathemical expressions
    example= 45*2+12
    """
    try:
        result=eval(expression)
        return str(result)
    except Exception as e:
        return f'calculation error: {str(e)}'

@tool
def get_stock_price(symbol:str)->dict:
    """
    Get the latest stock price using alpha vantage api key
    example:
    AAPL=APPLE,
    TSLA=TESLA,
    MSFT=MICROSOFT
    """
    url = (
            "https://www.alphavantage.co/query"
            f"?function=GLOBAL_QUOTE"
            f"&symbol={symbol}"
            f"&apikey={ALPHA_VANTAGE_API_KEY}"
        )
    try:
        response=requests.get(url,timeout=15)
        response.raise_for_status()
        data=response.json()
        return data

    except requests.RequestException as e:
        return  {
            'error': f'API request failed {str(e)}'
        }
tools=[Calculator,get_stock_price,search_tool]
llm_with_tools=model.bind_tools(tools)
# ============================================================
# BIND TOOLS TO MODEL
# ============================================================

# llm_with_tools = model.bind_tools(tools)


# ============================================================
# STATE
# ============================================================
class ChatState(TypedDict):
    messages:Annotated[list[BaseMessage],add_messages]
def Chat_Node(state:ChatState):
    messages=state['messages']
    response=llm_with_tools.invoke(messages)
    return {'messages':[response]}

graph=StateGraph(ChatState)
tool_node = ToolNode(tools)
graph.add_node('Chat_Node', Chat_Node)
graph.add_node('tools', tool_node)
graph.add_edge(START, 'Chat_Node')
graph.add_conditional_edges(
    'Chat_Node',
    tools_condition
)
graph.add_edge(
    'tools',
    'Chat_Node'
)
chatbot=graph.compile()
if __name__ == '__main__':
    result=chatbot.invoke(
        {
            'messages':[HumanMessage(content='78+12-45+87/2')]
        }
    )
    print(result['messages'][-1].content)


# class ChatState(TypedDict):
#     messages: Annotated[
#         list[BaseMessage],
#         add_messages
#     ]


# # ============================================================
# # CHAT NODE
# # ============================================================

# def chat_node(state: ChatState):

#     messages = state["messages"]

#     response = llm_with_tools.invoke(messages)

#     return {
#         "messages": [response]
#     }


# # ============================================================
# # TOOL NODE
# # ============================================================

# tool_node = ToolNode(tools)


# # ============================================================
# # CREATE GRAPH
# # ============================================================

# graph = StateGraph(ChatState)


# # Add nodes
# graph.add_node("chat", chat_node)
# graph.add_node("tools", tool_node)


# # ============================================================
# # EDGES
# # ============================================================

# # START → Chat
# graph.add_edge(
#     START,
#     "chat"
# )


# # Chat → Tool OR END
# graph.add_conditional_edges(
#     "chat",
#     tools_condition
# )


# # Tool → Chat
# graph.add_edge(
#     "tools",
#     "chat"
# )


# # ============================================================
# # COMPILE GRAPH
# # ============================================================

# chatbot = graph.compile()


# # ============================================================
# # TEST
# # ============================================================

# if __name__ == "__main__":

#     result = chatbot.invoke(
#         {
#             "messages": [
#                 HumanMessage(
#                     content="Search the web and tell me the latest news about OpenAI."
#                 )
#             ]
#         }
#     )

#     print("\n==============================")
#     print("FINAL RESPONSE")
#     print("==============================\n")

#     print(result["messages"][-1].content)