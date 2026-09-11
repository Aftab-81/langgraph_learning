from langgraph.graph import START, END, StateGraph
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode , tools_condition
from langchain_core.tools import tool
from langgraph.checkpoint.sqlite import SqliteSaver
import sqlite3
from langchain_community.tools import DuckDuckGoSearchRun

from dotenv import load_dotenv
import requests
import random
import os

load_dotenv()

# 2
llm_1 = HuggingFaceEndpoint(
    repo_id="zai-org/GLM-4.5",  #openai/gpt-oss-120b
    task="text-generation",
    huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"]
)

llm = ChatHuggingFace(llm=llm_1)

# 3
# 1
search_tool = DuckDuckGoSearchRun()

# 2
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
        Performs basic mathematical operation on two numbers.
        Supported operations: add, sub, mul, div.
    """
    try:
        if operation == "add":
            return first_num + second_num
        elif operation == "sub":
            return first_num - second_num
        elif operation == "mul":
            return first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "ZeroDivisionError"}
            return first_num + second_num
        else:
            return {"error": f"Invalid Operation: {operation}"}
    except Exception as e:
        return {"error": str(e)}

# 3
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for the given symbol (eg, 'AAPL', 'TSLA')
    using Alpha Vantage API key in the URL
    """
    url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey=97XR5TLBCBD2YAJC"
    response = requests.get(url)
    return response.json()

# 4

# Make a tool list
tools = [search_tool, calculator, get_stock_price]

# Bind tools with LLM
llm_with_tools = llm.bind_tools(tools = tools)

# 5
# Create a state
class ChatState(TypedDict):
    # add_messages -> Basically to add HumanMessage, AIMessage, ... in the list
    messages: Annotated[list[BaseMessage], add_messages]   

# 6
# Graph node
def chat_node(state: ChatState) -> ChatState:
    message = state["messages"]
    llm_response = llm_with_tools.invoke(message)
    return {"messages": [llm_response]}

tool_node = ToolNode(tools = tools)

# 7 Checkpoiner

conn = sqlite3.connect("chatbot.db", check_same_thread = False)
checkpointer = SqliteSaver(conn = conn)

# 8
graph = StateGraph(ChatState)
graph.set_entry_point("chat_node")

# Add nodes in graph
graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

# Define conditional edge
graph.add_conditional_edges("chat_node", tools_condition)

graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer = checkpointer)
chatbot

def retrieve_all_threads():
    all_thread = set()
    for checkpoint in checkpointer.list(None): # None -> Return all checkpoints from DB instead of specific one.
        all_thread.add(checkpoint.config["configurable"]["thread_id"])

    return list(all_thread)