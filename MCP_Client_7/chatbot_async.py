from typing import TypedDict, Annotated
import asyncio
import os
import queue
import sys
import threading

import aiosqlite
import requests
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_openrouter import ChatOpenRouter
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()

# =====================================================================
# 1. A background event loop
#    Streamlit is synchronous, but MCP tools are async-only.
#    So we run ONE async event loop in a background thread, and send
#    all async work to it. This avoids "event loop" errors in Streamlit.
# =====================================================================
_loop = asyncio.new_event_loop()
threading.Thread(target=_loop.run_forever, daemon=True).start()


def run_async(coro):
    """Run an async function on the background loop and wait for the result."""
    return asyncio.run_coroutine_threadsafe(coro, _loop).result()


# =====================================================================
# 2. API keys, LLM, normal tools
# =====================================================================
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OpenRouter API key not found. Check your .env file.")

# Optional: only needed for the stock price tool
alpha_key = os.getenv("ALPHAVANTAGE_API_KEY")

llm = ChatOpenRouter(
    model="nvidia/nemotron-3.5-lightning:free",
    api_key=api_key,
    temperature=0,
    max_tokens=1024,
)

search_tool = DuckDuckGoSearchRun()


@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for the given symbol (eg, 'AAPL', 'TSLA')
    using the Alpha Vantage API.
    """
    url = (
        "https://www.alphavantage.co/query"
        f"?function=GLOBAL_QUOTE&symbol={symbol}&apikey={alpha_key}"
    )
    response = requests.get(url)
    return response.json()


# =====================================================================
# 3. MCP client (your servers)
# =====================================================================
SERVER_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "maths_mcp_server.py"
)

client = MultiServerMCPClient(
    {
        "maths": {
            "transport": "stdio",
            "command": sys.executable,  # same Python as your venv
            "args": [SERVER_PATH],
        },
        "Kaggle Server": {
            "transport": "stdio",
            "command": "C:/Users/A/AppData/Local/Programs/Python/Python311/Scripts/uv.exe",
            "args": [
                "--directory",
                "D:/Multiple Projects/kaggle-server-mcp",
                "run",
                "main.py",
            ],
        },
    }
)

SERVER_NAMES = ["maths", "Kaggle Server"]

# =====================================================================
# 4. Chat history database
#    NOTE: change this name if your old chats are in a different file.
# =====================================================================
DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chatbot.db")


# =====================================================================
# 5. State and graph
# =====================================================================
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


async def _build():
    # Load tools from each MCP server. If one server fails, skip it
    # instead of crashing the whole chatbot.
    mcp_tools = []
    for name in SERVER_NAMES:
        try:
            mcp_tools += await client.get_tools(server_name=name)
        except Exception as e:
            print(f"[warning] Could not load tools from '{name}': {e}")

    tools = mcp_tools + [search_tool]
    if alpha_key:  # add the stock tool only if you have a key
        tools.append(get_stock_price)

    print("Loaded tools:", [t.name for t in tools])

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState) -> ChatState:
        response = await llm_with_tools.ainvoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    # Async SQLite checkpointer (saves chats)
    conn = await aiosqlite.connect(DB_PATH)
    checkpointer = AsyncSqliteSaver(conn)

    graph = StateGraph(ChatState)
    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.set_entry_point("chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")

    return graph.compile(checkpointer=checkpointer), checkpointer


# Build once when this file is first imported
chatbot, checkpointer = run_async(_build())


# =====================================================================
# 6. Helper functions for the Streamlit frontend (all synchronous)
# =====================================================================
_DONE = object()


def stream_chat(user_input: str, config: dict):
    """
    Sync generator. Yields (message_chunk, metadata) while the async
    graph streams in the background loop.
    """
    q = queue.Queue()

    async def _producer():
        try:
            async for item in chatbot.astream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
                stream_mode="messages",
            ):
                q.put(item)
        except Exception as e:
            q.put(e)
        finally:
            q.put(_DONE)

    asyncio.run_coroutine_threadsafe(_producer(), _loop)

    while True:
        item = q.get()
        if item is _DONE:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def _retrieve_all_threads():
    threads = []
    async for checkpoint in checkpointer.alist(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]
        if thread_id not in threads:
            threads.append(thread_id)
    return threads[::-1]  # oldest first


def retrieve_all_threads():
    return run_async(_retrieve_all_threads())


async def _load_conversation(thread_id):
    state = await chatbot.aget_state(
        config={"configurable": {"thread_id": str(thread_id)}}
    )
    return state.values.get("messages", [])


def load_conversation(thread_id):
    return run_async(_load_conversation(thread_id))


# =====================================================================
# 7. Quick test (run: python langgraph_mcp_backend.py)
# =====================================================================
if __name__ == "__main__":
    result = run_async(
        chatbot.ainvoke(
            {"messages": [HumanMessage(content="What is 25 multiplied by 4? Use the tools.")]},
            config={"configurable": {"thread_id": "test-thread"}},
        )
    )
    print(result["messages"][-1].content)