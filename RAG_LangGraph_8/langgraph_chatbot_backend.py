from typing import TypedDict, Annotated
import asyncio
import json
import os
import queue
import sys
import tempfile
import threading

import aiosqlite
import requests
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openrouter import ChatOpenRouter
from langchain_mcp_adapters.client import MultiServerMCPClient

from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

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

# Optional key for the stock tool (add ALPHA_VANTAGE_API_KEY to your .env)
alpha_key = os.getenv("ALPHA_VANTAGE_API_KEY")

llm = ChatOpenRouter(
    model="dots-studio/dots-3-note-preview:free",
    api_key=api_key,
    temperature=0,
    max_tokens=1024,
    max_retries=3,
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
# 3. RAG: PDF ingestion + rag_tool
#    Every chat (thread_id) gets its OWN PDF index, so one chat's PDF
#    never leaks into another chat. Indexes are also saved to disk so
#    old chats still work after you restart the app.
# =====================================================================
INDEX_DIR = os.path.join(BASE_DIR, "faiss_indexes")
os.makedirs(INDEX_DIR, exist_ok=True)

# Load the embedding model once (loading it again and again is slow)
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    encode_kwargs={"normalize_embeddings": True},
)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# In-memory cache: thread_id -> retriever / document info
_THREAD_RETRIEVERS: dict = {}
_THREAD_DOCS: dict = {}


def _index_path(thread_id: str) -> str:
    return os.path.join(INDEX_DIR, str(thread_id))


def _make_retriever(vector_store):
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})


def _load_thread(thread_id) -> bool:
    """Make sure this thread's index is in memory. Returns True if a PDF exists."""
    if not thread_id:
        return False
    thread_id = str(thread_id)

    if thread_id in _THREAD_RETRIEVERS:
        return True

    path = _index_path(thread_id)
    meta_file = os.path.join(path, "meta.json")
    if os.path.isdir(path) and os.path.exists(meta_file):
        store = FAISS.load_local(
            path, embedding, allow_dangerous_deserialization=True
        )
        with open(meta_file, "r", encoding="utf-8") as f:
            _THREAD_DOCS[thread_id] = json.load(f)
        _THREAD_RETRIEVERS[thread_id] = _make_retriever(store)
        return True
    return False


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str = "document.pdf") -> dict:
    """
    Read a PDF, split it into chunks, embed them, and store a FAISS index
    for this chat thread. Uploading a new PDF replaces the old one.

    Returns a small dict: {"filename", "documents" (pages), "chunks"}.
    """
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")
    thread_id = str(thread_id)

    # PyPDFLoader needs a real file path, so write the bytes to a temp file.
    # (delete=False + manual remove because Windows cannot reopen an open temp file)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    try:
        tmp.write(file_bytes)
        tmp.close()

        pages = PyPDFLoader(tmp.name).load()
    finally:
        try:
            os.remove(tmp.name)
        except OSError:
            pass

    for page in pages:
        page.metadata["source"] = filename  # show the real file name, not the temp path

    chunks = splitter.split_documents(pages)
    chunks = [c for c in chunks if c.page_content.strip()]
    if not chunks:
        raise ValueError(
            "No text found in this PDF. It may be a scanned image (needs OCR)."
        )

    vector_store = FAISS.from_documents(chunks, embedding)

    # Save to disk so the index survives app restarts
    path = _index_path(thread_id)
    os.makedirs(path, exist_ok=True)
    vector_store.save_local(path)

    info = {"filename": filename, "documents": len(pages), "chunks": len(chunks)}
    with open(os.path.join(path, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(info, f)

    _THREAD_RETRIEVERS[thread_id] = _make_retriever(vector_store)
    _THREAD_DOCS[thread_id] = info
    return info


def thread_has_document(thread_id) -> bool:
    return _load_thread(thread_id)


def thread_document_metadata(thread_id) -> dict:
    """Info about the PDF of this chat, or {} if no PDF was uploaded."""
    if _load_thread(thread_id):
        return _THREAD_DOCS.get(str(thread_id), {})
    return {}


@tool
def rag_tool(query: str, config: RunnableConfig) -> dict:
    """
    Search the PDF that the user uploaded in THIS chat and return the most
    relevant passages. Use this whenever the user asks about the PDF,
    the document, or their notes.
    """
    # `config` is filled in automatically by LangGraph (the model does not see it)
    thread_id = (config or {}).get("configurable", {}).get("thread_id")

    if not _load_thread(thread_id):
        return {
            "query": query,
            "error": "No PDF has been uploaded for this chat. "
                     "Ask the user to upload a PDF first.",
        }

    result = _THREAD_RETRIEVERS[str(thread_id)].invoke(query)
    return {
        "query": query,
        "context": [doc.page_content for doc in result],
        "metadata": [
            {
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page_label", doc.metadata.get("page")),
            }
            for doc in result
        ],
    }


# =====================================================================
# 4. MCP client (your servers)
# =====================================================================
# maths_mcp_server.py lives in the sibling folder MCP_Client_7, not in this RAG folder
SERVER_PATH = os.path.abspath(
    os.path.join(BASE_DIR, "..", "MCP_Client_7", "maths_mcp_server.py")
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
# 5. Chat history database
#    NOTE: change this name if your old chats are in a different file.
# =====================================================================
DB_PATH = os.path.join(BASE_DIR, "chatbot.db")


# =====================================================================
# 6. State and graph
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

    tools = mcp_tools + [search_tool, rag_tool]
    if alpha_key:  # add the stock tool only if you have a key
        tools.append(get_stock_price)

    print("Loaded tools:", [t.name for t in tools])

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState, config: RunnableConfig) -> ChatState:
        # Tell the model whether this chat has a PDF, so it knows when to use rag_tool
        thread_id = (config or {}).get("configurable", {}).get("thread_id")
        doc = thread_document_metadata(thread_id)

        system_text = "You are a helpful assistant."
        if doc:
            system_text += (
                f" The user uploaded a PDF named '{doc['filename']}' in this chat. "
                "For any question about the PDF or the document, call rag_tool and "
                "answer from the retrieved text. Mention the page number when you can."
            )
        else:
            system_text += (
                " No PDF is uploaded in this chat. If the user asks about a PDF "
                "or document, ask them to upload one using the sidebar."
            )

        messages = [SystemMessage(content=system_text)] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
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
# 7. Helper functions for the Streamlit frontend (all synchronous)
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
# 8. Quick test (run: python langgraph_chatbot_backend.py)
# =====================================================================
if __name__ == "__main__":
    result = run_async(
        chatbot.ainvoke(
            {"messages": [HumanMessage(content="What is 25 multiplied by 4? Use the tools.")]},
            config={"configurable": {"thread_id": "test-thread"}},
        )
    )
    print(result["messages"][-1].content)