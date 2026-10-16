import uuid

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from langgraph_chatbot_backend import (
    stream_chat,
    retrieve_all_threads,
    load_conversation,
    ingest_pdf,
    thread_document_metadata,
)

"""
    Observability:
        LangSmith Observability provides full visibility into your LLM application.
"""

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered",
)

# ************************************ Utility functions ************************************

# Generate unique thread_id (as a string, because the database stores it as text)
def generate_thread_id():
    return str(uuid.uuid4())


# Get plain text from a message (content can be a string or a list of parts)
def message_text(message):
    content = message.content
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part)
            for part in content
        )
    return content or ""


# Reset the chat
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    add_thread(thread_id)
    st.session_state.message_history = []


def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)


# ************************************ Session Setup ************************************

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_threads" not in st.session_state:
    # All saved thread_ids from the database
    st.session_state.chat_threads = retrieve_all_threads()

if "ingested_files" not in st.session_state:
    # thread_id -> set of files already indexed (so we do not index twice)
    st.session_state.ingested_files = {}

add_thread(st.session_state.thread_id)

# ************************************ Sidebar UI ************************************
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

# ---- PDF upload (one PDF per chat) ----
st.sidebar.header("PDF for this chat")

upload_thread_id = st.session_state.thread_id
uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF",
    type=["pdf"],
    key=f"uploader_{upload_thread_id}",  # new key per chat = fresh uploader
)

if uploaded_pdf is not None:
    file_key = f"{uploaded_pdf.name}-{uploaded_pdf.size}"
    done_files = st.session_state.ingested_files.setdefault(upload_thread_id, set())

    if file_key not in done_files:
        with st.sidebar.status("Indexing PDF...", expanded=True) as status_box:
            try:
                ingest_pdf(uploaded_pdf.getvalue(), upload_thread_id, uploaded_pdf.name)
                done_files.add(file_key)
                status_box.update(label="✅ PDF ready", state="complete", expanded=False)
            except Exception as e:
                status_box.update(label=f"❌ Could not read PDF: {e}", state="error")

# ---- Conversations ----
st.sidebar.header("Conversations")

for thread_id in st.session_state.chat_threads[::-1]:  # newest first
    if st.sidebar.button(str(thread_id), key=f"thread_{thread_id}"):
        st.session_state.thread_id = thread_id

        temp_messages = []
        for message in load_conversation(thread_id):
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, AIMessage):
                role = "assistant"
            else:
                continue  # skip tool messages

            text = message_text(message)
            if not text:
                continue  # skip AI messages that only call tools

            temp_messages.append({"role": role, "content": text})

        st.session_state.message_history = temp_messages

# ************************************ Main UI ************************************

# Show which PDF is attached to the current chat
doc_info = thread_document_metadata(st.session_state.thread_id)
if doc_info:
    st.caption(
        f"📄 {doc_info['filename']} · {doc_info['documents']} pages · "
        f"{doc_info['chunks']} chunks"
    )
else:
    st.caption("No PDF in this chat. Upload one in the sidebar to ask questions about it.")

# Show the conversation history
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.text(message["content"])

CONFIG = {
    "configurable": {"thread_id": st.session_state.thread_id},
    "metadata": {"thread_id": st.session_state.thread_id},
    "run_name": "chat_turn",
}

user_input = st.chat_input("Type here")

if user_input:

    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    with st.chat_message("assistant"):
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, _ in stream_chat(user_input, CONFIG):

                # Tool call requested by the LLM
                if isinstance(message_chunk, AIMessage) and message_chunk.tool_calls:
                    tool_name = message_chunk.tool_calls[0].get("name")
                    if tool_name:
                        if status_holder["box"] is None:
                            status_holder["box"] = st.status(
                                f"🔧 Using `{tool_name}` ...", expanded=True
                            )
                        else:
                            status_holder["box"].update(
                                label=f"🔧 Using `{tool_name}` ...",
                                state="running",
                                expanded=True,
                            )

                # Tool result coming back
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` ...", expanded=True
                        )
                    status_holder["box"].update(
                        label=f"✅ Used `{tool_name}`", state="complete", expanded=False
                    )

                # Only yield the assistant's text
                if isinstance(message_chunk, AIMessage):
                    text = message_text(message_chunk)
                    if text:
                        yield text

        ai_message = st.write_stream(ai_only_stream())

    st.session_state.message_history.append({"role": "assistant", "content": ai_message})