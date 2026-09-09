import streamlit as st
from langgraph_backend import chatbot, retrieve_all_threads
from langchain_core.messages import HumanMessage
import uuid


"""
    Observability:
        LangSmith Observability provides full visibility into your LLM application.
"""

st.set_page_config(
    page_title="LangGraph Chatbot",
    page_icon="🤖",
    layout="centered",
    # menu_items={
    #         'Get Help': 'https://www.gmail.com/help',
    #         'Report a bug': "https://www.gmail.com/bug",
    #         'About': "# This is a header. This is an *extremely* cool app!"
    #      }
)

# ************************************ Utility functions *******************************************************************************************

# Generate unique thread_id
def generate_thread_id():
    thread_id = uuid.uuid4()
    return thread_id

# Reset the chat
def reset_chat():
    thread_id = generate_thread_id()
    st.session_state.thread_id = thread_id
    add_thread(st.session_state.thread_id)
    st.session_state.message_history = []

def add_thread(thread_id):
    if thread_id not in st.session_state.chat_threads:
        st.session_state.chat_threads.append(thread_id)

def load_conversation(thread_id):
    # Returns a list of [HumanMessage(), AIMessage()]
    return chatbot.get_state(config = {"configurable": {"thread_id": thread_id}}).values["messages"]

# ************************************ Session Setup *******************************************************************************************

# st.session_state -> dict -> The info inside the dict won't erase when we type new msg in a conversation. Only get erased when we refresh the page.
if "message_history" not in st.session_state:
    st.session_state.message_history = [] # Here we add a key in session_state dict


if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_threads" not in st.session_state:
    # Storing all the thread_ids 
    st.session_state.chat_threads = retrieve_all_threads()

add_thread(st.session_state.thread_id)



# ************************************ Sidebar UI *******************************************************************************************
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()

st.sidebar.header("Conversations")

for thread_id in st.session_state.chat_threads[::-1]: # [::-1] for stack like feature on COnversation IDs
    if st.sidebar.button(str(thread_id)):

        # messages -> []
        messages = load_conversation(thread_id)
        st.session_state.thread_id = thread_id

        temp_messages = []

        for message in messages:
            if type(message).__name__ == "HumanMessage":
                role = "user"
            else:
                role = "assistant"

            content = message.content[-1]["text"] if isinstance(message.content, list) else message.content
            temp_messages.append({"role": role, "content": content})

        st.session_state.message_history = temp_messages

# ************************************ Main UI *******************************************************************************************

# Loading the conversation history  
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# {"role": "user", "content": "Hello, how are you?"}
# {"role": "assistant", "content": "I'm good, thank you! How can I assist you today?"}

CONFIG = {
            "configurable": {"thread_id": st.session_state.thread_id},
            "metadata": {   # Here we seperate each turn(conversation) according to the threads.
                "thread_id": st.session_state.thread_id
            },
            "run_name": "chat_turn"
        }

user_input = st.chat_input("Type here")

if user_input:

    # first add the message to message_history
    st.session_state.message_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)

    # first add the message to message_history
    
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            # Here chatbot.stream() -> ai_message_chunk, metadata
            ai_message_chunk.content[-1]["text"] for ai_message_chunk, _ in chatbot.stream( # This "for loop" is just like list comprehension..
                {"messages": [HumanMessage(content = user_input)]}, 
                config = CONFIG,
                stream_mode = "messages"
            )
            
            # Added because we are using Google AI Studio API & it has different response format without this line the error would happen. 
            # If you are using OpenAI or any other powerful models, it is OK to remove this line the code'll work properly.
            if ai_message_chunk.content and isinstance(ai_message_chunk.content, list) and ai_message_chunk.content[-1].get("type") == "text" 

        )

    st.session_state.message_history.append({"role": "assistant", "content": ai_message})
        