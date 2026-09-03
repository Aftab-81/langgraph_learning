import streamlit as st
from langgraph_backend import chatbot
from langchain_core.messages import HumanMessage

# st.session_state -> dict -> The info inside the dict won't erase when we type new msg in a cinversation. Only get erased when we refresh the page.
if "message_history" not in st.session_state:
    st.session_state.message_history = [] # Here we add a key in session_state dict



# Loading the conversation history  
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.text(message["content"])

# {"role": "user", "content": "Hello, how are you?"}
# {"role": "assistant", "content": "I'm good, thank you! How can I assist you today?"}

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
                config = {"configurable": {"thread_id": "1"}},
                stream_mode = "messages"
            )
            
            # Added because we are using Google AI Studio API & it has different response format without this line the error would happen. 
            # If using OpenAI or any other powerful models, it is OK to remove this line the code'll work properly.
            if ai_message_chunk.content and isinstance(ai_message_chunk.content, list) and ai_message_chunk.content[-1].get("type") == "text" 

        )

    st.session_state.message_history.append({"role": "assistant", "content": ai_message})
        