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

    response = chatbot.invoke({"messages": [HumanMessage(content = user_input)]}, config = {"configurable": {"thread_id": "1"}})
    ai_message = response["messages"][-1].content[-1]["text"]
    # first add the message to message_history
    st.session_state.message_history.append({"role": "assistant", "content": ai_message})
    with st.chat_message("assistant"):
        st.text(ai_message)