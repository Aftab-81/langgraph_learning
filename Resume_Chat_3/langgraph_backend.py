from langgraph.graph import StateGraph, START, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from typing import Annotated, Self
from typing_extensions import TypedDict
from dotenv import load_dotenv
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
)

# Define the state
class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

# Task 1
def chat_node(state: ChatState) -> Self:
    # Take input
    input = state["messages"]

    # Send it to LLM
    response = llm.invoke(input).content

    # Save response into the messages
    return {"messages": [AIMessage(response)]} 

memory = MemorySaver()

graph = StateGraph(ChatState)

# Add node
graph.add_node("chat_node", chat_node)

# Add edges
graph.set_entry_point("chat_node")
graph.set_finish_point("chat_node")

chatbot = graph.compile(checkpointer = memory)
