# from langgraph.graph import START, END, StateGraph
# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from pydantic import BaseModel, Field
# from typing import TypedDict
# from dotenv import load_dotenv
# from IPython.display import Image, display 

# load_dotenv()

# # Creating an LLM model

# llm1 = HuggingFaceEndpoint(
#     repo_id = "meta-llama/Llama-3.1-8B-Instruct",
#     task = "text-generation"
# )

# llm = ChatHuggingFace(llm = llm1)

# print(llm.invoke("Hello").content)

# # Graph's State
# class State(TypedDict):
#     topic: str
#     joke: str
#     improved_joke: str
#     final_joke: str 

# # Nodes
# def generate_joke(state: State):
#     """First LLM call to generate an initial joke"""

#     msg = llm.invoke(f"Write a short joke about {state['topic']}")
#     return {"joke": msg.content}

# def improve_joke(state: State):
#     """Second LLM call to improve the Joke"""

#     msg = llm.invoke(f"Add a surprising twist to this joke: {state['joke']}")
#     return {"improved_joke": msg.content}

# def final_joke(state: State):
#     """Third LLM call to get Final Joke"""

#     msg = llm.invoke(f"Make this joke funnier by adding wordplay: {state['improved_joke']}")
#     return {"final_joke": msg.content}

# # Build Workflow

# workflow = StateGraph(state_schema = State)

# workflow.add_node("generate_joke", generate_joke)
# workflow.add_node("improve_joke", improve_joke)
# workflow.add_node("final_joke", final_joke)

# workflow.add_edge(START, "generate_joke")
# workflow.add_edge("generate_joke", "improve_joke")
# workflow.add_edge("improve_joke", "final_joke")
# workflow.add_edge("final_joke", END)

# # Compile
# chain = workflow.compile()

# # Show workflow
# display(Image(chain.get_graph().draw_mermaid_png()))

# # Invoke
# state = chain.invoke({"topic": "Cat"})
# print(state)

# with open("graph.png", "wb") as f:
#     f.write(chain.get_graph().draw_mermaid_png())
