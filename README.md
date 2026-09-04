# Agentic AI using LangGraph

This repository contains practical implementations and demonstrations of **Agentic AI using LangGraph**.

The project focuses on understanding how to build **stateful, multi-step, tool-using AI agents and workflows** using LangGraph.

## 📚 Topics Covered

### 1. Agentic AI Fundamentals

Understanding the fundamentals of Agentic AI and how AI agents work.

The concepts include:

* Generative AI vs Agentic AI
* AI Agents
* Reasoning and Planning
* Tool Usage
* Agent Execution Loop

**Basic workflow:**

```text
Goal → Plan → Action → Observe → Decide → Result
```

---

### 2. LangGraph Fundamentals

Understanding the core components used to build LangGraph applications.

The concepts include:

* State
* Nodes
* Edges
* START and END
* Conditional Edges
* State Updates

**Basic workflow:**

```text
State → Node → Edge → Node → Result
```

---

### 3. LangGraph Workflows

Implementation of different workflow patterns for building multi-step AI applications.

The workflows include:

* Sequential Workflows
* Parallel Workflows
* Conditional Workflows
* Iterative Workflows

**Example:**

```text
Input
  ↓
Process
  ↓
Decision
 ├──→ Path A
 └──→ Path B
        ↓
      Output
```

---

### 4. Chatbots & Persistence

Building stateful conversational applications using LangGraph.

The concepts include:

* LangGraph Chatbot
* Conversation State
* Persistence
* Streaming
* Streamlit UI
* SQLite Integration

**Workflow:**

```text
User
 ↓
LangGraph
 ↓
State / Memory
 ↓
LLM
 ↓
Response
```

---

### 5. Tools, MCP & RAG

Extending AI agents with external tools and knowledge sources.

The concepts include:

* Tool Calling
* Custom Tools
* Model Context Protocol (MCP)
* Retrieval-Augmented Generation (RAG)
* Agentic RAG

**Agent workflow:**

```text
User
 ↓
Agent
 ↓
Tool / Retriever
 ↓
Result
 ↓
Agent
 ↓
Final Answer
```

---

### 6. Human-in-the-Loop & Subgraphs

Building controlled and modular agentic workflows.

The concepts include:

* Human-in-the-Loop (HITL)
* Interrupt and Resume
* Human Approval
* Subgraphs
* Modular Graph Design

**Example:**

```text
Agent
 ↓
Action
 ↓
Human Approval
 ├──→ Approve → Continue
 └──→ Reject  → Modify / Stop
```

---

### 7. Memory

Understanding how AI applications maintain information across interactions.

The concepts include:

* LLM Memory
* Short-Term Memory
* Long-Term Memory
* Persistent Memory

**Basic idea:**

```text
Conversation
     ↓
Memory
     ↓
Relevant Context
     ↓
LLM
```

---

### 8. End-to-End Agentic AI

An end-to-end agentic workflow combining multiple concepts.

```text
User Request
     ↓
Planning
     ↓
Research
     ↓
Information Processing
     ↓
Decision Making
     ↓
Content Generation
     ↓
Final Output
```

---

## 🛠️ Technologies Used

* **Python**
* **LangChain**
* **LangGraph**
* **LLMs / Chat Models**
* **Streamlit**
* **SQLite**
* **LangSmith**
* **RAG**
* **MCP**

---

## 🎯 Learning Objective

The main objective of this project is to understand how **Agentic AI systems can be designed and implemented using LangGraph**.

The overall progression is:

```text
Generative AI
      ↓
Agentic AI
      ↓
LangGraph
      ↓
Workflows
      ↓
State & Persistence
      ↓
Tools
      ↓
RAG / MCP
      ↓
Human-in-the-Loop
      ↓
Memory
      ↓
Agentic AI Applications
```

---

## 📌 Summary

| Concept           | Purpose                                   |
| ----------------- | ----------------------------------------- |
| **Agentic AI**    | Builds goal-oriented AI systems           |
| **LangGraph**     | Orchestrates stateful AI workflows        |
| **State**         | Maintains information during execution    |
| **Nodes & Edges** | Define graph execution                    |
| **Tools**         | Allow agents to perform actions           |
| **RAG**           | Provides external knowledge               |
| **MCP**           | Connects agents with external tools       |
| **HITL**          | Enables human control                     |
| **Memory**        | Maintains information across interactions |
| **Subgraphs**     | Creates modular workflows                 |
| **LangSmith**     | Provides tracing and observability        |

---

## 👨‍💻 Author

**Aftabalam Makandar**

This project is part of my learning journey in **Generative AI, LLMs, Agentic AI, LangChain, and LangGraph**.
