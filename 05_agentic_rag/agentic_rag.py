import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

# --- SAME DOCUMENTS AS EXERCISE 2 ---
documents = [
    Document(page_content="LangGraph is a library for building stateful, multi-agent applications with LLMs. It represents workflows as graphs with nodes and edges."),
    Document(page_content="Groq provides ultra-low latency LLM inference using custom hardware called LPUs (Language Processing Units), making it much faster than typical GPU-based inference."),
    Document(page_content="A checkpointer in LangGraph, such as MemorySaver, saves the state of a graph at every step, enabling memory across multiple invocations using a thread_id."),
    Document(page_content="RAG stands for Retrieval-Augmented Generation. It combines a retrieval step, pulling relevant documents from a knowledge base, with a generation step, where an LLM produces an answer using those documents as context."),
    Document(page_content="LangSmith is a platform for tracing, monitoring, and evaluating LLM applications. It shows the full execution path of a chain or graph, including token usage and latency."),
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# --- RETRIEVAL AS A TOOL ---
@tool
def search_knowledge_base(query: str) -> str:
    """Searches the internal knowledge base for information about LangGraph, Groq, RAG, or LangSmith."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

tools = [search_knowledge_base]

llm = ChatGroq(model="openai/gpt-oss-120b")
llm_with_tools = llm.bind_tools(tools)

def assistant(state: MessagesState):
    system_prompt = SystemMessage(content=(
        "You are a helpful assistant. Use the search_knowledge_base tool "
        "when a question is about LangGraph, Groq, RAG, or LangSmith. "
        "For general questions unrelated to these topics, answer directly without using the tool."
    ))
    messages = [system_prompt] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
graph = builder.compile()

# --- TEST: some need retrieval, some don't ---
test_inputs = [
    "What is RAG?",
    "What's 2 + 2?",
    "How does memory work in LangGraph?",
    "What's the capital of France?",
]

for user_input in test_inputs:
    print(f"\n{'='*60}")
    print(f"USER: {user_input}")
    print('='*60)
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
    for msg in result["messages"]:
        msg.pretty_print()
        