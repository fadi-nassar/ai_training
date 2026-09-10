import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- knowledge base documents (swap these for real company docs later) ---
documents = [
    Document(page_content="Our return policy allows returns within 30 days of purchase with a valid receipt."),
    Document(page_content="Standard shipping takes 5-7 business days. Express shipping takes 1-2 business days."),
    Document(page_content="We offer a 1-year warranty on all electronics products, covering manufacturing defects."),
    Document(page_content="Customer support is available Monday to Friday, 9 AM to 5 PM, via email or live chat."),
]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

@tool
def search_knowledge_base(query: str) -> str:
    """Searches company policy documents for information about returns, shipping, warranty, or support."""
    docs = retriever.invoke(query)
    return "\n\n".join(doc.page_content for doc in docs)

tools = [search_knowledge_base]
llm = ChatGroq(model="openai/gpt-oss-120b")
llm_with_tools = llm.bind_tools(tools)

def assistant(state: MessagesState):
    system_prompt = SystemMessage(content=(
    "You are a helpful assistant with access to company policy documents. "
    "Use the search_knowledge_base tool for questions about returns, shipping, warranty, or support. "
    "IMPORTANT: When answering from the knowledge base, use ONLY the information returned by the tool. "
    "Do not add details, policies, or steps that are not explicitly stated in the retrieved content. "
    "If the retrieved content doesn't fully answer the question, say so rather than inventing details. "
    "For unrelated questions, answer directly."
))
    messages = [system_prompt] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
rag_agent_graph = builder.compile()

if __name__ == "__main__":
    test_questions = [
        "What's your return policy?",
        "How fast is shipping?",
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"USER: {q}")
        print('='*60)
        result = rag_agent_graph.invoke({"messages": [HumanMessage(content=q)]})
        for msg in result["messages"]:
            msg.pretty_print()