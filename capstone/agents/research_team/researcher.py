import os
from pathlib import Path
from dotenv import load_dotenv
import wikipedia
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

@tool
def wikipedia_search(query: str) -> str:
    """Searches Wikipedia and returns a summary of the top matching article."""
    try:
        results = wikipedia.search(query, results=1)
        if not results:
            return "No results found."
        summary = wikipedia.summary(results[0], sentences=5)
        return f"Source: {results[0]}\n\n{summary}"
    except Exception as e:
        return f"Search error: {e}"

tools = [wikipedia_search]
llm = ChatGroq(model="openai/gpt-oss-120b")
llm_with_tools = llm.bind_tools(tools)

def researcher_assistant(state: MessagesState):
    system_prompt = SystemMessage(content=(
        "You are a research assistant. Use the wikipedia_search tool to gather "
        "factual information relevant to the user's research topic. "
        "Summarize what you found clearly."
    ))
    messages = [system_prompt] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

builder = StateGraph(MessagesState)
builder.add_node("researcher", researcher_assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "researcher")
builder.add_conditional_edges("researcher", tools_condition)
builder.add_edge("tools", "researcher")
researcher_graph = builder.compile()