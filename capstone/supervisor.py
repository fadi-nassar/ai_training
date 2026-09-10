import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "agents"))
sys.path.append(str(Path(__file__).resolve().parent / "agents" / "research_team"))

import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv(Path(__file__).resolve().parent / ".env")

# import each agent
from agents.sql_agent import sql_agent_graph
from agents.rag_agent import rag_agent_graph
from agents.conversation_agent import conversation_agent_graph
from agents.viz_agent import viz_assistant
from agents.research_team.sub_supervisor import run_research_team

llm = ChatGroq(model="openai/gpt-oss-120b")

class SupervisorState(TypedDict):
    user_input: str
    category: str
    response: str

# --- ROUTING NODE ---
def classify(state: SupervisorState):
    prompt = f"""Classify the following user request into exactly one category:
- 'sql' for questions about products, prices, stock, inventory (database questions)
- 'rag' for questions about company policies (returns, shipping, warranty, support)
- 'research' for questions requiring general knowledge lookup or research on a topic
- 'visualization' for requests to create a chart or graph
- 'conversation' for general chit-chat, greetings, or anything else

Respond with ONLY one word: sql, rag, research, visualization, or conversation.

User request: {state['user_input']}"""
    result = llm.invoke([HumanMessage(content=prompt)])
    category = result.content.strip().lower()
    return {"category": category}

# --- AGENT-CALLING NODES ---
def call_sql_agent(state: SupervisorState):
    result = sql_agent_graph.invoke({"messages": [HumanMessage(content=state["user_input"])]})
    return {"response": result["messages"][-1].content}

def call_rag_agent(state: SupervisorState):
    result = rag_agent_graph.invoke({"messages": [HumanMessage(content=state["user_input"])]})
    return {"response": result["messages"][-1].content}

def call_conversation_agent(state: SupervisorState):
    config = {"configurable": {"thread_id": "main-thread"}}
    result = conversation_agent_graph.invoke(
        {"messages": [HumanMessage(content=state["user_input"])]}, config
    )
    return {"response": result["messages"][-1].content}

def call_viz_agent(state: SupervisorState):
    chart = viz_assistant({"messages": [HumanMessage(content=state["user_input"])]})
    return {"response": f"Chart type: {chart.chart_type}\nTitle: {chart.title}\nLabels: {chart.labels}\nValues: {chart.values}"}

def call_research_team(state: SupervisorState):
    report = run_research_team(state["user_input"])
    return {"response": report}

# --- ROUTING LOGIC ---
def route(state: SupervisorState):
    category = state["category"]
    if "sql" in category:
        return "sql"
    elif "rag" in category:
        return "rag"
    elif "research" in category:
        return "research"
    elif "visual" in category:
        return "visualization"
    else:
        return "conversation"

# --- BUILD THE GRAPH ---
builder = StateGraph(SupervisorState)
builder.add_node("classify", classify)
builder.add_node("sql", call_sql_agent)
builder.add_node("rag", call_rag_agent)
builder.add_node("research", call_research_team)
builder.add_node("visualization", call_viz_agent)
builder.add_node("conversation", call_conversation_agent)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route)
builder.add_edge("sql", END)
builder.add_edge("rag", END)
builder.add_edge("research", END)
builder.add_edge("visualization", END)
builder.add_edge("conversation", END)

main_supervisor_graph = builder.compile()

if __name__ == "__main__":
    test_inputs = [
        "What's our most expensive product?",
        "What's your return policy?",
        "What is LangGraph?",
        "Create a chart of sales: Q1 100, Q2 200, Q3 150",
        "Hi, how are you?",
    ]

    for user_input in test_inputs:
        print(f"\n{'='*60}")
        print(f"USER: {user_input}")
        print('='*60)
        result = main_supervisor_graph.invoke({"user_input": user_input, "category": "", "response": ""})
        print(f"CATEGORY: {result['category']}")
        print(f"RESPONSE: {result['response']}")