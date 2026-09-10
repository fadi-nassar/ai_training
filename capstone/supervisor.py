import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parent / "agents"))
sys.path.append(str(Path(__file__).resolve().parent / "agents" / "research_team"))

import os
import time
import logging
import functools
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv(Path(__file__).resolve().parent / ".env")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("capstone.supervisor")

# How many times to retry an agent call after the first failure.
AGENT_MAX_RETRIES = 1
# Seconds to wait between the failed attempt and the retry.
AGENT_RETRY_DELAY = 1.0


def resilient_agent(agent_label: str):
    """Decorator for the agent-calling nodes.

    Wraps an agent node so that:
      1. If the agent raises, we log it and retry once (AGENT_MAX_RETRIES).
      2. If it still fails, we return a readable error string in `response`
         instead of letting the exception bubble up and crash the graph.

    The wrapped function body is untouched - routing and classification are
    unaffected. Each attempt still runs as its own child run in LangSmith,
    so failures remain visible in tracing.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(state: "SupervisorState"):
            last_error = None
            total_attempts = AGENT_MAX_RETRIES + 1
            for attempt in range(1, total_attempts + 1):
                try:
                    return func(state)
                except Exception as exc:  # noqa: BLE001 - deliberately broad at the boundary
                    last_error = exc
                    logger.warning(
                        "%s failed (attempt %d/%d): %s: %s",
                        agent_label, attempt, total_attempts,
                        type(exc).__name__, exc,
                    )
                    if attempt <= AGENT_MAX_RETRIES:
                        time.sleep(AGENT_RETRY_DELAY)
            logger.error(
                "%s failed after %d attempts, returning error to user.",
                agent_label, total_attempts, exc_info=last_error,
            )
            return {
                "response": (
                    f"[agent error] The {agent_label} couldn't complete your request "
                    f"after {total_attempts} attempts.\n\n"
                    f"Error: {type(last_error).__name__}: {last_error}\n\n"
                    "Please try again in a moment or rephrase your question."
                )
            }
        return wrapper
    return decorator

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
# Bodies are unchanged; @resilient_agent only adds retry + graceful-failure handling.
@resilient_agent("SQL Query Agent")
def call_sql_agent(state: SupervisorState):
    result = sql_agent_graph.invoke({"messages": [HumanMessage(content=state["user_input"])]})
    return {"response": result["messages"][-1].content}

@resilient_agent("RAG Agent")
def call_rag_agent(state: SupervisorState):
    result = rag_agent_graph.invoke({"messages": [HumanMessage(content=state["user_input"])]})
    return {"response": result["messages"][-1].content}

@resilient_agent("Conversation Agent")
def call_conversation_agent(state: SupervisorState):
    config = {"configurable": {"thread_id": "main-thread"}}
    result = conversation_agent_graph.invoke(
        {"messages": [HumanMessage(content=state["user_input"])]}, config
    )
    return {"response": result["messages"][-1].content}

@resilient_agent("Visualization Agent")
def call_viz_agent(state: SupervisorState):
    chart = viz_assistant({"messages": [HumanMessage(content=state["user_input"])]})
    return {"response": f"Chart type: {chart.chart_type}\nTitle: {chart.title}\nLabels: {chart.labels}\nValues: {chart.values}"}

@resilient_agent("Web Research Team")
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