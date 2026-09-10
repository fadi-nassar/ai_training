import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from pathlib import Path

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# STEP 1: connect to Postgres
engine = create_engine(os.getenv("DATABASE_URL"))
print("DATABASE_URL:", os.getenv("DATABASE_URL"))

# STEP 2: the tool - runs real SQL against the real database
@tool
def run_sql_query(query: str) -> str:
    """Executes a SQL SELECT query against the products database and returns the results.
    The products table has columns: id, name, category, price, stock."""
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            columns = result.keys()
            if not rows:
                return "No results found."
            formatted = "\n".join(str(dict(zip(columns, row))) for row in rows)
            return formatted
    except Exception as e:
        return f"SQL Error: {e}"

tools = [run_sql_query]
llm = ChatGroq(model="openai/gpt-oss-120b")
llm_with_tools = llm.bind_tools(tools)

def assistant(state: MessagesState):
    system_prompt = SystemMessage(content=(
        "You are a SQL assistant for a products database. "
        "The products table has columns: id, name, category, price, stock. "
        "Always use the run_sql_query tool to answer questions - never guess data. "
        "Write safe, read-only SELECT queries only. "
        "After getting results, explain them in clear natural language."
    ))
    messages = [system_prompt] + state["messages"]
    return {"messages": [llm_with_tools.invoke(messages)]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")
sql_agent_graph = builder.compile()

# quick test
if __name__ == "__main__":
    test_questions = [
        "What's our most expensive product?",
        "How many products are in the Furniture category?",
        "What's the total value of all inventory (price times stock)?",
    ]
    for q in test_questions:
        print(f"\n{'='*60}")
        print(f"USER: {q}")
        print('='*60)
        result = sql_agent_graph.invoke({"messages": [HumanMessage(content=q)]})
        for msg in result["messages"]:
            msg.pretty_print()