import os
from pathlib import Path
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# --- the exact shape we want the LLM's output to match ---
class ChartData(BaseModel):
    chart_type: str = Field(description="One of: 'bar', 'pie', or 'line'.")
    title: str = Field(description="A short title for the chart.")
    labels: list[str] = Field(description="The category labels for the chart.")
    values: list[float] = Field(description="The numeric values corresponding to each label.")

llm = ChatGroq(model="openai/gpt-oss-120b")
structured_llm = llm.with_structured_output(ChartData)

def viz_assistant(state: MessagesState):
    system_prompt = SystemMessage(content=(
        "You are a data visualization assistant. Given a request and some data, "
        "produce chart configuration data. Choose the most appropriate chart_type "
        "for the data (bar for comparisons, pie for proportions, line for trends over time)."
    ))
    messages = [system_prompt] + state["messages"]
    chart = structured_llm.invoke(messages)
    return chart

if __name__ == "__main__":
    test_requests = [
        "Create a chart showing product counts by category: Electronics has 2, Furniture has 3, Stationery has 1.",
        "Show a chart of quarterly sales: Q1 was $10000, Q2 was $15000, Q3 was $12000, Q4 was $18000.",
    ]

    for req in test_requests:
        print(f"\n{'='*60}")
        print(f"REQUEST: {req}")
        print('='*60)
        result = viz_assistant({"messages": [HumanMessage(content=req)]})
        print(f"Chart type: {result.chart_type}")
        print(f"Title:      {result.title}")
        print(f"Labels:     {result.labels}")
        print(f"Values:     {result.values}")