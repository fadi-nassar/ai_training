import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

@tool
def calculator(expression: str) -> str:
    """
    Evaluates a basic math expression, e.g. '3 + 4 * 2'.
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"


@tool
def word_length(word: str) -> str:
    """
    Returns the length of a given word.
    """
    return str(len(word))

@tool
def business_hours_lookup(day: str) -> str:
    """
    Returns the business hours for a given day of the week.
    """
    hours = {
        "monday": "9 AM - 5 PM",
        "tuesday": "9 AM - 5 PM",
        "wednesday": "9 AM - 5 PM",
        "thursday": "9 AM - 5 PM",
        "friday": "9 AM - 5 PM",
        "saturday": "Closed",
        "sunday": "Closed"
    }
    return hours.get(day.lower(), "unknown day")

tools = [calculator, word_length, business_hours_lookup]
llm = ChatGroq(model="openai/gpt-oss-120b")
llm_with_tools = llm.bind_tools(tools)

def assistant(state: MessagesState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", tools_condition)
builder.add_edge("tools", "assistant")

graph = builder.compile()

test_inputs = [
    "what is 24 times 7?",
    "how many letters are in 'programming'?",
    "Are you open on saturday?",
    "what's 15 + 15 and also how many letters does the word 'LangGraph' have?",
]

for user_input in test_inputs:
    print(f"\n{'='*60}")
    print(f"USER: {user_input}")
    print('='*60)
    result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
    for msg in result["messages"]:
        msg.pretty_print()