import os
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage

load_dotenv()

llm = ChatGroq(model="openai/gpt-oss-120b")

class State(TypedDict):
    messages: list
    category: str

def classify(state):
    print("---Classify---")
    last_message = state["messages"][-1].content
    prompt = f"Classify this message into exactly one word: 'billing', 'technical', or 'general'. Message: {last_message}"
    result = llm.invoke([HumanMessage(content=prompt)])
    category = result.content.strip().lower()
    return {"category": category}

def billing_node(state):
    print("---Billing Agent---")
    last_message = state["messages"][-1].content
    prompt = f"You are a billing support agent. Respond helpfully to: {last_message}"
    result = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": state["messages"] + [result]}

def technical_node(state):
    print("---Technical Agent---")
    last_message = state["messages"][-1].content
    prompt = f"You are a technical support agent. Respond helpfully to: {last_message}"
    result = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": state["messages"] + [result]}

def general_node(state):
    print("---General Agent---")
    last_message = state["messages"][-1].content
    prompt = f"You are a general support agent. Respond helpfully to: {last_message}"
    result = llm.invoke([HumanMessage(content=prompt)])
    return {"messages": state["messages"] + [result]}

def route(state):
    category = state["category"]
    if "billing" in category:
        return "billing"
    elif "technical" in category:
        return "technical"
    else:
        return "general"

builder = StateGraph(State)
builder.add_node("classify", classify)
builder.add_node("billing", billing_node)
builder.add_node("technical", technical_node)
builder.add_node("general", general_node)

builder.add_edge(START, "classify")
builder.add_conditional_edges("classify", route)
builder.add_edge("billing", END)
builder.add_edge("technical", END)
builder.add_edge("general", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

config = {"configurable": {"thread_id": "test-1"}}

test_inputs = [
    "I was charged twice for my subscription this month",
    "My app keeps crashing when I open it",
    "What are your business hours?",
]

for user_input in test_inputs:
    print(f"\n{'='*50}")
    print(f"USER: {user_input}")
    print('='*50)
    result = graph.invoke(
        {"messages": [HumanMessage(content=user_input)], "category": ""},
        config
    )
    print(f"CATEGORY: {result['category']}")
    print(f"RESPONSE: {result['messages'][-1].content}")