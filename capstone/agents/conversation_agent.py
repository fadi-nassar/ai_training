import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

llm = ChatGroq(model="openai/gpt-oss-120b")

def assistant(state: MessagesState):
    system_prompt = SystemMessage(content=(
        "You are a friendly, helpful general-purpose assistant. "
        "Answer naturally and conversationally. Use the conversation history "
        "to maintain context across turns."
    ))
    messages = [system_prompt] + state["messages"]
    response = llm.invoke(messages)
    return {"messages": [response]}

builder = StateGraph(MessagesState)
builder.add_node("assistant", assistant)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

memory = MemorySaver()
conversation_agent_graph = builder.compile(checkpointer=memory)

if __name__ == "__main__":
    config = {"configurable": {"thread_id": "test-conversation-1"}}

    turns = [
        "Hi, my name is Fadi and I'm based in Lebanon.",
        "What's my name?",
        "What country did I say I'm from?",
    ]

    for turn in turns:
        print(f"\n{'='*60}")
        print(f"USER: {turn}")
        print('='*60)
        result = conversation_agent_graph.invoke(
            {"messages": [HumanMessage(content=turn)]},
            config
        )
        print(f"AI: {result['messages'][-1].content}")