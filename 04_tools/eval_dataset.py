import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage

load_dotenv()

# --- same tools as before ---
@tool
def calculator(expression: str) -> str:
    """Evaluates a basic math expression, e.g. '3 + 4 * 2'."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

@tool
def word_length(word: str) -> str:
    """Returns the length of a given word."""
    return str(len(word))

@tool
def business_hours_lookup(day: str) -> str:
    """Returns the business hours for a given day of the week."""
    hours = {
        "monday": "9 AM - 5 PM", "tuesday": "9 AM - 5 PM", "wednesday": "9 AM - 5 PM",
        "thursday": "9 AM - 5 PM", "friday": "9 AM - 5 PM",
        "saturday": "Closed", "sunday": "Closed"
    }
    return hours.get(day.lower(), "unknown day")

tools = [calculator, word_length, business_hours_lookup]
llm = ChatGroq(model="openai/gpt-oss-120b")
llm_with_tools = llm.bind_tools(tools)

def assistant(state: MessagesState):
    system_prompt = SystemMessage(content=(
        "You must always use the provided tools for calculations, word length checks, "
        "and business hours lookups. Never compute or answer these yourself, even if "
        "the answer seems simple. Always call the appropriate tool first."
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

# --- THE EVALUATION DATASET ---
# each case: input question + which tool(s) we EXPECT to be called
eval_cases = [
    {"question": "What is 12 times 8?", "expected_tools": {"calculator"}},
    {"question": "How many letters in 'evaluation'?", "expected_tools": {"word_length"}},
    {"question": "Are you open on Sunday?", "expected_tools": {"business_hours_lookup"}},
    {"question": "What's 9 + 9 and how many letters does 'testing' have?", "expected_tools": {"calculator", "word_length"}},
]

# --- RUN EVAL ---
def get_called_tools(result):
    called = set()
    for msg in result["messages"]:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for call in msg.tool_calls:
                called.add(call["name"])
    return called

passed = 0
for case in eval_cases:
    result = graph.invoke({"messages": [HumanMessage(content=case["question"])]})
    actual_tools = get_called_tools(result)
    expected_tools = case["expected_tools"]

    correct = expected_tools.issubset(actual_tools)
    status = "PASS" if correct else "FAIL"
    if correct:
        passed += 1

    print(f"\n[{status}] {case['question']}")
    print(f"  Expected tools: {expected_tools}")
    print(f"  Actual tools:   {actual_tools}")
    print(f"  Final answer: {result['messages'][-1].content}")

print(f"\n{'='*50}")
print(f"Score: {passed}/{len(eval_cases)} passed")
