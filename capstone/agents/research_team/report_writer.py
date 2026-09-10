import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

llm = ChatGroq(model="openai/gpt-oss-120b")

def write_report(research_findings: str, topic: str) -> str:
    """Takes raw research findings and synthesizes them into a clean report."""
    system_prompt = SystemMessage(content=(
        "You are a report writer. Given research findings, write a clear, "
        "well-organized summary report (3-5 sentences) on the given topic. "
        "Base the report only on the provided findings - do not add outside information."
    ))
    human_prompt = HumanMessage(content=f"Topic: {topic}\n\nResearch findings:\n{research_findings}")
    response = llm.invoke([system_prompt, human_prompt])
    return response.content