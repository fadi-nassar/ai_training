import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

from researcher import researcher_graph
from report_writer import write_report
from langchain_core.messages import HumanMessage

def run_research_team(topic: str) -> str:
    """Coordinates the Researcher and Report Writer to produce a final report on a topic."""
    # STEP 1: Researcher gathers information
    research_result = researcher_graph.invoke({"messages": [HumanMessage(content=topic)]})
    research_findings = research_result["messages"][-1].content

    # STEP 2: Report Writer synthesizes it into a final report
    final_report = write_report(research_findings, topic)
    return final_report

if __name__ == "__main__":
    topic = "What is LangGraph?"
    print(f"RESEARCHING: {topic}\n")
    report = run_research_team(topic)
    print(f"\n{'='*60}")
    print("FINAL REPORT")
    print('='*60)
    print(report)