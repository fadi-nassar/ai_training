"""Sanity-check that LangSmith is tracing the whole supervisor flow end to end.

What it does:
  1. Prints the tracing-related environment variables and flags common mistakes.
  2. Sends one request through `main_supervisor_graph`.
  3. Pulls that trace back from LangSmith and prints the run tree, then checks
     that the expected spans are present:
        supervisor graph  ->  classify  ->  <agent node>  ->  LLM / tool calls

Run:
    python capstone/verify_tracing.py            # default: a SQL question
    python capstone/verify_tracing.py "What is your return policy?"
"""

import os
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

CAPSTONE_DIR = Path(__file__).resolve().parent
load_dotenv(CAPSTONE_DIR / ".env")

from supervisor import main_supervisor_graph  # noqa: E402


def check_env() -> str:
    """Print tracing env vars, return the effective project name."""
    print("=" * 64)
    print("ENVIRONMENT")
    print("=" * 64)

    tracing = os.getenv("LANGSMITH_TRACING") or os.getenv("LANGCHAIN_TRACING_V2")
    api_key = os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")
    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or "default"
    )
    endpoint = os.getenv("LANGSMITH_ENDPOINT") or "https://api.smith.langchain.com"

    print(f"  tracing flag      : {tracing!r}")
    print(f"  api key present   : {bool(api_key)}  ({'…' + api_key[-4:] if api_key else 'MISSING'})")
    print(f"  project           : {project!r}")
    print(f"  endpoint          : {endpoint}")

    problems = []
    if str(tracing).lower() != "true":
        problems.append("Tracing is not enabled - set LANGSMITH_TRACING=true")
    if not api_key:
        problems.append("No LANGSMITH_API_KEY / LANGCHAIN_API_KEY found")
    if os.getenv("LANGCHAIN_PROJECT") and not os.getenv("LANGSMITH_PROJECT"):
        print(
            "  note: only LANGCHAIN_PROJECT is set. It still works, but the modern "
            "name is LANGSMITH_PROJECT - consider adding it for consistency."
        )

    if problems:
        print("\n  PROBLEMS:")
        for p in problems:
            print(f"   - {p}")
    else:
        print("\n  env looks OK.")
    return project


def run_once(user_input: str):
    print("\n" + "=" * 64)
    print("INVOKING SUPERVISOR")
    print("=" * 64)
    print(f"  input: {user_input}")
    result = main_supervisor_graph.invoke(
        {"user_input": user_input, "category": "", "response": ""}
    )
    print(f"  category : {result['category']}")
    print(f"  response : {result['response'][:160]}")
    return result


def inspect_trace(project: str):
    try:
        from langsmith import Client
    except ImportError:
        print("\nlangsmith package not installed - cannot inspect the trace.")
        return

    client = Client()
    print("\n" + "=" * 64)
    print("FETCHING TRACE FROM LANGSMITH")
    print("=" * 64)

    root = None
    for attempt in range(1, 11):  # traces are flushed asynchronously; poll a bit
        runs = list(
            client.list_runs(
                project_name=project,
                filter='eq(is_root, true)',
                limit=1,
            )
        )
        if runs:
            root = runs[0]
            break
        print(f"  waiting for trace to land… ({attempt}/10)")
        time.sleep(3)

    if root is None:
        print("  No root run found. Tracing may be disabled or the project name is wrong.")
        return

    all_runs = sorted(
        client.list_runs(project_name=project, filter=f'eq(trace_id, "{root.trace_id}")'),
        key=lambda r: r.dt_start if hasattr(r, "dt_start") else r.start_time,
    )

    by_parent = {}
    for r in all_runs:
        by_parent.setdefault(r.parent_run_id, []).append(r)

    def walk(run, depth=0):
        status = "ERROR" if run.error else "ok"
        print(f"  {'  ' * depth}- [{run.run_type}] {run.name}  ({status})")
        if run.error:
            print(f"  {'  ' * (depth + 1)}! {str(run.error)[:120]}")
        for child in by_parent.get(run.id, []):
            walk(child, depth + 1)

    print(f"\n  Trace {root.trace_id}:")
    walk(root)

    names = {r.name for r in all_runs}
    types = {r.run_type for r in all_runs}
    print("\n" + "=" * 64)
    print("CHECKS")
    print("=" * 64)
    checks = [
        ("supervisor graph run is the root", root.run_type in ("chain", "graph")),
        ("'classify' node was traced", "classify" in names),
        ("an agent node was traced",
         any(n in names for n in ("sql", "rag", "research", "visualization", "conversation"))),
        ("at least one LLM call was traced", "llm" in types),
        ("no errored spans", not any(r.error for r in all_runs)),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")

    url = None
    try:
        url = client.get_run_url(run=root)
    except Exception:
        pass
    if url:
        print(f"\n  View it: {url}")


if __name__ == "__main__":
    query = sys.argv[1] if len(sys.argv) > 1 else "What is our most expensive product?"
    project = check_env()
    run_once(query)
    inspect_trace(project)
