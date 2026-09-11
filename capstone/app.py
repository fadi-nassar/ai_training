"""Streamlit chat front-end for the capstone multi-agent system.

Everything the user types is sent through `main_supervisor_graph` (supervisor.py).
The graph classifies the input, routes it to one of the five agents, and returns
`category` + `response`. This file only renders that - it does not do any routing.

Run it with:
    streamlit run capstone/app.py
"""

import os
from pathlib import Path
from dotenv import load_dotenv

CAPSTONE_DIR = Path(__file__).resolve().parent
# Load .env BEFORE importing the supervisor so LangSmith / Groq / DB env vars exist.
load_dotenv(CAPSTONE_DIR / ".env")

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage

st.set_page_config(page_title="Capstone Multi-Agent Assistant", page_icon="🤖", layout="wide")

# category (from SupervisorState) -> friendly name shown in the sidebar
CATEGORY_TO_AGENT = {
    "sql": "SQL Query Agent · Postgres",
    "rag": "RAG Agent · FAISS knowledge base",
    "research": "Web Research Team · Researcher + Report Writer",
    "visualization": "Visualization Agent · structured chart output",
    "conversation": "Conversation Agent · thread memory",
}


def agent_name_for(category: str) -> str:
    """Map a raw category string to a friendly agent label (mirrors supervisor.route)."""
    key = (category or "").strip().lower()
    for cat, name in CATEGORY_TO_AGENT.items():
        if cat[:5] in key:  # 'visual' matches 'visualization', etc.
            return name
    return f"Unrecognised category: {category!r}"


@st.cache_resource(show_spinner="Loading agents (first run downloads the embedding model)…")
def load_system():
    """Import the compiled supervisor graph once per Streamlit process.

    Importing `supervisor` builds every agent graph (FAISS index, HF embeddings,
    DB engine, etc.), so we cache it and reuse it across reruns.
    """
    from supervisor import main_supervisor_graph, get_metrics, reset_metrics
    from agents.viz_agent import viz_assistant
    return main_supervisor_graph, viz_assistant, get_metrics, reset_metrics


main_supervisor_graph, viz_assistant, get_metrics, reset_metrics = load_system()


def get_chart_data(user_input: str):
    """Re-run the Visualization Agent to get its *structured* Pydantic output.

    The supervisor node `call_viz_agent` flattens the chart into a text string
    (SupervisorState only has string fields), which we can't plot. Rather than
    modifying the supervisor's state schema, we call the same `viz_assistant`
    directly here to recover chart_type / labels / values for rendering.
    Cost: one extra (fast) Groq call on visualization turns only.
    """
    return viz_assistant({"messages": [HumanMessage(content=user_input)]})


def render_chart(chart):
    """Render a ChartData object with a Streamlit-native chart (or matplotlib for pie)."""
    st.caption(f"📊 {chart.title}")
    df = pd.DataFrame({"value": chart.values}, index=chart.labels)
    ctype = (chart.chart_type or "bar").strip().lower()

    if ctype == "line":
        st.line_chart(df)
    elif ctype == "pie":
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots()
        ax.pie(chart.values, labels=chart.labels, autopct="%1.1f%%", startangle=90)
        ax.axis("equal")
        st.pyplot(fig)
    else:  # 'bar' and any unexpected value fall back to a bar chart
        st.bar_chart(df)

    with st.expander("Raw chart data"):
        st.json(
            {
                "chart_type": chart.chart_type,
                "title": chart.title,
                "labels": chart.labels,
                "values": chart.values,
            }
        )


# --- session state -----------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []      # [{role, content, category?, chart?}]
if "routing_log" not in st.session_state:
    st.session_state.routing_log = []  # [{turn, question, category, agent}]


# --- main chat panel -------------------------------------------------------
st.title("🤖 Capstone Multi-Agent Assistant")
st.caption("One supervisor routes your question to the SQL, RAG, Research, Visualization, or Conversation agent.")

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("chart") is not None:
            render_chart(msg["chart"])
        if msg.get("category"):
            st.caption(f"Handled by: **{agent_name_for(msg['category'])}**")

prompt = st.chat_input("Ask about products, policies, a topic to research, or a chart to make…")

if prompt:
    st.session_state.history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Classifying and routing…"):
            try:
                result = main_supervisor_graph.invoke(
                    {"user_input": prompt, "category": "", "response": ""}
                )
                category = result.get("category", "")
                response = result.get("response", "") or "(no response returned)"
            except Exception as exc:  # graph-level failure (classify node, etc.)
                category = "error"
                response = f"[graph error] The supervisor graph failed: {type(exc).__name__}: {exc}"

        chart_obj = None
        is_error = response.lstrip().startswith(("[agent error]", "[graph error]"))
        if "visual" in (category or "").lower() and not is_error:
            try:
                chart_obj = get_chart_data(prompt)
            except Exception as exc:
                st.info(f"Chart data could not be generated: {exc}")

        st.markdown(response)
        if chart_obj is not None:
            render_chart(chart_obj)
        if category and category != "error":
            st.caption(f"Handled by: **{agent_name_for(category)}**")

    st.session_state.history.append(
        {"role": "assistant", "content": response, "category": category, "chart": chart_obj}
    )
    st.session_state.routing_log.append(
        {
            "turn": len(st.session_state.routing_log) + 1,
            "question": (prompt[:37] + "…") if len(prompt) > 38 else prompt,
            "category": category,
            "agent": agent_name_for(category),
        }
    )


# --- sidebar (rendered last so it always reflects the newest turn) ----------
with st.sidebar:
    st.header("Routing log")
    st.caption("Which agent handled each turn (from SupervisorState['category']).")
    if st.session_state.routing_log:
        st.dataframe(
            pd.DataFrame(st.session_state.routing_log),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.write("_No turns yet._")

    st.divider()
    st.subheader("Agents & categories")
    for cat, name in CATEGORY_TO_AGENT.items():
        st.markdown(f"- `{cat}` → {name}")

    st.divider()
    st.subheader("Metrics (session)")
    st.caption("Per agent call, from supervisor.get_metrics().")
    _metrics = get_metrics()
    if _metrics:
        _n = len(_metrics)
        _avg_rt = sum(m["duration_s"] for m in _metrics) / _n
        _fallbacks = sum(m["retry_count"] for m in _metrics)
        _completion = 100.0 * sum(1 for m in _metrics if m["succeeded"]) / _n
        _m1, _m2, _m3 = st.columns(3)
        _m1.metric("Avg response", f"{_avg_rt:.2f}s")
        _m2.metric("Fallbacks", _fallbacks)
        _m3.metric("Completion", f"{_completion:.0f}%")
        st.caption(f"{_n} agent call(s) recorded this session")
    else:
        st.write("_No agent calls yet._")

    st.divider()
    st.subheader("Observability")
    tracing_on = os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true"
    project = os.getenv("LANGSMITH_PROJECT") or os.getenv("LANGCHAIN_PROJECT") or "(default)"
    st.write(f"LangSmith tracing: {'✅ on' if tracing_on else '❌ off'}")
    st.write(f"Project: `{project}`")

    st.divider()
    if st.button("Clear conversation & metrics", use_container_width=True):
        st.session_state.history = []
        st.session_state.routing_log = []
        reset_metrics()  # clears the in-memory list; metrics_log.jsonl is kept
        st.rerun()
