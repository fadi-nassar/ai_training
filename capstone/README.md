# Capstone: Multi-Agent Assistant (LangGraph)

A single **Supervisor** graph that classifies each user request and routes it to
one of **five specialised agents**, then returns the answer. Includes a Streamlit
chat front-end, retry/error handling around every agent call, and LangSmith
tracing of the whole flow.

---

## What it does

You ask one question. The Supervisor decides what *kind* of question it is
(`sql` / `rag` / `research` / `visualization` / `conversation`) and hands it to
the agent built for that job. You get back the answer plus a note of which agent
produced it.

---

## The five agents

| Category | Agent | What it does | Key tech |
|----------------|--------------------------|--------------------------------------------------------------------------------|----------|
| `sql` | **SQL Query Agent** | Answers questions about products / prices / stock by writing and running read-only `SELECT` queries against Postgres. | SQLAlchemy + Postgres, tool-calling loop |
| `rag` | **RAG Agent** | Answers company-policy questions (returns, shipping, warranty, support) strictly from a small document set. Instructed not to invent details. | FAISS + HuggingFace `all-MiniLM-L6-v2` embeddings |
| `research` | **Web Research Team** | A sub-graph: a **Researcher** gathers facts from Wikipedia, a **Report Writer** turns them into a short report. Coordinated by a sub-supervisor. | Nested LangGraph, Wikipedia tool |
| `visualization`| **Visualization Agent** | Turns a data request into a structured chart spec (`chart_type`, `title`, `labels`, `values`) using a Pydantic schema. The front-end renders it. | `with_structured_output(ChartData)` |
| `conversation` | **Conversation Agent** | General chat / greetings / follow-ups. Remembers the conversation via a checkpointer keyed on a thread id. | `MemorySaver` checkpointer |

---

## Architecture — the Supervisor routing pattern

```
                 ┌──────────────┐
   user_input ──▶│  classify    │  LLM picks one of 5 categories
                 └──────┬───────┘
                        │ route(state["category"])   (conditional edge)
      ┌──────────┬──────┼─────────┬──────────────┐
      ▼          ▼      ▼         ▼              ▼
    sql        rag   research  visualization  conversation
      │          │      │         │              │
      └──────────┴──────┴────┬────┴──────────────┘
                             ▼
                            END        state["response"] is returned
```

- **State** (`SupervisorState`): `user_input`, `category`, `response` — all strings.
- **`classify`** node: one LLM call returns a single word; `route()` maps it to a node.
- **Agent nodes** (`call_sql_agent`, `call_rag_agent`, …): each invokes its own
  compiled agent graph and writes the final message into `response`.
- **`research`** is itself a LangGraph (Researcher → Report Writer), invoked as a
  sub-graph from the `research` node.

### Retry / error handling

Every agent node is wrapped with the `@resilient_agent(...)` decorator in
`supervisor.py`:

1. Call the agent.
2. On any exception: log it, wait `AGENT_RETRY_DELAY` (1s), **retry once**.
3. If it still fails: return a clear `[agent error] …` message in `response`
   instead of letting the exception crash the graph.

Routing and classification logic are untouched — the decorator only adds the
safety net. Each attempt is still a separate child run in LangSmith, so failures
stay visible.

### Metrics

A second, outer decorator `@record_metrics(...)` logs one row per agent call:
`duration_s` (wall-clock, including any retry + backoff), `retry_count`
(failed attempts seen), `retried`, and `succeeded`. Rows go to an in-memory
list (`supervisor.get_metrics()`) and are best-effort appended to
`capstone/metrics_log.jsonl`. The Streamlit sidebar shows session
**avg response time**, **fallback count**, and **completion rate** from this.
It does not modify `@resilient_agent` — retry counts are read from that
decorator's warning logs.

---

## Project layout

```
capstone/
├── supervisor.py            # entry point: builds & compiles main_supervisor_graph
├── app.py                   # Streamlit chat front-end
├── verify_tracing.py        # LangSmith end-to-end tracing check
├── metrics_log.jsonl        # per-agent-call metrics (generated, git-ignored)
├── .env                     # secrets & config (not committed)
├── requirements.txt
└── agents/
    ├── sql_agent.py         # sql_agent_graph
    ├── rag_agent.py         # rag_agent_graph
    ├── conversation_agent.py# conversation_agent_graph (+ MemorySaver)
    ├── viz_agent.py         # viz_assistant() -> ChartData
    └── research_team/
        ├── researcher.py    # researcher_graph
        ├── report_writer.py # write_report()
        └── sub_supervisor.py# run_research_team()
```

---

## Setup

### 1. Python environment

```powershell
# venv already at C:\dev\ai-training\ai-env
C:\dev\ai-training\ai-env\Scripts\python.exe -m pip install -r capstone\requirements.txt
```

Dependencies of note: `langgraph`, `langchain-groq`, `langchain-huggingface`,
`faiss-cpu`, `sentence-transformers`, `sqlalchemy`, `psycopg2-binary`,
`wikipedia`, `streamlit`, `matplotlib`, `pandas`, `langsmith`.

### 2. Postgres via Docker (port 5433)

```powershell
docker run -d --name capstone-postgres `
  -e POSTGRES_PASSWORD=capstone123 `
  -e POSTGRES_DB=capstone `
  -p 5433:5432 `
  postgres
```

Seed a `products` table (columns: `id, name, category, price, stock`), e.g.:

```sql
CREATE TABLE products (
  id       SERIAL PRIMARY KEY,
  name     TEXT NOT NULL,
  category TEXT NOT NULL,
  price    NUMERIC(10,2) NOT NULL,
  stock    INTEGER NOT NULL
);

INSERT INTO products (name, category, price, stock) VALUES
  ('Laptop Pro 15',   'Electronics', 1499.00, 12),
  ('Wireless Mouse',  'Electronics',   24.99, 130),
  ('Standing Desk',   'Furniture',    389.00,  8),
  ('Office Chair',    'Furniture',    179.00, 25),
  ('Notebook A5',     'Stationery',     4.50, 300),
  ('Gel Pen (12pk)',  'Stationery',     8.75, 210);
```

Connect with: `docker exec -it capstone-postgres psql -U postgres -d capstone`

### 3. `.env` (in `capstone/`)

```dotenv
GROQ_API_KEY=your_groq_key
DATABASE_URL=postgresql://postgres:capstone123@localhost:5433/capstone

# LangSmith tracing
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key
LANGSMITH_PROJECT=capstone          # modern name; LANGCHAIN_PROJECT also works
```

---

## Run it

### CLI smoke test

```powershell
C:\dev\ai-training\ai-env\Scripts\python.exe capstone\supervisor.py
```

Runs five example inputs (one per category) and prints category + response.

### Streamlit chat UI

```powershell
C:\dev\ai-training\ai-env\Scripts\python.exe -m streamlit run capstone\app.py
```

- Type a question; it goes through `main_supervisor_graph`.
- The **sidebar** shows a routing log (which agent handled each turn) and the
  agent/category legend.
- Visualization answers are **rendered as real charts** (`st.bar_chart` /
  `st.line_chart`, matplotlib for pie) using the agent's structured output.

### Verify LangSmith tracing

```powershell
C:\dev\ai-training\ai-env\Scripts\python.exe capstone\verify_tracing.py "What is our most expensive product?"
```

Checks env vars, runs one request, pulls the trace back, prints the run tree
(`supervisor → classify → agent → llm/tool calls`) and PASS/FAIL checks, and
prints a link to the trace.

---

## Example questions to try

| Ask this | Routed to |
|-----------------------------------------------------------|----------------|
| "What's our most expensive product?" | `sql` |
| "How many products are in the Furniture category?" | `sql` |
| "What's your return policy?" | `rag` |
| "How long does express shipping take?" | `rag` |
| "Give me a short research summary of what LangGraph is." | `research` |
| "Make a bar chart of sales: Q1 100, Q2 200, Q3 150" | `visualization` |
| "Show product counts by category as a pie chart: Electronics 2, Furniture 2, Stationery 2" | `visualization` |
| "Hi, how are you?" | `conversation` |
| "My name is Fadi — what did I just say my name was?" | `conversation` (memory) |
