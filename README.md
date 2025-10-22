# EduPolicy Agent — Agentic AI Assistant for University Regulations

EduPolicy Agent is a full-stack, agentic AI assistant designed to reason over
university regulation documents, structured student records and trusted web
sources.  It combines FastAPI, LangChain-based ReAct reasoning, Milvus vector
search, SQLite analytics and a Streamlit chat UI to deliver concise answers
supported by transparent reasoning traces.

## ✨ Features

- **ReAct Agent Orchestration** – multi-tool reasoning loop powered by
  LangChain and OpenRouter LLMs.
- **RAG over University Regulations** – PDF ingestion, chunking and embedding
  (intfloat/e5-large-v2) stored in Milvus for semantic retrieval.
- **NL2SQL Analytics** – Natural language questions translated into SQL against
  `student_records.db` for academic performance queries.
- **Live Web Search** – Tavily integration to pull in Ministry and education
  guidance when internal knowledge is insufficient.
- **Explainable Responses** – reasoning trace and tool observations surfaced in
  the UI for auditability.
- **Streamlit UI** – lightweight chat interface for demonstrations or can be
  replaced by OpenWebUI.
- **Dockerised Deployment** – Dockerfile + docker-compose for reproducible
  setup including Milvus standalone.

## 📁 Project Structure

```
edupolicy-agent/
├── app/
│   ├── main.py                # FastAPI entrypoint
│   ├── config.py              # Pydantic settings
│   ├── schemas.py             # Pydantic models
│   ├── utils.py               # Utilities & session memory
│   ├── agents/
│   │   ├── controller.py      # ReAct agent orchestrator
│   │   └── tools/             # Tool implementations
│   └── db/                    # Milvus & SQLite clients
├── data/
│   ├── regulations.pdf        # Provide your own regulation PDF (git-ignored)
│   ├── student_records.db     # Provide your own SQLite DB (git-ignored)
│   └── embeddings/            # Placeholder for exported vectors
├── ui/
│   ├── app_ui.py              # Streamlit chat UI
│   └── openwebui_config.json  # Front-end configuration
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🚀 Quick Start

1. **Clone repository & install dependencies**

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

2. **Configure environment**

   ```bash
   export OPENROUTER_API_KEY="your-openrouter-key"
   export TAVILY_API_KEY="optional-tavily-key"
   # Optional overrides
   export OPENROUTER_BASE_URL="https://openrouter.ai/api/v1"
   export MILVUS_URI="http://localhost:19530"
   ```

3. **Start Milvus** – run via docker compose or connect to an existing Milvus
   instance.

   ```bash
   docker compose up milvus -d
   ```

4. **Run FastAPI server**

   ```bash
   uvicorn app.main:app --reload
   ```

   On first start the agent ingests `data/regulations.pdf` into Milvus when the
   file is present. If the PDF is missing, ingestion is skipped until you add a
   document and restart the service.

5. **Launch the Streamlit UI (optional)**

   ```bash
   streamlit run ui/app_ui.py
   ```

   The UI communicates with the FastAPI backend at `http://localhost:8000`.

## 🧠 Agent Behaviour

- The **system prompt** enforces tool-aware behaviour, instructing the model to
  reason in Vietnamese and clearly state when tools are used.
- Toolset: `rag_tool`, `sql_tool`, `web_tool`, `summarizer`.
- Conversation history is persisted as JSON (`data/session_memory.json`) keyed by
  `session_id` allowing stateless API deployments with lightweight persistence.

## 📄 Data & Customisation

- Place an institution-specific PDF at `data/regulations.pdf`. The repository
  does not ship a sample for licensing reasons. The ingestion pipeline runs
  automatically on boot when the Milvus collection is empty and the file is
  available.
- Provision `data/student_records.db` with your academic dataset. A minimal
  starter schema can be created with:

  ```bash
  sqlite3 data/student_records.db <<'SQL'
  CREATE TABLE IF NOT EXISTS students (
      id INTEGER PRIMARY KEY,
      name TEXT NOT NULL,
      gpa REAL NOT NULL,
      status TEXT,
      warnings INTEGER DEFAULT 0,
      year INTEGER
  );
  INSERT INTO students (name, gpa, status, warnings, year) VALUES
      ('Nguyen Van A', 3.2, 'Good Standing', 0, 2023),
      ('Tran Thi B', 2.1, 'Probation', 2, 2023);
  SQL
  ```

  When the file is absent the SQL tool responds with a descriptive message so
  the backend remains operational.
- Fine-tune chunking or retrieval depth via `config.py`.

## 🌐 Docker Deployment

A production ready stack can be launched with:

```bash
docker compose up --build
```

Services started:

- **edupolicy-backend** – FastAPI + agent, exposed on `8000`.
- **milvus-standalone** – Vector database with persistent volume.
- **edupolicy-ui** – Streamlit UI exposed on `8501`.

Provide environment variables via `.env` for secrets. Mount real PDFs/databases
using volumes if required.

## ✅ API Reference

- `POST /chat` – body `{"session_id": "...", "message": "..."}`. Returns
  agent answer, reasoning and tool logs.
- `POST /rag/query` – semantic search over regulations, returns concatenated
  context and snippet list.
- `POST /sql/query` – direct access to SQL tool (useful for testing).
- `POST /web/query` – execute Tavily search.
- `GET /health` – health probe.

## 🧪 Testing the Agent

Example question flow once the backend is running:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "Điều kiện bị cảnh báo học vụ là gì?"}'
```

The agent will fetch relevant regulation snippets, optionally summarise and
respond. Inspect the reasoning trace to verify tool usage.

## 🔧 Development Notes

- Milvus operations require the server to be reachable at `settings.milvus_uri`.
  When running locally ensure the Docker container is healthy.
- OpenRouter requires the `HTTP-Referer` and `X-Title` headers; these are set in
  the LangChain client by default.
- The RAG ingestion uses e5-large-v2 embeddings (dimension 1024). Ensure the
  Milvus collection uses matching dimensionality.
- Tavily integration is optional; without a key the tool returns a descriptive
  message, allowing the agent to avoid failing requests.

## 📚 Extending the Project

- Swap Streamlit for OpenWebUI by pointing the UI configuration to the backend
  and mapping the API contract.
- Add additional tools (e.g., calendar lookups, policy comparison) by extending
  `app/agents/tools` and registering them in the controller.
- Integrate authentication or rate limiting at the FastAPI layer for production.

---

Crafted with ❤️ to simplify navigation of university academic policies.
