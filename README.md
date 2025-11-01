# EduPolicy Agent — Agentic AI Assistant for University Regulations

Due to difficulties in finding out and search for suitable regulations attachments when problem arise, I find this is a very complex due to the disorder and organize of how these documents are stored, therefore I create this agent, to help me search for relevent infomation with some simple prompt, no need to look hopelessly for hours just to find some very short answer.

## Features

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

## Project Structure

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
│   ├── all_regulations_files.pdf        # 
│   ├── student_records.db     
│   └── embeddings/            # Placeholder for exported vectors
├── ui/
│   ├── app_ui.py              # Streamlit UI
│   └── openwebui_config.json  # Configuration
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## Quick Start

1. **Clone repository & install dependencies**

   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1 (Command Prompt) or source .venv/bin/activate (Bash)
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

5. **Launch the Streamlit UI (optional)**

   ```bash
   streamlit run ui/app_ui.py
   ```

   The UI communicates with the FastAPI backend at `http://localhost:8000`.

## Embedding
All the pdf files will be embedded before put into Milvus database, BAAI/bge-m3 embedding model is chosen because of its ability in embedding multilingual, therefore it very suitable to use with Vietnamese.
The `data/Embedding.ipynb` notebook shows how to embed your PDFs into Milvus Lite using the `BAAI/bge-m3` model.

## Agent Behaviour

- The **system prompt** enforces tool-aware behaviour, instructing the model to
  reason in Vietnamese and clearly state when tools are used.
- Toolset: `rag_tool`, `sql_tool`, `web_tool`, `summarizer`.
- Conversation history is persisted as JSON (`data/session_memory.json`) keyed by
  `session_id` allowing stateless API deployments with lightweight persistence.

## Data & Customisation

- Place an institution-specific related-to-law PDF at `data/` folder. The repository
  does not ship a sample for licensing reasons. The ingestion pipeline runs
  automatically on boot when the Milvus collection is empty and the file is
  available.
- Fine-tune chunking or retrieval depth via `config.py`.

## Docker Deployment

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

## API Reference

- `POST /chat` – body `{"session_id": "...", "message": "..."}`. Returns
  agent answer, reasoning and tool logs.
- `POST /rag/query` – semantic search over regulations, returns concatenated
  context and snippet list.
- `POST /sql/query` – direct access to SQL tool (useful for testing).
- `POST /web/query` – execute Tavily search.
- `GET /health` – health probe.

## Testing the Agent

Example question flow once the backend is running:

```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"session_id": "demo", "message": "Điều kiện bị cảnh báo học vụ là gì?"}'
```

The agent will fetch relevant regulation snippets, optionally summarise and
respond. Inspect the reasoning trace to verify tool usage.
