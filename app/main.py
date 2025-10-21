"""FastAPI entrypoint wiring HTTP endpoints to the agent controller."""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .agents.controller import AgentController
from .schemas import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    RAGQueryRequest,
    SQLQueryRequest,
    ToolResponse,
    WebQueryRequest,
)

LOGGER = logging.getLogger(__name__)

app = FastAPI(title="EduPolicy Agent", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

controller: AgentController | None = None


@app.on_event("startup")
async def startup_event() -> None:
    """Initialise the agent controller on application start."""

    global controller
    if controller is None:
        try:
            controller = AgentController()
        except Exception as exc:  # pragma: no cover - ensures informative logs during startup
            LOGGER.exception("Failed to initialise AgentController")
            raise


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Simple health check endpoint."""

    return HealthResponse(status="ok")


@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest) -> ChatResponse:
    """Main chat endpoint bridging the UI and the agent."""

    controller: AgentController = globals().get("controller")  # type: ignore[assignment]
    if not controller:
        raise HTTPException(status_code=503, detail="Agent controller not initialised")
    try:
        response = controller.chat(request.session_id, request.message)
    except Exception as exc:  # pragma: no cover - surfaces agent errors
        LOGGER.exception("Agent execution failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return response


@app.post("/rag/query", response_model=ToolResponse)
async def rag_query(request: RAGQueryRequest) -> ToolResponse:
    controller: AgentController = globals().get("controller")  # type: ignore[assignment]
    if not controller:
        raise HTTPException(status_code=503, detail="Agent controller not initialised")
    context, snippets = controller.rag_query(request.query, top_k=request.top_k)
    return ToolResponse(result=context, source="rag_tool", context=snippets)


@app.post("/sql/query", response_model=ToolResponse)
async def sql_query(request: SQLQueryRequest) -> ToolResponse:
    controller: AgentController = globals().get("controller")  # type: ignore[assignment]
    if not controller:
        raise HTTPException(status_code=503, detail="Agent controller not initialised")
    result = controller.sql_tool.query_sql(request.question)
    return ToolResponse(result=result, source="sql_tool")


@app.post("/web/query", response_model=ToolResponse)
async def web_query(request: WebQueryRequest) -> ToolResponse:
    controller: AgentController = globals().get("controller")  # type: ignore[assignment]
    if not controller:
        raise HTTPException(status_code=503, detail="Agent controller not initialised")
    result = controller.web_tool.search_web(request.query, max_results=request.max_results)
    return ToolResponse(result=result, source="web_tool")
