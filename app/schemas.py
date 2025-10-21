"""Pydantic request/response schemas used by the FastAPI layer."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Response model for health checks."""

    status: str = Field(..., description="Health status string")


class ChatMessage(BaseModel):
    """Representation of a single chat message."""

    role: str = Field(..., description="Message author, e.g. 'user' or 'assistant'.")
    content: str = Field(..., description="Textual content of the message.")


class ChatRequest(BaseModel):
    """Inbound request body for the `/chat` endpoint."""

    session_id: str = Field(..., description="Conversation identifier used for memory persistence.")
    message: str = Field(..., description="User prompt to send to the agent.")
    stream: bool = Field(False, description="Whether to stream tokens (not yet implemented).")


class ToolResponse(BaseModel):
    """Generic response envelope for tool specific endpoints."""

    result: str
    source: str = Field(..., description="Name of the tool producing the response.")
    context: Optional[List[str]] = Field(default=None, description="Optional context snippets.")


class ChatResponse(BaseModel):
    """Structured response from the agent orchestrator."""

    answer: str
    session_id: str
    reasoning: List[str] = Field(default_factory=list, description="Agent reasoning trace.")
    tool_interactions: List[str] = Field(default_factory=list, description="Human readable view of tool usage.")


class RAGQueryRequest(BaseModel):
    query: str
    top_k: Optional[int] = Field(default=None)


class SQLQueryRequest(BaseModel):
    question: str


class WebQueryRequest(BaseModel):
    query: str
    max_results: Optional[int] = Field(default=None)
