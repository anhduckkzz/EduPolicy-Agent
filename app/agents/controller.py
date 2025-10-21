"""Agent orchestrator implementing a ReAct style reasoning loop."""

from __future__ import annotations

import logging
from typing import List

from langchain.agents import AgentType, Tool, initialize_agent
from langchain.schema import AIMessage, BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI

from ..config import settings
from ..schemas import ChatResponse
from ..utils import SessionMemory, build_tool_observation
from .tools.rag_tool import RAGTool
from .tools.sql_tool import SQLTool
from .tools.summarizer import Summarizer
from .tools.web_tool import WebSearchTool

LOGGER = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are EduPolicyAgent, an AI assistant that helps users understand and "
    "query university regulations. You can use the following tools: rag_tool "
    "(retrieve internal regulations), sql_tool (query student records), "
    "web_tool (search reputable sources online) and summarizer (compress long "
    "information). Always think through whether a tool is required before "
    "answering. Provide clear reasoning and end with a concise answer in "
    "Vietnamese."
)


class AgentController:
    """High level orchestrator for handling chat requests."""

    def __init__(self) -> None:
        if not settings.openrouter_api_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY is required. Please set it in the environment or .env file."
            )
        self.llm = ChatOpenAI(
            model=settings.openrouter_model,
            temperature=0.1,
            openai_api_base=settings.openrouter_base_url,
            openai_api_key=settings.openrouter_api_key,
            default_headers={
                "HTTP-Referer": "https://github.com/",
                "X-Title": "EduPolicy Agent",
            },
            max_retries=3,
        )
        self.memory = SessionMemory()
        self.rag_tool = RAGTool()
        self.summarizer = Summarizer(self.llm)
        self.sql_tool = SQLTool(self.llm)
        self.web_tool = WebSearchTool()
        self.tools = [
            Tool(
                name="rag_tool",
                func=self._rag_tool_wrapper,
                description=(
                    "Use this tool to retrieve information from the university regulations. "
                    "Input should be a natural language question or keywords."
                ),
            ),
            Tool(
                name="sql_tool",
                func=self.sql_tool.query_sql,
                description=(
                    "Use for questions about student records, warnings, GPA, statistics. "
                    "Input should be a clear question in Vietnamese."
                ),
            ),
            Tool(
                name="web_tool",
                func=self.web_tool.search_web,
                description=(
                    "Use to search trusted web sources such as the Ministry of Education. "
                    "Provide a short search query."
                ),
            ),
            Tool(
                name="summarizer",
                func=self.summarizer.summarise,
                description="Use to summarise long pieces of text into concise Vietnamese.",
            ),
        ]

    # ------------------------------------------------------------------
    def _rag_tool_wrapper(self, query: str) -> str:
        context, _ = self.rag_tool.query_rag(query)
        return context

    # ------------------------------------------------------------------
    def _build_history(self, session_id: str) -> List[BaseMessage]:
        history = self.memory.get_history(session_id)
        messages: List[BaseMessage] = []
        for item in history:
            if item["role"] == "user":
                messages.append(HumanMessage(content=item["content"]))
            else:
                messages.append(AIMessage(content=item["content"]))
        return messages

    # ------------------------------------------------------------------
    def chat(self, session_id: str, message: str) -> ChatResponse:
        LOGGER.info("Handling chat message for session %s", session_id)
        history_messages = self._build_history(session_id)
        agent_executor = initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            handle_parsing_errors=True,
            verbose=False,
            agent_kwargs={"system_message": SYSTEM_PROMPT},
        )
        result = agent_executor.invoke(
            {"input": message, "chat_history": history_messages},
            return_intermediate_steps=True,
        )
        output = result.get("output", "")
        intermediate_steps = result.get("intermediate_steps", [])

        reasoning: List[str] = []
        tool_interactions: List[str] = []
        for action, observation in intermediate_steps:
            reasoning.append(f"Suy nghĩ: sử dụng {action.tool} với đầu vào {action.tool_input}")
            obs_text = observation if isinstance(observation, str) else str(observation)
            tool_interactions.append(build_tool_observation(action.tool, obs_text))

        self.memory.append(session_id, "user", message)
        self.memory.append(session_id, "assistant", output)

        return ChatResponse(
            answer=output,
            session_id=session_id,
            reasoning=reasoning,
            tool_interactions=tool_interactions,
        )

    # ------------------------------------------------------------------
    def rag_query(self, query: str, top_k: int | None = None) -> tuple[str, List[str]]:
        return self.rag_tool.query_rag(query, top_k=top_k)
