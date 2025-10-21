"""Web search tool leveraging Tavily's API."""

from __future__ import annotations

import logging
from typing import List

from tavily import TavilyClient

from ...config import settings

LOGGER = logging.getLogger(__name__)


class WebSearchTool:
    """Simple wrapper for the Tavily search client."""

    def __init__(self) -> None:
        if not settings.tavily_api_key:
            LOGGER.warning("Tavily API key missing. Web search tool will be inactive until provided.")
        self.client = TavilyClient(api_key=settings.tavily_api_key) if settings.tavily_api_key else None

    def search_web(self, query: str, *, max_results: int | None = None) -> str:
        LOGGER.info("Searching the web for: %s", query)
        if not self.client:
            return "Web search hiện chưa được cấu hình (thiếu Tavily API key)."
        max_results = max_results or settings.tavily_max_results
        try:
            response = self.client.search(query=query, max_results=max_results)
        except Exception as exc:  # pragma: no cover - network failures
            LOGGER.exception("Tavily search failed")
            return f"Không thể truy vấn Tavily: {exc}"
        snippets: List[str] = []
        for result in response.get("results", []):
            title = result.get("title", "")
            content = result.get("content", "")
            url = result.get("url", "")
            snippets.append(f"{title}\n{content}\nNguồn: {url}")
        if not snippets:
            return "Không tìm thấy thông tin phù hợp trên web."
        return "\n\n".join(snippets)
