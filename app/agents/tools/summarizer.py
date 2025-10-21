"""Summarisation utility built on top of the primary LLM."""

from __future__ import annotations

from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import PromptTemplate

SUMMARY_PROMPT = PromptTemplate.from_template(
    """Bạn là trợ lý học thuật. Hãy tóm tắt nội dung sau thành 3-4 câu ngắn gọn, nêu rõ các quy định quan trọng hoặc số liệu chính.

Nội dung:
{content}
"""
)


class Summarizer:
    """Lightweight summariser used to condense long tool outputs."""

    def __init__(self, llm: BaseLanguageModel) -> None:
        self.llm = llm

    def summarise(self, content: str) -> str:
        if not content.strip():
            return ""
        return self.llm.invoke(SUMMARY_PROMPT.format(content=content)).content.strip()
