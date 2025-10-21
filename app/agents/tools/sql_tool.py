"""SQL tool enabling the agent to query structured records."""

from __future__ import annotations

import logging
from typing import List

from langchain.chains import create_sql_query_chain
from langchain_core.language_models import BaseLanguageModel

from ...db.sql_client import SQLiteClient

LOGGER = logging.getLogger(__name__)


class SQLTool:
    """Translate natural language questions into SQL and execute them."""

    def __init__(self, llm: BaseLanguageModel) -> None:
        self.client = SQLiteClient()
        self.llm = llm
        self.query_chain = (
            create_sql_query_chain(self.llm, self.client.db)
            if self.client.db is not None
            else None
        )

    def query_sql(self, question: str) -> str:
        LOGGER.info("SQLTool received question: %s", question)
        if not self.query_chain:
            return (
                "Cơ sở dữ liệu sinh viên chưa được cấu hình. Vui lòng cung cấp "
                "tệp data/student_records.db trước khi sử dụng truy vấn SQL."
            )
        try:
            sql_query = self.query_chain.invoke({"question": question})
            if isinstance(sql_query, dict):
                sql_query = sql_query.get("result") or sql_query.get("query") or ""
            if not isinstance(sql_query, str):
                raise ValueError("Unexpected response type from SQL chain")
        except Exception as exc:  # pragma: no cover - defensive handling
            LOGGER.exception("LLM failed to craft SQL")
            return f"Không thể tạo truy vấn SQL từ câu hỏi. Chi tiết: {exc}"
        LOGGER.debug("Generated SQL: %s", sql_query)
        try:
            rows = self.client.run_raw_query(sql_query)
        except Exception as exc:  # pragma: no cover
            LOGGER.exception("SQL execution failed")
            return f"Truy vấn SQL chạy thất bại: {exc}. Câu lệnh: {sql_query}"
        if not rows:
            return f"Không tìm thấy bản ghi phù hợp. SQL: {sql_query}"
        formatted_rows = self._format_rows(rows)
        return f"Kết quả truy vấn:\n{formatted_rows}\n\n(SQL: {sql_query})"

    @staticmethod
    def _format_rows(rows: List[List]) -> str:
        headers = ["col_" + str(i + 1) for i in range(len(rows[0]))]
        lines = [" | ".join(headers)]
        for row in rows:
            lines.append(" | ".join(str(item) for item in row))
        return "\n".join(lines)
