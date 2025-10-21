"""SQLite client and helper utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Sequence

from langchain_community.utilities import SQLDatabase

from ..config import settings

LOGGER = logging.getLogger(__name__)


class SQLiteClient:
    """Wrapper around ``SQLDatabase`` providing helper methods."""

    def __init__(self) -> None:
        self.path: Path = settings.sqlite_path
        if not self.path.exists():
            LOGGER.warning("SQLite database missing at %s. SQL tool will be disabled until provided.", self.path)
            self.db: Optional[SQLDatabase] = None
        else:
            self.db = SQLDatabase.from_uri(f"sqlite:///{self.path}")

    def run_raw_query(self, query: str) -> List[Sequence[str]]:
        if not self.db:
            raise FileNotFoundError(
                f"SQLite database not found at {self.path}. Populate data/student_records.db to enable SQL queries."
            )
        LOGGER.debug("Executing raw SQL: %s", query)
        return self.db.run(query)

    def get_table_info(self) -> str:
        if not self.db:
            raise FileNotFoundError(
                f"SQLite database not found at {self.path}. Populate data/student_records.db to enable SQL queries."
            )
        return self.db.get_table_info()
