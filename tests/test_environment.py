"""Task 1.1 verification: project dependencies are installed and importable."""

import sqlite3
import sys


def test_python_version_is_pinned_range():
    assert (3, 12) <= sys.version_info[:2] < (3, 13)


def test_core_dependencies_import():
    import aiosqlite  # noqa: F401
    import google.adk  # noqa: F401
    import google.genai  # noqa: F401
    import opentelemetry.sdk.trace  # noqa: F401
    import pydantic  # noqa: F401
    import pydantic_settings  # noqa: F401
    import structlog  # noqa: F401


def test_sqlite_supports_local_db_path():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE sanity (id INTEGER PRIMARY KEY)")
    conn.close()
