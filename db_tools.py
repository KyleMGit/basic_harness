"""Read-only database query helpers used by the native agent tools."""

import base64
import importlib
import json
import math
import os
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Callable, Dict


_ALLOWED_STATEMENTS = {"SELECT", "WITH", "SHOW", "DESCRIBE", "DESC", "EXPLAIN"}
_MAX_CONFIG_BYTES = 64 * 1024
_MAX_RESULT_CHARS = 16000
_MAX_CELL_CHARS = 4096
_MAX_COLUMN_CHARS = 512
_WRITE_KEYWORDS = re.compile(
    r"\b(?:INSERT|UPDATE|DELETE|MERGE|UPSERT|DROP|ALTER|CREATE|TRUNCATE|GRANT|REVOKE|CALL|EXECUTE)\b",
    re.IGNORECASE,
)


def validate_read_only_sql(sql: str, dialect: str = "generic") -> None:
    """Reject blank, stacked, or non-read-only SQL without naive semicolon splitting."""
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL must not be blank.")

    visible = []
    semicolons = []
    state = "code"
    index = 0
    while index < len(sql):
        char = sql[index]
        following = sql[index + 1] if index + 1 < len(sql) else ""
        if state == "line_comment":
            if char in "\r\n":
                state = "code"
                visible.append(char)
            else:
                visible.append(" ")
        elif state == "block_comment":
            if char == "*" and following == "/":
                visible.extend((" ", " "))
                index += 1
                state = "code"
            else:
                visible.append(" ")
        elif state == "single_quote":
            visible.append(char if char == "'" else " ")
            if dialect == "impala" and char == "\\" and following:
                visible.append(" ")
                index += 1
            elif char == "'":
                if following == "'":
                    visible.append(" ")
                    index += 1
                else:
                    state = "code"
        elif state == "double_quote":
            visible.append(char if char == '"' else " ")
            if char == '"':
                if following == '"':
                    visible.append(" ")
                    index += 1
                else:
                    state = "code"
        elif state == "backtick":
            visible.append(char if char == "`" else " ")
            if char == "`":
                if following == "`":
                    visible.append(" ")
                    index += 1
                else:
                    state = "code"
        elif char == "-" and following == "-":
            visible.extend((" ", " "))
            index += 1
            state = "line_comment"
        elif char == "/" and following == "*":
            visible.extend((" ", " "))
            index += 1
            state = "block_comment"
        elif char == "'":
            visible.append(char)
            state = "single_quote"
        elif char == '"':
            visible.append(char)
            state = "double_quote"
        elif dialect == "impala" and char == "`":
            visible.append(char)
            state = "backtick"
        else:
            visible.append(char)
            if char == ";":
                semicolons.append(len(visible) - 1)
        index += 1

    statement = "".join(visible)
    if state in {"block_comment", "single_quote", "double_quote", "backtick"}:
        raise ValueError("SQL contains an unterminated comment or quoted value.")
    if not statement.strip():
        raise ValueError("SQL must not be blank or comments only.")
    if semicolons:
        if len(semicolons) > 1 or statement[semicolons[0] + 1:].strip():
            raise ValueError("SQL must contain a single statement.")
        statement = statement[:semicolons[0]]
    if dialect == "teradata":
        locking = re.match(r"\s*LOCKING\b.+?\bFOR\s+ACCESS\s+(?=(?:SELECT|WITH)\b)",
                           statement, re.IGNORECASE | re.DOTALL)
        if locking:
            statement = statement[locking.end():]
    match = re.match(r"\s*([A-Za-z]+)\b", statement)
    if not match or match.group(1).upper() not in _ALLOWED_STATEMENTS:
        raise ValueError("SQL must be a read-only SELECT, WITH, SHOW, DESCRIBE/DESC, or EXPLAIN statement.")
    keyword_check = statement
    if dialect == "impala" and match and match.group(1).upper() == "SHOW":
        keyword_check = re.sub(r"^\s*SHOW\s+(?:CREATE|GRANT)\b", "SHOW", statement,
                               count=1, flags=re.IGNORECASE)
    if _WRITE_KEYWORDS.search(keyword_check):
        raise ValueError("SQL must be a read-only query or introspection statement.")


def _load_config() -> Dict[str, Dict[str, Any]]:
    path = os.environ.get("AGENT_DB_CONFIG")
    if not path:
        return {}
    try:
        if os.path.getsize(path) > _MAX_CONFIG_BYTES:
            raise RuntimeError(f"AGENT_DB_CONFIG file {path} exceeds 64 KiB.")
        with open(path, "rb") as handle:
            raw = handle.read(_MAX_CONFIG_BYTES + 1)
    except RuntimeError:
        raise
    except OSError:
        raise RuntimeError(f"Could not read AGENT_DB_CONFIG file: {path}") from None
    if len(raw) > _MAX_CONFIG_BYTES:
        raise RuntimeError(f"AGENT_DB_CONFIG file {path} exceeds 64 KiB.")
    try:
        config = json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError:
        raise RuntimeError(f"AGENT_DB_CONFIG file {path} must be valid UTF-8.") from None
    except json.JSONDecodeError:
        raise RuntimeError(f"AGENT_DB_CONFIG file {path} must contain valid JSON.") from None
    if not isinstance(config, dict):
        raise RuntimeError(f"AGENT_DB_CONFIG file {path} root must be an object.")
    for section in ("teradata", "impala"):
        if section in config and not isinstance(config[section], dict):
            raise RuntimeError(f"AGENT_DB_CONFIG key '{section}' in {path} must be an object.")
    return config


def _setting(section: Dict[str, Any], env_name: str, key: str, default: Any = None) -> Any:
    return os.environ[env_name] if env_name in os.environ else section.get(key, default)


def _required_settings(values: Dict[str, Any]) -> None:
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise RuntimeError("Missing required environment configuration: " + ", ".join(missing))


def _max_rows(value: int) -> int:
    try:
        return max(1, min(1000, int(value)))
    except (TypeError, ValueError):
        raise ValueError("max_rows must be an integer.")


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("+Infinity" if value > 0 else "-Infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")
    return str(value)


def _bounded_text(value: Any, limit: int) -> tuple[str, bool]:
    text = str(value)
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _serialize_result(database: str, description: Any, rows: Any, row_truncated: bool) -> str:
    truncated = row_truncated
    database, shortened = _bounded_text(database, _MAX_COLUMN_CHARS)
    truncated |= shortened
    columns = []
    column_json_chars = 2
    for column in description or ():
        if len(columns) >= 256:
            truncated = True
            break
        name, shortened = _bounded_text(column[0], _MAX_COLUMN_CHARS)
        name_chars = len(json.dumps(name, separators=(",", ":"), allow_nan=False))
        if column_json_chars + (1 if columns else 0) + name_chars > 4096:
            truncated = True
            break
        columns.append(name)
        column_json_chars += (1 if len(columns) > 1 else 0) + name_chars
        truncated |= shortened

    def convert_row(row: Any, cell_limit: int) -> tuple[list[Any], bool]:
        converted = []
        shortened_row = len(row) > len(columns)
        for index in range(len(columns)):
            value = row[index] if index < len(row) else None
            value = _json_value(value)
            if isinstance(value, str):
                value, shortened = _bounded_text(value, cell_limit)
                shortened_row |= shortened
            converted.append(value)
        return converted, shortened_row

    empty_result = {"database": database, "columns": columns, "rows": [],
                    "row_count": 0, "truncated": False}
    empty_size = len(json.dumps(empty_result, sort_keys=True, separators=(",", ":"),
                                allow_nan=False))
    converted_rows = []
    rows_chars = 0
    for row in rows:
        converted, shortened_row = convert_row(row, _MAX_CELL_CHARS)
        fragment = json.dumps(converted, separators=(",", ":"), allow_nan=False)
        next_count = len(converted_rows) + 1
        next_rows_chars = rows_chars + (1 if converted_rows else 0) + len(fragment)
        projected = empty_size + next_rows_chars + len(str(next_count)) - 1
        if projected > _MAX_RESULT_CHARS and not converted_rows:
            low, high = 0, _MAX_CELL_CHARS
            while low < high:
                middle = (low + high + 1) // 2
                candidate, _ = convert_row(row, middle)
                candidate_fragment = json.dumps(candidate, separators=(",", ":"), allow_nan=False)
                if empty_size + len(candidate_fragment) <= _MAX_RESULT_CHARS:
                    low = middle
                else:
                    high = middle - 1
            converted, shortened_row = convert_row(row, low)
            fragment = json.dumps(converted, separators=(",", ":"), allow_nan=False)
            next_rows_chars = len(fragment)
            projected = empty_size + next_rows_chars
        if projected > _MAX_RESULT_CHARS:
            truncated = True
            break
        converted_rows.append(converted)
        rows_chars = next_rows_chars
        truncated |= shortened_row
    result = {"database": database, "columns": columns, "rows": converted_rows,
              "row_count": len(converted_rows), "truncated": bool(truncated)}
    return json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _execute(connect: Callable[[], Any], sql: str, max_rows: int, database: str,
             database_kind: str) -> str:
    connection = None
    cursor = None
    try:
        connection = connect()
        cursor = connection.cursor()
        cursor.execute(sql)
        fetched = list(cursor.fetchmany(max_rows + 1))
        rows = fetched[:max_rows]
        return _serialize_result(database, cursor.description, rows, len(fetched) > max_rows)
    except Exception:
        raise RuntimeError(f"{database_kind} query failed during connection, execution, or fetch.") from None
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def query_teradata(sql: str, max_rows: int = 100) -> str:
    validate_read_only_sql(sql, "teradata")
    limit = _max_rows(max_rows)
    section = _load_config().get("teradata", {})
    values = {name: _setting(section, name, key) for name, key in (
        ("TERADATA_HOST", "host"), ("TERADATA_USER", "user"), ("TERADATA_PASSWORD", "password"))}
    _required_settings(values)
    try:
        driver = importlib.import_module("teradatasql")
    except ImportError:
        raise RuntimeError("Teradata driver is unavailable; install it with: pip install teradatasql") from None
    options: Dict[str, Any] = {
        "host": values["TERADATA_HOST"], "user": values["TERADATA_USER"],
        "password": values["TERADATA_PASSWORD"],
    }
    database = _setting(section, "TERADATA_DATABASE", "database", "")
    logmech = _setting(section, "TERADATA_LOGMECH", "logmech")
    if database:
        options["database"] = database
    if logmech:
        options["logmech"] = logmech
    return _execute(lambda: driver.connect(**options), sql, limit, database, "Teradata")


def query_impala(sql: str, max_rows: int = 100) -> str:
    validate_read_only_sql(sql, "impala")
    limit = _max_rows(max_rows)
    section = _load_config().get("impala", {})
    connection_string = _setting(section, "IMPALA_CONNECTION_STRING", "connection_string")
    _required_settings({"IMPALA_CONNECTION_STRING": connection_string})
    try:
        pyodbc = importlib.import_module("pyodbc")
    except ImportError:
        raise RuntimeError("Impala driver is unavailable; install it with: pip install pyodbc") from None
    try:
        timeout = int(_setting(section, "IMPALA_ODBC_TIMEOUT", "timeout", 30))
        if timeout <= 0:
            raise ValueError
    except (TypeError, ValueError):
        raise RuntimeError("IMPALA_ODBC_TIMEOUT must be a positive integer.") from None
    database = _setting(section, "IMPALA_DATABASE", "database", "default")
    return _execute(lambda: pyodbc.connect(connection_string, autocommit=True, timeout=timeout),
                    sql, limit, database, "Impala")
