"""Test the configured Teradata connection through teradatasql directly.

This bypasses teradataml and SQLAlchemy while using the same database.json and
environment-variable precedence as db_tools.py.

Examples:
    python diagnose_teradatasql.py
    python diagnose_teradatasql.py --without-database
"""

import argparse
import os
import re
from typing import Any

import db_tools


_PLACEHOLDER_PREFIXES = ("YOUR_", "OPTIONAL_", "TERADATA_")


def _setting(
        section: dict[str, Any], env_name: str, json_name: str,
        default: Any = None) -> tuple[Any, str]:
    if env_name in os.environ:
        return os.environ[env_name], env_name
    return section.get(json_name, default), "database.json"


def _redact(message: str, values: tuple[Any, ...]) -> str:
    for value in values:
        if value is not None and str(value):
            message = message.replace(str(value), "<redacted>")
    return re.sub(
        r"(?i)\b(password|pwd)\s*[=:]\s*[^,;\s]+",
        r"\1=<redacted>",
        message,
    )


def _format_exception_chain(
        exc: BaseException, redacted_values: tuple[Any, ...]) -> list[str]:
    lines = []
    current: BaseException | None = exc
    seen = set()
    first = True
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        prefix = "" if first else "Caused by "
        lines.append(
            f"{prefix}{type(current).__name__}: "
            f"{_redact(str(current), redacted_values)}")
        current = current.__cause__ or current.__context__
        first = False
    return lines


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test database.json through teradatasql without teradataml or "
            "SQLAlchemy."))
    parser.add_argument(
        "--without-database",
        action="store_true",
        help=(
            "Omit the optional database parameter to isolate host/LDAP "
            "authentication."),
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        section = db_tools._load_config().get("teradata", {})
    except RuntimeError as exc:
        print("Configuration error:", exc)
        return 2

    host, host_source = _setting(section, "TERADATA_HOST", "host")
    user, user_source = _setting(section, "TERADATA_USER", "user")
    password, password_source = _setting(
        section, "TERADATA_PASSWORD", "password")
    database, database_source = _setting(
        section, "TERADATA_DATABASE", "database", "")
    logmech, logmech_source = _setting(
        section, "TERADATA_LOGMECH", "logmech", "")

    print("Effective setting sources:")
    print("  host:", host_source)
    print("  user:", user_source)
    print("  password:", password_source)
    print("  database:", database_source)
    print("  logmech:", logmech_source)
    print("  logmech is LDAP:", str(logmech).upper() == "LDAP")
    print("  database included:", bool(database) and not args.without_database)

    values = {
        "host": host,
        "user": user,
        "password": password,
        "database": database,
        "logmech": logmech,
    }
    invalid_required = False
    for name, value in values.items():
        text = "" if value is None else str(value)
        if not text and name in {"host", "user", "password"}:
            print(f"WARNING: {name} is missing")
            invalid_required = True
        elif text.startswith(_PLACEHOLDER_PREFIXES):
            print(f"WARNING: {name} still contains a placeholder")
            if name in {"host", "user", "password"}:
                invalid_required = True

    if invalid_required:
        print("Connection test skipped because required configuration is invalid.")
        return 2

    try:
        import teradatasql
    except ImportError:
        print("Import failed: install the driver with: pip install teradatasql")
        return 2

    options = {
        "host": host,
        "user": user,
        "password": password,
    }
    if logmech:
        options["logmech"] = logmech
    if database and not args.without_database:
        options["database"] = database

    stage = "connect"
    connection = None
    cursor = None
    try:
        connection = teradatasql.connect(**options)
        print("connect: SUCCESS")

        stage = "execute"
        cursor = connection.cursor()
        cursor.execute("SELECT CURRENT_DATE AS current_date")
        print("execute: SUCCESS")

        stage = "fetch"
        rows = cursor.fetchmany(2)
        print("fetch: SUCCESS")
        print("Rows returned:", len(rows))
        return 0
    except Exception as exc:
        print(f"{stage}: FAILED")
        print("Redacted exception chain:")
        print("\n".join(_format_exception_chain(
            exc, (password, user, host, database)))[:4000])
        return 1
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


if __name__ == "__main__":
    raise SystemExit(main())
