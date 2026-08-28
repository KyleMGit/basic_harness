"""Diagnose Teradata configuration, connection, execution, and fetch failures.

Run this script on the same server and in the same environment as the agent:

    python diagnose_teradata.py

It reports where a minimal read-only query fails while redacting configured
connection values from the driver error. It never prints the password or the
contents of database.json.
"""

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


def main() -> int:
    config = db_tools._load_config()
    section = config.get("teradata", {})

    host, host_source = _setting(section, "TERADATA_HOST", "host")
    user, user_source = _setting(section, "TERADATA_USER", "user")
    password, password_source = _setting(section, "TERADATA_PASSWORD", "password")
    database, database_source = _setting(section, "TERADATA_DATABASE", "database", "")
    logmech, logmech_source = _setting(section, "TERADATA_LOGMECH", "logmech", "")

    print("Effective setting sources:")
    print("  host:", host_source)
    print("  user:", user_source)
    print("  password:", password_source)
    print("  database:", database_source)
    print("  logmech:", logmech_source)
    print("  logmech is LDAP:", str(logmech).upper() == "LDAP")

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
        import teradataml
    except ImportError:
        print("Import failed: install the driver with: pip install teradataml")
        return 2

    options = {
        "host": host,
        "username": user,
        "password": password,
    }
    if database:
        options["database"] = database
    if logmech:
        options["logmech"] = logmech

    stage = "create_context"
    cursor = None
    context_created = False
    try:
        teradataml.create_context(**options)
        context_created = True
        print("create_context: SUCCESS")

        stage = "execute_sql"
        cursor = teradataml.execute_sql(
            statement="SELECT CURRENT_DATE AS current_date")
        print("execute_sql: SUCCESS")

        stage = "fetchmany"
        rows = cursor.fetchmany(2)
        print("fetchmany: SUCCESS")
        print("Rows returned:", len(rows))
        return 0
    except Exception as exc:
        message = _redact(
            str(exc), (password, user, host, database))
        print(f"{stage}: FAILED")
        print("Exception type:", type(exc).__name__)
        print("Redacted driver message:")
        print(message[:2000])
        return 1
    finally:
        if cursor is not None:
            try:
                cursor.close()
            except Exception:
                pass
        if context_created:
            try:
                teradataml.remove_context()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
