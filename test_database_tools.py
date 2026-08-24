import json
import math
import os
import tempfile
import unittest
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import db_tools
from tools import registry


class FakeCursor:
    def __init__(self, rows=(), description=(), execute_error=None, fetch_error=None):
        self.rows = list(rows)
        self.description = description
        self.execute_error = execute_error
        self.fetch_error = fetch_error
        self.closed = False
        self.executed = None

    def execute(self, sql):
        self.executed = sql
        if self.execute_error:
            raise self.execute_error

    def fetchmany(self, size):
        if self.fetch_error:
            raise self.fetch_error
        return self.rows[:size]

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.closed = False

    def cursor(self):
        return self._cursor

    def close(self):
        self.closed = True


class DatabaseToolTests(unittest.TestCase):
    def test_registry_schemas_and_read_only_dispatch(self):
        schemas = {item["function"]["name"]: item["function"] for item in registry.schemas}
        for name in ("query_teradata", "query_impala"):
            self.assertIn(name, schemas)
            self.assertEqual(schemas[name]["parameters"]["required"], ["sql"])
            self.assertEqual(set(schemas[name]["parameters"]["properties"]), {"sql", "max_rows"})
            self.assertEqual(schemas[name]["parameters"]["properties"]["max_rows"]["minimum"], 1)
            self.assertEqual(schemas[name]["parameters"]["properties"]["max_rows"]["maximum"], 1000)
            self.assertFalse(registry.is_write_tool(name))

        with patch("tools.db_tools.query_teradata", return_value='{"database":"td"}') as td:
            self.assertEqual(registry.execute("query_teradata", {"sql": "SELECT 1"}, read_only=True),
                             '{"database":"td"}')
            td.assert_called_once_with("SELECT 1", 100)
        with patch("tools.db_tools.query_impala", return_value='{"database":"default"}') as impala:
            registry.execute("query_impala", {"sql": "SHOW TABLES"}, read_only=True)
            impala.assert_called_once_with("SHOW TABLES", 100)

    def test_teradata_formats_serializable_bounded_results_and_maps_environment(self):
        cursor = FakeCursor(
            rows=[(Decimal("1.25"), date(2026, 1, 2), datetime(2026, 1, 2, 3, 4, 5), b"hi"),
                  (Decimal("2"), None, None, b"extra")],
            description=[("amount",), ("day",), ("timestamp",), ("payload",)],
        )
        connection = FakeConnection(cursor)
        driver = MagicMock()
        driver.connect.return_value = connection
        env = {
            "TERADATA_HOST": "td.example", "TERADATA_USER": "analyst",
            "TERADATA_PASSWORD": "hidden", "TERADATA_DATABASE": "warehouse",
            "TERADATA_LOGMECH": "LDAP",
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
                db_tools.importlib, "import_module", return_value=driver):
            result = json.loads(db_tools.query_teradata("/* lead */ SELECT ';' AS x;", max_rows=1))

        driver.connect.assert_called_once_with(host="td.example", user="analyst", password="hidden",
                                               database="warehouse", logmech="LDAP")
        self.assertEqual(result, {
            "database": "warehouse", "columns": ["amount", "day", "timestamp", "payload"],
            "rows": [["1.25", "2026-01-02", "2026-01-02T03:04:05", "aGk="]],
            "row_count": 1, "truncated": True,
        })
        self.assertTrue(cursor.closed)
        self.assertTrue(connection.closed)

    def test_impala_lazily_imports_pyodbc_and_maps_opaque_connection(self):
        cursor = FakeCursor(rows=[(index,) for index in range(150)], description=[("one",)])
        connection = FakeConnection(cursor)
        pyodbc = MagicMock()
        pyodbc.connect.return_value = connection
        env = {
            "IMPALA_CONNECTION_STRING": "DSN=opaque-secret", "IMPALA_DATABASE": "analytics",
            "IMPALA_ODBC_TIMEOUT": "45",
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
                db_tools.importlib, "import_module", return_value=pyodbc) as importer:
            result = json.loads(db_tools.query_impala("WITH x AS (SELECT 1) SELECT * FROM x", max_rows=125))

        importer.assert_called_once_with("pyodbc")
        pyodbc.connect.assert_called_once_with("DSN=opaque-secret", autocommit=True, timeout=45)
        self.assertEqual(result["database"], "analytics")
        self.assertEqual(result["row_count"], 125)
        self.assertTrue(result["truncated"])
        self.assertEqual(cursor.executed, "WITH x AS (SELECT 1) SELECT * FROM x")
        self.assertTrue(cursor.closed)
        self.assertTrue(connection.closed)

    def test_write_stacked_and_blank_sql_are_rejected_before_import(self):
        invalid = ["", " -- only a comment\n", "DELETE FROM t", "SELECT 1; DROP TABLE t",
                   "/* comment */ INSERT INTO t VALUES (1)",
                   "WITH old AS (SELECT 1) DELETE FROM t"]
        for query in invalid:
            with self.subTest(query=query), patch.object(db_tools.importlib, "import_module") as importer:
                with self.assertRaisesRegex(ValueError, "read-only|single|blank"):
                    db_tools.query_impala(query)
                importer.assert_not_called()

    def test_allowed_introspection_and_semicolons_in_quotes(self):
        for query in ["SHOW TABLES", "DESCRIBE x", "DESC x", "EXPLAIN SELECT 1",
                      "-- lead\nSELECT 'a;''b'", '/* x */ SELECT "a;b"']:
            with self.subTest(query=query):
                db_tools.validate_read_only_sql(query)

    def test_dialect_specific_read_only_sql(self):
        for query in ["SHOW CREATE TABLE `odd-name`", "SHOW GRANT USER bob",
                      r"SELECT 'it\'s;safe' FROM `odd-name`"]:
            with self.subTest(query=query):
                db_tools.validate_read_only_sql(query, "impala")
        db_tools.validate_read_only_sql(
            "LOCKING warehouse.t FOR ACCESS SELECT * FROM warehouse.t", "teradata")

        for query in [r"SELECT 'safe\'; DELETE FROM t", r"SELECT 'x\'; DROP TABLE t"]:
            with self.subTest(query=query), self.assertRaises(ValueError):
                db_tools.validate_read_only_sql(query, "teradata")
        for dialect in ("impala", "teradata"):
            for query in ["EXPLAIN INSERT INTO t SELECT 1", "WITH x AS (DELETE FROM t) SELECT * FROM x",
                          "SELECT 1 /* ; */; -- comment\nDROP TABLE t"]:
                with self.subTest(dialect=dialect, query=query), self.assertRaises(ValueError):
                    db_tools.validate_read_only_sql(query, dialect)

    def test_missing_configuration_and_drivers_are_actionable_without_secrets(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "TERADATA_HOST.*TERADATA_USER.*TERADATA_PASSWORD"):
                db_tools.query_teradata("SELECT 1")
            with self.assertRaisesRegex(RuntimeError, "IMPALA_CONNECTION_STRING"):
                db_tools.query_impala("SELECT 1")

        with patch.dict(os.environ, {"TERADATA_HOST": "h", "TERADATA_USER": "u",
                                     "TERADATA_PASSWORD": "super-secret"}, clear=True), patch.object(
                db_tools.importlib, "import_module", side_effect=ImportError("no module")):
            with self.assertRaises(RuntimeError) as caught:
                db_tools.query_teradata("SELECT 1")
        self.assertIn("pip install teradatasql", str(caught.exception))
        self.assertNotIn("super-secret", str(caught.exception))

        with patch.dict(os.environ, {"IMPALA_CONNECTION_STRING": "DSN=super-secret"}, clear=True), patch.object(
                db_tools.importlib, "import_module", side_effect=ImportError("super-secret")):
            with self.assertRaises(RuntimeError) as caught:
                db_tools.query_impala("SELECT 1")
        self.assertIn("pip install pyodbc", str(caught.exception))
        self.assertNotIn("super-secret", str(caught.exception))

    def test_json_config_loading_and_environment_precedence(self):
        config = {
            "teradata": {"host": "json-host", "user": "json-user", "password": "json-pass",
                         "database": "json-db", "logmech": "LDAP"},
            "impala": {"connection_string": "DSN=json-secret", "database": "json-default", "timeout": 12},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "database.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            cursor = FakeCursor(description=[])
            driver = MagicMock()
            driver.connect.return_value = FakeConnection(cursor)
            env = {"AGENT_DB_CONFIG": path, "TERADATA_HOST": "env-host",
                   "IMPALA_CONNECTION_STRING": "DSN=env-secret", "IMPALA_ODBC_TIMEOUT": "31"}
            with patch.dict(os.environ, env, clear=True), patch.object(
                    db_tools.importlib, "import_module", return_value=driver):
                td_result = json.loads(db_tools.query_teradata("SELECT 1"))
                td_call = driver.connect.call_args
                driver.reset_mock()
                driver.connect.return_value = FakeConnection(FakeCursor(description=[]))
                impala_result = json.loads(db_tools.query_impala("SELECT 1"))

        self.assertEqual(td_call.kwargs, {"host": "env-host", "user": "json-user", "password": "json-pass",
                                          "database": "json-db", "logmech": "LDAP"})
        driver.connect.assert_called_once_with("DSN=env-secret", autocommit=True, timeout=31)
        self.assertEqual(td_result["database"], "json-db")
        self.assertEqual(impala_result["database"], "json-default")

    def test_json_config_errors_are_actionable_and_secret_safe(self):
        cases = [(b'{"impala":', "valid JSON"), (b'[]', "root.*object"),
                 (b'{"impala":"secret-value"}', "impala.*object"), (b'\xff', "UTF-8")]
        with tempfile.TemporaryDirectory() as directory:
            for index, (content, expected) in enumerate(cases):
                path = os.path.join(directory, f"bad-{index}.json")
                with open(path, "wb") as handle:
                    handle.write(content)
                with self.subTest(path=path), patch.dict(os.environ, {"AGENT_DB_CONFIG": path}, clear=True):
                    with self.assertRaisesRegex(RuntimeError, expected) as caught:
                        db_tools.query_impala("SELECT 1")
                    self.assertNotIn("secret-value", str(caught.exception))
            path = os.path.join(directory, "large.json")
            with open(path, "wb") as handle:
                handle.write(b" " * (65536 + 1))
            with patch.dict(os.environ, {"AGENT_DB_CONFIG": path}, clear=True):
                with self.assertRaisesRegex(RuntimeError, "64 KiB"):
                    db_tools.query_impala("SELECT 1")

    def test_result_is_strict_json_and_character_bounded(self):
        cases = [
            ([("x" * 50000,)], [("huge",)]),
            ([tuple("v" * 1000 for _ in range(100))], [("column-" + "n" * 500,)] * 100),
            ([(math.nan, math.inf, -math.inf)], [("nan",), ("positive",), ("negative",)]),
        ]
        for rows, description in cases:
            cursor = FakeCursor(rows=rows, description=description)
            driver = MagicMock()
            driver.connect.return_value = FakeConnection(cursor)
            with self.subTest(width=len(description)), patch.dict(
                    os.environ, {"IMPALA_CONNECTION_STRING": "DSN=secret"}, clear=True), patch.object(
                    db_tools.importlib, "import_module", return_value=driver):
                payload = db_tools.query_impala("SELECT 1")
            self.assertLessEqual(len(payload), 16000)
            result = json.loads(payload, parse_constant=lambda value: self.fail(value))
            if any(isinstance(value, float) and not math.isfinite(value) for value in rows[0]):
                self.assertEqual(result["rows"], [["NaN", "+Infinity", "-Infinity"]])
            else:
                self.assertTrue(result["truncated"])

    def test_large_result_does_not_repeatedly_serialize_the_whole_payload(self):
        description = [(f"column_{index}",) for index in range(20)]
        rows = [tuple(f"{row_index}:" + "x" * 496 for _ in range(20))
                for row_index in range(1000)]
        original_dumps = json.dumps
        with patch.object(db_tools.json, "dumps", wraps=original_dumps) as dumps:
            payload = db_tools._serialize_result("analytics", description, rows, False)

        whole_result_calls = [call for call in dumps.call_args_list
                              if call.args and isinstance(call.args[0], dict)
                              and "rows" in call.args[0]]
        self.assertLessEqual(len(whole_result_calls), 2)
        self.assertLessEqual(len(payload), 16000)
        self.assertTrue(json.loads(payload)["truncated"])

    def test_wide_cells_preserve_at_least_one_aligned_row(self):
        description = [(f"column_{index}",) for index in range(4)]
        rows = [tuple("x" * 4096 for _ in range(4)) for _ in range(100)]

        payload = db_tools._serialize_result("analytics", description, rows, False)
        result = json.loads(payload)

        self.assertLessEqual(len(payload), 16000)
        self.assertGreaterEqual(result["row_count"], 1)
        self.assertEqual(result["row_count"], len(result["rows"]))
        self.assertTrue(all(len(row) == len(result["columns"]) for row in result["rows"]))
        self.assertTrue(result["truncated"])

    def test_small_result_serialization_is_unchanged(self):
        payload = db_tools._serialize_result(
            "analytics", [("integer",), ("text",), ("empty",)],
            [(7, "exact value", None)], False)

        self.assertEqual(payload, json.dumps({
            "database": "analytics", "columns": ["integer", "text", "empty"],
            "rows": [[7, "exact value", None]], "row_count": 1, "truncated": False,
        }, sort_keys=True, separators=(",", ":"), allow_nan=False))

    def test_resources_close_on_execute_and_fetch_failures(self):
        for failure_kind in ("execute", "fetch"):
            cursor = FakeCursor(execute_error=RuntimeError("execute failed") if failure_kind == "execute" else None,
                                fetch_error=RuntimeError("fetch failed") if failure_kind == "fetch" else None)
            connection = FakeConnection(cursor)
            driver = MagicMock()
            driver.connect.return_value = connection
            with self.subTest(failure_kind=failure_kind), patch.dict(
                    os.environ, {"IMPALA_CONNECTION_STRING": "DSN=host"}, clear=True), patch.object(
                    db_tools.importlib, "import_module", return_value=driver):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    db_tools.query_impala("SELECT 1")
            self.assertTrue(cursor.closed)
            self.assertTrue(connection.closed)


if __name__ == "__main__":
    unittest.main()
