import json
import csv
import io
import math
import os
import sys
import tempfile
import types
import unittest
from datetime import date, datetime
from decimal import Decimal
from unittest.mock import MagicMock, patch

import db_tools
import tools
from tools import registry


class FakeCursor:
    def __init__(self, rows=(), description=(), execute_error=None, fetch_error=None):
        self.rows = list(rows)
        self.description = description
        self.execute_error = execute_error
        self.fetch_error = fetch_error
        self.closed = False
        self.executed = None
        self.offset = 0
        self.fetch_sizes = []

    def execute(self, sql):
        self.executed = sql
        if self.execute_error:
            raise self.execute_error

    def fetchmany(self, size):
        self.fetch_sizes.append(size)
        if self.fetch_error:
            raise self.fetch_error
        result = self.rows[self.offset:self.offset + size]
        self.offset += len(result)
        return result

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
    def test_export_registry_schema_write_classification_and_dispatch(self):
        schemas = {item["function"]["name"]: item["function"] for item in registry.schemas}
        for name in ("export_teradata_csv", "export_impala_csv"):
            schema = schemas[name]["parameters"]
            self.assertEqual(schema["required"], ["sql", "file_path"])
            self.assertFalse(schema["additionalProperties"])
            self.assertEqual(set(schema["properties"]),
                             {"sql", "file_path", "batch_size", "overwrite"})
            self.assertEqual(schema["properties"]["batch_size"], {
                "type": "integer", "description": "Rows fetched and written per batch (1..10000).",
                "minimum": 1, "maximum": 10000, "default": 1000})
            self.assertEqual(schema["properties"]["overwrite"]["default"], False)
            self.assertTrue(registry.is_write_tool(name))
        self.assertFalse(registry.is_write_tool("query_teradata"))
        self.assertFalse(registry.is_write_tool("query_impala"))
        blocked = registry.execute("export_impala_csv", {
            "sql": "SELECT 1", "file_path": "x.csv"}, read_only=True)
        self.assertIn("not executed", blocked)

        with tempfile.TemporaryDirectory(dir=os.getcwd()) as root, patch.object(
                tools.terminal_session, "workspace_root", root), patch.object(
                tools.terminal_session, "_cwd", root), patch(
                "tools.db_tools.export_impala_csv", return_value='{"completed":true}') as export:
            result = registry.execute("export_impala_csv", {
                "sql": "SELECT 1", "file_path": "nested/out.csv",
                "batch_size": 7, "overwrite": False})
        self.assertEqual(result, '{"completed":true}')
        export.assert_called_once_with(
            "SELECT 1", os.path.join(root, "nested", "out.csv"), 7, root, False)

    def test_export_destination_guards_run_before_backend_and_create_parents(self):
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as root, patch.object(
                tools.terminal_session, "workspace_root", root), patch.object(
                tools.terminal_session, "_cwd", root), patch(
                "tools.db_tools.export_impala_csv") as export:
            for path in ("bad.txt", "../escape.csv", ".ssh/secret.csv"):
                with self.subTest(path=path):
                    result = registry.execute("export_impala_csv", {"sql": "SELECT 1", "file_path": path})
                    self.assertTrue(result.startswith("Error"))
            existing = os.path.join(root, "existing.csv")
            with open(existing, "w", encoding="utf-8") as handle:
                handle.write("keep")
            result = registry.execute("export_impala_csv", {
                "sql": "SELECT 1", "file_path": "existing.csv"})
            self.assertIn("already exists", result)
            export.assert_not_called()
            registry.execute("export_impala_csv", {
                "sql": "SELECT 1", "file_path": "new/ok.csv", "overwrite": True})
            self.assertTrue(os.path.isdir(os.path.join(root, "new")))

    def test_impala_export_streams_multiple_batches_and_csv_types_atomically(self):
        rows = [
            ('comma,value', 'line1\nline2', None, Decimal("1.20"),
             date(2026, 1, 2), datetime(2026, 1, 2, 3, 4, 5), b"hi", math.nan),
            ('quote"value', "plain", "", Decimal("2"), None, None, b"", math.inf),
            ("third", "row", None, Decimal("3"), None, None, b"x", -math.inf),
        ]
        cursor = FakeCursor(rows=rows, description=[(name,) for name in
            ("text", "lines", "null", "amount", "day", "stamp", "bytes", "float")])
        connection = FakeConnection(cursor)
        driver = MagicMock()
        driver.connect.return_value = connection
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as root:
            output = os.path.join(root, "nested", "data.csv")
            os.makedirs(os.path.dirname(output))
            with patch.dict(os.environ, {"IMPALA_HOST": "impala.example",
                                         "IMPALA_DATABASE": "analytics"}, clear=True), patch.object(
                    db_tools.importlib, "import_module", return_value=driver):
                manifest = json.loads(db_tools.export_impala_csv(
                    "SELECT * FROM t", output, 2, root))
            with open(output, newline="", encoding="utf-8") as handle:
                parsed = list(csv.reader(handle))
            self.assertEqual(parsed[0], [item[0] for item in cursor.description])
            self.assertEqual(parsed[1], ["comma,value", "line1\nline2", "", "1.20",
                                         "2026-01-02", "2026-01-02T03:04:05", "aGk=", "NaN"])
            self.assertEqual(parsed[2][-1], "+Infinity")
            self.assertEqual(parsed[3][-1], "-Infinity")
            self.assertEqual(cursor.fetch_sizes, [2, 2, 2])
            self.assertEqual(cursor.executed, "SELECT * FROM t")
            self.assertTrue(cursor.closed)
            self.assertTrue(connection.closed)
            self.assertEqual(manifest["row_count"], 3)
            self.assertEqual(manifest["file_path"], "nested/data.csv")
            self.assertEqual(manifest["byte_size"], os.path.getsize(output))
            self.assertEqual(manifest["batch_size"], 2)
            self.assertTrue(manifest["completed"])
            self.assertEqual(len(manifest["sql_sha256"]), 64)
            self.assertNotIn("rows", manifest)
            self.assertNotIn("SELECT", json.dumps(manifest))

    def test_export_empty_result_writes_header_and_bounded_manifest(self):
        long_name = "c" * 1000
        cursor = FakeCursor(description=[(long_name,)] * 400)
        driver = MagicMock()
        driver.connect.return_value = FakeConnection(cursor)
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as root:
            output = os.path.join(root, "empty.csv")
            with patch.dict(os.environ, {"IMPALA_HOST": "host"}, clear=True), patch.object(
                    db_tools.importlib, "import_module", return_value=driver):
                raw = db_tools.export_impala_csv("SELECT 1", output, 1000, root)
            manifest = json.loads(raw)
            self.assertEqual(manifest["row_count"], 0)
            self.assertLessEqual(len(manifest["columns"]), 256)
            self.assertLessEqual(len(raw), 16000)
            self.assertEqual(cursor.fetch_sizes, [1000])
            with open(output, newline="", encoding="utf-8") as handle:
                self.assertEqual(len(next(csv.reader(handle))), 400)

    def test_export_rejects_sql_and_batch_before_driver_import(self):
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as root, patch.object(
                db_tools.importlib, "import_module") as importer:
            for sql, batch in (("DELETE FROM t", 10), ("SELECT 1", 0), ("SELECT 1", 10001)):
                with self.subTest(sql=sql, batch=batch), self.assertRaises((ValueError, RuntimeError)):
                    db_tools.export_impala_csv(sql, os.path.join(root, "x.csv"), batch, root)
            importer.assert_not_called()

    def test_export_failure_removes_temp_and_preserves_destination(self):
        cursor = FakeCursor(rows=[("one",), ("two",)], description=[("value",)])
        cursor.fetchmany = MagicMock(side_effect=[[('one',)], RuntimeError("row secret")])
        connection = FakeConnection(cursor)
        driver = MagicMock()
        driver.connect.return_value = connection
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as root:
            output = os.path.join(root, "keep.csv")
            with open(output, "w", encoding="utf-8") as handle:
                handle.write("original")
            with patch.dict(os.environ, {"IMPALA_HOST": "secret-host"}, clear=True), patch.object(
                    db_tools.importlib, "import_module", return_value=driver):
                with self.assertRaisesRegex(RuntimeError, "Impala CSV export failed") as caught:
                    db_tools.export_impala_csv("SELECT secret_column FROM t", output, 1, root, True)
            self.assertNotIn("secret", str(caught.exception).lower())
            with open(output, encoding="utf-8") as handle:
                self.assertEqual(handle.read(), "original")
            self.assertEqual(os.listdir(root), ["keep.csv"])
            self.assertTrue(cursor.closed)
            self.assertTrue(connection.closed)

    def test_teradata_export_context_lifecycle_and_partial_cleanup(self):
        cursor = FakeCursor(rows=[(1,), (2,)], description=[("n",)])
        teradataml = MagicMock()
        teradataml.get_context.return_value = None
        teradataml.execute_sql.return_value = cursor
        env = {"TERADATA_HOST": "host", "TERADATA_USER": "user",
               "TERADATA_PASSWORD": "password", "TERADATA_DATABASE": "warehouse"}
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as root, patch.dict(
                os.environ, env, clear=True), patch.object(
                db_tools.importlib, "import_module", return_value=teradataml):
            result = json.loads(db_tools.export_teradata_csv(
                "SELECT n FROM t", os.path.join(root, "td.csv"), 1, root))
        self.assertEqual(cursor.fetch_sizes, [1, 1, 1])
        teradataml.create_context.assert_called_once()
        teradataml.execute_sql.assert_called_once_with(statement="SELECT n FROM t")
        teradataml.remove_context.assert_called_once_with()
        self.assertEqual(result["backend"], "Teradata")

        partial = MagicMock()
        partial.get_context.side_effect = [None, object()]
        partial.create_context.side_effect = RuntimeError("credential secret")
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as root, patch.dict(
                os.environ, env, clear=True), patch.object(
                db_tools.importlib, "import_module", return_value=partial):
            with self.assertRaisesRegex(RuntimeError, "Teradata CSV export failed"):
                db_tools.export_teradata_csv("SELECT 1", os.path.join(root, "x.csv"), 1, root)
            self.assertEqual(os.listdir(root), [])
        partial.remove_context.assert_called_once_with()

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
        teradataml = MagicMock()
        teradataml.get_context.return_value = None
        teradataml.execute_sql.return_value = cursor
        env = {
            "TERADATA_HOST": "td.example", "TERADATA_USER": "analyst",
            "TERADATA_PASSWORD": "hidden", "TERADATA_DATABASE": "warehouse",
            "TERADATA_LOGMECH": "LDAP",
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
                db_tools.importlib, "import_module", return_value=teradataml) as importer:
            result = json.loads(db_tools.query_teradata("/* lead */ SELECT ';' AS x;", max_rows=1))

        importer.assert_called_once_with("teradataml")
        teradataml.create_context.assert_called_once_with(
            host="td.example", username="analyst", password="hidden",
            database="warehouse", logmech="LDAP")
        teradataml.execute_sql.assert_called_once_with(statement="/* lead */ SELECT ';' AS x;")
        self.assertEqual(result, {
            "database": "warehouse", "columns": ["amount", "day", "timestamp", "payload"],
            "rows": [["1.25", "2026-01-02", "2026-01-02T03:04:05", "aGk="]],
            "row_count": 1, "truncated": True,
        })
        self.assertTrue(cursor.closed)
        teradataml.remove_context.assert_called_once_with()

    def test_teradata_omits_absent_optional_context_kwargs(self):
        teradataml = MagicMock()
        teradataml.get_context.return_value = None
        teradataml.execute_sql.return_value = FakeCursor(description=[])
        env = {"TERADATA_HOST": "host", "TERADATA_USER": "user", "TERADATA_PASSWORD": "password"}
        with patch.dict(os.environ, env, clear=True), patch.object(
                db_tools.importlib, "import_module", return_value=teradataml):
            db_tools.query_teradata("SELECT 1")

        teradataml.create_context.assert_called_once_with(
            host="host", username="user", password="password")

    def test_teradata_cleanup_on_execute_and_fetch_failures(self):
        for failure_kind in ("execute", "fetch"):
            teradataml = MagicMock()
            teradataml.get_context.return_value = None
            cursor = FakeCursor(fetch_error=RuntimeError("fetch failed for password=password-secret"))
            if failure_kind == "execute":
                teradataml.execute_sql.side_effect = RuntimeError("execute failed for host-secret")
            else:
                teradataml.execute_sql.return_value = cursor
            env = {"TERADATA_HOST": "host-secret", "TERADATA_USER": "user-secret",
                   "TERADATA_PASSWORD": "password-secret"}
            with self.subTest(failure_kind=failure_kind), patch.dict(
                    os.environ, env, clear=True), patch.object(
                    db_tools.importlib, "import_module", return_value=teradataml):
                with self.assertRaisesRegex(RuntimeError, "Teradata query failed") as caught:
                    db_tools.query_teradata("SELECT 1")
            for secret in env.values():
                self.assertNotIn(secret, str(caught.exception))
            if failure_kind == "fetch":
                self.assertTrue(cursor.closed)
            teradataml.remove_context.assert_called_once_with()

    def test_teradata_query_error_reports_stage_and_redacted_exception_chain_to_agent(self):
        class TeradataMlException(Exception):
            pass

        class OperationalError(Exception):
            errorcode = 3706
            sqlstate = "42000"

        inner = OperationalError(
            "[Error 3706] Syntax error near host-secret; password=password-secret")
        outer = TeradataMlException("TDML_2000: Failed to execute SQL")
        outer.__cause__ = inner
        teradataml = MagicMock()
        teradataml.get_context.return_value = None
        teradataml.execute_sql.side_effect = outer
        env = {"TERADATA_HOST": "host-secret", "TERADATA_USER": "user-secret",
               "TERADATA_PASSWORD": "password-secret", "TERADATA_DATABASE": "db-secret"}

        with patch.dict(os.environ, env, clear=True), patch.object(
                db_tools.importlib, "import_module", return_value=teradataml):
            result = registry.execute("query_teradata", {"sql": "SELECT broken FROM t"})

        self.assertIn("stage=execute", result)
        self.assertIn("TeradataMlException", result)
        self.assertIn("OperationalError", result)
        self.assertIn("TDML_2000", result)
        self.assertIn("3706", result)
        self.assertIn("sqlstate=42000", result)
        self.assertIn("Syntax error", result)
        for secret in env.values():
            self.assertNotIn(secret, result)
        teradataml.remove_context.assert_called_once_with()

    def test_teradata_does_not_remove_context_when_creation_fails(self):
        teradataml = MagicMock()
        teradataml.get_context.return_value = None
        teradataml.create_context.side_effect = RuntimeError("host-secret")
        env = {"TERADATA_HOST": "host-secret", "TERADATA_USER": "user-secret",
               "TERADATA_PASSWORD": "password-secret"}
        with patch.dict(os.environ, env, clear=True), patch.object(
                db_tools.importlib, "import_module", return_value=teradataml):
            with self.assertRaisesRegex(RuntimeError, "Teradata query failed") as caught:
                db_tools.query_teradata("SELECT 1")
        self.assertNotIn("secret", str(caught.exception))
        teradataml.remove_context.assert_not_called()

    def test_teradata_removes_partially_created_context_when_creation_fails(self):
        teradataml = MagicMock()
        contexts = iter((None, object()))
        teradataml.get_context.side_effect = lambda: next(contexts)
        teradataml.create_context.side_effect = RuntimeError("host-secret")
        env = {"TERADATA_HOST": "host-secret", "TERADATA_USER": "user-secret",
               "TERADATA_PASSWORD": "password-secret"}
        with patch.dict(os.environ, env, clear=True), patch.object(
                db_tools.importlib, "import_module", return_value=teradataml):
            with self.assertRaisesRegex(RuntimeError, "Teradata query failed") as caught:
                db_tools.query_teradata("SELECT 1")
        self.assertNotIn("secret", str(caught.exception))
        teradataml.remove_context.assert_called_once_with()

    def test_teradata_recovers_from_partial_context_when_public_cleanup_raises(self):
        teradataml = types.ModuleType("teradataml")
        context_package = types.ModuleType("teradataml.context")
        context_module = types.ModuleType("teradataml.context.context")
        disposed = []
        create_attempts = []

        class UnderlyingEngine:
            def dispose(self):
                disposed.append(True)

        class EngineWrapper:
            def __init__(self):
                self.engine = UnderlyingEngine()

        context_module.td_connection = None
        context_module.td_sqlalchemy_engine = None

        def get_context():
            return context_module.td_sqlalchemy_engine

        def create_context(**_options):
            create_attempts.append(True)
            context_module.td_sqlalchemy_engine = EngineWrapper()
            raise RuntimeError("host-secret")

        def remove_context():
            context_module.td_connection.close()

        teradataml.get_context = get_context
        teradataml.create_context = create_context
        teradataml.remove_context = remove_context
        context_package.context = context_module
        teradataml.context = context_package
        fake_modules = {
            "teradataml": teradataml,
            "teradataml.context": context_package,
            "teradataml.context.context": context_module,
        }
        env = {"TERADATA_HOST": "host-secret", "TERADATA_USER": "user-secret",
               "TERADATA_PASSWORD": "password-secret"}

        with patch.dict(os.environ, env, clear=True), patch.dict(sys.modules, fake_modules):
            for _ in range(2):
                with self.assertRaisesRegex(RuntimeError, "Teradata query failed") as caught:
                    db_tools.query_teradata("SELECT 1")
                self.assertNotIn("secret", str(caught.exception))
                self.assertIsNone(context_module.td_connection)
                self.assertIsNone(context_module.td_sqlalchemy_engine)

        self.assertEqual(create_attempts, [True, True])
        self.assertEqual(disposed, [True, True])

    def test_teradata_rejects_preexisting_context_without_modifying_it(self):
        teradataml = MagicMock()
        teradataml.get_context.return_value = object()
        env = {"TERADATA_HOST": "host", "TERADATA_USER": "user", "TERADATA_PASSWORD": "password"}
        with patch.dict(os.environ, env, clear=True), patch.object(
                db_tools.importlib, "import_module", return_value=teradataml):
            with self.assertRaisesRegex(RuntimeError, "active Teradata context"):
                db_tools.query_teradata("SELECT 1")
        teradataml.create_context.assert_not_called()
        teradataml.remove_context.assert_not_called()

    def test_impala_lazily_imports_dbapi_and_maps_full_connection(self):
        cursor = FakeCursor(rows=[(index,) for index in range(150)], description=[("one",)])
        connection = FakeConnection(cursor)
        dbapi = MagicMock()
        dbapi.connect.return_value = connection
        env = {
            "IMPALA_HOST": "impala.example", "IMPALA_PORT": "21051",
            "IMPALA_DATABASE": "analytics", "IMPALA_TIMEOUT": "45",
            "IMPALA_AUTH_MECHANISM": "PLAIN", "IMPALA_USER": "analyst",
            "IMPALA_PASSWORD": "hidden", "IMPALA_USE_SSL": "yes",
            "IMPALA_CA_CERT": "/certs/ca.pem", "IMPALA_KERBEROS_SERVICE_NAME": "hive",
            "IMPALA_USE_HTTP_TRANSPORT": "1", "IMPALA_HTTP_PATH": "cliservice",
            "IMPALA_VERIFY_CERT": "on",
        }
        with patch.dict(os.environ, env, clear=True), patch.object(
                db_tools.importlib, "import_module", return_value=dbapi) as importer:
            result = json.loads(db_tools.query_impala("WITH x AS (SELECT 1) SELECT * FROM x", max_rows=125))

        importer.assert_called_once_with("impala.dbapi")
        dbapi.connect.assert_called_once_with(
            host="impala.example", port=21051, database="analytics", timeout=45,
            auth_mechanism="PLAIN", user="analyst", password="hidden", use_ssl=True,
            ca_cert="/certs/ca.pem", kerberos_service_name="hive",
            use_http_transport=True, http_path="cliservice", verify_cert=True)
        self.assertEqual(result["database"], "analytics")
        self.assertEqual(result["row_count"], 125)
        self.assertTrue(result["truncated"])
        self.assertEqual(cursor.executed, "WITH x AS (SELECT 1) SELECT * FROM x")
        self.assertTrue(cursor.closed)
        self.assertTrue(connection.closed)

    def test_impala_connection_defaults_and_optional_fields_are_omitted(self):
        dbapi = MagicMock()
        dbapi.connect.return_value = FakeConnection(FakeCursor(description=[]))
        with patch.dict(os.environ, {"IMPALA_HOST": "impala.example"}, clear=True), patch.object(
                db_tools.importlib, "import_module", return_value=dbapi):
            db_tools.query_impala("SELECT 1")

        dbapi.connect.assert_called_once_with(
            host="impala.example", port=21050, database="default", timeout=30,
            auth_mechanism="NOSASL", use_ssl=False, kerberos_service_name="impala",
            use_http_transport=False, http_path="", verify_cert=False)

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
            with self.assertRaisesRegex(
                    RuntimeError,
                    "Missing required database configuration: TERADATA_HOST.*TERADATA_USER.*TERADATA_PASSWORD"):
                db_tools.query_teradata("SELECT 1")
            with self.assertRaisesRegex(
                    RuntimeError, "Missing required database configuration: IMPALA_HOST"):
                db_tools.query_impala("SELECT 1")

        with patch.dict(os.environ, {"TERADATA_HOST": "h", "TERADATA_USER": "u",
                                     "TERADATA_PASSWORD": "super-secret"}, clear=True), patch.object(
                db_tools.importlib, "import_module", side_effect=ImportError("no module")):
            with self.assertRaises(RuntimeError) as caught:
                db_tools.query_teradata("SELECT 1")
        self.assertIn("pip install teradataml", str(caught.exception))
        self.assertNotIn("super-secret", str(caught.exception))

        with patch.dict(os.environ, {"IMPALA_HOST": "super-secret"}, clear=True), patch.object(
                db_tools.importlib, "import_module", side_effect=ImportError("super-secret")):
            with self.assertRaises(RuntimeError) as caught:
                db_tools.query_impala("SELECT 1")
        self.assertIn("pip install impyla", str(caught.exception))
        self.assertNotIn("super-secret", str(caught.exception))

    def test_json_config_loading_and_environment_precedence(self):
        config = {
            "teradata": {"host": "json-host", "user": "json-user", "password": "json-pass",
                         "database": "json-db", "logmech": "LDAP"},
            "impala": {"host": "json-host", "port": 21052, "database": "json-default",
                       "timeout": 12, "auth_mechanism": "LDAP", "user": "json-user",
                       "password": "json-pass", "use_ssl": False, "ca_cert": "json-ca.pem",
                       "kerberos_service_name": "json-service", "use_http_transport": False,
                       "http_path": "json-path", "verify_cert": False},
        }
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "database.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            teradataml = MagicMock()
            teradataml.get_context.return_value = None
            teradataml.execute_sql.return_value = FakeCursor(description=[])
            env = {"AGENT_DB_CONFIG": os.path.join(directory, "ignored.json"),
                   "TERADATA_HOST": "env-host",
                   "IMPALA_HOST": "env-impala", "IMPALA_PORT": "21053",
                   "IMPALA_TIMEOUT": "31", "IMPALA_USE_SSL": "true",
                   "IMPALA_USE_HTTP_TRANSPORT": "yes", "IMPALA_VERIFY_CERT": "1"}
            with patch.dict(os.environ, env, clear=True), patch.object(
                    db_tools, "__file__", os.path.join(directory, "db_tools.py")), patch.object(
                    db_tools.importlib, "import_module", return_value=teradataml):
                td_result = json.loads(db_tools.query_teradata("SELECT 1"))
                td_call = teradataml.create_context.call_args
                dbapi = MagicMock()
                dbapi.connect.return_value = FakeConnection(FakeCursor(description=[]))
                db_tools.importlib.import_module.return_value = dbapi
                impala_result = json.loads(db_tools.query_impala("SELECT 1"))

        self.assertEqual(td_call.kwargs, {"host": "env-host", "username": "json-user", "password": "json-pass",
                                          "database": "json-db", "logmech": "LDAP"})
        dbapi.connect.assert_called_once_with(
            host="env-impala", port=21053, database="json-default", timeout=31,
            auth_mechanism="LDAP", user="json-user", password="json-pass", use_ssl=True,
            ca_cert="json-ca.pem", kerberos_service_name="json-service",
            use_http_transport=True, http_path="json-path", verify_cert=True)
        self.assertEqual(td_result["database"], "json-db")
        self.assertEqual(impala_result["database"], "json-default")

    def test_json_config_loads_from_db_tools_directory_without_agent_db_config(self):
        config = {"impala": {"host": "adjacent-host"}}
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "database.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            with patch.dict(os.environ, {}, clear=True), patch.object(
                    db_tools, "__file__", os.path.join(directory, "db_tools.py")):
                self.assertEqual(db_tools._load_config(), config)

    def test_json_config_loading_is_independent_of_current_working_directory(self):
        config = {"teradata": {"host": "adjacent-host"}}
        with tempfile.TemporaryDirectory() as module_directory, tempfile.TemporaryDirectory() as cwd:
            path = os.path.join(module_directory, "database.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(config, handle)
            previous_cwd = os.getcwd()
            try:
                os.chdir(cwd)
                with patch.dict(os.environ, {}, clear=True), patch.object(
                        db_tools, "__file__", os.path.join(module_directory, "db_tools.py")):
                    self.assertEqual(db_tools._load_config(), config)
            finally:
                os.chdir(previous_cwd)

    def test_impala_rejects_invalid_native_settings_before_import_or_connection(self):
        cases = [
            ({"IMPALA_PORT": "0"}, "IMPALA_PORT.*1.*65535"),
            ({"IMPALA_PORT": "65536"}, "IMPALA_PORT.*1.*65535"),
            ({"IMPALA_PORT": "not-an-int"}, "IMPALA_PORT.*1.*65535"),
            ({"IMPALA_TIMEOUT": "0"}, "IMPALA_TIMEOUT.*positive integer"),
            ({"IMPALA_TIMEOUT": "1.5"}, "IMPALA_TIMEOUT.*positive integer"),
            ({"IMPALA_USE_SSL": "maybe"}, "IMPALA_USE_SSL.*boolean"),
            ({"IMPALA_USE_HTTP_TRANSPORT": "2"}, "IMPALA_USE_HTTP_TRANSPORT.*boolean"),
            ({"IMPALA_VERIFY_CERT": ""}, "IMPALA_VERIFY_CERT.*boolean"),
        ]
        for extra, expected in cases:
            env = {"IMPALA_HOST": "impala.example"}
            env.update(extra)
            with self.subTest(extra=extra), patch.dict(os.environ, env, clear=True), patch.object(
                    db_tools.importlib, "import_module") as importer:
                with self.assertRaisesRegex(RuntimeError, expected):
                    db_tools.query_impala("SELECT 1")
                importer.assert_not_called()

    def test_json_config_errors_are_actionable_and_secret_safe(self):
        cases = [(b'{"impala":', "valid JSON"), (b'[]', "root.*object"),
                 (b'{"impala":"secret-value"}', "impala.*object"), (b'\xff', "UTF-8")]
        for content, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                path = os.path.join(directory, "database.json")
                with open(path, "wb") as handle:
                    handle.write(content)
                with patch.dict(os.environ, {}, clear=True), patch.object(
                        db_tools, "__file__", os.path.join(directory, "db_tools.py")):
                    with self.assertRaisesRegex(RuntimeError, expected) as caught:
                        db_tools.query_impala("SELECT 1")
                    self.assertNotIn("secret-value", str(caught.exception))
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "database.json")
            with open(path, "wb") as handle:
                handle.write(b" " * (65536 + 1))
            with patch.dict(os.environ, {}, clear=True), patch.object(
                    db_tools, "__file__", os.path.join(directory, "db_tools.py")):
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
                    os.environ, {"IMPALA_HOST": "impala.example"}, clear=True), patch.object(
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
                    os.environ, {"IMPALA_HOST": "impala.example"}, clear=True), patch.object(
                    db_tools.importlib, "import_module", return_value=driver):
                with self.assertRaisesRegex(RuntimeError, "failed"):
                    db_tools.query_impala("SELECT 1")
            self.assertTrue(cursor.closed)
            self.assertTrue(connection.closed)

    def test_query_error_redacts_configured_secret_embedded_in_driver_text(self):
        rendered = str(db_tools._format_query_error(
            "Teradata", "execute",
            RuntimeError("driver payload=prefixsecret-passsuffix"),
            ("secret-pass",),
        ))

        self.assertNotIn("secret-pass", rendered)
        self.assertIn("prefix<redacted>suffix", rendered)

    def test_query_error_diagnostics_are_bounded_and_cycle_safe(self):
        outer = RuntimeError("outer " + "x" * 5000)
        inner = RuntimeError("password=secret-pass")
        outer.__cause__ = inner
        inner.__context__ = outer

        rendered = str(db_tools._format_query_error(
            "Teradata", "execute", outer, ("secret-pass",)))

        self.assertLessEqual(len(rendered), db_tools._MAX_ERROR_CHARS)
        self.assertIn("truncated", rendered)
        self.assertNotIn("additional exception chain", rendered)
        self.assertNotIn("secret-pass", rendered)

    def test_impala_query_error_reports_exact_stage_code_and_redacted_message(self):
        class HiveServer2Error(Exception):
            errno = 100

        cases = {
            "connect": (None, HiveServer2Error("connect failed to secret-host as secret-user")),
            "execute": (FakeCursor(execute_error=HiveServer2Error(
                "AnalysisException code 100: unknown column; pwd=secret-pass")), None),
            "fetch": (FakeCursor(fetch_error=HiveServer2Error(
                "fetch failed for dsn=secret-dsn")), None),
        }
        env = {"IMPALA_HOST": "secret-host", "IMPALA_USER": "secret-user",
               "IMPALA_PASSWORD": "secret-pass", "IMPALA_DATABASE": "secret-dsn"}
        for expected_stage, (cursor, connect_error) in cases.items():
            driver = MagicMock()
            if connect_error is not None:
                driver.connect.side_effect = connect_error
            else:
                driver.connect.return_value = FakeConnection(cursor)
            with self.subTest(stage=expected_stage), patch.dict(
                    os.environ, env, clear=True), patch.object(
                    db_tools.importlib, "import_module", return_value=driver):
                result = registry.execute("query_impala", {"sql": "SELECT broken FROM t"})

            self.assertIn(f"stage={expected_stage}", result)
            self.assertIn("HiveServer2Error", result)
            self.assertIn("errno=100", result)
            for secret in env.values():
                self.assertNotIn(secret, result)


if __name__ == "__main__":
    unittest.main()
