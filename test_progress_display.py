import io
import json
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import agent as agent_module


class ProgressConfigurationTests(unittest.TestCase):
    def parse(self, *arguments):
        with patch.object(sys, "argv", ["agent.py", *arguments]):
            return agent_module.parse_args()

    def test_cli_progress_default_and_accepted_values(self):
        self.assertEqual(self.parse().progress, "concise")
        self.assertEqual(self.parse("--progress", "concise").progress, "concise")
        self.assertEqual(self.parse("--progress", "verbose").progress, "verbose")

    def test_cli_rejects_invalid_progress_value(self):
        with self.assertRaises(SystemExit):
            self.parse("--progress", "chatty")

    def test_constructor_progress_default_and_explicit_value(self):
        default_agent = agent_module.HermesCodingAgent(
            read_only=True, enable_skills=False, enable_memory=False
        )
        verbose_agent = agent_module.HermesCodingAgent(
            progress_mode="verbose", read_only=True,
            enable_skills=False, enable_memory=False,
        )
        self.assertEqual(default_agent.progress_mode, "concise")
        self.assertEqual(verbose_agent.progress_mode, "verbose")


class ProgressDisplayTests(unittest.TestCase):
    @staticmethod
    def database_result(database="warehouse", rows=None, row_count=0):
        rows = [] if rows is None else rows
        return json.dumps({
            "database": database,
            "columns": ["value"],
            "rows": rows,
            "row_count": row_count,
            "truncated": False,
        })

    @staticmethod
    def message(content, calls=None):
        tool_calls = []
        for index, (name, arguments) in enumerate(calls or []):
            tool_calls.append(SimpleNamespace(
                id=f"call_{index}",
                function=SimpleNamespace(name=name, arguments=json.dumps(arguments)),
            ))
        payload = {
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in tool_calls
            ],
        }
        return SimpleNamespace(
            content=content,
            tool_calls=tool_calls,
            model_dump=lambda exclude_none=True: payload,
        )

    def run_agent(self, progress_mode, first_message):
        instance = agent_module.HermesCodingAgent(
            progress_mode=progress_mode,
            max_iterations=2,
            read_only=True,
            enable_skills=False,
            enable_memory=False,
            confirm_all_terminal_commands=False,
        )
        instance.logger = MagicMock()
        instance.manage_context = MagicMock()
        instance.context_manager.estimate_tokens = MagicMock(return_value=1)
        instance.step = MagicMock(side_effect=[first_message, self.message("finished")])
        with patch.object(agent_module.registry, "execute", return_value="ok"):
            output = io.StringIO()
            with patch("sys.stdout", output):
                result = instance.run("task")
        self.assertEqual(result, "finished")
        return output.getvalue()

    def run_cycles(self, calls, results, progress_mode="concise", xml=False, instance=None):
        instance = instance or agent_module.HermesCodingAgent(
            progress_mode=progress_mode,
            max_iterations=len(calls) + 1,
            read_only=True,
            enable_skills=False,
            enable_memory=False,
            confirm_all_terminal_commands=False,
            use_hermes_xml_protocol=xml,
        )
        instance.logger = MagicMock()
        instance.manage_context = MagicMock()
        instance.context_manager.estimate_tokens = MagicMock(return_value=1)
        messages = [self.message("hidden", cycle) for cycle in calls]
        messages.append(self.message("finished"))
        instance.step = MagicMock(side_effect=messages)
        with patch.object(agent_module.registry, "execute", side_effect=results):
            output = io.StringIO()
            with patch("sys.stdout", output):
                result = instance.run("task")
        self.assertEqual(result, "finished")
        return output.getvalue(), instance

    def test_concise_suppresses_multiline_thought_and_prints_one_bounded_line(self):
        secret_thought = "private reasoning line one\nprivate reasoning line two"
        long_path = "folder/" + ("very-long-name-" * 30) + "target.py"
        output = self.run_agent(
            "concise", self.message(secret_thought, [("read_file", {"file_path": long_path})])
        )
        progress_lines = [line for line in output.splitlines() if "[Agent Progress]:" in line]
        self.assertEqual(len(progress_lines), 1)
        self.assertNotIn(secret_thought, output)
        self.assertTrue(progress_lines[0].startswith("[Agent Progress]: Reading "))
        self.assertLessEqual(len(progress_lines[0]), 180)

    def test_verbose_retains_full_thought(self):
        thought = "inspect the first file\nthen compare the second"
        output = self.run_agent(
            "verbose", self.message(thought, [("read_file", {"file_path": "README.md"})])
        )
        self.assertIn(f"[Agent Thought]:\n{thought}", output)
        self.assertNotIn("[Agent Progress]:", output)

    def test_concise_multiple_tools_emit_one_line_and_note_remainder(self):
        output = self.run_agent(
            "concise",
            self.message("hidden", [
                ("grep_search", {"query": "needle\nwith whitespace"}),
                ("read_file", {"file_path": "README.md"}),
                ("list_directory", {"directory_path": "."}),
            ]),
        )
        progress_lines = [line for line in output.splitlines() if "[Agent Progress]:" in line]
        self.assertEqual(len(progress_lines), 1)
        self.assertIn("Searching for needle with whitespace", progress_lines[0])
        self.assertIn("(+2 more tools)", progress_lines[0])

    def test_unknown_tool_with_empty_arguments_is_graceful(self):
        output = self.run_agent(
            "concise", self.message("hidden", [("mystery_tool", {})])
        )
        progress_lines = [line for line in output.splitlines() if "[Agent Progress]:" in line]
        self.assertEqual(progress_lines, ["[Agent Progress]: Using mystery_tool."])

    def test_teradata_failure_then_metadata_inspection(self):
        output, _ = self.run_cycles([
            [("query_teradata", {"sql": "select bad"})],
            [("query_teradata", {"sql": "HELP TABLE sales"})],
        ], ["Error executing tool 'query_teradata': bad column", "columns"])
        self.assertIn("Troubleshooting Teradata SQL by inspecting schema metadata.", output)

    def test_explain_and_revised_sql_methods(self):
        output, _ = self.run_cycles([
            [("query_teradata", {"sql": "select bad"})],
            [("query_teradata", {"sql": "EXPLAIN select * from sales"})],
            [("query_teradata", {"sql": "select fixed from sales"})],
        ], ["Error: bad SQL", "Error: plan exposed problem", "rows"])
        self.assertIn("Troubleshooting Teradata SQL by checking the query plan.", output)
        self.assertIn("Troubleshooting Teradata SQL by retrying with revised SQL.", output)
        self.assertNotIn("select fixed from sales", next(
            line for line in output.splitlines() if "retrying with revised SQL" in line
        ))

    def test_connection_diagnostics_and_multiple_tool_suffix(self):
        output, _ = self.run_cycles([
            [("query_teradata", {"sql": "select bad"})],
            [("run_terminal_command", {"command": "python diagnose_teradatasql.py"}),
             ("read_file", {"file_path": "log.txt"})],
        ], ["Error: connection failed", "diagnostic output", "log"])
        self.assertIn("Troubleshooting Teradata SQL by running connection diagnostics. (+1 more tool)", output)

    def test_state_persists_across_intermediate_step_and_clears_on_success(self):
        output, _ = self.run_cycles([
            [("query_teradata", {"sql": "select bad"})],
            [("read_file", {"file_path": "diagnostic.log"})],
            [("query_teradata", {"sql": "select fixed"})],
            [("read_file", {"file_path": "after.txt"})],
        ], ["Error: bad SQL", "details", "rows", "after"])
        self.assertIn("Troubleshooting Teradata SQL by reading diagnostic.log.", output)
        self.assertIn("Troubleshooting Teradata SQL by retrying with revised SQL.", output)
        self.assertIn("[Agent Progress]: Reading after.txt.", output)
        self.assertNotIn("Troubleshooting Teradata SQL by reading after.txt.", output)

    def test_latest_failed_backend_is_impala(self):
        output, _ = self.run_cycles([
            [("query_teradata", {"sql": "select bad"})],
            [("query_impala", {"sql": "select also_bad"})],
            [("query_impala", {"sql": "select * from INFORMATION_SCHEMA.TABLES"})],
        ], ["Error: td", "Error: impala", "rows"])
        self.assertIn("Troubleshooting Impala SQL by inspecting schema metadata.", output)

    def test_troubleshooting_resets_between_top_level_runs(self):
        first, instance = self.run_cycles(
            [[("query_teradata", {"sql": "select bad"})]], ["Error: bad"]
        )
        second, _ = self.run_cycles(
            [[("read_file", {"file_path": "new-task.txt"})]], ["ok"], instance=instance
        )
        self.assertNotIn("Troubleshooting", second)
        self.assertIn("[Agent Progress]: Reading new-task.txt.", second)

    def test_verbose_failure_flow_remains_unchanged(self):
        output, _ = self.run_cycles([
            [("query_teradata", {"sql": "select bad"})],
            [("read_file", {"file_path": "diagnostic.log"})],
        ], ["Error: bad", "details"], progress_mode="verbose")
        self.assertNotIn("[Agent Progress]:", output)
        self.assertNotIn("Troubleshooting Teradata SQL", output)
        self.assertIn("[Agent Thought]:\nhidden", output)

    def test_xml_protocol_troubleshooting_integration(self):
        def xml_message(name=None, arguments=None, final=None):
            content = final if final is not None else (
                "hidden\n<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"
            )
            return SimpleNamespace(content=content, tool_calls=[], finish_reason="stop")

        instance = agent_module.HermesCodingAgent(
            progress_mode="concise", max_iterations=3, read_only=True,
            enable_skills=False, enable_memory=False,
            confirm_all_terminal_commands=False, use_hermes_xml_protocol=True,
        )
        instance.logger = MagicMock()
        instance.manage_context = MagicMock()
        instance.context_manager.estimate_tokens = MagicMock(return_value=1)
        instance.step = MagicMock(side_effect=[
            xml_message("query_impala", {"sql": "select bad"}),
            xml_message("query_impala", {"sql": "DESCRIBE sales"}),
            xml_message(final="finished"),
        ])
        with patch.object(agent_module.registry, "execute", side_effect=["Error: bad", "columns"]):
            output = io.StringIO()
            with patch("sys.stdout", output):
                self.assertEqual(instance.run("task"), "finished")
        self.assertIn("Troubleshooting Impala SQL by inspecting schema metadata.", output.getvalue())

    def test_zero_results_general_query_is_piece_by_piece_diagnostic(self):
        output, _ = self.run_cycles([
            [("query_teradata", {"sql": "SELECT * FROM sales JOIN customers USING (id)"})],
            [("query_teradata", {"sql": "SELECT * FROM sales"})],
        ], [self.database_result(), self.database_result()])
        self.assertIn(
            "Investigating zero Teradata results by testing the query piece by piece.", output
        )
        self.assertNotIn("SELECT * FROM sales", next(
            line for line in output.splitlines() if "Investigating zero" in line
        ))

    def test_zero_results_query_methods(self):
        cases = [
            ("SELECT COUNT(*) FROM sales", "checking intermediate row counts"),
            ("HELP TABLE sales", "inspecting schema metadata"),
            ("select * from INFORMATION_SCHEMA.TABLES", "inspecting schema metadata"),
            ("EXPLAIN SELECT * FROM sales", "checking the query plan"),
            ("  select   * FROM sales ; ", "rerunning the original query"),
        ]
        for diagnostic_sql, method in cases:
            with self.subTest(method=method):
                output, _ = self.run_cycles([
                    [("query_teradata", {"sql": "SELECT * FROM sales"})],
                    [("query_teradata", {"sql": diagnostic_sql})],
                ], [self.database_result(), self.database_result()])
                self.assertIn(
                    f"Investigating zero Teradata results by {method}.", output
                )

    def test_repeated_zero_result_updates_origin_and_remains_active(self):
        output, _ = self.run_cycles([
            [("query_teradata", {"sql": "SELECT * FROM joined_sales"})],
            [("query_teradata", {"sql": "SELECT * FROM sales"})],
            [("query_teradata", {"sql": " select *  from SALES; "})],
        ], [self.database_result(), self.database_result(), self.database_result()])
        self.assertIn("testing the query piece by piece", output)
        self.assertIn("rerunning the original query", output)

    def test_nonempty_result_clears_empty_result_state(self):
        output, _ = self.run_cycles([
            [("query_teradata", {"sql": "SELECT * FROM sales"})],
            [("query_teradata", {"sql": "SELECT COUNT(*) FROM sales"})],
            [("read_file", {"file_path": "after.txt"})],
        ], [
            self.database_result(),
            self.database_result(rows=[[3]], row_count=1),
            "after",
        ])
        self.assertIn("checking intermediate row counts", output)
        self.assertIn("[Agent Progress]: Reading after.txt.", output)
        self.assertNotIn("Investigating zero Teradata results by reading after.txt.", output)

    def test_error_after_zero_switches_to_existing_error_mode(self):
        output, _ = self.run_cycles([
            [("query_teradata", {"sql": "SELECT * FROM sales"})],
            [("query_teradata", {"sql": "SELECT COUNT(*) FROM sales"})],
            [("query_teradata", {"sql": "HELP TABLE sales"})],
        ], [self.database_result(), "Error: count failed", "columns"])
        self.assertIn("checking intermediate row counts", output)
        self.assertIn("Troubleshooting Teradata SQL by inspecting schema metadata.", output)

    def test_final_answer_after_zero_prints_no_diagnostic_line(self):
        output, _ = self.run_cycles([
            [("query_teradata", {"sql": "SELECT * FROM sales"})],
        ], [self.database_result()])
        self.assertNotIn("Investigating zero", output)

    def test_only_strict_database_envelope_triggers_empty_state(self):
        invalid_results = [
            '{"message":"row_count: 0, rows: []"}',
            json.dumps({"database": "db", "columns": ["note"], "rows": [["rows: []"]],
                        "row_count": 1, "truncated": False}),
            json.dumps({"database": "db", "columns": [], "rows": [],
                        "row_count": False, "truncated": False}),
            json.dumps({"database": "db", "columns": [], "rows": [], "row_count": 0}),
            "not json rows=[] row_count=0",
        ]
        for result in invalid_results:
            with self.subTest(result=result):
                output, _ = self.run_cycles([
                    [("query_teradata", {"sql": "SELECT * FROM sales"})],
                    [("read_file", {"file_path": "diagnostic.txt"})],
                ], [result, "ok"])
                self.assertNotIn("Investigating zero", output)

    def test_impala_other_tool_uses_bounded_action_as_method(self):
        output, _ = self.run_cycles([
            [("query_impala", {"sql": "SELECT * FROM sales"})],
            [("read_file", {"file_path": "diagnostic.log"}),
             ("list_directory", {"directory_path": "."})],
        ], [self.database_result("impala"), "details", "files"])
        self.assertIn(
            "Investigating zero Impala results by reading diagnostic.log. (+1 more tool)", output
        )

    def test_verbose_zero_result_flow_remains_unchanged(self):
        output, _ = self.run_cycles([
            [("query_teradata", {"sql": "SELECT * FROM sales"})],
            [("query_teradata", {"sql": "SELECT COUNT(*) FROM sales"})],
        ], [self.database_result(), self.database_result()], progress_mode="verbose")
        self.assertNotIn("[Agent Progress]:", output)
        self.assertNotIn("Investigating zero", output)
        self.assertIn("[Agent Thought]:\nhidden", output)

    def test_xml_protocol_empty_result_integration(self):
        def xml_message(name=None, arguments=None, final=None):
            content = final if final is not None else (
                "hidden\n<tool_call>" + json.dumps({"name": name, "arguments": arguments}) + "</tool_call>"
            )
            return SimpleNamespace(content=content, tool_calls=[], finish_reason="stop")

        instance = agent_module.HermesCodingAgent(
            progress_mode="concise", max_iterations=3, read_only=True,
            enable_skills=False, enable_memory=False,
            confirm_all_terminal_commands=False, use_hermes_xml_protocol=True,
        )
        instance.logger = MagicMock()
        instance.manage_context = MagicMock()
        instance.context_manager.estimate_tokens = MagicMock(return_value=1)
        instance.step = MagicMock(side_effect=[
            xml_message("query_impala", {"sql": "SELECT * FROM sales"}),
            xml_message("query_impala", {"sql": "SELECT COUNT(*) FROM sales"}),
            xml_message(final="finished"),
        ])
        with patch.object(agent_module.registry, "execute", side_effect=[
            self.database_result("impala"), self.database_result("impala")
        ]):
            output = io.StringIO()
            with patch("sys.stdout", output):
                self.assertEqual(instance.run("task"), "finished")
        self.assertIn("Investigating zero Impala results by checking intermediate row counts.", output.getvalue())

    def test_empty_result_state_resets_between_top_level_runs(self):
        first, instance = self.run_cycles(
            [[("query_teradata", {"sql": "SELECT * FROM sales"})]],
            [self.database_result()],
        )
        second, _ = self.run_cycles(
            [[("read_file", {"file_path": "new-task.txt"})]], ["ok"], instance=instance
        )
        self.assertNotIn("Investigating zero", first)
        self.assertNotIn("Investigating zero", second)
        self.assertIn("[Agent Progress]: Reading new-task.txt.", second)


if __name__ == "__main__":
    unittest.main()
