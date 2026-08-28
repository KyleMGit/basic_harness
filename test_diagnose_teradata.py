import unittest

import diagnose_teradata


class DiagnoseTeradataTests(unittest.TestCase):
    def test_exception_chain_includes_redacted_underlying_driver_error(self):
        cause = ConnectionError(
            "host=td.example user=alice password=secret Error 8017")
        outer = RuntimeError("Failed to connect to Teradata Vantage.")
        outer.__cause__ = cause

        lines = diagnose_teradata._format_exception_chain(
            outer, ("secret", "alice", "td.example"))
        output = "\n".join(lines)

        self.assertIn("RuntimeError: Failed to connect to Teradata Vantage.", output)
        self.assertIn("Caused by ConnectionError:", output)
        self.assertIn("Error 8017", output)
        self.assertNotIn("secret", output)
        self.assertNotIn("alice", output)
        self.assertNotIn("td.example", output)


if __name__ == "__main__":
    unittest.main()
