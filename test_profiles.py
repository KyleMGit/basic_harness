import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import agent as agent_module
import tools


class ProfilePersistenceTests(unittest.TestCase):
    def setUp(self):
        snapshot = {
            "history": agent_module.ACTIVE_HISTORY_DB,
            "workspace": tools.terminal_session.cwd,
            "skill_dir": tools.skill_store.storage_dir,
            "user": (
                tools.user_profile_manager.storage_dir,
                tools.user_profile_manager.file_path,
                tools.user_profile_manager.allow_root_fallback,
            ),
            "memory": (
                tools.project_memory_manager.storage_dir,
                tools.project_memory_manager.file_path,
                tools.project_memory_manager.allow_root_fallback,
            ),
        }

        def restore():
            agent_module.ACTIVE_HISTORY_DB = snapshot["history"]
            tools.terminal_session.cwd = snapshot["workspace"]
            tools.skill_store.storage_dir = snapshot["skill_dir"]
            (tools.user_profile_manager.storage_dir,
             tools.user_profile_manager.file_path,
             tools.user_profile_manager.allow_root_fallback) = snapshot["user"]
            (tools.project_memory_manager.storage_dir,
             tools.project_memory_manager.file_path,
             tools.project_memory_manager.allow_root_fallback) = snapshot["memory"]

        self.addCleanup(restore)

    def parse(self, *arguments):
        with patch.object(sys, "argv", ["agent.py", *arguments]):
            return agent_module.parse_args()

    def test_cli_defaults_preserve_launch_cwd_and_source_adjacent_state_roots(self):
        with tempfile.TemporaryDirectory() as launch:
            with patch.object(os, "getcwd", return_value=launch):
                args = self.parse()
        self.assertIsNone(args.profile)
        self.assertEqual(args.workspace, str(Path(launch).resolve()))
        self.assertEqual(
            args.profiles_dir,
            str(Path(agent_module.__file__).resolve().parent / ".agent_profiles"),
        )
        self.assertEqual(
            args.workspaces_dir,
            str(Path(agent_module.__file__).resolve().parent / ".agent_workspaces"),
        )

    def test_cli_accepts_named_profile_custom_root_and_independent_workspace(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as workspace:
            args = self.parse(
                "--profile", "alice-_2", "--profiles-dir", root,
                "--workspace", workspace,
            )
        self.assertEqual(args.profile, "alice-_2")
        self.assertEqual(args.profiles_dir, str(Path(root).resolve()))
        self.assertEqual(args.workspace, str(Path(workspace).resolve()))

    def test_named_profile_omitted_workspace_bootstraps_exact_sibling_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            profiles = Path(temp, "profiles")
            workspaces = Path(temp, "workspaces")
            args = self.parse(
                "--profile", "alice", "--profiles-dir", str(profiles),
                "--workspaces-dir", str(workspaces),
            )
            expected = (workspaces / "alice").resolve()
            self.assertEqual(Path(args.workspace), expected)
            self.assertFalse(expected.exists())

            config = agent_module.configure_runtime(args)

            self.assertEqual(config.workspace, expected)
            self.assertTrue(expected.is_dir())
            profile = profiles / "alice"
            self.assertTrue((profile / "memories" / "USER.md").is_file())
            self.assertTrue((profile / "memories" / "MEMORY.md").is_file())
            self.assertTrue((profile / "skills").is_dir())
            self.assertTrue((profile / "history.db").is_file())
            for obsolete in (".agent_memories", ".agent_skills", ".agent_history.db", "workspace"):
                self.assertFalse((profile / obsolete).exists())

    def test_named_profile_defaults_are_distinct_and_bind_terminal_and_files(self):
        with tempfile.TemporaryDirectory() as temp:
            profiles, workspaces = Path(temp, "profiles"), Path(temp, "workspaces")
            common = ("--profiles-dir", str(profiles), "--workspaces-dir", str(workspaces))
            alice = self.parse("--profile", "alice", *common)
            alice_config = agent_module.configure_runtime(alice)
            self.assertEqual(Path(tools.terminal_session.workspace_root), alice_config.workspace)

            tools.write_file("alice.txt", "alice")
            alice_file = alice_config.workspace / "alice.txt"
            bob = self.parse("--profile", "bob", *common)
            bob_config = agent_module.configure_runtime(bob)
            self.assertEqual(Path(tools.terminal_session.workspace_root), bob_config.workspace)
            self.assertNotEqual(alice_config.workspace, bob_config.workspace)
            tools.write_file("bob.txt", "bob")
            bob_file = bob_config.workspace / "bob.txt"
            self.assertEqual(alice_config.workspace, (workspaces / "alice").resolve())
            self.assertEqual(bob_config.workspace, (workspaces / "bob").resolve())
            self.assertEqual(alice_file, alice_config.workspace / "alice.txt")
            self.assertEqual(bob_file, bob_config.workspace / "bob.txt")
            self.assertEqual(alice_file.read_text(encoding="utf-8"), "alice")
            self.assertEqual(bob_file.read_text(encoding="utf-8"), "bob")

    def test_invalid_profile_names_and_missing_workspace_are_rejected(self):
        invalid = ("", ".", "..", "a/b", "a\\b", "../alice", "C:\\alice", "a b", "x" * 65)
        for name in invalid:
            with self.subTest(name=name), self.assertRaises(SystemExit):
                self.parse("--profile", name)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(SystemExit):
                self.parse("--workspace", str(Path(root) / "missing"))

    def test_writable_profile_bootstrap_and_all_consumers_are_rebound(self):
        with tempfile.TemporaryDirectory() as temp, tempfile.TemporaryDirectory() as workspace:
            root = Path(temp, "profiles")
            defaults = Path(temp, "workspaces")
            args = self.parse(
                "--profile", "alice", "--profiles-dir", str(root),
                "--workspaces-dir", str(defaults), "--workspace", workspace,
            )
            config = agent_module.configure_runtime(args)
            profile = Path(root, "alice")
            self.assertEqual(config.state_root, profile.resolve())
            self.assertTrue((profile / "memories" / "USER.md").is_file())
            self.assertTrue((profile / "memories" / "MEMORY.md").is_file())
            self.assertTrue((profile / "skills").is_dir())
            self.assertTrue((profile / "history.db").is_file())
            self.assertFalse(defaults.exists())
            self.assertEqual(Path(tools.skill_store.storage_dir), profile / "skills")
            self.assertEqual(Path(tools.user_profile_manager.file_path), profile / "memories" / "USER.md")
            self.assertEqual(Path(tools.project_memory_manager.file_path), profile / "memories" / "MEMORY.md")
            self.assertEqual(Path(tools.terminal_session.workspace_root), Path(workspace).resolve())
            instance = agent_module.HermesCodingAgent(enable_memory=False, enable_skills=False)
            self.assertEqual(Path(instance.logger.db_path), profile / "history.db")
            self.assertIs(instance.memory_extractor.user_manager, tools.user_profile_manager)
            self.assertIs(instance.skill_extractor.skill_store, tools.skill_store)

    def test_two_profiles_share_workspace_without_memory_skill_or_history_bleed(self):
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as workspace:
            alice = self.parse("--profile", "alice", "--profiles-dir", root, "--workspace", workspace)
            agent_module.configure_runtime(alice)
            tools.user_profile_manager.save_profile("# Alice")
            tools.project_memory_manager.save_memory("# Alice memory")
            tools.skill_store.save_skill("alice_only", "Alice", "Do Alice work")
            a = agent_module.HermesCodingAgent(enable_memory=False, enable_skills=False)
            a.logger.start_session("alice-session", "private")

            bob = self.parse("--profile", "bob", "--profiles-dir", root, "--workspace", workspace)
            agent_module.configure_runtime(bob)
            self.assertNotIn("Alice", tools.user_profile_manager.load_profile())
            self.assertNotIn("Alice memory", tools.project_memory_manager.load_memory())
            self.assertNotIn("alice_only", tools.skill_store.list_skills())
            b = agent_module.HermesCodingAgent(enable_memory=False, enable_skills=False)
            self.assertFalse(b.resume_session("alice-session"))
            self.assertEqual(Path(tools.terminal_session.workspace_root), Path(workspace).resolve())

    def test_read_only_and_stateless_never_create_missing_profile(self):
        for flag in ("--read-only", "--stateless"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as workspace:
                args = self.parse("--profile", "missing", "--profiles-dir", root, "--workspace", workspace, flag)
                with self.assertRaisesRegex(ValueError, "does not exist"):
                    agent_module.configure_runtime(args)
                self.assertFalse(Path(root, "missing").exists())

    def test_read_only_and_stateless_reject_missing_default_workspace_without_writes(self):
        for flag in ("--read-only", "--stateless"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as root:
                profile = Path(root, "alice")
                profile.mkdir()
                before = tuple(profile.iterdir())
                workspaces = Path(root, "workspaces")
                args = self.parse(
                    "--profile", "alice", "--profiles-dir", root,
                    "--workspaces-dir", str(workspaces), flag,
                )

                with self.assertRaisesRegex(ValueError, "default workspace.*does not exist"):
                    agent_module.configure_runtime(args)

                self.assertEqual(tuple(profile.iterdir()), before)
                self.assertFalse(workspaces.exists())

    def test_cli_missing_read_only_profile_exits_nonzero_without_creating_state(self):
        with tempfile.TemporaryDirectory() as temp:
            profiles_root = Path(temp, "profiles")
            workspace = Path(temp, "workspace")
            workspace.mkdir()

            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(agent_module.__file__).resolve()),
                    "--profile", "missing",
                    "--profiles-dir", str(profiles_root),
                    "--workspace", str(workspace),
                    "--read-only",
                ],
                input="",
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("Named profile 'missing' does not exist", result.stderr)
            self.assertFalse(profiles_root.exists())

    def test_cli_writable_named_profile_bootstraps_on_eof_without_llm_request(self):
        with tempfile.TemporaryDirectory() as temp:
            profiles_root = Path(temp, "profiles")
            profile = profiles_root / "alice"
            workspaces_root = Path(temp, "workspaces")
            workspace = workspaces_root / "alice"

            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(agent_module.__file__).resolve()),
                    "--profile", "alice",
                    "--profiles-dir", str(profiles_root),
                    "--workspaces-dir", str(workspaces_root),
                ],
                input="",
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn("Profile:    alice", result.stdout)
            self.assertIn(f"State:      {profile.resolve()}", result.stdout)
            self.assertIn(f"Workspace:  {workspace.resolve()}", result.stdout)
            self.assertIn("Shutting down.", result.stdout)
            self.assertTrue((profile / "memories" / "USER.md").is_file())
            self.assertTrue((profile / "memories" / "MEMORY.md").is_file())
            self.assertTrue((profile / "skills").is_dir())
            self.assertTrue((profile / "history.db").is_file())
            self.assertTrue(workspace.is_dir())
            self.assertFalse((workspace / ".agent_memories").exists())
            self.assertFalse((workspace / ".agent_skills").exists())
            self.assertFalse((workspace / ".agent_history.db").exists())

    def test_partial_named_profile_read_only_and_stateless_create_nothing(self):
        for flag in ("--read-only", "--stateless"):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as temp:
                profiles = Path(temp, "profiles")
                profile = profiles / "alice"
                (profile / "memories").mkdir(parents=True)
                (profile / "memories" / "USER.md").write_text("existing", encoding="utf-8")
                workspaces = Path(temp, "workspaces")
                before = sorted(str(path.relative_to(temp)) for path in Path(temp).rglob("*"))
                args = self.parse(
                    "--profile", "alice", "--profiles-dir", str(profiles),
                    "--workspaces-dir", str(workspaces), flag,
                )
                with self.assertRaisesRegex(ValueError, "default workspace.*does not exist"):
                    agent_module.configure_runtime(args)
                after = sorted(str(path.relative_to(temp)) for path in Path(temp).rglob("*"))
                self.assertEqual(after, before)

    def test_legacy_runtime_uses_launch_cwd_locations_exactly(self):
        with tempfile.TemporaryDirectory() as launch:
            with patch.object(os, "getcwd", return_value=launch):
                args = self.parse()
            config = agent_module.configure_runtime(args)
            self.assertTrue(config.legacy)
            self.assertEqual(Path(tools.user_profile_manager.storage_dir), Path(launch, ".agent_memories"))
            self.assertEqual(Path(tools.skill_store.storage_dir), Path(launch, ".agent_skills"))
            instance = agent_module.HermesCodingAgent(enable_memory=False, enable_skills=False, read_only=True)
            self.assertEqual(Path(instance.logger.db_path), Path(launch, ".agent_history.db"))


if __name__ == "__main__":
    unittest.main()
