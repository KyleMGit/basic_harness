"""Focused regression tests for create-only direct skill saves."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from skills import AutoSkillExtractor, SkillStore
from tools import registry, skill_store as registry_skill_store


class TestDirectSkillSaveSafety(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.skill_dir = Path(self.temp_dir.name, "skills")
        self.previous_storage_dir = registry_skill_store.storage_dir
        registry_skill_store.storage_dir = str(self.skill_dir)

    def tearDown(self):
        registry_skill_store.storage_dir = self.previous_storage_dir
        self.temp_dir.cleanup()

    def _direct_save(self, name, description, instructions):
        return registry.execute("save_skill", {
            "name": name,
            "description": description,
            "instructions": instructions,
        })

    @staticmethod
    def _client_for(payload):
        client = MagicMock()
        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=json.dumps(payload)))]
        client.chat.completions.create.return_value = response
        return client

    @staticmethod
    def _messages():
        return [
            {"role": "user", "content": "Set up pytest coverage."},
            {"role": "assistant", "content": "Starting."},
            {"role": "tool", "content": "Environment configured."},
            {"role": "assistant", "content": "Done."},
        ]

    def test_direct_same_normalized_name_refuses_and_preserves_pair(self):
        first = self._direct_save(
            "Deploy Skill", "Safely deploy the service", "Run the original deployment steps."
        )
        self.assertIn("successfully saved", first)
        md_path = self.skill_dir / "deploy_skill.md"
        json_path = self.skill_dir / "deploy_skill.json"
        original_md = md_path.read_bytes()
        original_json = json_path.read_bytes()

        result = self._direct_save(
            "deploy_skill.json", "Replacement description", "Overwrite the existing procedure."
        )

        self.assertIn("refused", result.lower())
        self.assertIn("Deploy Skill", result)
        self.assertIn("load", result.lower())
        self.assertEqual(original_md, md_path.read_bytes())
        self.assertEqual(original_json, json_path.read_bytes())

    def test_direct_check_and_write_use_consistent_canonicalization(self):
        first = self._direct_save(
            "parse_json", "Parse JSON safely", "Use the original JSON parsing procedure."
        )
        self.assertIn("successfully saved", first)
        md_path = self.skill_dir / "parse_json.md"
        json_path = self.skill_dir / "parse_json.json"
        original_md = md_path.read_bytes()
        original_json = json_path.read_bytes()

        result = self._direct_save(
            "parse.json\t", "Unrelated description", "Unrelated new content."
        )

        self.assertIn("successfully saved", result)
        self.assertEqual(original_md, md_path.read_bytes())
        self.assertEqual(original_json, json_path.read_bytes())
        self.assertIn("Unrelated new content", (self.skill_dir / "parse.md").read_text())
        self.assertIn("Unrelated new content", (self.skill_dir / "parse.json").read_text())
        self.assertEqual(2, len(SkillStore(str(self.skill_dir)).get_all_skills()))

    def test_direct_high_similarity_distinct_name_creates_second_pair(self):
        first = self._direct_save(
            "pytest_environment_setup",
            "Configure python virtual environment pytest coverage dependencies",
            "Original environment instructions.",
        )
        self.assertIn("successfully saved", first)
        result = self._direct_save(
            "python_pytest_environment",
            "Configure python virtual environment pytest coverage dependencies",
            "Different instructions that must not be published.",
        )

        self.assertIn("successfully saved", result)
        self.assertTrue((self.skill_dir / "python_pytest_environment.md").is_file())
        self.assertTrue((self.skill_dir / "python_pytest_environment.json").is_file())
        self.assertEqual(2, len(SkillStore(str(self.skill_dir)).get_all_skills()))
        self.assertIn("Original environment instructions", (self.skill_dir / "pytest_environment_setup.md").read_text())
        self.assertIn("Different instructions", (self.skill_dir / "python_pytest_environment.md").read_text())

    def test_direct_novel_skill_succeeds(self):
        result = self._direct_save(
            "sqlite_integrity_check",
            "Validate a SQLite database before migration",
            "Run PRAGMA integrity_check and inspect every returned row.",
        )
        self.assertIn("successfully saved", result)
        self.assertTrue((self.skill_dir / "sqlite_integrity_check.md").is_file())
        self.assertTrue((self.skill_dir / "sqlite_integrity_check.json").is_file())

    def test_reflection_explicit_update_remains_authorized(self):
        store = SkillStore(str(self.skill_dir))
        store.save_skill("pytest_setup", "Configure pytest tooling", "Original instructions.")
        extractor = AutoSkillExtractor(store)
        client = self._client_for({
            "action": "UPDATE",
            "target_skill_name": "pytest_setup",
            "name": "ignored_model_name",
            "description": "Configure pytest tooling with coverage",
            "instructions": "Install pytest and pytest-cov.",
        })

        result = extractor.extract_and_save(client, "model", self._messages(), "Improve pytest setup")

        self.assertEqual("UPDATE", result["action"])
        self.assertEqual("pytest_setup", result["name"])
        self.assertEqual(1, len(store.get_all_skills()))
        self.assertIn("pytest-cov", store.load_skill("pytest_setup"))

    def test_reflection_update_trailing_whitespace_extension_updates_resolved_pair_only(self):
        store = SkillStore(str(self.skill_dir))
        store.save_skill("foo", "Original description", "Original instructions.")
        extractor = AutoSkillExtractor(store)
        client = self._client_for({
            "action": "UPDATE",
            "target_skill_name": "foo.md ",
            "name": "ignored_model_name",
            "description": "Updated description",
            "instructions": "Updated instructions.",
        })

        result = extractor.extract_and_save(client, "model", self._messages(), "Update foo")

        self.assertEqual("UPDATE", result["action"])
        self.assertEqual("foo.md", result["name"])
        self.assertIn("Updated instructions", (self.skill_dir / "foo.md").read_text())
        self.assertIn("Updated instructions", (self.skill_dir / "foo.json").read_text())
        self.assertFalse((self.skill_dir / "foo_md.md").exists())
        self.assertFalse((self.skill_dir / "foo_md.json").exists())
        self.assertEqual(1, len(store.get_all_skills()))

    def test_reflection_update_missing_target_returns_error_without_writes(self):
        store = SkillStore(str(self.skill_dir))
        extractor = AutoSkillExtractor(store)
        client = self._client_for({
            "action": "UPDATE",
            "target_skill_name": "missing",
            "name": "ignored_model_name",
            "description": "Missing description",
            "instructions": "Must not be written.",
        })

        result = extractor.extract_and_save(client, "model", self._messages(), "Update missing")

        self.assertEqual("ERROR", result["action"])
        self.assertFalse(self.skill_dir.exists())

    def test_reflection_update_ambiguous_target_returns_error_and_preserves_all_bytes(self):
        self.skill_dir.mkdir(parents=True)
        for directory, instructions in (("first", "First copy."), ("second", "Second copy.")):
            nested = self.skill_dir / directory
            nested.mkdir()
            (nested / "SKILL.md").write_text(
                SkillStore.format_markdown_skill("foo", "Duplicate foo", instructions),
                encoding="utf-8",
            )
        before = {
            path.relative_to(self.skill_dir): path.read_bytes()
            for path in self.skill_dir.rglob("*") if path.is_file()
        }
        store = SkillStore(str(self.skill_dir))
        extractor = AutoSkillExtractor(store)
        client = self._client_for({
            "action": "UPDATE",
            "target_skill_name": "foo",
            "name": "ignored_model_name",
            "description": "Updated description",
            "instructions": "Must not replace either copy.",
        })

        result = extractor.extract_and_save(client, "model", self._messages(), "Update foo")

        after = {
            path.relative_to(self.skill_dir): path.read_bytes()
            for path in self.skill_dir.rglob("*") if path.is_file()
        }
        self.assertEqual("ERROR", result["action"])
        self.assertEqual(before, after)

    def test_direct_nested_skill_collision_refuses_without_top_level_pair(self):
        nested = self.skill_dir / "nested"
        nested.mkdir(parents=True)
        skill_path = nested / "SKILL.md"
        skill_path.write_text(
            SkillStore.format_markdown_skill("nested", "Nested skill", "Original instructions."),
            encoding="utf-8",
        )
        original = skill_path.read_bytes()

        result = self._direct_save("nested", "Replacement", "Must not be created.")

        self.assertIn("refused", result.lower())
        self.assertEqual(original, skill_path.read_bytes())
        self.assertFalse((self.skill_dir / "nested.md").exists())
        self.assertFalse((self.skill_dir / "nested.json").exists())

    def test_reflection_similar_create_remains_create(self):
        store = SkillStore(str(self.skill_dir))
        store.save_skill(
            "pytest_environment_setup",
            "Configure python virtual environment pytest coverage dependencies",
            "Original instructions.",
        )
        extractor = AutoSkillExtractor(store)
        client = self._client_for({
            "action": "CREATE",
            "target_skill_name": "",
            "name": "python_pytest_environment",
            "description": "Configure python virtual environment pytest coverage dependencies",
            "instructions": "Curated instructions with pytest-cov.",
        })

        result = extractor.extract_and_save(client, "model", self._messages(), "Improve pytest setup")

        self.assertEqual("CREATE", result["action"])
        self.assertEqual("python_pytest_environment", result["name"])
        self.assertEqual(2, len(store.get_all_skills()))
        self.assertTrue((self.skill_dir / "python_pytest_environment.md").is_file())
        self.assertIn("Original instructions", store.load_skill("pytest_environment_setup"))
        self.assertIn("pytest-cov", store.load_skill("python_pytest_environment"))

    def test_reflection_low_similarity_normalized_name_create_refuses_without_writes(self):
        store = SkillStore(str(self.skill_dir))
        store.save_skill(
            "Deploy Skill",
            "Release a service safely",
            "Original deployment instructions.",
        )
        original_md = (self.skill_dir / "deploy_skill.md").read_bytes()
        original_json = (self.skill_dir / "deploy_skill.json").read_bytes()
        original_files = {path.name for path in self.skill_dir.iterdir()}
        extractor = AutoSkillExtractor(store)
        client = self._client_for({
            "action": "CREATE",
            "target_skill_name": "",
            "name": "deploy_skill.json",
            "description": "Diagnose lunar telemetry corruption",
            "instructions": "Use the curated replacement procedure.",
        })

        result = extractor.extract_and_save(client, "model", self._messages(), "Improve deployment")

        self.assertEqual("ERROR", result["action"])
        self.assertIn("refused", result["description"].lower())
        self.assertEqual(original_md, (self.skill_dir / "deploy_skill.md").read_bytes())
        self.assertEqual(original_json, (self.skill_dir / "deploy_skill.json").read_bytes())
        self.assertEqual(1, len(store.get_all_skills()))
        self.assertEqual(original_files, {path.name for path in self.skill_dir.iterdir()})

    def test_model_schema_has_no_update_or_overwrite_bypass(self):
        schema = next(
            item["function"] for item in registry.schemas
            if item["function"]["name"] == "save_skill"
        )
        properties = schema["parameters"]["properties"]
        self.assertEqual({"name", "description", "instructions"}, set(properties))
        self.assertFalse({"update", "overwrite", "allow_update"} & set(properties))
        self.assertIn("create", schema["description"].lower())
        self.assertIn("refus", schema["description"].lower())


if __name__ == "__main__":
    unittest.main()
