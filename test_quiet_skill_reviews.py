"""Display-only muting for asynchronous skill review owner notices."""
from io import StringIO
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
from unittest.mock import MagicMock, patch

import pytest

import agent as agent_module
from skill_review import Admission, Owner, ReviewService, load_roster, read_auth
from skills import SkillStore
from test_async_skill_review import create_proposal, evidence, owner_for, rows, setup_roster, wait_for
from test_skill_review_integration import answer_message
from test_skill_review_process import boundary_openai, command, prepared_episode


ROOT = Path(__file__).resolve().parent


def _agent(tmp_path, *, quiet):
    roster = setup_roster(tmp_path)
    previous = agent_module.skill_store.storage_dir
    instance = None
    try:
        agent_module.skill_store.storage_dir = str(tmp_path / "profile-0" / "skills")
        with patch.object(agent_module, "ACTIVE_HISTORY_DB", str(tmp_path / "history.db")):
            instance = agent_module.HermesCodingAgent(
                model=roster.model,
                base_url=roster.base_url,
                enable_memory=False,
                auto_learn_skills=True,
                review_roster=roster,
                review_profile="user-0",
                quiet_skill_reviews=quiet,
            )
    finally:
        agent_module.skill_store.storage_dir = previous
    return instance, roster


def _all_notices_settled(mailbox, minimum=1):
    if not Path(mailbox).exists():
        return False
    try:
        with sqlite3.connect(mailbox) as conn:
            total, pending = conn.execute(
                "SELECT count(*),sum(CASE WHEN delivered IS NULL THEN 1 ELSE 0 END) FROM notices"
            ).fetchone()
            return total >= minimum and pending == 0
    except sqlite3.Error:
        return False


def test_parser_exposes_quiet_flag_with_false_default(monkeypatch):
    help_result = subprocess.run(
        command("agent.py", "--help"), cwd=ROOT, capture_output=True, text=True, timeout=5
    )
    assert help_result.returncode == 0
    assert "--quiet-skill-reviews" in help_result.stdout

    monkeypatch.setattr(sys, "argv", ["agent.py"])
    assert agent_module.parse_args().quiet_skill_reviews is False
    monkeypatch.setattr(sys, "argv", ["agent.py", "--quiet-skill-reviews"])
    assert agent_module.parse_args().quiet_skill_reviews is True


def test_default_owner_still_prints_and_acknowledges_valid_notices(tmp_path):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, background=False)
    try:
        assert owner.enqueue(evidence()).status == "ACCEPTED"
        owner.pump()
        assert ReviewService(roster, provider=lambda _: create_proposal("visible_skill")).once() == 1
        owner.pump()
        output = StringIO()
        assert owner.deliver_notices(output) == 2
        assert "[Skill Review] Review started." in output.getvalue()
        assert "[Skill Review] Created skill 'visible_skill'" in output.getvalue()
        assert all(row["delivered"] and row["delivered"] > 0 for row in rows(owner, "notices"))
    finally:
        owner.close()


def test_quiet_startup_consumes_pending_backlog_and_unmuted_reconnect_does_not_replay(tmp_path, capsys):
    roster = setup_roster(tmp_path)
    first = owner_for(roster, background=False)
    try:
        assert first.enqueue(evidence()).status == "ACCEPTED"
        first.pump()
        assert ReviewService(roster, provider=lambda _: create_proposal("startup_skill")).once() == 1
        first.pump()
        assert len(rows(first, "notices")) == 2
        assert all(row["delivered"] is None for row in rows(first, "notices"))
    finally:
        first.close()

    quiet = owner_for(roster, quiet=True, background=True)
    try:
        wait_for(lambda: _all_notices_settled(quiet.entry.mailbox, minimum=2))
        delivered = [row["delivered"] for row in rows(quiet, "notices")]
        assert all(value and value > 0 for value in delivered)
        assert "[Skill Review]" not in capsys.readouterr().out
    finally:
        quiet.close()

    visible = owner_for(roster, background=False)
    try:
        output = StringIO()
        assert visible.deliver_notices(output) == 0
        assert output.getvalue() == ""
    finally:
        visible.close()


def test_quiet_safe_shutdown_publishes_result_and_acknowledges_without_output(tmp_path, capsys):
    roster = setup_roster(tmp_path)
    owner = owner_for(roster, quiet=True, background=False)
    assert owner.enqueue(evidence()).status == "ACCEPTED"
    owner.pump()
    assert ReviewService(roster, provider=lambda _: create_proposal("shutdown_skill")).once() == 1
    owner.close()

    assert (tmp_path / "profile-0" / "skills" / "shutdown_skill.md").exists()
    with sqlite3.connect(owner.entry.mailbox) as conn:
        status = conn.execute("SELECT status FROM jobs ORDER BY seq DESC LIMIT 1").fetchone()[0]
        delivered = [row[0] for row in conn.execute("SELECT delivered FROM notices ORDER BY seq")]
    assert status == "APPLIED"
    assert len(delivered) == 2 and all(value and value > 0 for value in delivered)
    assert "[Skill Review]" not in capsys.readouterr().out


def test_quiet_invalid_notice_stays_invalid_while_valid_notice_is_dismissed(tmp_path):
    roster = setup_roster(tmp_path)
    first = owner_for(roster, background=False)
    try:
        assert first.enqueue(evidence()).status == "ACCEPTED"
        first.pump()
        assert ReviewService(roster, provider=lambda _: {"action": "NONE"}).once() == 1
        first.pump()
    finally:
        first.close()

    with sqlite3.connect(first.entry.mailbox) as conn:
        conn.execute("UPDATE notices SET payload_json='{}' WHERE kind='DETAIL'")
        conn.commit()

    quiet = owner_for(roster, quiet=True, background=False)
    try:
        quiet.pump()
        values = {row["kind"]: row["delivered"] for row in rows(quiet, "notices")}
        assert values["START"] and values["START"] > 0
        assert values["DETAIL"] == -1
    finally:
        quiet.close()


@pytest.mark.parametrize(
    "settings",
    [
        dict(auto_learn_skills=False),
        dict(auto_learn_skills=True, read_only=True),
        dict(auto_learn_skills=True, stateless=True),
        dict(auto_learn_skills=True, enable_skills=False),
    ],
)
def test_quiet_does_not_bypass_disabled_or_nonwriting_mode_fences(tmp_path, settings):
    roster = setup_roster(tmp_path)
    previous = agent_module.skill_store.storage_dir
    instance = None
    try:
        agent_module.skill_store.storage_dir = str(tmp_path / "profile-0" / "skills")
        with patch.object(agent_module, "ACTIVE_HISTORY_DB", str(tmp_path / "history.db")):
            instance = agent_module.HermesCodingAgent(
                model=roster.model,
                base_url=roster.base_url,
                enable_memory=False,
                review_roster=roster,
                review_profile="user-0",
                quiet_skill_reviews=True,
                **settings,
            )
        assert instance.deliver_skill_review_notices(StringIO()) == 0
        assert not Path(roster.profiles[0].mailbox).exists()
    finally:
        if instance:
            instance.shutdown_skill_reviews()
        agent_module.skill_store.storage_dir = previous


def test_quiet_does_not_bypass_foreign_profile_authorization(tmp_path):
    roster = setup_roster(tmp_path)
    with pytest.raises(ValueError):
        Owner(
            roster,
            "foreign-profile",
            SkillStore(str(tmp_path / "foreign" / "skills")),
            quiet=True,
            background=False,
        )


def test_quiet_silences_setup_refusal_error_and_begin_turn_prints_but_preserves_answer(tmp_path, capsys):
    unavailable = object.__new__(agent_module.HermesCodingAgent)
    unavailable.auto_learn_skills = True
    unavailable.read_only = False
    unavailable.stateless = False
    unavailable.quiet_skill_reviews = True
    unavailable.skill_review_owner = None
    unavailable._task_evidence = MagicMock()
    unavailable.last_skill_admission = None
    unavailable.run_auto_skill_synthesis("setup failure")

    instance, _ = _agent(tmp_path, quiet=True)
    try:
        instance._task_evidence = MagicMock()
        with patch.object(
            instance.skill_review_owner,
            "capture_turn",
            return_value=Admission("REFUSED", "stage=episode.admission reason=test"),
        ):
            instance.run_auto_skill_synthesis("refused")

        instance._task_evidence = MagicMock()
        with patch.object(instance.skill_review_owner, "capture_turn", side_effect=ValueError("test")):
            instance.run_auto_skill_synthesis("error")

        with patch.object(
            instance.skill_review_owner,
            "begin_turn",
            return_value=Admission("FAILED", "stage=capture.begin_turn reason=test"),
        ), patch.object(instance, "step", return_value=answer_message("Quiet business answer.")), patch.object(
            instance, "manage_context"
        ), patch.object(instance, "run_auto_skill_synthesis"):
            assert instance.run("Keep the business answer visible") == "Quiet business answer."
        output = capsys.readouterr().out
        assert "Quiet business answer." in output
        assert "[Skill Review]" not in output
    finally:
        instance.shutdown_skill_reviews()


def test_default_begin_turn_failure_remains_visible(tmp_path, capsys):
    instance, _ = _agent(tmp_path, quiet=False)
    try:
        with patch.object(
            instance.skill_review_owner,
            "begin_turn",
            return_value=Admission("FAILED", "stage=capture.begin_turn reason=visible-test"),
        ), patch.object(instance, "step", return_value=answer_message("Visible business answer.")), patch.object(
            instance, "manage_context"
        ), patch.object(instance, "run_auto_skill_synthesis"):
            assert instance.run("Keep default behavior") == "Visible business answer."
        output = capsys.readouterr().out
        assert "[Skill Review] FAILED: stage=capture.begin_turn reason=visible-test" in output
        assert "Visible business answer." in output
    finally:
        instance.shutdown_skill_reviews()


def test_partially_constructed_agent_keeps_visible_default(capsys):
    instance = object.__new__(agent_module.HermesCodingAgent)
    instance._print_skill_review("legacy construction")
    assert "[Skill Review] legacy construction" in capsys.readouterr().out


def test_real_quiet_cli_consumes_result_while_foreground_busy_and_unmuted_cli_has_no_replay(tmp_path):
    with boundary_openai() as (endpoint, foreground_entered, foreground_release, _):
        profile = tmp_path / "profiles" / "alice"
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        roster_path = tmp_path / "roster.json"
        roster_path.write_text(
            json.dumps(
                dict(
                    control_dir=str(tmp_path / "control"),
                    model="test-model",
                    base_url=endpoint,
                    profiles=[
                        dict(
                            profile_id="alice",
                            mailbox=str(profile / "skill_review.db"),
                            store_id=SkillStore(str(profile / "skills")).store_id,
                        )
                    ],
                )
            ),
            encoding="utf-8",
        )
        roster = load_roster(roster_path)
        prepared_episode(roster, profile)
        env = dict(
            os.environ,
            PYTHONUNBUFFERED="1",
            PYTHONIOENCODING="utf-8",
            OPENAI_API_KEY="local-fake-test-key",
        )
        base_args = (
            "agent.py",
            "--profile",
            "alice",
            "--profiles-dir",
            tmp_path / "profiles",
            "--workspace",
            workspace,
            "--model",
            "test-model",
            "--base-url",
            endpoint,
            "--auto-skills",
            "--no-memory",
            "--skill-review-roster",
            roster_path,
        )
        output = []
        owner = subprocess.Popen(
            command(*base_args, "--quiet-skill-reviews"),
            cwd=ROOT,
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        reader = threading.Thread(target=lambda: [output.append(line) for line in owner.stdout], daemon=True)
        reader.start()
        try:
            wait_for(lambda: bool((read_auth(roster, roster.entry("alice")) or {}).get("owner_id")))
            owner.stdin.write("Start a blocked foreground response\n")
            owner.stdin.flush()
            assert foreground_entered.wait(5)

            service = subprocess.run(
                command("skill_review.py", "once", "--roster", roster_path, "--timeout", "3"),
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
                timeout=8,
            )
            assert service.returncode == 0 and '"processed":1' in service.stdout
            wait_for(
                lambda: (profile / "skills" / "boundary_skill.md").exists()
                and _all_notices_settled(profile / "skill_review.db", minimum=2)
            )
            assert "Boundary foreground answer complete" not in "".join(output)
            assert "[Skill Review]" not in "".join(output)

            with sqlite3.connect(profile / "skill_review.db") as conn:
                status = conn.execute("SELECT status FROM jobs ORDER BY seq LIMIT 1").fetchone()[0]
                delivered = [row[0] for row in conn.execute("SELECT delivered FROM notices ORDER BY seq")]
            assert status == "APPLIED"
            assert len(delivered) == 2 and all(value and value > 0 for value in delivered)

            foreground_release.set()
            wait_for(lambda: "Boundary foreground answer complete" in "".join(output))
            owner.stdin.write("exit\n")
            owner.stdin.flush()
            owner.wait(timeout=5)
            reader.join(2)
            assert owner.returncode == 0
            assert "Boundary foreground answer complete" in "".join(output)
            assert "[Skill Review]" not in "".join(output)
        finally:
            foreground_release.set()
            if owner.poll() is None:
                owner.terminate()
                owner.wait(timeout=5)
            owner.stdin.close()
            owner.stdout.close()

        reconnect = subprocess.run(
            command(*base_args),
            cwd=ROOT,
            env=env,
            input="exit\n",
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=8,
        )
        assert reconnect.returncode == 0
        assert "[Skill Review]" not in reconnect.stdout + reconnect.stderr
