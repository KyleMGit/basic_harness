"""Public owner CLI plus REAL separate service and deterministic local HTTP SDK."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import agent as agent_module
from skills import SkillStore
from test_async_skill_review import wait_for, create_proposal, setup_roster, owner_for, evidence, rows
from skill_review import Evidence, Owner, ReviewService, load_roster
from test_skill_review_integration import answer_message


ROOT = Path(__file__).resolve().parent


@contextmanager
def fake_openai(*, delay=False):
    entered, release = threading.Event(), threading.Event()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            review = "private skill snapshot" in body["messages"][0]["content"]
            if review:
                entered.set()
                if delay:
                    release.wait(10)
            content = json.dumps(create_proposal("from_service")) if review else "The subprocess task is complete."
            raw = json.dumps(dict(id="fake-response", object="chat.completion", created=1,
                model=body["model"], choices=[dict(index=0, finish_reason="stop",
                message=dict(role="assistant", content=content))], usage=dict(prompt_tokens=10, completion_tokens=10, total_tokens=20))).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            try:
                self.wfile.write(raw)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/v1", entered, release, requests
    finally:
        release.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


@contextmanager
def boundary_openai(*, permission=False):
    """Fake endpoint that blocks a foreground answer or emits a reviewed tool call."""
    foreground_entered, foreground_release = threading.Event(), threading.Event()
    requests = []
    foreground_calls = 0

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            nonlocal foreground_calls
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            review = "private skill snapshot" in body["messages"][0]["content"]
            if review:
                content = json.dumps(create_proposal("boundary_skill"))
                message = dict(role="assistant", content=content)
                reason = "stop"
            else:
                foreground_calls += 1
                foreground_entered.set()
                if permission and foreground_calls == 1:
                    message = dict(role="assistant", content="", tool_calls=[dict(
                        id="permission-call", type="function", function=dict(
                            name="run_terminal_command",
                            arguments=json.dumps({"command": "Write-Output boundary-test"}),
                        ))])
                    reason = "tool_calls"
                else:
                    if not permission:
                        foreground_release.wait(10)
                    message = dict(role="assistant", content="Boundary foreground answer complete.")
                    reason = "stop"
            raw = json.dumps(dict(
                id="fake-boundary", object="chat.completion", created=1, model=body["model"],
                choices=[dict(index=0, finish_reason=reason, message=message)],
                usage=dict(prompt_tokens=10, completion_tokens=10, total_tokens=20),
            )).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield (f"http://127.0.0.1:{server.server_port}/v1", foreground_entered,
               foreground_release, requests)
    finally:
        foreground_release.set()
        server.shutdown()
        server.server_close()
        thread.join(2)


def command(*args):
    return [sys.executable, *map(str, args)]


def http_agent(tmp_path, endpoint):
    setup_roster(tmp_path)
    roster_path = tmp_path / "roster.json"
    value = json.loads(roster_path.read_text())
    value["base_url"] = endpoint
    value["profiles"][0]["profile_id"] = "user-0"
    roster_path.write_text(json.dumps(value))
    roster = load_roster(roster_path)
    previous = agent_module.skill_store.storage_dir
    try:
        agent_module.skill_store.storage_dir = str(tmp_path / "profile-0" / "skills")
        with patch.object(agent_module, "ACTIVE_HISTORY_DB", str(tmp_path / "history.db")):
            instance = agent_module.HermesCodingAgent(
                model=roster.model, base_url=roster.base_url, enable_memory=False,
                auto_learn_skills=True, read_only=False, review_roster=roster,
                review_profile="user-0",
            )
    finally:
        agent_module.skill_store.storage_dir = previous
    return instance, roster, roster_path


def native_tool_message(sql, call_id):
    call = SimpleNamespace(id=call_id, function=SimpleNamespace(
        name="query_teradata", arguments=json.dumps({"sql": sql})))
    message = SimpleNamespace(content="", tool_calls=[call])
    message.model_dump = lambda exclude_none=True: {
        "role": "assistant", "content": "", "tool_calls": [{
            "id": call_id, "type": "function",
            "function": {"name": "query_teradata", "arguments": json.dumps({"sql": sql})},
        }],
    }
    return message


def capture_large_multiround_agent_episode(instance):
    metadata = json.dumps({
        "database": "warehouse", "columns": ["column_name", "data_type"],
        "rows": [["RETAINED_AGENT_METADATA_" + "m" * 15000, "VARCHAR"]],
        "row_count": 1, "truncated": False,
    })
    business = json.dumps({
        "database": "warehouse", "columns": ["amount"],
        "rows": [["PRIVATE-AGENT-BUSINESS-ROW" * 5000]],
        "row_count": 1, "truncated": False,
    })
    replies = [
        native_tool_message("SELECT column_name, data_type FROM information_schema.columns", "metadata"),
        native_tool_message(
            "SELECT * FROM sales QUALIFY ROW_NUMBER() OVER (ORDER BY sale_date DESC)=1 "
            "/* RETAINED_AGENT_SQL_" + "x" * 20000 + " */", "business"),
        answer_message("Verified the large multi-round procedure."),
    ]
    with patch.object(instance, "step", side_effect=replies), \
         patch.object(instance, "manage_context"), \
         patch.object(agent_module.registry, "execute", side_effect=[metadata, business]):
        assert instance.run(
            "No, that procedure is wrong; inspect metadata and use the verified QUALIFY correction "
            "password=synthetic-private-value") == "Verified the large multi-round procedure."
    instance.skill_review_owner.flush_session(instance.session_id)
    instance.skill_review_owner.pump()
    assert rows(instance.skill_review_owner, "jobs")[-1]["status"] == "PREPARED"


def prepared_episode(roster, profile, text="No, that procedure is wrong; use QUALIFY password=private-value"):
    item = Evidence("process-session", "process-episode")
    item.add({"role": "user", "content": text})
    item.add({"role": "assistant", "content": "", "tool_calls": [{
        "id": "metadata", "type": "function", "function": {
            "name": "query_teradata", "arguments": json.dumps({
                "sql": "SELECT column_name, data_type FROM information_schema.columns"})},
    }]})
    item.add({"role": "tool", "tool_call_id": "metadata", "content": json.dumps({
        "database": "warehouse", "columns": ["column_name", "data_type"],
        "rows": [["RETAINED_METADATA_CONTEXT_" + ("m" * 10000), "VARCHAR"]],
        "row_count": 1, "truncated": False,
    })})
    item.add({"role": "assistant", "content": "", "tool_calls": [{
        "id": "sql", "type": "function", "function": {
            "name": "query_teradata", "arguments": json.dumps({
                "sql": "SELECT * FROM sales QUALIFY ROW_NUMBER() OVER (ORDER BY sale_date DESC)=1 /* RETAINED_SQL_CONTEXT "
                       + ("x" * 20000) + " */"})},
    }]})
    item.add({"role": "tool", "tool_call_id": "sql", "content": json.dumps({
        "database": "warehouse", "columns": ["amount"], "rows": [["private-row" * 5000]],
        "row_count": 1, "truncated": False,
    })})
    item.add({"role": "assistant", "content": "Verified execution completed."})
    owner = Owner(roster, "alice", SkillStore(str(profile / "skills")), background=False)
    try:
        assert owner.capture_turn(item.finish()).status == "ELIGIBLE"
        owner.flush_session("process-session")
        owner.pump()
        assert rows(owner, "jobs")[0]["status"] == "PREPARED"
    finally:
        owner.close()


def test_real_owner_and_service_process_with_blocked_http_idle_apply_and_singleton(tmp_path):
    with fake_openai(delay=True) as (endpoint, entered, release, requests):
        profile = tmp_path / "profiles" / "alice"
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        roster_path = tmp_path / "roster.json"
        roster_path.write_text(json.dumps(dict(control_dir=str(tmp_path / "control"), model="test-model", base_url=endpoint,
            profiles=[dict(profile_id="alice", mailbox=str(profile / "skill_review.db"),
                               store_id=SkillStore(str(profile / "skills")).store_id)])))
        prepared_episode(load_roster(roster_path), profile)
        env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", OPENAI_API_KEY="local-fake-test-key")
        output = []
        owner = subprocess.Popen(command("agent.py", "--profile", "alice", "--profiles-dir", tmp_path / "profiles",
            "--workspace", workspace, "--model", "test-model", "--base-url", endpoint, "--auto-skills",
            "--no-memory", "--skill-review-roster", roster_path), cwd=ROOT, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
        reader = threading.Thread(target=lambda: [output.append(line) for line in owner.stdout], daemon=True)
        reader.start()
        service = None
        try:
            def prepared():
                if not (profile / "skill_review.db").exists():
                    return False
                with sqlite3.connect(profile / "skill_review.db") as conn:
                    return conn.execute("SELECT 1 FROM jobs WHERE status='PREPARED'").fetchone()

            wait_for(prepared)
            owner.stdin.write("/ski")
            owner.stdin.flush()
            time.sleep(.15)
            service = subprocess.Popen(command("skill_review.py", "run", "--roster", roster_path, "--timeout", "3"),
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8")
            service_errors = []
            service_error_reader = threading.Thread(
                target=lambda: [service_errors.append(line) for line in service.stderr], daemon=True)
            service_error_reader.start()
            assert entered.wait(5)
            # The service has recorded worker entry, but input() remains an
            # output-free boundary until the user submits the current line.
            time.sleep(.15)
            assert "[Skill Review] Review started." not in "".join(output)
            # A safe drain exposes the recorded start while review HTTP remains blocked.
            owner.stdin.write("lls\n")
            owner.stdin.flush()
            wait_for(lambda: any("No skills found" in line for line in output))
            combined = "".join(output)
            assert combined.index("[Skill Review] Review started.") < combined.index("No skills found")
            assert "[Skill Review] Created skill 'from_service'" not in combined
            wait_for(lambda: "status=STARTED" in "".join(service_errors))
            service_lifecycle = "".join(service_errors)
            assert "status=REQUESTED" in service_lifecycle
            assert "status=APPLIED" not in service_lifecycle
            assert "profile=alice" in service_lifecycle
            assert not (profile / "skills" / "from_service.md").exists()
            duplicate = subprocess.run(command("skill_review.py", "once", "--roster", roster_path), cwd=ROOT,
                env=env, capture_output=True, text=True, timeout=5)
            assert duplicate.returncode == 2 and "busy" in duplicate.stdout
            release.set()
            wait_for(lambda: (profile / "skills" / "from_service.md").exists())
            wait_for(lambda: "status=APPLIED" in "".join(service_errors))
            assert service.poll() is None  # `run` remains alive after owner-final publication.
            service_lifecycle = "".join(service_errors)
            assert service_lifecycle.index("status=REQUESTED") < service_lifecycle.index(
                "status=STARTED") < service_lifecycle.index("status=APPLIED")
            assert "action=CREATE" in service_lifecycle and 'skill="from_service"' in service_lifecycle
            assert "private-value" not in service_lifecycle and "private-row" not in service_lifecycle
            review_request = requests[-1]
            assert review_request["model"] == "test-model" and review_request["max_tokens"] == 4096
            assert "private-value" not in json.dumps(review_request)
            assert "private-row" not in json.dumps(review_request)
            assert "RETAINED_SQL_CONTEXT" in json.dumps(review_request)
            assert "RETAINED_METADATA_CONTEXT" in json.dumps(review_request)
            assert "Global Operator Instructions" not in review_request["messages"][-1]["content"]
            # Background publication never repaints the live input line. A
            # notice that arrives while input() is idle/partially typed waits
            # for submission, and the partial bytes remain the actual task.
            time.sleep(.15)
            assert "[Skill Review] Created skill 'from_service'" not in "".join(output)
            owner.stdin.write("Complete this ordinary")
            owner.stdin.flush()
            time.sleep(.15)
            assert "[Skill Review] Created skill 'from_service'" not in "".join(output)
            assert len(requests) == 1
            owner.stdin.write(" foreground task\n")
            owner.stdin.flush()
            wait_for(lambda: any("The subprocess task is complete." in line for line in output))
            combined = "".join(output)
            assert combined.index("[Skill Review] Created skill 'from_service'") < combined.index("[Task Complete]")
            assert combined.index("[Skill Review] Review started.") < combined.index(
                "[Skill Review] Created skill 'from_service'")
            foreground_request = requests[-1]
            assert foreground_request["messages"][-1]["content"] == "Complete this ordinary foreground task"
            owner.stdin.write("exit\n")
            owner.stdin.flush()
            owner.wait(timeout=5)
            reader.join(2)
            assert owner.returncode == 0
            service.terminate()
            service.wait(timeout=5)
            service_error_reader.join(2)
            service.stdout.read()
            smoke = subprocess.run(command("skill_review.py", "once", "--roster", roster_path), cwd=ROOT,
                env=env, capture_output=True, text=True, timeout=5)
            assert smoke.returncode == 0 and '"processed":0' in smoke.stdout
        finally:
            release.set()
            for process in (service, owner):
                if process and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            for handle in (owner.stdin, owner.stdout, service.stdout if service else None,
                           service.stderr if service else None):
                if handle:
                    handle.close()


def boundary_case(tmp_path, endpoint):
    profile = tmp_path / "profiles" / "alice"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    roster_path = tmp_path / "roster.json"
    roster_path.write_text(json.dumps(dict(
        control_dir=str(tmp_path / "control"), model="test-model", base_url=endpoint,
        profiles=[dict(profile_id="alice", mailbox=str(profile / "skill_review.db"),
                       store_id=SkillStore(str(profile / "skills")).store_id)],
    )))
    prepared_episode(load_roster(roster_path), profile)
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
               OPENAI_API_KEY="local-fake-test-key")
    output = []
    owner = subprocess.Popen(command(
        "agent.py", "--profile", "alice", "--profiles-dir", tmp_path / "profiles",
        "--workspace", workspace, "--model", "test-model", "--base-url", endpoint,
        "--auto-skills", "--no-memory", "--skill-review-roster", roster_path,
    ), cwd=ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    reader = threading.Thread(target=lambda: [output.append(line) for line in owner.stdout], daemon=True)
    reader.start()
    return profile, roster_path, env, output, owner, reader


def pending_notice(profile):
    if not (profile / "skill_review.db").exists():
        return False
    with sqlite3.connect(profile / "skill_review.db") as conn:
        return bool(conn.execute("SELECT 1 FROM notices WHERE delivered IS NULL").fetchone())


def test_actual_cli_notice_arriving_during_foreground_response_waits_until_answer_complete(tmp_path):
    with boundary_openai() as (endpoint, foreground_entered, foreground_release, _):
        profile, roster_path, env, output, owner, reader = boundary_case(tmp_path, endpoint)
        try:
            owner.stdin.write("Start the blocked foreground response\n")
            owner.stdin.flush()
            assert foreground_entered.wait(5)
            service = subprocess.run(command("skill_review.py", "once", "--roster", roster_path, "--timeout", "3"),
                                     cwd=ROOT, env=env, capture_output=True, text=True, timeout=8)
            assert service.returncode == 0 and '"processed":1' in service.stdout
            wait_for(lambda: pending_notice(profile))
            time.sleep(.15)
            assert "[Skill Review] Review started." not in "".join(output)
            assert "boundary_skill" not in "".join(output)
            assert "Boundary foreground answer complete" not in "".join(output)

            foreground_release.set()
            wait_for(lambda: "boundary_skill" in "".join(output))
            combined = "".join(output)
            assert combined.index("[Skill Review] Review started.") < combined.index(
                "[Skill Review] Created skill 'boundary_skill'")
            assert combined.index("Boundary foreground answer complete") < combined.index(
                "[Skill Review] Created skill 'boundary_skill'")
            owner.stdin.write("exit\n")
            owner.stdin.flush()
            owner.wait(timeout=5)
            reader.join(2)
            assert owner.returncode == 0
        finally:
            foreground_release.set()
            if owner.poll() is None:
                owner.terminate()
                owner.wait(timeout=5)
            owner.stdin.close()
            owner.stdout.close()


def test_actual_cli_permission_input_holds_notice_until_decision_then_delivers_before_tool_output(tmp_path):
    with boundary_openai(permission=True) as (endpoint, foreground_entered, _, _):
        profile, roster_path, env, output, owner, reader = boundary_case(tmp_path, endpoint)
        try:
            owner.stdin.write("Request a reviewed terminal command\n")
            owner.stdin.flush()
            assert foreground_entered.wait(5)
            wait_for(lambda: "Options: [Enter/y] Run" in "".join(output))
            service = subprocess.run(command("skill_review.py", "once", "--roster", roster_path, "--timeout", "3"),
                                     cwd=ROOT, env=env, capture_output=True, text=True, timeout=8)
            assert service.returncode == 0 and '"processed":1' in service.stdout
            wait_for(lambda: pending_notice(profile))
            time.sleep(.15)
            assert "[Skill Review] Review started." not in "".join(output)
            assert "boundary_skill" not in "".join(output)

            owner.stdin.write("n\n")
            owner.stdin.flush()
            wait_for(lambda: "Boundary foreground answer complete" in "".join(output))
            combined = "".join(output)
            assert combined.index("[Skill Review] Review started.") < combined.index(
                "[Skill Review] Created skill 'boundary_skill'")
            assert combined.index("[Skill Review] Created skill 'boundary_skill'") < combined.index(
                "Command rejected by user")
            owner.stdin.write("exit\n")
            owner.stdin.flush()
            owner.wait(timeout=5)
            reader.join(2)
            assert owner.returncode == 0
        finally:
            if owner.poll() is None:
                owner.terminate()
                owner.wait(timeout=5)
            owner.stdin.close()
            owner.stdout.close()


def test_actual_agent_large_multiround_capture_to_separate_service_real_sdk_http_and_ack(tmp_path):
    with fake_openai() as (endpoint, _, _, requests):
        instance, _, roster_path = http_agent(tmp_path, endpoint)
        try:
            capture_large_multiround_agent_episode(instance)
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                       OPENAI_API_KEY="local-fake-test-key")
            result = subprocess.run(
                command("skill_review.py", "once", "--roster", roster_path, "--timeout", "3"),
                cwd=ROOT, env=env, capture_output=True, text=True, timeout=8,
            )
            assert result.returncode == 0 and '"processed":1' in result.stdout
            instance.skill_review_owner.pump()
            assert "from_service" in instance.skill_store.list_skills()
            assert rows(instance.skill_review_owner, "jobs")[-1]["status"] == "APPLIED"
            rendered = json.dumps(requests[-1])
            assert "RETAINED_AGENT_METADATA_" in rendered
            assert "RETAINED_AGENT_SQL_" in rendered
            assert "PRIVATE-AGENT-BUSINESS-ROW" not in rendered
            assert "synthetic-private-value" not in rendered
            assert requests[-1]["model"] == "test-model"
        finally:
            instance.shutdown_skill_reviews()


def test_foreground_question_progresses_while_separate_process_prepares_episode(tmp_path):
    with fake_openai() as (endpoint, _, _, requests):
        instance, _, roster_path = http_agent(tmp_path, endpoint)
        marker = tmp_path / "preparation-entered"
        release = tmp_path / "preparation-release"
        child = None
        try:
            capture_large_multiround_agent_episode(instance)
            child_code = "\n".join([
                "from pathlib import Path",
                "import sys,time",
                "from skill_review import ReviewService,load_roster",
                "marker,release,roster = Path(sys.argv[1]),Path(sys.argv[2]),sys.argv[3]",
                "original = ReviewService._episode_request",
                "def blocked(self, entry, job):",
                "    marker.write_text('entered', encoding='utf-8')",
                "    deadline = time.monotonic() + 8",
                "    while not release.exists() and time.monotonic() < deadline: time.sleep(.01)",
                "    return original(self, entry, job)",
                "ReviewService._episode_request = blocked",
                "raise SystemExit(0 if ReviewService(load_roster(roster), timeout=3).once() == 1 else 2)",
            ])
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8",
                       OPENAI_API_KEY="local-fake-test-key")
            child = subprocess.Popen(
                command("-B", "-c", child_code, marker, release, roster_path),
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8",
            )
            wait_for(marker.exists)
            started = time.monotonic()
            with patch.object(instance, "step", return_value=answer_message("Synthetic foreground answer.")), \
                 patch.object(instance, "manage_context"):
                assert instance.run("What is the next foreground answer?") == "Synthetic foreground answer."
            synthetic_elapsed = time.monotonic() - started
            assert synthetic_elapsed < .75
            assert child.poll() is None
            release.write_text("release", encoding="utf-8")
            output, _ = child.communicate(timeout=8)
            assert child.returncode == 0, output
            instance.skill_review_owner.pump()
            assert "from_service" in instance.skill_store.list_skills()
            assert len(requests) == 1
        finally:
            release.write_text("release", encoding="utf-8")
            if child and child.poll() is None:
                child.terminate()
                child.wait(timeout=5)
            if child and child.stdout:
                child.stdout.close()
            instance.shutdown_skill_reviews()


def test_real_sdk_timeout_is_finite_and_provider_not_retried(tmp_path):
    with fake_openai(delay=True) as (endpoint, entered, release, requests):
        setup_roster(tmp_path)
        roster_path = tmp_path / "roster.json"
        value = json.loads(roster_path.read_text())
        value["base_url"] = endpoint
        roster_path.write_text(json.dumps(value))
        roster = load_roster(roster_path)
        owner = owner_for(roster, background=False)
        try:
            owner.enqueue(evidence())
            owner.pump()
            before = time.monotonic()
            assert ReviewService(roster, timeout=.15).once() == 1
            assert time.monotonic() - before < 2
            assert entered.is_set() and len(requests) == 1
            owner.pump()
            assert rows(owner, "jobs")[0]["status"] == "FAILED"
            release.set()
            assert not owner.store.get_all_skills()
        finally:
            release.set()
            owner.close()


def test_unauthorized_cli_profile_refuses_before_creating_profile_state(tmp_path):
    setup_roster(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    profiles = tmp_path / "profiles"
    result = subprocess.run(command("agent.py", "--profile", "forged", "--profiles-dir", profiles,
        "--workspace", workspace, "--auto-skills", "--skill-review-roster", tmp_path / "roster.json"),
        cwd=ROOT, capture_output=True, text=True, timeout=5)
    assert result.returncode == 2
    assert not profiles.exists()
