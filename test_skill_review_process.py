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

from skills import SkillStore
from test_async_skill_review import wait_for, create_proposal, setup_roster, owner_for, evidence, rows
from skill_review import ReviewService, load_roster


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


def command(*args):
    return [sys.executable, *map(str, args)]


def test_real_owner_and_service_process_with_blocked_http_idle_apply_and_singleton(tmp_path):
    with fake_openai(delay=True) as (endpoint, entered, release, requests):
        profile = tmp_path / "profiles" / "alice"
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        roster_path = tmp_path / "roster.json"
        roster_path.write_text(json.dumps(dict(control_dir=str(tmp_path / "control"), model="test-model", base_url=endpoint,
            profiles=[dict(profile_id="alice", mailbox=str(profile / "skill_review.db"),
                           store_id=SkillStore(str(profile / "skills")).store_id)])))
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
            owner.stdin.write("Complete this reusable work password=private-value\n")
            owner.stdin.flush()
            wait_for(lambda: any("[Skill Review] ACCEPTED" in line for line in output), 8)

            def prepared():
                if not (profile / "skill_review.db").exists():
                    return False
                with sqlite3.connect(profile / "skill_review.db") as conn:
                    return conn.execute("SELECT 1 FROM jobs WHERE status='PREPARED'").fetchone()

            wait_for(prepared)
            service = subprocess.Popen(command("skill_review.py", "once", "--roster", roster_path, "--timeout", "3"),
                cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
            assert entered.wait(5)
            # The owner handles a new interactive command while review HTTP is blocked.
            owner.stdin.write("/skills\n")
            owner.stdin.flush()
            wait_for(lambda: any("No skills found" in line for line in output))
            assert not (profile / "skills" / "from_service.md").exists()
            duplicate = subprocess.run(command("skill_review.py", "once", "--roster", roster_path), cwd=ROOT,
                env=env, capture_output=True, text=True, timeout=5)
            assert duplicate.returncode == 2 and "busy" in duplicate.stdout
            release.set()
            service_out, _ = service.communicate(timeout=6)
            assert service.returncode == 0 and '"processed":1' in service_out
            wait_for(lambda: (profile / "skills" / "from_service.md").exists())
            review_request = requests[-1]
            assert review_request["model"] == "test-model" and review_request["max_tokens"] == 4096
            assert "private-value" not in json.dumps(review_request)
            assert "Global Operator Instructions" not in review_request["messages"][-1]["content"]
            owner.stdin.write("exit\n")
            owner.stdin.flush()
            owner.wait(timeout=5)
            reader.join(2)
            assert owner.returncode == 0
            assert any("The subprocess task is complete." in line for line in output)
            smoke = subprocess.run(command("skill_review.py", "once", "--roster", roster_path), cwd=ROOT,
                env=env, capture_output=True, text=True, timeout=5)
            assert smoke.returncode == 0 and '"processed":0' in smoke.stdout
        finally:
            release.set()
            for process in (service, owner):
                if process and process.poll() is None:
                    process.terminate()
                    process.wait(timeout=5)
            for handle in (owner.stdin, owner.stdout, service.stdout if service else None):
                if handle:
                    handle.close()


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
