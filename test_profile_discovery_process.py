"""Sequential real owner/service processes against a local fake OpenAI server."""
from contextlib import contextmanager, ExitStack
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import threading

import pytest

from profile_paths import control_directory
from test_async_skill_review import create_proposal, wait_for


REPO = Path(__file__).resolve().parent
ENV = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8", OPENAI_API_KEY="local-fake-test-key")


@contextmanager
def private_provider():
    entered, release = threading.Event(), threading.Event()
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(body)
            review = "private skill snapshot" in body["messages"][0]["content"]
            prompt = json.dumps(body["messages"])
            who = "alice" if "TASK_ALICE" in prompt else "bob"
            if review:
                entered.set()
                release.wait(10)
            proposal = create_proposal(who + "_skill")
            proposal["description"] = who + " private workflow"
            content = json.dumps(proposal) if review else who + " task complete"
            raw = json.dumps(dict(id="fake", object="chat.completion", created=1, model=body["model"],
                choices=[dict(index=0, finish_reason="stop", message=dict(role="assistant", content=content))],
                usage=dict(prompt_tokens=10, completion_tokens=10, total_tokens=20))).encode()
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
def process(*args):
    child = subprocess.Popen([sys.executable, *map(str, args)], cwd=REPO, env=ENV,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8")
    output = []
    reader = threading.Thread(target=lambda: [output.append(line) for line in child.stdout], daemon=True)
    reader.start()
    try:
        yield child, output
    finally:
        if child.poll() is None:
            child.terminate()
            child.wait(timeout=5)
        reader.join(2)
        child.stdin.close()
        child.stdout.close()


def send(child, message):
    child.stdin.write(message + "\n")
    child.stdin.flush()


def await_output(child, output, needle):
    def ready():
        assert child.poll() is None, "".join(output)
        return any(needle in line for line in output)
    wait_for(ready, 8)


def job_status(root, who, status):
    mailbox = root / who / "skill_review.db"
    if not mailbox.exists():
        return False
    with sqlite3.connect(mailbox.as_uri() + "?mode=ro", uri=True, timeout=.05) as conn:
        return conn.execute("SELECT 1 FROM jobs WHERE status=?", (status,)).fetchone()


@pytest.mark.parametrize("existing", [True, False, True], ids=["existing-live-add", "service-first", "live-add-repeat"])
def test_same_service_pid_discovers_new_real_owners_during_inflight_review(tmp_path, existing):
    root = tmp_path / "profiles"
    with private_provider() as (endpoint, entered, release, requests), ExitStack() as stack:
        owners = {}

        def start_owner(who):
            workspace = tmp_path / (who + "-workspace")
            workspace.mkdir()
            (workspace / "private.txt").write_text(who + " workspace")
            schema = tmp_path / (who + "-schema")
            schema.mkdir()
            owner, output = stack.enter_context(process("agent.py", "--profile", who, "--profiles-dir", root,
                "--workspace", workspace, "--read-only-dir", schema, "--max-tokens", "8192",
                "--auto-skills", "--no-memory", "--model", "test-model", "--base-url", endpoint))
            owners[who] = owner, output
            send(owner, f"Complete TASK_{who.upper()} password={who.upper()}_SECRET")
            await_output(owner, output, "[Skill Review] ACCEPTED")
            wait_for(lambda: job_status(root, who, "PREPARED"))

        if existing:
            start_owner("alice")
        service, service_output = stack.enter_context(process("skill_review.py", "run", "--profiles-dir", root,
            "--model", "test-model", "--base-url", endpoint, "--discovery-interval", ".05", "--timeout", "8"))
        service_pid = service.pid
        wait_for(lambda: (control_directory(root) / "policy.json").is_file())
        if not existing:
            assert not root.exists()  # The real service never provisions its root.
            start_owner("alice")
        assert entered.wait(5), "".join(service_output)
        assert job_status(root, "alice", "RUNNING")
        start_owner("bob")  # A new directory while the same singleton holds a blocked request.
        alice, output = owners["alice"]
        send(alice, "/skills")
        await_output(alice, output, "No skills found")
        assert service.pid == service_pid and service.poll() is None
        reviews = lambda: [r for r in requests if "private skill snapshot" in r["messages"][0]["content"]]
        assert len(reviews()) == 1
        assert job_status(root, "alice", "RUNNING")
        release.set()
        for who in owners:
            wait_for(lambda: job_status(root, who, "APPLIED"), 8)
            assert (root / who / "skills" / (who + "_skill.md")).is_file()
            other = "bob" if who == "alice" else "alice"
            assert not (root / who / "skills" / (other + "_skill.md")).exists()
            assert (tmp_path / (who + "-workspace") / "private.txt").read_text() == who + " workspace"
        assert len(reviews()) == 2
        for request in reviews():
            text = json.dumps(request)
            assert ("TASK_ALICE" in text) != ("TASK_BOB" in text)
            assert "ALICE_SECRET" not in text and "BOB_SECRET" not in text
            assert request["model"] == "test-model" and request["max_tokens"] == 4096
        assert service.pid == service_pid and service.poll() is None
        for owner, output in owners.values():
            send(owner, "exit")
            owner.wait(timeout=5)
            assert owner.returncode == 0, "".join(output)


@pytest.mark.parametrize("mode", ["read-only", "stateless", "no-skills"])
def test_real_automatic_mode_revocation_and_reenable_fence_late_writes(tmp_path, mode):
    from test_profile_discovery import run_agent, tree_bytes
    root, workspace = tmp_path / "profiles", tmp_path / "workspace"
    workspace.mkdir()
    provision = run_agent(root, workspace, "--no-memory")
    assert provision.returncode == 0, provision.stdout + provision.stderr
    assert not (root / "alice" / "skill_review.db").exists()
    with private_provider() as (endpoint, entered, release, requests), ExitStack() as stack:
        owner, output = stack.enter_context(process("agent.py", "--profile", "alice", "--profiles-dir", root,
            "--workspace", workspace, "--auto-skills", "--read-only", "--no-memory", "--model", "test-model", "--base-url", endpoint))
        await_output(owner, output, "Commands:")
        assert not (root / "alice" / "skill_review.db").exists()
        send(owner, "/mode normal")
        send(owner, "Complete TASK_ALICE password=ALICE_SECRET")
        await_output(owner, output, "[Skill Review] ACCEPTED")
        service, service_output = stack.enter_context(process("skill_review.py", "run", "--profiles-dir", root,
            "--model", "test-model", "--base-url", endpoint, "--discovery-interval", ".05", "--timeout", "8"))
        assert entered.wait(5), "".join(service_output)
        send(owner, "/mode " + mode)
        acknowledgement = {"read-only": "Read-Only mode:", "stateless": "Stateless Benchmark mode:",
                           "no-skills": "Skills disabled;"}[mode]
        await_output(owner, output, acknowledgement)
        before = tree_bytes(root / "alice")
        release.set()
        bob_workspace = tmp_path / "bob-workspace"
        bob_workspace.mkdir()
        bob, bob_output = stack.enter_context(process("agent.py", "--profile", "bob", "--profiles-dir", root,
            "--workspace", bob_workspace, "--auto-skills", "--no-memory", "--model", "test-model", "--base-url", endpoint))
        send(bob, "Complete TASK_BOB")
        await_output(bob, bob_output, "[Skill Review] ACCEPTED")
        wait_for(lambda: job_status(root, "bob", "APPLIED"), 8)
        # Bob's result proves the one service slot finished Alice's revoked call.
        assert tree_bytes(root / "alice") == before
        assert not (root / "alice" / "skills" / "alice_skill.md").exists()
        send(owner, "/mode normal")
        send(owner, "Complete new TASK_ALICE")
        wait_for(lambda: job_status(root, "alice", "APPLIED"), 8)
        with sqlite3.connect((root / "alice" / "skill_review.db").as_uri() + "?mode=ro", uri=True) as conn:
            assert conn.execute("SELECT count(*) FROM jobs").fetchone()[0] == 1
        assert len([r for r in requests if "private skill snapshot" in r["messages"][0]["content"]]) == 3
