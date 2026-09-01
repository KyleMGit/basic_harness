"""
Hermes-Refined Coding Agent Harness.
An open, persistent, multi-protocol coding agent harness in Python.

Key Features:
1. Codebase Navigation & Search (grep_search, find_files_by_pattern).
2. Dual Persistent Memory System: USER.md (operator profile) & MEMORY.md (project architecture facts).
3. Configurable Testing & Benchmark Modes (--read-only, --stateless, --no-skills, --no-memory).
4. Autonomous Reflection Engines: AutoSkillExtractor & AutoMemoryExtractor.
5. Session Continuity & Resumption (--resume <session_id> or /resume command).
6. Configured for Qwen-32B with 40K (40,960) Token Context Budget.
7. Local Model Support without API Keys (Qwen-32b, Ollama, vLLM, LM Studio, llama.cpp).
8. Production-Grade Context Checkpoint Compaction & Summarization.
9. Interactive User Prompt for Every Terminal Command (approve, edit, reject, feedback).
10. Stateful Terminal Execution (persistent cwd, cd management, output clipping).
11. Dual Protocol Tool Calling (OpenAI JSON API & Hermes XML <tool_call> syntax).
12. SQLite Trajectory Logging & Export.
"""

import argparse
import json
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from openai import OpenAI
except ImportError:
    print("Error: The 'openai' package is required. Install it using: pip install openai")
    sys.exit(1)

from tools import registry, terminal_session, skill_store
from skills import AutoSkillExtractor
from protocol import ToolProtocol
from storage import TrajectoryLogger
from compaction import ContextManager
from memory import user_profile_manager, project_memory_manager, auto_memory_extractor
from safety import screen_prompt_content


READ_THIS_PATH = Path(__file__).resolve().parent / "READ_THIS.md"
READ_THIS_MAX_CHARS = 20_000
READ_THIS_START_MARKER = "<!-- READ_THIS.md:START -->"
READ_THIS_END_MARKER = "<!-- READ_THIS.md:END -->"


def load_read_this_block() -> str:
    """Load the required, source-relative operator prompt layer."""
    try:
        content = READ_THIS_PATH.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise RuntimeError(f"Required READ_THIS.md is missing: {READ_THIS_PATH}") from exc
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"Required READ_THIS.md is not valid UTF-8: {READ_THIS_PATH}") from exc
    except (PermissionError, OSError) as exc:
        raise RuntimeError(f"Required READ_THIS.md is unreadable: {READ_THIS_PATH}: {exc}") from exc
    content = content.strip()
    if not content:
        raise RuntimeError(f"Required READ_THIS.md is empty: {READ_THIS_PATH}")
    if READ_THIS_START_MARKER in content or READ_THIS_END_MARKER in content:
        raise RuntimeError("Required READ_THIS.md contains a reserved READ_THIS.md marker.")
    if len(content) > READ_THIS_MAX_CHARS:
        raise RuntimeError(
            f"Required READ_THIS.md exceeds the maximum of {READ_THIS_MAX_CHARS} characters: {READ_THIS_PATH}"
        )
    accepted, reason = screen_prompt_content(content)
    if not accepted:
        raise RuntimeError(f"Required READ_THIS.md failed prompt screening: {reason}")
    return f"{READ_THIS_START_MARKER}\n{content}\n{READ_THIS_END_MARKER}"


def _read_this_snapshot(prompt: str) -> Optional[str]:
    if prompt.count(READ_THIS_START_MARKER) != prompt.count(READ_THIS_END_MARKER):
        raise RuntimeError("Saved system prompt contains an incomplete READ_THIS.md marker block.")
    if prompt.count(READ_THIS_START_MARKER) > 1:
        raise RuntimeError("Saved system prompt contains more than one READ_THIS.md marker block.")
    start = prompt.find(READ_THIS_START_MARKER)
    if start < 0:
        return None
    end = prompt.find(READ_THIS_END_MARKER)
    if end < start:
        raise RuntimeError("Saved system prompt contains READ_THIS.md markers out of order.")
    return prompt[start:end + len(READ_THIS_END_MARKER)]


def _append_read_this(prompt: str, block: Optional[str] = None) -> str:
    if READ_THIS_START_MARKER in prompt or READ_THIS_END_MARKER in prompt:
        raise RuntimeError("Fresh system prompt contains a reserved READ_THIS.md marker.")
    return f"{prompt.rstrip()}\n\n### Global Operator Instructions (READ_THIS.md)\n{block or load_read_this_block()}\n"


def _inject_read_this(prompt: str, block: Optional[str] = None) -> str:
    if _read_this_snapshot(prompt) is not None:
        return prompt
    return _append_read_this(prompt, block)


class HermesCodingAgent:
    """
    Stateful, autonomous coding agent harness supporting both
    OpenAI structured tool calling and Hermes XML function calling protocols,
    with interactive human-in-the-loop confirmation for every system command,
    automated context compaction/summarization, local model support (Qwen-32b, etc.),
    dual memory snapshots (USER.md / MEMORY.md), session resumption,
    and flexible testing modes (read-only, stateless, no-skills, no-memory).
    """

    SYSTEM_PROMPT_TEMPLATE = """You are an autonomous AI software engineer and terminal agent.
You have access to tools that allow you to inspect the system, manage files, search codebases, and execute terminal commands.

### Important Protocol:
- **Interactive Terminal Commands**: Every system/terminal command you propose will be reviewed interactively by the user before execution. The user may approve it, modify it, or provide feedback.
- **Context Compaction**: In long-running tasks, earlier conversation segments may be compressed into structured summary blocks. Use the summary to maintain continuity.
- **Persistent Workspace**: The terminal environment maintains your working directory (`cwd`) across tool calls. Use `cd <dir>` to navigate projects.
- **Codebase Exploration**: Use `grep_search` and `find_files_by_pattern` for fast multi-file navigation instead of reading entire files repeatedly.
- **Verification & Testing**: Always verify changes by running tests, linters, or checking file content.
- **Analyze Output**: Inspect command outputs (stdout, stderr, exit codes). If errors occur, diagnose and repair them iteratively.
- **Database Results**: Database query tools return bounded previews. Use only the requested columns and deterministic ordering. A truncated preview is not the complete dataset, so use aggregate or otherwise answer-shaped SQL when the database can compute the requested answer. For a complete row-set request (for example, “all products”), answer inline only when the preview is complete; if it is truncated, run the matching database export tool and provide the complete CSV manifest instead of presenting the preview as complete.
- **Complete CSV Exports**: Never reconstruct a complete CSV from query-preview rows or pass those rows to `write_file`. For complete database CSV requests, call `export_teradata_csv` or `export_impala_csv` with the validated SQL and report the returned manifest.

### Memory Evolution & Learning Protocol:
- **Operator Preferences (USER.md)**: When the user expresses a personal preference, workflow habit, formatting requirement, or gives you a correction (e.g. "I prefer pytest", "don't use pip", "keep answers under 3 bullets"), immediately call `update_user_profile(category, preference)`.
- **Project Architecture & Facts (MEMORY.md)**: When you discover new architectural facts, environment variables, server ports, or resolve recurring bug patterns, immediately call `update_project_memory(category, fact)`.

### Operator Profile & Preferences (USER.md):
{user_profile}

### Project Architecture & Environment Facts (MEMORY.md):
{project_memory}

### Learned Skills & Best Practices:
Below is the catalog of learned project skills. When a task relates to any available skill, use `load_skill(name="<skill_name>")` or invoke the skill directly by name:
{skills_catalog}

- **Conciseness**: Summarize your work clearly when the task is achieved.
"""

    def __init__(
        self,
        model: str = "Qwen-32b",
        compaction_model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_iterations: int = 30,
        max_context_tokens: int = 40960,     # Default to 40K (40,960 tokens) for Qwen-32B
        compaction_threshold: float = 0.70, # Triggers compaction at ~28,672 tokens
        compaction_max_context_tokens: Optional[int] = None,
        compaction_output_tokens: Optional[int] = None,
        confirm_all_terminal_commands: bool = True,
        enable_skills: bool = True,
        enable_memory: bool = True,
        auto_learn_skills: bool = False,
        auto_learn_memory: bool = False,
        read_only: bool = False,
        stateless: bool = False,
        use_hermes_xml_protocol: bool = False,
    ):
        self.model = model or "Qwen-32b"
        self.compaction_model = compaction_model or self.model
        self.max_iterations = max_iterations
        self.confirm_all_terminal_commands = confirm_all_terminal_commands
        self.enable_skills = enable_skills
        self.enable_memory = enable_memory
        self.read_only = read_only
        self.stateless = stateless
        configured_auto_skills = auto_learn_skills and enable_skills
        configured_auto_memory = auto_learn_memory and enable_memory
        self.auto_learn_skills = configured_auto_skills and not read_only
        self.auto_learn_memory = configured_auto_memory and not read_only
        self.use_hermes_xml_protocol = use_hermes_xml_protocol
        self._startup_config = (enable_skills, enable_memory, configured_auto_skills, configured_auto_memory)
        
        # Local models running on Ollama/vLLM/LMStudio do not require a real API key,
        # but the OpenAI SDK requires a non-empty string.
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or "local-no-key-required"
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:11434/v1"

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url
        )
        
        self.logger = TrajectoryLogger(write_enabled=not self.read_only)
        self.context_manager = ContextManager(
            max_context_tokens=max_context_tokens,
            trigger_threshold=compaction_threshold,
            compaction_max_context_tokens=compaction_max_context_tokens,
            compaction_output_tokens=compaction_output_tokens,
        )
        self.skill_extractor = AutoSkillExtractor(skill_store=skill_store)
        self.memory_extractor = auto_memory_extractor
        
        self.session_id = str(uuid.uuid4())[:8]
        self.step_counter = 0

        # Construct system prompt with live skill catalog & memory snapshots
        system_content = self._build_system_prompt()
        if self.use_hermes_xml_protocol:
            system_content = ToolProtocol.format_hermes_system_prompt(system_content, registry.schemas_for(self.enable_memory, self.enable_skills))

        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]

    def _build_system_prompt(self) -> str:
        """Build system prompt embedding live skills, USER.md, and MEMORY.md."""
        catalog_xml = skill_store.format_catalog_prompt() if self.enable_skills else "<available_skills>\nNone (Skills disabled for testing).\n</available_skills>"
        user_profile_xml = user_profile_manager.format_system_prompt_block() if self.enable_memory else "<user_profile>\nDefault testing profile.\n</user_profile>"
        project_mem_xml = project_memory_manager.format_system_prompt_block() if self.enable_memory else "<project_memory>\nDefault testing memory.\n</project_memory>"
        base_prompt = self.SYSTEM_PROMPT_TEMPLATE.format(
            skills_catalog=catalog_xml,
            user_profile=user_profile_xml,
            project_memory=project_mem_xml
        )
        return _append_read_this(base_prompt)

    def refresh_system_prompt(self):
        """System prompts are frozen for a session; updates apply next session."""
        return

    def _persist_session_state(self):
        """Persist the active transcript without the rebuildable system prompt."""
        active_messages = self.messages[1:] if self.messages else []
        self.logger.save_session_state(
            self.session_id,
            active_messages,
            self.step_counter,
            self.context_manager.snapshot_state(),
        )

    def set_testing_mode(self, mode: str):
        """Configure agent testing/learning modes dynamically."""
        mode_clean = mode.lower().strip()
        if mode_clean == "normal":
            self.enable_skills, self.enable_memory, configured_skills, configured_memory = self._startup_config
            self.read_only = False
            self.stateless = False
            self.auto_learn_skills = configured_skills
            self.auto_learn_memory = configured_memory
            self.logger.set_write_enabled(True)
            print(f"[Mode Updated] Normal mode restored (skills={self.enable_skills}, memory={self.enable_memory}, auto-skills={self.auto_learn_skills}, auto-memory={self.auto_learn_memory}).")
        elif mode_clean in ("read-only", "readonly", "freeze"):
            self.enable_skills, self.enable_memory = self._startup_config[:2]
            self.read_only = True
            self.stateless = False
            self.auto_learn_skills = False
            self.auto_learn_memory = False
            self.logger.set_write_enabled(False)
            print("[Mode Updated] Read-Only mode: Existing skills & memories readable, but ZERO saving/writing to disk.")
        elif mode_clean in ("stateless", "benchmark", "isolated"):
            self.enable_skills = False
            self.enable_memory = False
            self.read_only = True
            self.stateless = True
            self.auto_learn_skills = False
            self.auto_learn_memory = False
            self.logger.set_write_enabled(False)
            print("[Mode Updated] Stateless Benchmark mode: No skills, No memories, Zero writes to disk.")
        else:
            print(f"Unknown mode '{mode}'. Options: normal | read-only | stateless")
            return
        
        # Privacy boundary: no transcript from the prior mode crosses modes.
        self.session_id = str(uuid.uuid4())[:8]
        self.step_counter = 0
        self.context_manager.restore_state()
        system_content = self._build_system_prompt()
        schemas = registry.schemas_for(self.enable_memory, self.enable_skills)
        if self.use_hermes_xml_protocol:
            system_content = ToolProtocol.format_hermes_system_prompt(system_content, schemas)
        self.messages = [{"role": "system", "content": system_content}]

    def resume_session(self, target_session_id: str) -> bool:
        """Resume a past conversation session from .agent_history.db."""
        if self.stateless:
            print("[Testing Mode] Stateless mode: persisted sessions are unavailable.")
            return False
        task, legacy_messages = self.logger.load_session_messages(target_session_id)
        saved_state = self.logger.load_session_state(target_session_id)
        if saved_state:
            restored_msgs = saved_state.get("messages") or []
            restored_step = int(saved_state.get("step_counter", len(restored_msgs)))
            self.context_manager.restore_state(saved_state.get("context_state"))
        else:
            restored_msgs = legacy_messages
            restored_step = len(restored_msgs)
            self.context_manager.restore_state()
        if not restored_msgs:
            return False

        repaired = []
        index = 0
        while index < len(restored_msgs):
            message = restored_msgs[index]
            repaired.append(message)
            index += 1
            calls = (message.get("tool_calls") or []) if message.get("role") == "assistant" else []
            if not calls:
                continue
            answered = set()
            while index < len(restored_msgs) and restored_msgs[index].get("role") == "tool":
                result = restored_msgs[index]
                repaired.append(result)
                answered.add(result.get("tool_call_id"))
                index += 1
            for call in calls:
                call_id = call.get("id")
                if call_id and call_id not in answered:
                    repaired.append({
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": "Interrupted tool call: no result was persisted before session recovery.",
                    })
        restored_msgs = repaired

        self.session_id = target_session_id
        self.step_counter = restored_step

        saved_prompt = self.logger.load_session_system_prompt(target_session_id)
        if self.enable_memory and self.enable_skills:
            system_content = _inject_read_this(saved_prompt) if saved_prompt else self._build_system_prompt()
        else:
            # A frozen prompt may contain capabilities or persisted material that
            # the current process explicitly disabled at startup.
            system_content = self._build_system_prompt()
            if saved_prompt:
                saved_read_this = _read_this_snapshot(saved_prompt)
                if saved_read_this:
                    current_read_this = _read_this_snapshot(system_content)
                    system_content = system_content.replace(current_read_this, saved_read_this, 1)
        if self.use_hermes_xml_protocol and "<tools>" not in system_content:
            system_content = ToolProtocol.format_hermes_system_prompt(system_content, registry.schemas_for(self.enable_memory, self.enable_skills))

        self.messages = [{"role": "system", "content": system_content}] + restored_msgs
        print(f"\n[Session Resumed] Successfully reloaded session '{target_session_id}' ({len(restored_msgs)} turns restored).")
        if task:
            print(f"  Original Task: {task}")
        return True

    def prompt_user_for_command(self, command: str, args: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        """Interactive prompt for every system command."""
        print("\n" + "-" * 60)
        print("[SYSTEM COMMAND REVIEW]")
        print(f"   Command: {command}")
        print(f"   In CWD:  {terminal_session.cwd}")
        if terminal_session.is_destructive(command):
            print("   [!] WARNING: Command matches potentially destructive pattern!")
        print("-" * 60)
        print("Options: [Enter/y] Run | [n] Deny | [e] Edit command | [text] Send feedback")

        try:
            user_input = input(">> Action: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[!] Command aborted.")
            return False, command, "User interrupted command execution."

        if user_input.lower() in ("", "y", "yes"):
            return True, command, None
        elif user_input.lower() in ("n", "no", "cancel", "skip"):
            print("[-] Command rejected by user.")
            return False, command, "Execution denied by user."
        elif user_input.lower() in ("e", "edit"):
            print(f"Current command: {command}")
            new_cmd = input(">> Enter new command: ").strip()
            if not new_cmd:
                print("[-] Empty command. Cancelled.")
                return False, command, "Execution cancelled (empty edit)."
            return True, new_cmd, None
        else:
            print(f"[+] Sending user feedback to agent: '{user_input}'")
            return False, command, f"User declined to run this command and provided feedback: '{user_input}'. Adjust your plan accordingly."

    def manage_context(self, force: bool = False):
        """Perform context health check, pruning, or summarization compaction."""
        if self.enable_memory and not self.stateless:
            try:
                memory_context = (
                    f"Project Memory (MEMORY.md):\n{project_memory_manager.load_memory()}\n\n"
                    f"User Profile (USER.md):\n{user_profile_manager.load_profile()}"
                )
            except Exception:
                memory_context = "None."
        else:
            memory_context = "None."
        compacted_msgs, was_compacted, msg = self.context_manager.compact(
            client=self.client,
            model=self.compaction_model,
            messages=self.messages,
            current_step=self.step_counter,
            force=force,
            tool_schemas=registry.schemas_for(self.enable_memory, self.enable_skills),
            memory_context=memory_context,
        )
        if was_compacted:
            print(f"\n[Context] {msg}")
            self.messages = compacted_msgs
            self.logger.log_step(
                self.session_id,
                self.step_counter,
                "system_compaction",
                content=msg
            )
            self._persist_session_state()
        elif force:
            print(f"\n[Info] {msg}")

    @staticmethod
    def _is_context_limit_error(exc: Exception) -> bool:
        status = getattr(exc, "status_code", None)
        code = str(getattr(exc, "code", "") or "").lower()
        message = str(exc).lower()
        markers = ("context length", "context_length", "maximum context", "too many tokens", "token limit", "context window")
        return any(marker in message or marker in code for marker in markers) or (status == 400 and "token" in message)

    def run_auto_memory_reflection(self, task_summary: str):
        """Analyze trajectory to automatically extract and record user preferences and project facts."""
        if not self.auto_learn_memory or self.read_only:
            return

        print("\n[Memory Reflection] Reviewing this task (one opt-in provider call)...")

        result = self.memory_extractor.extract_and_update(
            client=self.client,
            model=self.model,
            messages=self.messages,
            task_summary=task_summary
        )
        refreshed = False
        if result.get("user_updated"):
            print(f"\n[Memory Evolution] User preference recorded in USER.md: {result['user_updated']}")
            refreshed = True
        if result.get("project_updated"):
            print(f"\n[Memory Evolution] Project fact recorded in MEMORY.md: {result['project_updated']}")
            refreshed = True

        errors = [result.get("user_error"), result.get("project_error")]
        errors = [error for error in errors if error]
        for error in errors:
            print(f"\n[Memory Reflection Error] {error}")

        if refreshed:
            self.refresh_system_prompt()
        elif not errors:
            print("[Memory Reflection] No safe durable updates applied.")

    def run_auto_skill_synthesis(self, task_summary: str):
        """Analyze trajectory against catalog to automatically extract or refine skills."""
        if not self.auto_learn_skills or self.read_only:
            return

        print("\n[Skill Reflection] Reviewing this task (one opt-in provider call)...")

        result = self.skill_extractor.extract_and_save(
            client=self.client,
            model=self.model,
            messages=self.messages,
            task_summary=task_summary
        )
        if result:
            action = result.get("action", "SAVED")
            name = result["name"]
            desc = result.get("description", "")
            if action == "ERROR":
                print(f"\n[Skill Reflection] Skill '{name}' was not saved: {desc}")
                return
            if action in ("SKIP", "PROPOSE"):
                print(f"\n[Skill Reflection] Existing skill '{name}' retained; review required for replacement.")
                return
            action_label = "Refined existing skill" if action == "UPDATE" else "Synthesized new skill"
            
            print(f"\n[Self-Improvement] {action_label}: '{name}'")
            print(f"  Description: {desc}")
            
            self.refresh_system_prompt()
            
            self.logger.log_step(
                self.session_id,
                self.step_counter,
                "skill_synthesis",
                content=f"{action_label}: {name}"
            )
        else:
            print("[Skill Reflection] No safe skill proposal applied.")

    def step(self) -> Any:
        """Invoke the LLM for one turn."""
        request_messages = self.messages
        if getattr(self, "_active_skill_injection", None):
            request_messages = [dict(message) for message in self.messages]
            for message in reversed(request_messages):
                if message.get("role") == "user":
                    message["content"] = f"{self._active_skill_injection}\n\n[ACTIVE USER TASK]:\n{message.get('content', '')}"
                    break
        if self.use_hermes_xml_protocol:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=request_messages,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=request_messages,
                tools=registry.schemas_for(self.enable_memory, self.enable_skills) or None,
                tool_choice="auto" if registry.schemas_for(self.enable_memory, self.enable_skills) else None,
            )
        self._last_finish_reason = getattr(response.choices[0], "finish_reason", None)
        return response.choices[0].message

    def run(self, user_task: str) -> str:
        """Execute the autonomous agent loop for a given task."""
        user_message_content = user_task
        self._active_skill_injection = None
        if not self.messages or len(self.messages) <= 1:
            self.session_id = str(uuid.uuid4())[:8]
            self.step_counter = 0
            self.context_manager.restore_state()
            self.logger.start_session(self.session_id, user_task, self.messages[0].get("content", ""))

        # 1. Pre-Turn Skill Matching (if enabled)
        if self.enable_skills:
            relevant_skills = skill_store.find_relevant_skills(user_task)
            if relevant_skills:
                skill_blocks = []
                for sk in relevant_skills:
                    skill_blocks.append(
                        f"=== RELEVANT SKILL: {sk['name']} ===\n"
                        f"Description: {sk.get('description', '')}\n"
                        f"Instructions to Follow:\n{sk.get('instructions', '')}\n"
                        f"======================================"
                    )
                
                skill_injection = (
                    "[RELEVANT LEARNED SKILLS AUTO-INJECTED]:\n"
                    "The following established skills directly match this task. Apply their procedures:\n\n"
                    + "\n\n".join(skill_blocks)
                )
                print(f"\n[Skill Retrieval] Found {len(relevant_skills)} matching skill(s): {', '.join(s['name'] for s in relevant_skills)}")
                self._active_skill_injection = skill_injection

        # 2. Append user task
        self.messages.append({"role": "user", "content": user_message_content})
        self.step_counter += 1
        self.logger.log_step(
            self.session_id, self.step_counter, "user", content=user_message_content
        )
        self._persist_session_state()

        print("\n" + "=" * 65)
        print(f"[Agent Session {self.session_id}] Task: {user_task}")
        print(f"[Model]: {self.model}")
        print(f"[Mode]:  {'Stateless Benchmark' if not self.enable_skills and not self.enable_memory else ('Read-Only (Saving Disabled)' if self.read_only else 'Full Persistence')}")
        print(f"[Initial CWD]: {terminal_session.cwd}")
        print("=" * 65)

        partial_answer = ""
        for iteration in range(1, self.max_iterations + 1):
            self.manage_context()

            curr_tokens = self.context_manager.estimate_tokens(self.messages, registry.schemas_for(self.enable_memory, self.enable_skills))
            max_t = self.context_manager.max_context_tokens
            pct = (curr_tokens / max_t) * 100
            print(f"\n[Iteration {iteration}/{self.max_iterations}] Thinking ({self.model}) [Context: ~{curr_tokens:,}/{max_t:,} tokens ({pct:.1f}%)]...")

            try:
                self._last_finish_reason = None
                raw_message = self.step()
            except Exception as e:
                if self._is_context_limit_error(e):
                    self.manage_context(force=True)
                    try:
                        raw_message = self.step()
                    except Exception as retry_error:
                        e = retry_error
                    else:
                        e = None
                if e is None:
                    pass
                else:
                    err_msg = f"LLM API Error: {str(e)}"
                    print(f"[!] {err_msg}")
                    self.logger.end_session(self.session_id, status="FAILED")
                    self._active_skill_injection = None
                    return err_msg

            thought_content, tool_calls = ToolProtocol.extract_tool_calls(raw_message)

            self.step_counter += 1
            if self.use_hermes_xml_protocol:
                assistant_text = raw_message.content or ""
                self.messages.append({"role": "assistant", "content": assistant_text})
                self.logger.log_step(self.session_id, self.step_counter, "assistant", content=assistant_text)
            else:
                assistant_message = raw_message.model_dump(exclude_none=True)
                if not isinstance(assistant_message, dict):
                    assistant_message = {"role": "assistant", "content": raw_message.content or ""}
                protocol_errors = {tc["id"]: tc for tc in tool_calls if tc["name"] == "__protocol_error__"}
                if tool_calls and not assistant_message.get("tool_calls"):
                    assistant_message["content"] = thought_content or ""
                    assistant_message["tool_calls"] = [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["arguments"]),
                            },
                        }
                        for tc in tool_calls
                    ]
                for stored_call in assistant_message.get("tool_calls") or []:
                    repaired = protocol_errors.get(stored_call.get("id"))
                    if repaired:
                        stored_call["function"] = {
                            "name": "__protocol_error__",
                            "arguments": json.dumps(repaired["arguments"]),
                        }
                self.messages.append(assistant_message)
                self.logger.log_step(
                    self.session_id,
                    self.step_counter,
                    "assistant",
                    content=raw_message.content,
                    tool_calls=assistant_message.get("tool_calls")
                )

            # Case 1: Agent invokes tools
            if tool_calls:
                if thought_content:
                    print(f"\n[Agent Thought]:\n{thought_content.strip()}")

                assistant_message_ref = self.messages[-1]
                assistant_step_index = self.step_counter
                xml_tool_responses = []
                for tc in tool_calls:
                    fn_name = tc["name"]
                    fn_args = tc["arguments"]
                    call_id = tc["id"]

                    if fn_name == "__protocol_error__":
                        tool_result = "Protocol repair required: " + str(fn_args.get("error", "malformed tool call"))
                        print(f"\n[Protocol Error]: {tool_result}")
                    elif fn_name == "run_terminal_command" and self.confirm_all_terminal_commands:
                        orig_cmd = fn_args.get("command", "")
                        should_run, final_cmd, feedback = self.prompt_user_for_command(orig_cmd, fn_args)
                        
                        if should_run:
                            fn_args["command"] = final_cmd
                            message_calls = assistant_message_ref.get("tool_calls") or []
                            for message_call in message_calls:
                                if message_call.get("id") == call_id:
                                    message_call["function"]["arguments"] = json.dumps(fn_args)
                            if message_calls:
                                self.logger.update_step_tool_calls(
                                    self.session_id,
                                    assistant_step_index,
                                    message_calls,
                                )
                            tool_result = registry.execute(fn_name, fn_args, read_only=self.read_only, memory_disabled=not self.enable_memory, skills_disabled=not self.enable_skills)
                            if final_cmd != orig_cmd:
                                if self.read_only and registry.is_write_tool(fn_name):
                                    provenance = (
                                        "[Command provenance]\n"
                                        f"Proposed: {orig_cmd}\nApproved: {final_cmd}\n"
                                        "Not executed: blocked by read-only policy"
                                    )
                                else:
                                    provenance = (
                                        "[Command provenance]\n"
                                        f"Proposed: {orig_cmd}\nExecuted: {final_cmd}"
                                    )
                                tool_result = f"{provenance}\n\n{tool_result}"
                        else:
                            tool_result = feedback or "Execution aborted by user."
                    else:
                        print(f"\n[Tool Request]: {fn_name}")
                        print(f"    Arguments: {json.dumps(fn_args, indent=2)}")
                        tool_result = registry.execute(fn_name, fn_args, read_only=self.read_only, memory_disabled=not self.enable_memory, skills_disabled=not self.enable_skills)

                    preview = tool_result if len(tool_result) < 350 else tool_result[:350] + "\n... [TRUNCATED]"
                    print(f"[Tool Result]:\n{preview}\n" + "-" * 50)

                    if fn_name in ("update_user_profile", "update_project_memory", "save_skill") and not self.read_only:
                        self.refresh_system_prompt()

                    if self.use_hermes_xml_protocol:
                        tool_resp_str = ToolProtocol.format_hermes_tool_response(
                            fn_name, tool_result, tool_call_id=call_id
                        )
                        xml_tool_responses.append(tool_resp_str)
                    else:
                        self.step_counter += 1
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_result
                        })
                        self.logger.log_step(
                            self.session_id,
                            self.step_counter,
                            "tool",
                            content=tool_result,
                            tool_call_id=call_id,
                        )
                if self.use_hermes_xml_protocol and xml_tool_responses:
                    self.step_counter += 1
                    combined_response = "\n".join(xml_tool_responses)
                    self.messages.append({"role": "user", "content": combined_response})
                    self.logger.log_step(
                        self.session_id, self.step_counter, "user", content=combined_response
                    )
                self._persist_session_state()

            # Case 2: Final response
            else:
                finish_reason = getattr(raw_message, "finish_reason", None) or self._last_finish_reason
                is_truncated = finish_reason == "length"
                is_empty = not (thought_content or "").strip()
                if is_truncated or is_empty:
                    if thought_content:
                        partial_answer += thought_content
                    repair_prompt = (
                        "Continue the truncated response from exactly where it stopped. "
                        "Return only the continuation and finish the answer."
                        if is_truncated else
                        "The previous assistant response was empty. Provide the complete answer now."
                    )
                    self.step_counter += 1
                    self.messages.append({"role": "user", "content": repair_prompt})
                    self.logger.log_step(
                        self.session_id, self.step_counter, "user", content=repair_prompt
                    )
                    self._persist_session_state()
                    continue

                final_answer = partial_answer + (thought_content or "")
                print(f"\n[Task Complete]:\n{final_answer}\n" + "=" * 65)
                self._persist_session_state()
                self.logger.end_session(self.session_id, status="COMPLETED")

                # Run post-task reflection (if not disabled/read-only)
                self.run_auto_memory_reflection(user_task)
                self.run_auto_skill_synthesis(user_task)
                self._active_skill_injection = None
                return final_answer

        continue_instruction = 'Type "Continue" to continue the analysis.'
        print(f"\n[!] Agent reached iteration limit ({self.max_iterations}).\n{continue_instruction}")
        self.logger.end_session(self.session_id, status="MAX_ITERATIONS")
        self._active_skill_injection = None
        if partial_answer:
            return f"Task incomplete: iteration limit reached after partial response: {partial_answer}\n{continue_instruction}"
        return f"Task incomplete: iteration limit reached without a complete response.\n{continue_instruction}"


# ==============================================================================
# CLI ARGUMENT PARSER & ENTRY POINT
# ==============================================================================

def export_current_trajectory(agent: "HermesCodingAgent", filename: str) -> str:
    """Export one session and report whether a file was actually written."""
    if agent.logger.export_jsonl(agent.session_id, filename):
        return f"Exported session trajectory to {filename}"
    return "[!] Export skipped: trajectory writes are disabled or no history exists."


def parse_args():
    parser = argparse.ArgumentParser(
        description="Hermes-Refined Coding Agent Harness for Local & Remote LLMs."
    )
    parser.add_argument(
        "-m", "--model",
        type=str,
        default=os.environ.get("AGENT_MODEL", "Qwen-32b"),
        help="Model identifier (e.g. 'Qwen-32b', 'qwen2.5-coder:32b'). Default: Qwen-32b"
    )
    parser.add_argument(
        "-u", "--base-url",
        type=str,
        default=os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1"),
        help="Local or remote LLM endpoint (default: http://localhost:11434/v1)."
    )
    parser.add_argument(
        "-k", "--api-key",
        type=str,
        default=os.environ.get("OPENAI_API_KEY", "local"),
        help="API Key (optional for local models, defaults to 'local')."
    )
    parser.add_argument(
        "--xml",
        action="store_true",
        default=os.environ.get("HERMES_XML", "false").lower() in ("true", "1"),
        help="Use Hermes XML <tool_call> format instead of standard OpenAI function calling."
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=int(os.environ.get("AGENT_MAX_TOKENS", "40960")),
        help="Context window token limit before compaction triggers (default: 40960 / 40K tokens for Qwen-32B)."
    )
    parser.add_argument(
        "--compaction-model",
        default=os.environ.get("AGENT_COMPACTION_MODEL") or None,
        help="Optional model used only for checkpoint compaction (default: primary model).",
    )
    parser.add_argument(
        "--compaction-max-tokens",
        type=int,
        default=os.environ.get("AGENT_COMPACTION_MAX_TOKENS") or None,
        help="Compactor context capacity (default: inherit --max-tokens).",
    )
    parser.add_argument(
        "--compaction-output-tokens",
        type=int,
        default=os.environ.get("AGENT_COMPACTION_OUTPUT_TOKENS") or None,
        help="Maximum checkpoint output tokens (default: bounded automatic value).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Session ID to resume from .agent_history.db."
    )
    # Testing & Evaluation Modes
    parser.add_argument(
        "--read-only", "--freeze",
        action="store_true",
        help="Freeze skills and memories: readable, but disables all saving/writing to disk."
    )
    parser.add_argument(
        "--stateless", "--benchmark",
        action="store_true",
        help="Run in pure stateless benchmark mode: zero skills, zero memories, and zero writes to disk."
    )
    parser.add_argument(
        "--no-skills",
        action="store_true",
        help="Disable skill catalog injection and pre-turn skill retrieval."
    )
    parser.add_argument(
        "--no-memory",
        action="store_true",
        help="Disable USER.md and MEMORY.md injection."
    )
    parser.add_argument(
        "--auto-skills",
        action="store_true",
        help="Opt in to one visible post-task provider call for skill proposals (costs time/tokens)."
    )
    parser.add_argument(
        "--auto-memory",
        action="store_true",
        help="Opt in to one visible post-task provider call for memory reflection (costs time/tokens)."
    )
    parser.add_argument(
        "--no-auto-skills",
        action="store_true",
        help="Disable automatic post-task skill synthesis."
    )
    parser.add_argument(
        "--no-auto-memory",
        action="store_true",
        help="Disable automatic post-task memory reflection."
    )
    args = parser.parse_args()
    if args.max_tokens <= 0:
        parser.error("--max-tokens must be positive")
    if args.compaction_max_tokens is not None and args.compaction_max_tokens <= 0:
        parser.error("--compaction-max-tokens must be positive")
    if args.compaction_output_tokens is not None and args.compaction_output_tokens <= 0:
        parser.error("--compaction-output-tokens must be positive")
    compaction_capacity = args.compaction_max_tokens or args.max_tokens
    if (
        args.compaction_output_tokens is not None
        and args.compaction_output_tokens >= compaction_capacity
    ):
        parser.error(
            "--compaction-output-tokens must be smaller than the compactor context capacity"
        )
    return args


def handle_cli_command(agent: HermesCodingAgent, prompt: str) -> bool:
    """Handle small persistence-inspection commands at the agent policy boundary."""
    command = prompt.lower().strip()
    if command in ("/user", "/profile", "user", "profile"):
        print("\n=== USER.md Profile ===")
        print(user_profile_manager.load_profile() if agent.enable_memory else "Capability disabled: memory is unavailable for this agent.")
        return True
    if command in ("/memory", "memory"):
        print("\n=== MEMORY.md Project Facts ===")
        print(project_memory_manager.load_memory() if agent.enable_memory else "Capability disabled: memory is unavailable for this agent.")
        return True
    if command in ("/skills", "skills"):
        print("\n" + (skill_store.list_skills() if agent.enable_skills else "Capability disabled: skills are unavailable for this agent."))
        return True
    if command in ("/sessions", "sessions"):
        if agent.stateless:
            print("Capability disabled: persisted sessions are unavailable in stateless mode.")
            return True
        past_sessions = agent.logger.list_sessions(limit=10)
        if not past_sessions:
            print("No past sessions found in database.")
        else:
            print("\nPast Recorded Sessions:")
            for ps in past_sessions:
                print(f"  [{ps['session_id']}] {ps['date']} | Status: {ps['status']:<10} | Steps: {ps['step_count']:<3} | Task: {ps['task'][:45]}")
        return True
    if command.startswith("/resume") or command.startswith("resume"):
        parts = prompt.split()
        if len(parts) > 1:
            if not agent.resume_session(parts[1]):
                print(f"[!] Session '{parts[1]}' not found or unavailable.")
        else:
            print("Usage: /resume <session_id>")
        return True
    return False


def main():
    args = parse_args()

    # Determine testing modes
    is_stateless = args.stateless
    is_read_only = args.read_only or is_stateless
    enable_skills = not (args.no_skills or is_stateless)
    enable_memory = not (args.no_memory or is_stateless)
    auto_learn_skills = args.auto_skills and not (args.no_auto_skills or is_read_only)
    auto_learn_memory = args.auto_memory and not (args.no_auto_memory or is_read_only)

    print("=" * 65)
    print(" Hermes-Refined Coding Agent Harness")
    print("=" * 65)
    print(f"Model:      {args.model}")
    print(f"Endpoint:   {args.base_url}")
    print(f"Max Tokens: {args.max_tokens} (Compaction threshold: ~{int(args.max_tokens * 0.70)} tokens)")
    compactor_model = args.compaction_model or args.model
    compactor_capacity = args.compaction_max_tokens or args.max_tokens
    compactor_output = args.compaction_output_tokens or "auto"
    print(
        f"Compactor:  {compactor_model} "
        f"(context={compactor_capacity}, output={compactor_output})"
    )
    print(f"Protocol:   {'Hermes XML (<tool_call>)' if args.xml else 'OpenAI JSON Tool Calling'}")
    print(f"Mode:       {'Stateless Benchmark' if is_stateless else ('Read-Only (No Saving)' if is_read_only else 'Normal Persistence')}")
    print("Security:   Interactive user review is active for EVERY system command.\n")
    print("Commands:   /skills | /user | /memory | /mode [normal|read-only|stateless] | /context | exit")

    agent = HermesCodingAgent(
        model=args.model,
        compaction_model=args.compaction_model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_context_tokens=args.max_tokens,
        compaction_max_context_tokens=args.compaction_max_tokens,
        compaction_output_tokens=args.compaction_output_tokens,
        confirm_all_terminal_commands=True,
        enable_skills=enable_skills,
        enable_memory=enable_memory,
        auto_learn_skills=auto_learn_skills,
        auto_learn_memory=auto_learn_memory,
        read_only=is_read_only,
        stateless=is_stateless,
        use_hermes_xml_protocol=args.xml
    )

    if args.resume:
        resumed = agent.resume_session(args.resume)
        if not resumed:
            print(f"[!] Warning: Could not find session '{args.resume}' to resume. Starting new session.")

    while True:
        try:
            prompt = input("\nUser > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                print("Exiting agent harness. Trajectories saved.")
                break
            if handle_cli_command(agent, prompt):
                continue
            if prompt.lower().startswith("/mode"):
                parts = prompt.split()
                if len(parts) > 1:
                    agent.set_testing_mode(parts[1])
                else:
                    curr_mode = "Stateless Benchmark" if not agent.enable_skills and not agent.enable_memory else ("Read-Only" if agent.read_only else "Normal")
                    print(f"\nCurrent Mode: {curr_mode}")
                    print(f"  • Skills Enabled:     {agent.enable_skills}")
                    print(f"  • Memory Enabled:     {agent.enable_memory}")
                    print(f"  • Read-Only (Freeze): {agent.read_only}")
                    print(f"  • Auto-Learn Skills:  {agent.auto_learn_skills}")
                    print(f"  • Auto-Learn Memory:  {agent.auto_learn_memory}")
                    print("\nChange mode with: /mode [normal | read-only | stateless]")
                continue
            if prompt.lower() in ("/user", "/profile", "user", "profile"):
                print("\n=== USER.md Profile ===")
                print(user_profile_manager.load_profile())
                continue
            if prompt.lower() in ("/memory", "memory"):
                print("\n=== MEMORY.md Project Facts ===")
                print(project_memory_manager.load_memory())
                continue
            if prompt.lower() in ("/skills", "skills"):
                print("\n" + skill_store.list_skills())
                continue
            if prompt.lower() in ("/sessions", "sessions"):
                past_sessions = agent.logger.list_sessions(limit=10)
                if not past_sessions:
                    print("No past sessions found in database.")
                else:
                    print("\nPast Recorded Sessions:")
                    for ps in past_sessions:
                        print(f"  [{ps['session_id']}] {ps['date']} | Status: {ps['status']:<10} | Steps: {ps['step_count']:<3} | Task: {ps['task'][:45]}")
                continue
            if prompt.lower().startswith("/resume") or prompt.lower().startswith("resume"):
                parts = prompt.split()
                if len(parts) > 1:
                    target_id = parts[1]
                    res = agent.resume_session(target_id)
                    if not res:
                        print(f"[!] Session '{target_id}' not found.")
                else:
                    print("Usage: /resume <session_id>")
                continue
            if prompt.lower() in ("/context", "context"):
                total_tokens = agent.context_manager.estimate_tokens(agent.messages)
                max_t = agent.context_manager.max_context_tokens
                trigger_t = int(max_t * agent.context_manager.trigger_threshold)
                pct = (total_tokens / max_t) * 100
                bar_len = 25
                filled = int(bar_len * (total_tokens / max_t))
                bar = "█" * filled + "░" * (bar_len - filled)

                sys_tokens = agent.context_manager.estimate_tokens([agent.messages[0]]) if agent.messages else 0
                history_msgs = agent.messages[1:] if len(agent.messages) > 1 else []
                compaction_msgs = [m for m in history_msgs if "[CONTEXT COMPACTION" in str(m.get("content", ""))]
                turn_msgs = [m for m in history_msgs if "[CONTEXT COMPACTION" not in str(m.get("content", ""))]
                
                compaction_tokens = agent.context_manager.estimate_tokens(compaction_msgs)
                turn_tokens = agent.context_manager.estimate_tokens(turn_msgs)
                headroom = max(0, trigger_t - total_tokens)

                print("\n" + "=" * 55)
                print("CONTEXT WINDOW BUDGET & BREAKDOWN")
                print("=" * 55)
                print(f"Usage:       [{bar}] {pct:.1f}%")
                print(f"Total:       ~{total_tokens:,} / {max_t:,} tokens")
                print(f"Threshold:   ~{trigger_t:,} tokens (70.0% compaction trigger)")
                print(f"Headroom:    ~{headroom:,} tokens remaining before compaction\n")
                print("Breakdown:")
                print(f"  • System Prompt (Base + Skills + USER/MEMORY.md): ~{sys_tokens:,} tokens")
                if compaction_tokens > 0:
                    print(f"  • Historical Compaction Summary:                  ~{compaction_tokens:,} tokens")
                print(f"  • Active Conversation Turns ({len(turn_msgs)} turns):            ~{turn_tokens:,} tokens")
                print("=" * 55)
                continue
            if prompt.lower() in ("/compact", "compact"):
                agent.manage_context(force=True)
                continue
            if prompt.lower().startswith("export-trajectory"):
                parts = prompt.split()
                filename = parts[1] if len(parts) > 1 else "trajectory.jsonl"
                print(export_current_trajectory(agent, filename))
                continue

            agent.run(prompt)

        except (KeyboardInterrupt, EOFError):
            print("\nShutting down.")
            break


if __name__ == "__main__":
    main()
