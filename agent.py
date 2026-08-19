"""
Hermes-Refined Coding Agent Harness.
An open, persistent, multi-protocol coding agent harness in Python.

Key Features:
1. Dynamic Skill Catalog & Pre-Turn Relevant Skill Auto-Injection.
2. Catalog-Aware Skill Deduplication & Merging (Hermes Learning Loop).
3. Configured for Qwen-32B with 40K (40,960) Token Context Budget.
4. Local Model Support without API Keys (Qwen-32b, Ollama, vLLM, LM Studio, llama.cpp).
5. Production-Grade Context Checkpoint Compaction & Summarization.
6. Interactive User Prompt for Every Terminal Command (approve, edit, reject, feedback).
7. Stateful Terminal Execution (persistent cwd, cd management, output clipping).
8. Dual Protocol Tool Calling (OpenAI JSON API & Hermes XML <tool_call> syntax).
9. SQLite Trajectory Logging & Export.
"""

import argparse
import json
import os
import sys
import uuid
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


class HermesCodingAgent:
    """
    Stateful, autonomous coding agent harness supporting both
    OpenAI structured tool calling and Hermes XML function calling protocols,
    with interactive human-in-the-loop confirmation for every system command,
    automated context compaction/summarization, and local model support (Qwen-32b, etc.).
    """

    SYSTEM_PROMPT_TEMPLATE = """You are an autonomous AI software engineer and terminal agent.
You have access to tools that allow you to inspect the system, manage files, and execute terminal commands.

### Important Protocol:
- **Interactive Terminal Commands**: Every system/terminal command you propose will be reviewed interactively by the user before execution. The user may approve it, modify it, or provide feedback.
- **Context Compaction**: In long-running tasks, earlier conversation segments may be compressed into structured summary blocks. Use the summary to maintain continuity.
- **Persistent Workspace**: The terminal environment maintains your working directory (`cwd`) across tool calls. Use `cd <dir>` to navigate projects.
- **Verification & Testing**: Always verify changes by running tests, linters, or checking file content.
- **Analyze Output**: Inspect command outputs (stdout, stderr, exit codes). If errors occur, diagnose and repair them iteratively.
- **Learned Skills & Best Practices**: Below is the catalog of learned project skills. When a task relates to any available skill, use `load_skill(name)` or apply its established procedure:
{skills_catalog}

- **Conciseness**: Summarize your work clearly when the task is achieved.
"""

    def __init__(
        self,
        model: str = "Qwen-32b",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_iterations: int = 30,
        max_context_tokens: int = 40960,     # Default to 40K (40,960 tokens) for Qwen-32B
        compaction_threshold: float = 0.70, # Triggers compaction at ~28,672 tokens (leaving ~12,288 tokens headroom)
        confirm_all_terminal_commands: bool = True,
        auto_learn_skills: bool = True,
        use_hermes_xml_protocol: bool = False,
    ):
        self.model = model or "Qwen-32b"
        self.max_iterations = max_iterations
        self.confirm_all_terminal_commands = confirm_all_terminal_commands
        self.auto_learn_skills = auto_learn_skills
        self.use_hermes_xml_protocol = use_hermes_xml_protocol
        
        # Local models running on Ollama/vLLM/LMStudio do not require a real API key,
        # but the OpenAI SDK requires a non-empty string.
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or "local-no-key-required"
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:11434/v1"

        self.client = OpenAI(
            api_key=resolved_api_key,
            base_url=resolved_base_url
        )
        
        self.logger = TrajectoryLogger()
        self.context_manager = ContextManager(
            max_context_tokens=max_context_tokens,
            trigger_threshold=compaction_threshold
        )
        self.skill_extractor = AutoSkillExtractor(skill_store=skill_store)
        
        self.session_id = str(uuid.uuid4())[:8]
        self.step_counter = 0

        # Construct system prompt with live skill catalog
        system_content = self._build_system_prompt()
        if self.use_hermes_xml_protocol:
            system_content = ToolProtocol.format_hermes_system_prompt(system_content, registry.schemas)

        self.messages: List[Dict[str, Any]] = [
            {"role": "system", "content": system_content}
        ]

    def _build_system_prompt(self) -> str:
        """Build system prompt embedding the live catalog of available skills."""
        catalog_xml = skill_store.format_catalog_prompt()
        return self.SYSTEM_PROMPT_TEMPLATE.format(skills_catalog=catalog_xml)

    def refresh_system_prompt(self):
        """Update system prompt with newly learned skills."""
        new_prompt = self._build_system_prompt()
        if self.use_hermes_xml_protocol:
            new_prompt = ToolProtocol.format_hermes_system_prompt(new_prompt, registry.schemas)
        if self.messages and self.messages[0].get("role") == "system":
            self.messages[0]["content"] = new_prompt

    def prompt_user_for_command(self, command: str, args: Dict[str, Any]) -> Tuple[bool, str, Optional[str]]:
        """
        Interactive prompt for every system command.
        Returns:
            (should_execute: bool, final_command: str, custom_feedback: Optional[str])
        """
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

        # Case 1: Approve (Enter / y / yes)
        if user_input.lower() in ("", "y", "yes"):
            return True, command, None

        # Case 2: Reject (n / no / cancel / skip)
        elif user_input.lower() in ("n", "no", "cancel", "skip"):
            print("[-] Command rejected by user.")
            return False, command, "Execution denied by user."

        # Case 3: Edit command before running (e / edit)
        elif user_input.lower() in ("e", "edit"):
            print(f"Current command: {command}")
            new_cmd = input(">> Enter new command: ").strip()
            if not new_cmd:
                print("[-] Empty command. Cancelled.")
                return False, command, "Execution cancelled (empty edit)."
            return True, new_cmd, None

        # Case 4: Custom guidance / feedback message
        else:
            print(f"[+] Sending user feedback to agent: '{user_input}'")
            return False, command, f"User declined to run this command and provided feedback: '{user_input}'. Adjust your plan accordingly."

    def manage_context(self, force: bool = False):
        """Perform context health check, pruning, or summarization compaction."""
        compacted_msgs, was_compacted, msg = self.context_manager.compact(
            client=self.client,
            model=self.model,
            messages=self.messages,
            current_step=self.step_counter,
            force=force
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
        elif force:
            print(f"\n[Info] {msg}")

    def run_auto_skill_synthesis(self, task_summary: str):
        """Analyze trajectory against catalog to automatically extract or refine skills."""
        if not self.auto_learn_skills:
            return

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
            action_label = "Refined existing skill" if action == "UPDATE" else "Synthesized new skill"
            
            print(f"\n[Self-Improvement] {action_label}: '{name}'")
            print(f"  Description: {desc}")
            
            # Refresh system prompt with updated skills catalog
            self.refresh_system_prompt()
            
            self.logger.log_step(
                self.session_id,
                self.step_counter,
                "skill_synthesis",
                content=f"{action_label}: {name}"
            )

    def step(self) -> Any:
        """Invoke the LLM for one turn."""
        if self.use_hermes_xml_protocol:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
            )
        else:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=registry.schemas if registry.schemas else None,
                tool_choice="auto" if registry.schemas else None,
            )
        return response.choices[0].message

    def run(self, user_task: str) -> str:
        """Execute the autonomous agent loop for a given task."""
        self.session_id = str(uuid.uuid4())[:8]
        self.step_counter = 0
        self.logger.start_session(self.session_id, user_task)

        # 1. Pre-Turn Skill Matching: Automatically search for relevant skills
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
            self.messages.append({"role": "user", "content": skill_injection})
            self.step_counter += 1

        # 2. Append user task
        self.messages.append({"role": "user", "content": user_task})
        self.step_counter += 1
        self.logger.log_step(self.session_id, self.step_counter, "user", content=user_task)

        print("\n" + "=" * 65)
        print(f"[Agent Session {self.session_id}] Task: {user_task}")
        print(f"[Model]: {self.model}")
        print(f"[Initial CWD]: {terminal_session.cwd}")
        print("=" * 65)

        for iteration in range(1, self.max_iterations + 1):
            # Check and compact context before LLM call
            self.manage_context()

            print(f"\n[Iteration {iteration}/{self.max_iterations}] Thinking ({self.model})...")

            try:
                raw_message = self.step()
            except Exception as e:
                err_msg = f"LLM API Error: {str(e)}"
                print(f"[!] {err_msg}")
                self.logger.end_session(self.session_id, status="FAILED")
                return err_msg

            # Parse content and tool calls using dual protocol parser
            thought_content, tool_calls = ToolProtocol.extract_tool_calls(raw_message)

            # Record assistant turn in history
            self.step_counter += 1
            if self.use_hermes_xml_protocol:
                assistant_text = raw_message.content or ""
                self.messages.append({"role": "assistant", "content": assistant_text})
                self.logger.log_step(self.session_id, self.step_counter, "assistant", content=assistant_text)
            else:
                self.messages.append(raw_message.model_dump(exclude_none=True))
                self.logger.log_step(
                    self.session_id,
                    self.step_counter,
                    "assistant",
                    content=raw_message.content,
                    tool_calls=[{"name": tc["name"], "args": tc["arguments"]} for tc in tool_calls]
                )

            # Case 1: Agent invokes tools
            if tool_calls:
                if thought_content:
                    print(f"\n[Agent Thought]:\n{thought_content.strip()}")

                for tc in tool_calls:
                    fn_name = tc["name"]
                    fn_args = tc["arguments"]
                    call_id = tc["id"]

                    # If this is a terminal / system command, prompt user for input/confirmation
                    if fn_name == "run_terminal_command" and self.confirm_all_terminal_commands:
                        orig_cmd = fn_args.get("command", "")
                        should_run, final_cmd, feedback = self.prompt_user_for_command(orig_cmd, fn_args)
                        
                        if should_run:
                            fn_args["command"] = final_cmd
                            tool_result = registry.execute(fn_name, fn_args)
                        else:
                            tool_result = feedback or "Execution aborted by user."
                    else:
                        print(f"\n[Tool Request]: {fn_name}")
                        print(f"    Arguments: {json.dumps(fn_args, indent=2)}")
                        tool_result = registry.execute(fn_name, fn_args)

                    # Display result preview
                    preview = tool_result if len(tool_result) < 350 else tool_result[:350] + "\n... [TRUNCATED]"
                    print(f"[Tool Result]:\n{preview}\n" + "-" * 50)

                    # Append tool result in appropriate protocol format
                    self.step_counter += 1
                    if self.use_hermes_xml_protocol:
                        tool_resp_str = ToolProtocol.format_hermes_tool_response(fn_name, tool_result)
                        self.messages.append({"role": "user", "content": tool_resp_str})
                        self.logger.log_step(self.session_id, self.step_counter, "tool_response", content=tool_resp_str)
                    else:
                        self.messages.append({
                            "role": "tool",
                            "tool_call_id": call_id,
                            "content": tool_result
                        })
                        self.logger.log_step(self.session_id, self.step_counter, "tool", content=tool_result)

            # Case 2: Agent completed its work and provided final response
            else:
                final_answer = thought_content or "[Task Finished]"
                print(f"\n[Task Complete]:\n{final_answer}\n" + "=" * 65)
                self.logger.end_session(self.session_id, status="COMPLETED")

                # Run post-task automatic skill synthesis & deduplicating reflection
                self.run_auto_skill_synthesis(user_task)

                return final_answer

        print(f"\n[!] Agent reached iteration limit ({self.max_iterations}).")
        self.logger.end_session(self.session_id, status="MAX_ITERATIONS")
        return "Task halted: Max iterations reached."


# ==============================================================================
# CLI ARGUMENT PARSER & ENTRY POINT
# ==============================================================================

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
        "--no-auto-skills",
        action="store_true",
        help="Disable automatic post-task skill synthesis."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 65)
    print(" Hermes-Refined Coding Agent Harness")
    print("=" * 65)
    print(f"Model:      {args.model}")
    print(f"Endpoint:   {args.base_url}")
    print(f"Max Tokens: {args.max_tokens} (Compaction threshold: ~{int(args.max_tokens * 0.70)} tokens)")
    print(f"Protocol:   {'Hermes XML (<tool_call>)' if args.xml else 'OpenAI JSON Tool Calling'}")
    print("Security:   Interactive user review is active for EVERY system command.")
    print("Features:   Auto-Skill Synthesis & Deduplication | Skill Auto-Retrieval\n")
    print("Commands:   /skills | /compact (force compaction) | /context | exit")

    agent = HermesCodingAgent(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_context_tokens=args.max_tokens,
        confirm_all_terminal_commands=True,
        auto_learn_skills=not args.no_auto_skills,
        use_hermes_xml_protocol=args.xml
    )

    while True:
        try:
            prompt = input("\nUser > ").strip()
            if not prompt:
                continue
            if prompt.lower() in ("exit", "quit"):
                print("Exiting agent harness. Trajectories saved.")
                break
            if prompt.lower() in ("/skills", "skills"):
                print("\n" + skill_store.list_skills())
                continue
            if prompt.lower() in ("/context", "context"):
                tokens = agent.context_manager.estimate_tokens(agent.messages)
                max_t = agent.context_manager.max_context_tokens
                pct = (tokens / max_t) * 100
                print(f"\n[Context Status]: ~{tokens}/{max_t} tokens ({pct:.1f}% capacity) across {len(agent.messages)} messages.")
                continue
            if prompt.lower() in ("/compact", "compact"):
                agent.manage_context(force=True)
                continue
            if prompt.lower().startswith("export-trajectory"):
                parts = prompt.split()
                filename = parts[1] if len(parts) > 1 else "trajectory.jsonl"
                agent.logger.export_jsonl(agent.session_id, filename)
                print(f"Exported session trajectory to {filename}")
                continue

            agent.run(prompt)

        except (KeyboardInterrupt, EOFError):
            print("\nShutting down.")
            break


if __name__ == "__main__":
    main()
