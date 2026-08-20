# Hermes-Refined Coding Agent Harness

An open-source, terminal-native Python agent harness optimized for local models (**Qwen-32b**, etc.) and remote models, inspired by [**NousResearch/hermes-agent**](https://github.com/nousresearch/hermes-agent).

---

## 🌟 Key Features

1. **Ripgrep-Style Codebase Search & File Finder ([`tools.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/tools.py))**:
   - `grep_search`: Fast multi-file regex/literal search returning file paths, line numbers, and matching snippets.
   - `find_files_by_pattern`: Glob search (`*.py`, `src/**/*.ts`) across workspace folders without wasting context.
2. **Dual Persistent Memory System (`USER.md` & `MEMORY.md` / [`memory.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/memory.py))**:
   - `USER.md`: Stores operator profile, communication style, technical background, and safety constraints (`<user_profile>`).
   - `MEMORY.md`: Stores project architecture facts, tech stack details, and environment conventions (`<project_memory>`).
   - Tools `read_user_profile`, `update_user_profile`, `read_project_memory`, `update_project_memory`.
3. **Session Resumption & Trajectory Continuity ([`storage.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/storage.py))**:
   - Pick up past sessions directly via `--resume <session_id>` or interactive `/resume <session_id>`.
   - View past session logs, dates, and step counts with `/sessions`.
4. **Hermes Skill System & Intelligent Deduplication ([`skills.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/skills.py))**:
   - Injects `<available_skills>` catalog and pre-turn keyword auto-injection.
   - Catalog-aware deduplication: updates existing skills (`UPDATE`) instead of creating duplicates.
5. **Two-Phase Context Compaction & Summarization ([`compaction.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/compaction.py))**:
   - Implements the **Security & Provenance Context-Checkpoint Summarizer** contract.
   - Host-side deterministic extraction for `<EXACT_ANCHORS>` and `<VERBATIM_USER_MESSAGES>` at the 40K token limit.
6. **Interactive Review for Every System Command ([`agent.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/agent.py))**:
   - Every terminal command pauses for user approval: Run (`[Enter]/y`), Deny (`n`), Edit (`e`), or Steer with feedback.
7. **Stateful Terminal Engine ([`terminal.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/terminal.py))**:
   - `cd <dir>` and working directory state persist across tool calls.
8. **Dual-Protocol Tool Execution ([`protocol.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/protocol.py))**:
   - Supports both standard OpenAI JSON tool calling and Hermes XML ChatML (`<tool_call>...`).
9. **Local Model Native (Qwen-32B, Ollama, vLLM, LM Studio, llama.cpp)**:
   - Zero API key requirement with full CLI argument configurability.

---

## 🚀 Running with Qwen-32B Locally

### 1. With vLLM (Default: `http://localhost:8000/v1`)
```bash
vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000 --max-model-len 40960 --gpu-memory-utilization 0.95

# Launch the harness:
python agent.py --model Qwen/Qwen2.5-32B-Instruct --base-url http://localhost:8000/v1
```

### 2. With Ollama (Default: `http://localhost:11434/v1`)
```bash
ollama run qwen2.5:32b

# Launch the harness:
python agent.py --model qwen2.5:32b
```

### 3. Resuming a Past Session
```bash
python agent.py --model Qwen-32b --resume <session_id>
```

---

## ⚙️ CLI Arguments

| Argument | Shorthand | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `-m` | `Qwen-32b` | Model name or tag |
| `--base-url` | `-u` | `http://localhost:11434/v1` | LLM HTTP API endpoint |
| `--api-key` | `-k` | `local` | API Key (optional for local models) |
| `--xml` | | `False` | Use Hermes XML `<tool_call>` protocol |
| `--max-tokens` | | `40960` | Context capacity limit (40K tokens for Qwen-32B) |
| `--resume` | | `None` | Session ID to resume from `.agent_history.db` |
| `--no-auto-skills`| | `False` | Disable automatic post-task skill synthesis |

---

## 💬 Interactive In-Session Commands

- `/user` — View active operator profile (`USER.md`).
- `/memory` — View project architecture and environment facts (`MEMORY.md`).
- `/sessions` — List past recorded sessions with status and step counts.
- `/resume <id>` — Switch to and resume a past conversation trajectory.
- `/skills` — List all available learned skills.
- `/context` — Show current token count and percentage of maximum context budget.
- `/compact` — Force an immediate context compaction pass.
- `export-trajectory [filename.jsonl]` — Export current session steps to JSONL.
- `exit` or `quit` — Exit the session.
