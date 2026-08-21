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
   - Tools: `read_user_profile`, `update_user_profile`, `read_project_memory`, `update_project_memory`.
3. **Session Resumption & Trajectory Continuity ([`storage.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/storage.py))**:
   - Pick up past sessions directly via `--resume <session_id>` or interactive `/resume <session_id>`.
   - View past session logs, dates, and step counts with `/sessions`.
4. **Hermes Skill System & Intelligent Deduplication ([`skills.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/skills.py))**:
   - Injects `<available_skills>` catalog and pre-turn keyword auto-injection.
   - Catalog-aware deduplication; autonomous reflection proposes/skips changes to existing procedures instead of overwriting them.
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

## 💬 1. In-Session Chat Commands (Typed at `User >`)

| Command | Aliases | Description |
| :--- | :--- | :--- |
| **`/mode`** | | Shows current testing mode or switches mode: `/mode [normal \| read-only \| stateless]`. |
| **`/context`** | `context` | Displays a visual progress bar, total token usage, compaction headroom, and breakdown (System Prompt vs. History vs. Checkpoints). |
| **`/compact`** | `compact` | Forces an immediate 2-phase context checkpoint compaction without waiting for the 70% threshold. |
| **`/user`** | `/profile`, `user` | Displays the active operator profile and preferences contract ([`USER.md`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/.agent_memories/USER.md)). |
| **`/memory`** | `memory` | Displays the persistent project architecture and environment facts ([`MEMORY.md`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/.agent_memories/MEMORY.md)). |
| **`/skills`** | `skills` | Lists all learned procedures and recipes stored in the repository ([`.agent_skills/`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/.agent_skills)). |
| **`/sessions`** | `sessions` | Displays past recorded sessions from [`.agent_history.db`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/.agent_history.db) with dates, status, and turn counts. |
| **`/resume <id>`** | `resume <id>` | Switches to and resumes a past conversation session by its ID. |
| **`export-trajectory [file.jsonl]`** | | Exports the current session trajectory to a JSONL dataset file. |
| **`exit`** | `quit` | Saves the active trajectory and gracefully exits the harness. |

---

## 🛡️ 2. Interactive Safety Review (Human-in-the-Loop)

Whenever the agent proposes a system or terminal command, execution pauses for review:

| Option | Input | Action |
| :--- | :--- | :--- |
| **Approve & Run** | `[Enter]` or `y` / `yes` | Executes the command in the persistent terminal session. |
| **Deny** | `n` or `no` / `cancel` | Cancels execution and informs the agent the command was rejected. |
| **Edit** | `e` or `edit` | Prompts you to modify the command string before running it. |
| **Send Feedback** | Any custom text | Sends your text as guidance/feedback back to the model without running the command. |

---

## ⚙️ 3. CLI Startup Flags (`python agent.py [OPTIONS]`)

| Flag | Short | Default | Description |
| :--- | :--- | :--- | :--- |
| **`--model`** | `-m` | `Qwen-32b` | Model identifier (e.g. `Qwen-32b`, `qwen2.5-coder:32b`, `Qwen/Qwen2.5-32B-Instruct`). |
| **`--base-url`** | `-u` | `http://localhost:11434/v1` | LLM HTTP API endpoint (vLLM `:8000/v1`, Ollama `:11434/v1`, LM Studio `:1234/v1`). |
| **`--api-key`** | `-k` | `local` | API Key (optional for local models). |
| **`--max-tokens`** | | `40960` | Max context token capacity (compaction triggers at 70% $\approx$ 28,672 tokens). |
| **`--resume <id>`** | | `None` | Session ID to resume from `.agent_history.db` on startup. |
| **`--read-only`** | `--freeze` | `False` | **Testing Mode**: Existing memories/skills are readable, but zero writes/saves to disk. |
| **`--stateless`** | `--benchmark` | `False` | **Benchmark Baseline**: Disables skills, memory, and disk saving (pure zero-shot). |
| **`--no-skills`** | | `False` | Completely disables skill catalog and skill retrieval. |
| **`--no-memory`** | | `False` | Completely disables USER.md and MEMORY.md injection. |
| **`--auto-skills`** | | `False` | Opts in to a visible post-task skill reflection provider call (extra latency/tokens). Existing procedures are not autonomously overwritten. |
| **`--auto-memory`** | | `False` | Opts in to a visible post-task memory reflection provider call (extra latency/tokens). |
| **`--no-auto-skills`**| | `False` | Compatibility alias that disables `--auto-skills`. |
| **`--no-auto-memory`**| | `False` | Compatibility alias that disables `--auto-memory`. |
| **`--xml`** | | `False` | Switches from OpenAI JSON tool calling to Hermes XML `<tool_call>` syntax. |

---

## 🛠️ 4. Agent Tools (Autonomous Capabilities)

| Tool Name | Parameters | Description |
| :--- | :--- | :--- |
| **`grep_search`** | `query`, `search_path`, `is_regex`, `file_pattern`, `max_results` | Bounded regex/literal search confined to the canonical workspace; includes relevant hidden config folders and excludes VCS/runtime/cache folders and credential files. |
| **`find_files_by_pattern`** | `pattern`, `search_path`, `max_results` | Confined glob search (`*.py`, `src/**/*.ts`, `*router*`) across workspace folders. |
| **`run_terminal_command`** | `command`, `timeout` | Executes terminal commands with persistent `cwd` across turns. |
| **`read_file`** | `file_path`, `start_line`, `end_line` | Bounded file reads inside the configured workspace. Traversal, symlink escape, and sensitive credential targets are denied. |
| **`write_file`** | `file_path`, `content` | Writes/creates normal source files inside the canonical workspace; outside and sensitive targets are denied. |
| **`patch_file`** | `file_path`, `search_content`, `replace_content` | Performs targeted search-and-replace on existing files. |
| **`list_directory`** | `directory_path` | Inspects directory contents and file sizes. |
| **`load_skill` / `<skill_name>()`** | `name` | Reads instructions and workflow details for any learned project skill. |
| **`save_skill`** | `name`, `description`, `instructions` | Saves a newly discovered procedural workflow to `.agent_skills/`. |
| **`read_user_profile`** | *(none)* | Reads operator profile from `USER.md`. |
| **`update_user_profile`** | `category`, `preference` | Appends or updates preferences in `USER.md`. |
| **`read_project_memory`** | *(none)* | Reads project architecture facts from `MEMORY.md`. |
| **`update_project_memory`** | `category`, `fact` | Appends or updates technical facts in `MEMORY.md`. |
