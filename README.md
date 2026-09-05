# Hermes-Refined Coding Agent Harness

## Required global prompt layer

Repository-root `READ_THIS.md` is the operator-managed global instruction layer. It is resolved relative to `agent.py`, never the process working directory, validated as UTF-8, screened by the existing prompt-content scanner, and limited to **20,000 characters**. Missing, unreadable, invalid-UTF-8, empty, oversized, scanner-rejected, or reserved-marker-containing content stops startup or a mode rebuild instead of being omitted.

Prompt construction order is: the replaceable base template with `USER.md`, `MEMORY.md`, and the skill catalog filled in; then a separately marked `READ_THIS.md` block; then Hermes tool XML when `--xml` is enabled. Because injection happens outside template formatting, a replacement base/persona template does not need a `READ_THIS.md` placeholder. The block is mandatory in normal, no-memory, no-skills, read-only, stateless, mode-rebuilt, and XML prompts.

The complete system prompt is frozen and saved per session. Resume preserves an existing marked `READ_THIS.md` snapshot exactly; a legacy saved prompt without the marker receives the current validated block once. When disabled startup capabilities require the base prompt to be rebuilt, any saved marked `READ_THIS.md` snapshot is still retained.

Agent-facing `write_file` and `patch_file` deny only the canonical repository-root `READ_THIS.md`; `read_file` remains allowed, and unrelated nested files with that name are ordinary workspace files. This is an agent file-tool boundary, not an operating-system permission: an operator can still edit the root file through an explicitly user-approved terminal command.

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
   - Preflights the exact summarizer request, reserves bounded output capacity, and chunks oversized history without splitting native or XML tool exchanges.
   - Can use a separately configured compactor model/context while defaulting to the primary model and context capacity.
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

### 4. Named Profiles with an Independent Workspace
```bash
# Omit --workspace for separate, automatically created profile workspaces.
python agent.py --profile alice
python agent.py --profile bob

# Explicitly override the default to share one existing checkout instead.
python agent.py --profile alice --workspace C:\src\shared-project
python agent.py --profile bob --workspace C:\src\shared-project

# Store named profiles somewhere else.
python agent.py --profile alice --profiles-dir D:\agent-state --workspace C:\src\shared-project
```

Named profiles live under `.agent_profiles/<name>/` beside `agent.py` by default.
Each contains `.agent_memories/USER.md`, `.agent_memories/MEMORY.md`,
`.agent_skills/`, `.agent_history.db`, and an automatically created `workspace/`:

```text
.agent_profiles/
├── alice/
│   ├── .agent_memories/
│   ├── .agent_skills/
│   ├── .agent_history.db
│   └── workspace/
└── bob/
    └── workspace/
```

Profile names are 1-64 letters, digits, hyphens, or underscores. An explicit
`--workspace` overrides the profile default for terminal/file-tool confinement
and must name an already-existing directory; it is not created automatically.

Without `--profile`, legacy behavior is unchanged: persistence remains in the
process launch directory (`.agent_memories`, `.agent_skills`, and
`.agent_history.db`) and the launch directory is also the default workspace.
Read-only/stateless startup never creates a named profile and fails clearly if
the selected named profile does not already exist. When `--workspace` is omitted
in those modes, the profile's existing `workspace/` must also already exist.

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
| **`--profile <name>`** | | `None` (legacy mode) | Select an isolated named persistence profile. |
| **`--profiles-dir <path>`** | | `.agent_profiles` beside `agent.py` | Root containing named profile state; independent of the workspace. |
| **`--workspace <path>`** | | Profile `workspace/`; launch directory in legacy mode | Override with an existing directory used as the canonical terminal/file-tool workspace. |
| **`--max-tokens`** | | `40960` | Max context token capacity (compaction triggers at 70% $\approx$ 28,672 tokens). |
| **`--compaction-model`** | | Primary model | Optional model used only for checkpoint generation (`AGENT_COMPACTION_MODEL`). |
| **`--compaction-max-tokens`** | | Primary context | Compactor context capacity (`AGENT_COMPACTION_MAX_TOKENS`). Every compactor request is preflighted against it. |
| **`--compaction-output-tokens`** | | Automatic | Explicit checkpoint output reservation (`AGENT_COMPACTION_OUTPUT_TOKENS`); must be smaller than the compactor context. |
| **`--resume <id>`** | | `None` | Session ID to resume from `.agent_history.db` on startup. |
| **`--read-only`** | `--freeze` | `False` | **Testing Mode**: Existing memories/skills are readable, but zero writes/saves to disk. |
| **`--stateless`** | `--benchmark` | `False` | **Benchmark Baseline**: Disables skills, memory, and disk saving (pure zero-shot). |
| **`--no-skills`** | | `False` | Completely disables skill catalog and skill retrieval. |
| **`--no-memory`** | | `False` | Completely disables USER.md and MEMORY.md injection. |
| **`--auto-skills`** | | `False` | Opts in to a visible post-task skill reflection provider call (extra latency/tokens). Once enabled, safe CREATE and UPDATE proposals are applied automatically, including deduplicated CREATE proposals rerouted to an existing skill. |
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
| **`query_teradata`** | `sql`, `max_rows` | Runs one bounded, read-only Teradata query using environment configuration. |
| **`query_impala`** | `sql`, `max_rows` | Runs one bounded, read-only Hadoop Impala query using environment configuration. |
| **`export_teradata_csv`** | `sql`, `file_path`, `batch_size`, `overwrite` | Streams a read-only Teradata query to an atomic workspace CSV file. |
| **`export_impala_csv`** | `sql`, `file_path`, `batch_size`, `overwrite` | Streams a read-only Impala query to an atomic workspace CSV file. |
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

### Optional database query drivers

Install the drivers only when their tools are needed:

```bash
pip install teradataml impyla
```

Configure credentials outside model-visible tool arguments. Teradata supports
`TERADATA_HOST`, `TERADATA_USER`, `TERADATA_PASSWORD`, and the optional
`TERADATA_DATABASE` and `TERADATA_LOGMECH`. The `impyla` distribution exposes the
`impala.dbapi` module used by this harness. Impala requires `IMPALA_HOST`; optional
settings are `IMPALA_PORT` (default `21050`), `IMPALA_DATABASE` (default
`default`), `IMPALA_TIMEOUT` (default `30`), `IMPALA_AUTH_MECHANISM` (default
`NOSASL`), `IMPALA_USER`, `IMPALA_PASSWORD`, `IMPALA_USE_SSL`, `IMPALA_CA_CERT`,
`IMPALA_KERBEROS_SERVICE_NAME` (default `impala`), `IMPALA_USE_HTTP_TRANSPORT`,
`IMPALA_HTTP_PATH` (default empty), and `IMPALA_VERIFY_CERT`. Boolean environment
values accept `true/false`, `1/0`, `yes/no`, or `on/off`.

Alternatively, copy `database.example.json` to `database.json` in the same
directory as `db_tools.py` and fill in the UTF-8 JSON file:

```json
{
  "teradata": {
    "host": "TERADATA_HOSTNAME",
    "user": "TERADATA_USERNAME",
    "password": "TERADATA_PASSWORD",
    "database": "OPTIONAL_DATABASE",
    "logmech": "OPTIONAL_LOGON_MECHANISM"
  },
  "impala": {
    "host": "IMPALA_HOSTNAME",
    "port": 21050,
    "database": "default",
    "timeout": 30,
    "auth_mechanism": "NOSASL",
    "user": "OPTIONAL_USERNAME",
    "password": "OPTIONAL_PASSWORD",
    "use_ssl": false,
    "ca_cert": "OPTIONAL_CA_CERT_PATH",
    "kerberos_service_name": "impala",
    "use_http_transport": false,
    "http_path": "",
    "verify_cert": false
  }
}
```

`database.json` is ignored by Git; keep its permissions restricted to the account
running the harness. It is loaded from the fixed location beside `db_tools.py`
only when a database query runs, regardless of the current working directory.
If it is absent, configuration comes only from the environment. Individual
`TERADATA_*` and `IMPALA_*` environment values override their corresponding JSON
values. `database.example.json` remains the tracked template.

Query tools accept only `sql` and optional `max_rows` (1 through 1000). To
request more than the default 100 rows, for example, call `query_impala` with
`{"sql":"SELECT * FROM events","max_rows":500}`. Every result is also capped
at 16,000 characters, so large cells or wide results can return fewer requested
rows with `truncated` set to `true`.

Failed queries identify the exact `connect`, `execute`, `fetch`, or `serialize`
stage and include a bounded, sanitized exception chain. Exception classes,
SQLSTATEs, and vendor error codes are preserved when available so the agent can
correct SQL rather than seeing only a generic query failure. Configured hosts,
users, passwords, databases/DSNs, connection values, and common credential
assignments are redacted before the error reaches the agent.

CSV exports are the unbounded streaming alternative to those previews. For
example, call `export_impala_csv` with
`{"sql":"SELECT * FROM events","file_path":"exports/events.csv","batch_size":2000}`.
The destination must end in `.csv`, remain inside the canonical workspace, and
must not be a sensitive path. `batch_size` defaults to 1000 and accepts 1 through
10000; `overwrite` defaults to `false`. Export tools execute the query once and
write each fetched batch directly to a same-directory temporary file, publishing
the destination only after success. Their compact result is a manifest, not query
rows. Unlike query previews, exports are write tools and are blocked in read-only
agent mode. Never place credentials in prompts or SQL tool calls.
