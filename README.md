# Hermes-Refined Coding Agent Harness

An open-source, terminal-native Python agent harness optimized for local models (**Qwen-32b**, etc.) and remote models, inspired by [**NousResearch/hermes-agent**](https://github.com/nousresearch/hermes-agent).

---

## 🌟 Key Features

1. **Local Model Support without API Keys**:
   - Zero configuration needed for local endpoints (Ollama, vLLM, LM Studio, llama.cpp, LocalAI).
   - Automatically handles dummy API keys required by client SDKs without failing.
2. **CLI & Environment Variable Model Configuration**:
   - Easily swap models and endpoints via `-m / --model` and `-u / --base-url`.
3. **Two-Phase Context Checkpoint Compaction & Summarization ([`compaction.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/compaction.py))**:
   - Implements the **Security & Provenance Context-Checkpoint Summarizer** contract.
   - Host-side deterministic extraction for `<EXACT_ANCHORS>` (paths, URLs, error text, commit SHAs) and `<VERBATIM_USER_MESSAGES>`.
   - Iterative delta checkpointing via `<PREVIOUS_CHECKPOINT>`.
4. **Automatic Skill Synthesis & Writing ([`skills.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/skills.py))**:
   - Reflects on successful trajectories and automatically synthesizes newly learned engineering procedures into `.agent_skills/`.
5. **Interactive Review for Every System Command ([`agent.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/agent.py))**:
   - Every terminal command pauses for user approval: Run (`[Enter]/y`), Deny (`n`), Edit (`e`), or Steer with feedback text.
6. **Stateful Terminal Engine ([`terminal.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/terminal.py))**:
   - `cd <dir>` and working directory state persist across tool calls.
7. **Dual-Protocol Tool Execution ([`protocol.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/protocol.py))**:
   - Supports both standard OpenAI JSON tool calling and Hermes XML ChatML (`<tool_call>...`).
8. **Trajectory Logger ([`storage.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/storage.py))**:
   - SQLite `.agent_history.db` + JSONL export for dataset creation.

---

## 🚀 Running with Qwen-32B Locally

### 1. With Ollama (Default: `http://localhost:11434/v1`)
```bash
# Start your local model in Ollama:
ollama run qwen2.5:32b
# or: ollama run qwen2.5-coder:32b

# Launch the harness:
python agent.py --model qwen2.5:32b
```

### 2. With vLLM (Default: `http://localhost:8000/v1`)
```bash
# Launch vLLM server:
vllm serve Qwen/Qwen2.5-32B-Instruct --port 8000

# Launch the harness:
python agent.py --model Qwen/Qwen2.5-32B-Instruct --base-url http://localhost:8000/v1
```

### 3. With LM Studio (Default: `http://localhost:1234/v1`)
```bash
python agent.py --model Qwen-32b --base-url http://localhost:1234/v1
```

### 4. With llama.cpp server (Default: `http://localhost:8080/v1`)
```bash
python agent.py --model Qwen-32b --base-url http://localhost:8080/v1
```

---

## ⚙️ CLI Arguments

| Argument | Shorthand | Default | Description |
| :--- | :--- | :--- | :--- |
| `--model` | `-m` | `Qwen-32b` | Model name or tag |
| `--base-url` | `-u` | `http://localhost:11434/v1` | LLM HTTP API endpoint |
| `--api-key` | `-k` | `local` | API Key (optional for local models) |
| `--xml` | | `False` | Use Hermes XML `<tool_call>` protocol |
| `--max-tokens` | | `40960` | Context capacity limit (40K / 40,960 tokens for Qwen-32B) |
| `--no-auto-skills`| | `False` | Disable automatic post-task skill synthesis |

---

## 💬 Interactive In-Session Commands

- `/context` — Show current token count and percentage of maximum context budget.
- `/compact` — Force an immediate context compaction pass.
- `/skills` — List all learned and saved skills in the repository.
- `export-trajectory [filename.jsonl]` — Export current session steps to JSONL.
- `exit` or `quit` — Exit the session.
