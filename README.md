# Hermes-Refined Coding Agent Harness

An open-source, terminal-native Python agent harness inspired by [**NousResearch/hermes-agent**](https://github.com/nousresearch/hermes-agent).

---

## 🌟 Key Architecture & Context Practices

### 1. Two-Phase Context Compaction & Summarization ([`compaction.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/compaction.py))
- **Phase 1: Tool Output Pruning**: Verbose outputs from older tool calls (such as large terminal logs or multi-page file reads) are compressed in history to compact stubs while preserving exit codes and head/tail snippets.
- **Phase 2: LLM Conversation Compaction**: When token budget crosses the configured threshold (default 65%), older turns are summarized into a structured `[CONVERSATION COMPACTION BLOCK]` preserving the user goal, modified files, terminal command history, and pending tasks.
- **Anti-Thrashing Cooldown**: Enforces cooldown step intervals between compaction passes to prevent infinite compression loops.
- **Manual `/compact` & `/context`**: Inspect capacity or force compaction on demand.

### 2. Interactive Terminal Review & Human-in-the-Loop ([`agent.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/agent.py))
- Every terminal command pauses for user approval.
- Allows executing (`[Enter]/y`), denying (`n`), editing the command (`e`), or sending steering feedback text.

### 3. Stateful Terminal Engine ([`terminal.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/terminal.py))
- Maintains persistent working directory (`cwd`) and environment variables across multiple tool calls (`cd` persists across steps).
- Automatic detection of destructive commands.

### 4. Dual Protocol Tool Calling ([`protocol.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/protocol.py))
- Supports standard OpenAI function calling and Hermes XML ChatML (`<tools>`, `<tool_call>`, `<tool_response>`).

### 5. Persistent Skill Store ([`skills.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/skills.py))
- Hermes-inspired self-improving skill library (`save_skill`, `load_skill`, `list_skills`).

### 6. Trajectory Storage ([`storage.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/storage.py))
- Records complete execution graphs and step metadata to SQLite (`.agent_history.db`) with JSONL export support.

---

## 📁 Repository Structure

- [`agent.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/agent.py): ReAct orchestrator, interactive CLI, and safety loops.
- [`compaction.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/compaction.py): Two-phase context pruning and LLM summarization engine.
- [`terminal.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/terminal.py): Stateful terminal execution session with safety filters.
- [`tools.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/tools.py): Tool registry with terminal, file patch/read/write, and skill tools.
- [`skills.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/skills.py): Hermes persistent skill store.
- [`protocol.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/protocol.py): Dual protocol parser (JSON function calling + Hermes XML).
- [`storage.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/storage.py): SQLite trajectory logger & JSONL exporter.
- [`test_tools.py`](file:///C:/Users/Owner/.gemini/antigravity/scratch/coding_agent/test_tools.py): Comprehensive unit tests.

---

## 🚀 Interactive Commands

Inside the agent CLI:
- `/context` — Show current token count and percentage of maximum context budget.
- `/compact` — Force an immediate context compaction pass.
- `export-trajectory [filename.jsonl]` — Export current session steps to JSONL.
- `exit` or `quit` — Exit the session.
