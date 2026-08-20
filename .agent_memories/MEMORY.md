# Project Memory & Architecture Facts (MEMORY.md)

## Codebase Architecture & Tech Stack
- Primary Language: Python
- Key Modules: Terminal session engine, Context compaction summarizer, Hermes skill repository, SQLite trajectory logger.

## Environment & Configuration
- Workspace: Local project repository
- LLM Protocol: OpenAI JSON Tool Calling & Hermes XML ChatML

## Key Patterns & Conventions
- Unit Tests: Standard Python unittest suite in test_tools.py
- Skill Storage: Native Markdown (.md) with YAML frontmatter in .agent_skills/

## Known Gotchas & Resolved Issues
- Local LLM Token Limits: Always use context compaction threshold to stay safely within context limits.
- Process State: Working directory (cwd) persists across tool calls via stateful terminal session.
