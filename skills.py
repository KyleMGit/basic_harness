"""
Hermes-inspired Persistent Skill Store & Automatic Skill Synthesis.
Allows the agent to record reusable procedures, scripts, and learned patterns
into persistent storage automatically upon successful task completions.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional


class SkillStore:
    """
    Manages a local library of skills and reusable workflows.
    Stored as markdown / JSON files in the workspace or user directory.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = os.path.abspath(storage_dir or os.path.join(os.getcwd(), ".agent_skills"))
        os.makedirs(self.storage_dir, exist_ok=True)

    def save_skill(self, name: str, description: str, instructions: str, tags: Optional[List[str]] = None) -> str:
        """Save a new skill or update an existing one."""
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).lower()
        file_path = os.path.join(self.storage_dir, f"{safe_name}.json")

        skill_data = {
            "name": name,
            "description": description,
            "instructions": instructions,
            "tags": tags or [],
        }

        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(skill_data, f, indent=2)
            return f"Skill '{name}' successfully saved to '{file_path}'."
        except Exception as e:
            return f"Error saving skill '{name}': {str(e)}"

    def load_skill(self, name: str) -> str:
        """Load and read instructions for a specific skill."""
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).lower()
        file_path = os.path.join(self.storage_dir, f"{safe_name}.json")

        if not os.path.exists(file_path):
            return f"Skill '{name}' not found."

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return f"=== SKILL: {data.get('name')} ===\nDescription: {data.get('description')}\n\nInstructions:\n{data.get('instructions')}"
        except Exception as e:
            return f"Error loading skill '{name}': {str(e)}"

    def list_skills(self) -> str:
        """List all available skills with summaries."""
        if not os.path.exists(self.storage_dir):
            return "No skills saved yet."

        skills = []
        for filename in os.listdir(self.storage_dir):
            if filename.endswith(".json"):
                path = os.path.join(self.storage_dir, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        skills.append(f"- **{data.get('name')}**: {data.get('description')}")
                except Exception:
                    continue

        if not skills:
            return "No skills found in skill repository."

        return "Available Learned Skills:\n" + "\n".join(skills)


class AutoSkillExtractor:
    """
    Analyzes finished agent trajectories to automatically synthesize
    and save reusable procedures into the SkillStore.
    """

    REFLECTION_PROMPT = """You are an autonomous AI skill synthesis and learning engine.
Review the completed conversation and determine if a reusable, non-trivial engineering procedure, debugging method, setup recipe, or script pattern was discovered or created.

Criteria for creating a skill:
- Multi-step build, testing, or environment configuration sequence.
- Diagnostic pattern or error resolution technique that could apply to future tasks.
- Reusable script or automation pattern.

Do NOT create a skill for:
- Simple one-off questions, basic chit-chat, or trivial commands (e.g. echo, ls).
- Failed tasks or trivial non-reusable edits.

Respond ONLY with a JSON object in this format:
If a skill should be created:
{
  "should_save": true,
  "name": "concise_snake_case_name",
  "description": "Clear 1-2 sentence description of what this skill does and when to use it.",
  "instructions": "Markdown formatted step-by-step procedure, code template, commands, and caveats."
}

If no skill should be created:
{
  "should_save": false
}
"""

    def __init__(self, skill_store: SkillStore):
        self.skill_store = skill_store

    def extract_and_save(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        task_summary: str
    ) -> Optional[Dict[str, str]]:
        """
        Runs an evaluation pass over the trajectory to automatically extract and save a skill.
        Returns the skill metadata dict if a skill was saved, otherwise None.
        """
        # Skip if session is too short (fewer than 3 messages)
        if len(messages) < 4:
            return None

        # Build transcript excerpt
        transcript_parts = [f"Task: {task_summary}\n"]
        for msg in messages:
            role = msg.get("role", "").upper()
            content = msg.get("content") or ""
            if msg.get("tool_calls"):
                content += f"\n[Tool Calls: {json.dumps(msg.get('tool_calls'))}]"
            transcript_parts.append(f"[{role}]:\n{content}")

        transcript_text = "\n\n".join(transcript_parts)
        # Limit transcript size for extraction prompt
        if len(transcript_text) > 8000:
            transcript_text = transcript_text[:4000] + "\n...[TRUNCATED]...\n" + transcript_text[-4000:]

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.REFLECTION_PROMPT},
                    {"role": "user", "content": f"Analyze this session and extract a skill if applicable:\n\n{transcript_text}"}
                ],
                temperature=0.1,
            )

            raw_resp = response.choices[0].message.content or ""
            # Extract JSON block
            json_match = re.search(r"\{.*\}", raw_resp, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group(0))
            if data.get("should_save") and data.get("name") and data.get("instructions"):
                name = data["name"]
                desc = data.get("description", "Learned procedure.")
                instr = data["instructions"]

                self.skill_store.save_skill(name, desc, instr)
                return {"name": name, "description": desc}

        except Exception as e:
            # Silent fallback if reflection fails
            return None

        return None
