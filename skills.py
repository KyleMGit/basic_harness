"""
Hermes-inspired Persistent Skill Store, Semantic/Keyword Retrieval,
and Catalog-Aware Deduplicating Skill Synthesis.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple


class SkillStore:
    """
    Manages a persistent library of skills and reusable workflows.
    Provides indexing, keyword/overlap retrieval, deduplication, and catalog formatting.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = os.path.abspath(storage_dir or os.path.join(os.getcwd(), ".agent_skills"))
        os.makedirs(self.storage_dir, exist_ok=True)

    def _safe_filename(self, name: str) -> str:
        safe_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name).strip("_").lower()
        return f"{safe_name}.json"

    def save_skill(self, name: str, description: str, instructions: str, tags: Optional[List[str]] = None) -> str:
        """Save a new skill or update an existing one."""
        file_path = os.path.join(self.storage_dir, self._safe_filename(name))
        
        skill_data = {
            "name": name.strip(),
            "description": description.strip(),
            "instructions": instructions.strip(),
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
        file_path = os.path.join(self.storage_dir, self._safe_filename(name))

        if not os.path.exists(file_path):
            return f"Skill '{name}' not found."

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return f"=== SKILL: {data.get('name')} ===\nDescription: {data.get('description')}\n\nInstructions:\n{data.get('instructions')}"
        except Exception as e:
            return f"Error loading skill '{name}': {str(e)}"

    def get_all_skills(self) -> List[Dict[str, Any]]:
        """Retrieve all stored skills with full contents."""
        skills = []
        if not os.path.exists(self.storage_dir):
            return skills

        for filename in sorted(os.listdir(self.storage_dir)):
            if filename.endswith(".json"):
                path = os.path.join(self.storage_dir, filename)
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        if "name" in data and "instructions" in data:
                            skills.append(data)
                except Exception:
                    continue
        return skills

    def get_skills_index(self) -> List[Dict[str, str]]:
        """Retrieve a lightweight catalog index of all available skills."""
        return [
            {
                "name": s["name"],
                "description": s.get("description", ""),
                "tags": ", ".join(s.get("tags", []))
            }
            for s in self.get_all_skills()
        ]

    def format_catalog_prompt(self) -> str:
        """Format available skills into a prompt-friendly catalog."""
        skills = self.get_skills_index()
        if not skills:
            return "<available_skills>\nNone stored yet.\n</available_skills>"

        lines = ["<available_skills>"]
        for s in skills:
            lines.append(f'  - skill: "{s["name"]}"')
            lines.append(f'    when_to_use: "{s["description"]}"')
        lines.append("</available_skills>")
        return "\n".join(lines)

    def find_relevant_skills(self, query: str, top_k: int = 2, threshold: float = 0.10) -> List[Dict[str, Any]]:
        """
        Find skills relevant to a task query using token overlap and keyword matching.
        """
        all_skills = self.get_all_skills()
        if not all_skills:
            return []

        def tokenize(text: str) -> Set[str]:
            # Replace underscores and hyphens with spaces to extract constituent subwords
            clean_text = re.sub(r"[_\-/\\]", " ", text.lower())
            words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", clean_text))
            # Also keep exact original tokens
            words.update(re.findall(r"\b[a-zA-Z0-9_\-]{3,}\b", text.lower()))
            return words

        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        scored_skills = []
        for skill in all_skills:
            skill_text = f"{skill['name']} {skill.get('description', '')} {' '.join(skill.get('tags', []))}"
            skill_tokens = tokenize(skill_text)
            
            if not skill_tokens:
                continue

            intersection = query_tokens.intersection(skill_tokens)
            score = len(intersection) / len(query_tokens.union(skill_tokens))
            
            # Boost score if query words match the skill name subwords directly
            name_tokens = tokenize(skill["name"])
            if query_tokens.intersection(name_tokens):
                score += 0.35

            if score >= threshold:
                scored_skills.append((score, skill))

        # Sort descending by score
        scored_skills.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored_skills[:top_k]]

    def list_skills(self) -> str:
        """List all available skills with summaries."""
        skills = self.get_all_skills()
        if not skills:
            return "No skills found in skill repository (.agent_skills/)."

        output = ["Available Learned Skills:"]
        for s in skills:
            output.append(f"- **{s['name']}**: {s.get('description', 'No description')}")
        return "\n".join(output)

    def delete_skill(self, name: str) -> bool:
        """Delete a skill by name."""
        file_path = os.path.join(self.storage_dir, self._safe_filename(name))
        if os.path.exists(file_path):
            os.remove(file_path)
            return True
        return False


class AutoSkillExtractor:
    """
    Catalog-aware skill synthesis & curation engine.
    Analyzes finished trajectories against existing skills to:
    1. Avoid duplicate/redundant skills.
    2. Merge & update existing skills when better techniques are discovered.
    3. Synthesize novel skills only when genuinely new capabilities are formed.
    """

    REFLECTION_PROMPT = """You are an autonomous AI Skill Curator and Synthesis Engine.
Review the completed conversation trajectory against the existing library of skills.

Your goal is to maintain a high-quality, non-redundant, durable library of engineering skills.

=== EXISTING SKILLS IN REPOSITORY ===
{existing_skills_catalog}
=====================================

### Instructions:
1. **DEDUPLICATION RULE**: Carefully check if the completed procedure is already covered by an existing skill (even if worded differently).
   - If an existing skill already covers this: choose action "NONE".
   - If an existing skill covers this BUT the current session discovered a better method, fixed a bug, or added important edge cases: choose action "UPDATE" and refine that specific skill.
   - Only choose action "CREATE" if this is a genuinely novel, non-trivial, reusable workflow not represented in the repository.
2. **QUALITY RULE**: Do NOT save skills for trivial one-off tasks (e.g. 'echo', simple questions, basic file viewing, or failed attempts).
3. **INSTRUCTION QUALITY**: Instructions must be concrete, step-by-step markdown with exact commands, code snippets, and configuration caveats.

Respond ONLY with a JSON object in this format:
{
  "action": "CREATE" | "UPDATE" | "NONE",
  "target_skill_name": "name_of_existing_skill_to_update_or_blank",
  "name": "concise_snake_case_name",
  "description": "Clear explanation of what this skill accomplishes and when to use it.",
  "instructions": "Step-by-step markdown instructions, commands, code patterns, and caveats."
}
"""

    def __init__(self, skill_store: SkillStore):
        self.skill_store = skill_store

    def _compute_text_similarity(self, text_a: str, text_b: str) -> float:
        words_a = set(re.findall(r"\b\w{3,}\b", text_a.lower()))
        words_b = set(re.findall(r"\b\w{3,}\b", text_b.lower()))
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def extract_and_save(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        task_summary: str
    ) -> Optional[Dict[str, str]]:
        """
        Runs a catalog-aware evaluation pass over the trajectory to synthesize or refine skills.
        """
        # Skip trivial or empty sessions
        if len(messages) < 4:
            return None

        # Build transcript excerpt
        transcript_parts = [f"User Goal: {task_summary}\n"]
        for msg in messages:
            role = msg.get("role", "").upper()
            content = str(msg.get("content") or "")
            if len(content) > 800:
                content = content[:400] + "\n...[TRUNCATED]...\n" + content[-300:]
            if msg.get("tool_calls"):
                content += f"\n[Tool Calls: {json.dumps(msg.get('tool_calls'))}]"
            transcript_parts.append(f"[{role}]:\n{content}")

        transcript_text = "\n\n".join(transcript_parts)
        if len(transcript_text) > 8000:
            transcript_text = transcript_text[:4000] + "\n...[TRUNCATED]...\n" + transcript_text[-4000:]

        existing_catalog = self.skill_store.format_catalog_prompt()
        prompt = self.REFLECTION_PROMPT.replace("{existing_skills_catalog}", existing_catalog)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": f"Evaluate and curate skills for this session:\n\n{transcript_text}"}
                ],
                temperature=0.1,
            )

            raw_resp = response.choices[0].message.content or ""
            json_match = re.search(r"\{.*\}", raw_resp, re.DOTALL)
            if not json_match:
                return None

            data = json.loads(json_match.group(0))
            action = data.get("action", "").upper()

            if action in ("CREATE", "UPDATE"):
                name = (data.get("target_skill_name") if action == "UPDATE" and data.get("target_skill_name") else data.get("name", "")).strip()
                desc = data.get("description", "").strip()
                instr = data.get("instructions", "").strip()

                if not name or not instr:
                    return None

                # Additional host-side deduplication guard:
                # If creating, check if an existing skill has very high similarity (> 0.55)
                if action == "CREATE":
                    for existing in self.skill_store.get_all_skills():
                        sim = self._compute_text_similarity(
                            f"{name} {desc}",
                            f"{existing['name']} {existing.get('description', '')}"
                        )
                        if sim > 0.55:
                            # Re-route to updating the existing skill rather than creating duplicate
                            name = existing["name"]
                            action = "UPDATE"
                            break

                self.skill_store.save_skill(name, desc, instr)
                return {
                    "action": action,
                    "name": name,
                    "description": desc
                }

        except Exception:
            return None

        return None
