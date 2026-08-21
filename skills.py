"""
Hermes-inspired Persistent Skill Store, Semantic/Keyword Retrieval,
and Catalog-Aware Deduplicating Skill Synthesis.
Supports native Markdown (SKILL.md, .md with YAML frontmatter) and JSON format.
"""

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from safety import screen_prompt_content


class SkillStore:
    """
    Manages a persistent library of skills and reusable workflows.
    Supports native Hermes Markdown (.md, SKILL.md with YAML frontmatter)
    and JSON formats interchangeably.
    """

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = os.path.abspath(storage_dir or os.path.join(os.getcwd(), ".agent_skills"))

    def _safe_name(self, name: str) -> str:
        base = os.path.basename(name)
        for ext in (".md", ".json", ".yaml", ".yml"):
            if base.lower().endswith(ext):
                base = base[:-len(ext)]
        return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in base).strip("_").lower()

    @staticmethod
    def parse_markdown_skill(content: str, default_name: str = "") -> Dict[str, Any]:
        """Parse Markdown file with optional YAML frontmatter into a skill dictionary."""
        name = default_name
        description = ""
        tags = []
        instructions = content.strip()

        # Check for YAML frontmatter block (--- ... ---)
        fm_match = re.match(r"^---\s*\n(.*?)\n---\s*\n?(.*)$", content, re.DOTALL)
        if fm_match:
            frontmatter_text = fm_match.group(1)
            instructions = fm_match.group(2).strip()

            for line in frontmatter_text.splitlines():
                line = line.strip()
                if line.startswith("name:"):
                    name = line.split(":", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("description:"):
                    description = line.split(":", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("tags:"):
                    raw_tags = line.split(":", 1)[1].strip().strip("[]")
                    tags = [t.strip().strip('"').strip("'") for t in raw_tags.split(",") if t.strip()]

        # If description was not in frontmatter, extract from ## Description section
        if not description:
            desc_match = re.search(r"##\s*Description\s*\n+([^\n#]+)", instructions, re.IGNORECASE)
            if desc_match:
                description = desc_match.group(1).strip()

        # If instructions contain ## Instructions header, keep clean body
        return {
            "name": name or default_name or "unnamed_skill",
            "description": description or f"Skill: {name or default_name}",
            "instructions": instructions,
            "tags": tags,
        }

    @staticmethod
    def format_markdown_skill(name: str, description: str, instructions: str, tags: Optional[List[str]] = None) -> str:
        """Format skill data into standard Hermes Markdown with YAML frontmatter."""
        tags_str = f"[{', '.join(tags)}]" if tags else "[]"
        return f"""---
name: {name}
description: {description}
tags: {tags_str}
---

# {name}

## Description
{description}

## Instructions
{instructions}
"""

    def resolve_skill_file(self, name: str) -> Optional[str]:
        """
        Locate the skill file across various naming and extension conventions (.md, .json, SKILL.md).
        """
        safe = self._safe_name(name)
        candidates = [
            os.path.join(self.storage_dir, f"{safe}.md"),
            os.path.join(self.storage_dir, safe, "SKILL.md"),
            os.path.join(self.storage_dir, f"{safe}.json"),
        ]

        root_real = os.path.realpath(self.storage_dir)
        for walk_root, dirs, files in os.walk(self.storage_dir, followlinks=False):
            dirs[:] = [d for d in dirs if os.path.commonpath((root_real, os.path.realpath(os.path.join(walk_root, d)))) == root_real]
            if "SKILL.md" in files:
                candidate = os.path.join(walk_root, "SKILL.md")
                try:
                    with open(candidate, "r", encoding="utf-8") as handle:
                        content = handle.read()
                    parsed = self.parse_markdown_skill(content, default_name=os.path.basename(walk_root))
                    if self._safe_name(parsed.get("name", "")) == safe or self._safe_name(os.path.basename(walk_root)) == safe:
                        candidates.append(candidate)
                except OSError:
                    continue
        found = []
        for cand in candidates:
            if cand and os.path.isfile(cand) and os.path.commonpath((root_real, os.path.realpath(cand))) == root_real:
                if os.path.realpath(cand) not in found:
                    found.append(os.path.realpath(cand))
        if len(found) > 1:
            # Markdown/JSON siblings are one logical root skill; nested duplicates are ambiguous.
            nested = [p for p in found if os.path.basename(p).lower() == "skill.md"]
            if len(nested) > 1 or (nested and any(os.path.dirname(p) != self.storage_dir for p in found if p not in nested)):
                raise ValueError(f"Ambiguous skill name '{name}': {len(found)} confined candidates found.")
        if found:
            return next((p for p in found if p.lower().endswith(".md")), found[0])
        return None

    def save_skill(self, name: str, description: str, instructions: str, tags: Optional[List[str]] = None) -> str:
        """
        Save skill as standard Hermes Markdown (.md) and JSON for full backward & tool compatibility.
        """
        safe = self._safe_name(name)
        os.makedirs(self.storage_dir, exist_ok=True)
        md_path = os.path.join(self.storage_dir, f"{safe}.md")
        json_path = os.path.join(self.storage_dir, f"{safe}.json")

        clean_name = name.strip()
        clean_desc = description.strip()
        clean_instr = instructions.strip()
        safe_content, status = screen_prompt_content(f"{clean_name}\n{clean_desc}\n{clean_instr}")
        if not safe_content:
            return status

        # 1. Write Markdown format (.md)
        md_content = self.format_markdown_skill(clean_name, clean_desc, clean_instr, tags)
        skill_dict = {
            "name": clean_name,
            "description": clean_desc,
            "instructions": clean_instr,
            "tags": tags or [],
            "format": "markdown",
            "file": f"{safe}.md"
        }
        tmp_md = tmp_json = None
        old_md = Path(md_path).read_bytes() if os.path.exists(md_path) else None
        old_json = Path(json_path).read_bytes() if os.path.exists(json_path) else None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.storage_dir, delete=False) as f:
                tmp_md = f.name; f.write(md_content)
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.storage_dir, delete=False) as f:
                tmp_json = f.name; json.dump(skill_dict, f, indent=2)
            os.replace(tmp_md, md_path); tmp_md = None
            os.replace(tmp_json, json_path); tmp_json = None
        except Exception as e:
            for pending in (tmp_md, tmp_json):
                if pending and os.path.exists(pending):
                    os.unlink(pending)
            try:
                if old_md is None:
                    if os.path.exists(md_path): os.unlink(md_path)
                else:
                    with open(md_path, "wb") as f: f.write(old_md)
                if old_json is None:
                    if os.path.exists(json_path): os.unlink(json_path)
                else:
                    with open(json_path, "wb") as f: f.write(old_json)
            except OSError:
                pass
            return f"Error saving coherent skill '{name}': {e}"

        return f"Skill '{clean_name}' successfully saved as '{md_path}' and '{json_path}'."

    def load_skill(self, name: str) -> str:
        """Load and read instructions for a specific skill from .md or .json file."""
        try:
            file_path = self.resolve_skill_file(name)
        except ValueError as exc:
            return f"Error: {exc}"
        if not file_path:
            return f"Skill '{name}' not found in '{self.storage_dir}' (checked .md, .json, and SKILL.md)."

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            if file_path.endswith(".json"):
                data = json.loads(content)
                skill_name = data.get("name", name)
                desc = data.get("description", "")
                instr = data.get("instructions", "")
            else:
                data = self.parse_markdown_skill(content, default_name=self._safe_name(name))
                skill_name = data.get("name")
                desc = data.get("description")
                instr = data.get("instructions")

            safe_content, status = screen_prompt_content(f"{desc}\n{instr}")
            if not safe_content:
                return status

            return f"=== SKILL: {skill_name} ===\nFile: {file_path}\nDescription: {desc}\n\nInstructions:\n{instr}"
        except Exception as e:
            return f"Error loading skill from '{file_path}': {str(e)}"

    def get_all_skills(self) -> List[Dict[str, Any]]:
        """Retrieve all stored skills with full contents from .md, .json, and SKILL.md files."""
        skills_map: Dict[str, Dict[str, Any]] = {}
        if not os.path.exists(self.storage_dir):
            return []

        # 1. Scan storage directory (and subdirectories for SKILL.md)
        for root, _, files in os.walk(self.storage_dir):
            for filename in sorted(files):
                full_path = os.path.join(root, filename)
                try:
                    if os.path.commonpath((os.path.realpath(self.storage_dir), os.path.realpath(full_path))) != os.path.realpath(self.storage_dir):
                        continue
                except ValueError:
                    continue
                
                # Check markdown files (.md, SKILL.md)
                if filename.endswith(".md"):
                    try:
                        with open(full_path, "r", encoding="utf-8") as f:
                            content = f.read()
                        parsed = self.parse_markdown_skill(content, default_name=self._safe_name(filename))
                        if not screen_prompt_content(f"{parsed.get('description', '')}\n{parsed.get('instructions', '')}")[0]:
                            continue
                        norm_name = self._safe_name(parsed["name"])
                        parsed["file_path"] = full_path
                        skills_map[norm_name] = parsed
                    except Exception:
                        continue

                # Check JSON files (.json) only if not already loaded from .md
                elif filename.endswith(".json"):
                    norm_name = self._safe_name(filename)
                    if norm_name not in skills_map:
                        try:
                            with open(full_path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            if "name" in data and "instructions" in data:
                                if not screen_prompt_content(
                                    f"{data.get('description', '')}\n{data.get('instructions', '')}"
                                )[0]:
                                    continue
                                data["file_path"] = full_path
                                skills_map[norm_name] = data
                        except Exception:
                            continue

        unambiguous = []
        for norm_name in sorted(skills_map):
            try:
                self.resolve_skill_file(norm_name)
            except ValueError:
                continue
            unambiguous.append(skills_map[norm_name])
        return unambiguous

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
            clean_text = re.sub(r"[_\-/\\]", " ", text.lower())
            words = set(re.findall(r"\b[a-zA-Z0-9]{3,}\b", clean_text))
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

        scored_skills.sort(key=lambda x: x[0], reverse=True)
        return [skill for _, skill in scored_skills[:top_k]]

    def list_skills(self) -> str:
        """List all available skills with summaries and file formats."""
        skills = self.get_all_skills()
        if not skills:
            return "No skills found in skill repository (.agent_skills/)."

        output = ["Available Learned Skills:"]
        for s in skills:
            rel_file = os.path.basename(s.get("file_path", ""))
            output.append(f"- **{s['name']}** (`{rel_file}`): {s.get('description', 'No description')}")
        return "\n".join(output)

    def delete_skill(self, name: str) -> bool:
        """Delete a skill and its corresponding .md and .json files."""
        safe = self._safe_name(name)
        deleted = False
        for ext in (".md", ".json"):
            path = os.path.join(self.storage_dir, f"{safe}{ext}")
            if os.path.exists(path):
                os.remove(path)
                deleted = True
        return deleted


class AutoSkillExtractor:
    """
    Catalog-aware skill synthesis & curation engine.
    Analyzes finished trajectories against existing skills to:
    1. Avoid duplicate/redundant skills.
    2. Merge & update existing skills when better techniques are discovered.
    3. Synthesize novel skills into standard Markdown & JSON.
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
                            name = existing["name"]
                            action = "UPDATE"
                            break

                if action == "UPDATE" and self.skill_store.resolve_skill_file(name):
                    return {"action": "SKIP", "name": name, "description": "Existing skill retained; autonomous replacement requires review."}
                save_status = self.skill_store.save_skill(name, desc, instr)
                if "successfully saved" not in str(save_status).lower():
                    return {
                        "action": "ERROR",
                        "name": name,
                        "description": f"Skill was not saved: {save_status}",
                    }
                return {
                    "action": action,
                    "name": name,
                    "description": desc
                }

        except Exception:
            return None

        return None
