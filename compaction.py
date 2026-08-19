"""
Hermes-inspired Context Management, Summarization, and Compaction Engine.
Implements:
1. Two-phase compaction (Phase 1: Tool output pruning, Phase 2: LLM conversation summarization).
2. Token budget monitoring and trigger thresholds (e.g. 50-75% budget).
3. Anti-thrashing guards to prevent repetitive compaction loops.
4. Manual '/compact' trigger.
"""

import json
from typing import Any, Dict, List, Optional, Tuple


class ContextManager:
    """
    Manages LLM context window health via progressive pruning and compaction.
    """

    COMPACTION_SYSTEM_PROMPT = """You are a context compression expert for an autonomous coding agent.
Your task is to produce a structured, high-density summary of the conversation history provided below.

Your summary MUST preserve:
1. **Core Goal & Constraints**: What task the user requested and specific constraints.
2. **Files & Environment**: Files inspected, created, edited, or deleted, and current working directory.
3. **Execution History**: Crucial terminal commands run, their outcomes, and key errors diagnosed/fixed.
4. **Current State & Next Steps**: What has been completed and what remains to be done.

Be concise, dense, and objective. Avoid pleasantries.
Format your output with clear markdown headings.
"""

    def __init__(
        self,
        max_context_tokens: int = 16000,
        trigger_threshold: float = 0.65,
        keep_recent_turns: int = 6,
        cooldown_steps: int = 3,
    ):
        self.max_context_tokens = max_context_tokens
        self.trigger_threshold = trigger_threshold
        self.keep_recent_turns = keep_recent_turns
        self.cooldown_steps = cooldown_steps
        self.last_compaction_step = -100
        self.compaction_count = 0

    @staticmethod
    def estimate_tokens(messages: List[Dict[str, Any]]) -> int:
        """
        Fast token estimation based on character count and structured payloads.
        (~4 chars per token rule of thumb).
        """
        char_count = 0
        for msg in messages:
            content = msg.get("content") or ""
            char_count += len(str(content))
            
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                char_count += len(json.dumps(tool_calls))
                
        return max(1, char_count // 4)

    def prune_tool_outputs(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Phase 1 Compaction: Prune verbose tool outputs from older turns
        while preserving the most recent turns verbatim.
        """
        if len(messages) <= self.keep_recent_turns + 1:
            return messages

        cutoff_idx = max(1, len(messages) - self.keep_recent_turns)
        pruned_messages = []

        for idx, msg in enumerate(messages):
            # System prompt and recent turns are kept untouched
            if idx == 0 or idx >= cutoff_idx:
                pruned_messages.append(msg)
                continue

            role = msg.get("role")
            content = str(msg.get("content") or "")

            # If it's a tool output message or hermes tool response with large output
            if role == "tool" and len(content) > 300:
                head = content[:120].strip()
                tail = content[-80:].strip()
                summary_content = f"{head}\n\n... [PRUNED TOOL OUTPUT ({len(content)} chars) - Exit details retained] ...\n\n{tail}"
                
                new_msg = dict(msg)
                new_msg["content"] = summary_content
                pruned_messages.append(new_msg)
            elif "<tool_response>" in content and len(content) > 350:
                head = content[:150].strip()
                tail = content[-100:].strip()
                summary_content = f"{head}\n\n... [PRUNED TOOL OUTPUT] ...\n\n{tail}"
                new_msg = dict(msg)
                new_msg["content"] = summary_content
                pruned_messages.append(new_msg)
            else:
                pruned_messages.append(msg)

        return pruned_messages

    def summarize_history(
        self,
        client: Any,
        model: str,
        messages_to_summarize: List[Dict[str, Any]]
    ) -> str:
        """
        Generate a dense structured summary of older messages using an auxiliary prompt.
        """
        transcript_lines = []
        for msg in messages_to_summarize:
            role = msg.get("role", "unknown").upper()
            content = msg.get("content") or ""
            if msg.get("tool_calls"):
                content += f"\n[Tool Calls: {json.dumps(msg.get('tool_calls'))}]"
            transcript_lines.append(f"### {role}:\n{content}")

        history_transcript = "\n\n".join(transcript_lines)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": self.COMPACTION_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Summarize this conversation segment:\n\n{history_transcript}"}
                ],
                temperature=0.2,
            )
            summary = response.choices[0].message.content or "Summary generation returned empty."
            return summary.strip()
        except Exception as e:
            # Fallback heuristic summary if API call fails
            return f"[Automatic Compaction Fallback]: Previous {len(messages_to_summarize)} turns compressed. (Summary API unavailable: {str(e)})"

    def compact(
        self,
        client: Any,
        model: str,
        messages: List[Dict[str, Any]],
        current_step: int,
        force: bool = False
    ) -> Tuple[List[Dict[str, Any]], bool, str]:
        """
        Execute context compaction if triggered by budget threshold or force flag.
        Returns: (compacted_messages, was_compacted, status_message)
        """
        initial_tokens = self.estimate_tokens(messages)
        trigger_limit = int(self.max_context_tokens * self.trigger_threshold)

        # Check if compaction is needed
        if not force and initial_tokens < trigger_limit:
            return messages, False, f"Context healthy ({initial_tokens}/{self.max_context_tokens} est. tokens)."

        # Anti-thrashing check
        if not force and (current_step - self.last_compaction_step) < self.cooldown_steps:
            return messages, False, f"Compaction in cooldown ({current_step - self.last_compaction_step}/{self.cooldown_steps} steps)."

        # Phase 1: Tool Output Pruning
        pruned_messages = self.prune_tool_outputs(messages)
        pruned_tokens = self.estimate_tokens(pruned_messages)

        # If Phase 1 pruned enough below threshold, return without full LLM summarization
        if not force and pruned_tokens < trigger_limit:
            self.last_compaction_step = current_step
            saved = initial_tokens - pruned_tokens
            return pruned_messages, True, f"[Phase 1 Compaction] Pruned tool outputs, saved ~{saved} tokens ({pruned_tokens}/{self.max_context_tokens})."

        # Phase 2: Full Conversation Compaction & Summarization
        if len(pruned_messages) <= self.keep_recent_turns + 2:
            return pruned_messages, False, "Not enough message depth to compact."

        system_msg = pruned_messages[0]
        older_messages = pruned_messages[1:-self.keep_recent_turns]
        recent_messages = pruned_messages[-self.keep_recent_turns:]

        # Generate summary of older messages
        summary = self.summarize_history(client, model, older_messages)

        compaction_block = {
            "role": "user",
            "content": (
                f"=== [CONVERSATION COMPACTION BLOCK #{self.compaction_count + 1}] ===\n"
                f"The following is a condensed summary of the prior {len(older_messages)} turns:\n\n"
                f"{summary}\n"
                f"============================================================"
            )
        }
        ack_block = {
            "role": "assistant",
            "content": "Understood. I have integrated the historical context summary and will continue the task from the current state."
        }

        compacted = [system_msg, compaction_block, ack_block] + recent_messages
        final_tokens = self.estimate_tokens(compacted)
        
        self.compaction_count += 1
        self.last_compaction_step = current_step
        saved = initial_tokens - final_tokens

        msg = f"[Phase 2 Compaction Complete] Compressed {len(older_messages)} turns into summary. Saved ~{saved} tokens ({final_tokens}/{self.max_context_tokens})."
        return compacted, True, msg
