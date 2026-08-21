"""
Hermes and OpenAI Tool Protocol Parser.
Handles:
1. Standard OpenAI function calling format.
2. Hermes XML format (<tools> in system prompt, <tool_call> in assistant response, <tool_response> in tool response).
"""

import json
import re
from typing import Any, Dict, List, Optional, Tuple


class ToolProtocol:
    """Parses tool calls from raw model text or structured API objects."""

    HERMES_TOOL_CALL_REGEX = re.compile(
        r"<tool_call>\s*(.*?)\s*</tool_call>",
        re.DOTALL
    )

    @classmethod
    def format_hermes_system_prompt(cls, base_prompt: str, tool_schemas: List[Dict[str, Any]]) -> str:
        """Embed tool definitions in Hermes <tools> XML tags."""
        tools_json = json.dumps(tool_schemas, indent=2)
        return (
            f"{base_prompt}\n\n"
            f"# Tool Definitions\n"
            f"You have access to the following tools:\n"
            f"<tools>\n{tools_json}\n</tools>\n\n"
            f"To call a tool, respond with an XML block specifying the exact tool name from <tools>, for example:\n"
            f"<tool_call>\n"
            f'{{"name": "read_file", "arguments": {{"file_path": "README.md"}}}}\n'
            f"</tool_call>\n"
        )

    @classmethod
    def extract_tool_calls(cls, message_obj: Any) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Extract content and tool calls from either:
        1. Native OpenAI message object with `tool_calls`
        2. Raw string text containing Hermes `<tool_call>...</tool_call>`
        """
        tool_calls = []
        content = getattr(message_obj, "content", None) or (message_obj if isinstance(message_obj, str) else "")

        # 1. Native API tool calls
        if hasattr(message_obj, "tool_calls") and message_obj.tool_calls:
            for idx, tc in enumerate(message_obj.tool_calls):
                fn_name = tc.function.name
                raw_args = tc.function.arguments
                try:
                    args = json.loads(raw_args)
                    if not isinstance(args, dict):
                        raise ValueError("arguments must decode to an object")
                except Exception as exc:
                    tool_calls.append({"id": tc.id or f"call_{idx}", "name": "__protocol_error__",
                                       "arguments": {"error": f"Malformed arguments for {fn_name}: {exc}"}})
                    continue
                tool_calls.append({
                    "id": tc.id or f"call_{idx}",
                    "name": fn_name,
                    "arguments": args
                })
            return content, tool_calls

        # 2. Hermes XML parsing from text (<tool_call>...</tool_call>)
        if content:
            matches = list(cls.HERMES_TOOL_CALL_REGEX.finditer(content))
            if matches:
                clean_text = content
                for idx, match in enumerate(matches):
                    raw_json = match.group(1).strip()
                    try:
                        parsed = json.loads(raw_json)
                        fn_name = parsed.get("name")
                        args = parsed.get("arguments", {})
                        if isinstance(args, str):
                            args = json.loads(args)
                        if fn_name:
                            tool_calls.append({
                                "id": f"hermes_call_{idx}",
                                "name": fn_name,
                                "arguments": args
                            })
                    except Exception as exc:
                        tool_calls.append({"id": f"hermes_call_{idx}", "name": "__protocol_error__",
                                           "arguments": {"error": f"Malformed Hermes tool call: {exc}"}})
                
                clean_text = cls.HERMES_TOOL_CALL_REGEX.sub("", content).strip()
                return clean_text, tool_calls

        return content, []

    @classmethod
    def format_hermes_tool_response(
        cls, tool_name: str, content: str, tool_call_id: Optional[str] = None
    ) -> str:
        """Format tool result for Hermes ChatML <tool_response>."""
        payload_data = {"name": tool_name, "content": content}
        if tool_call_id:
            payload_data["tool_call_id"] = tool_call_id
        payload = json.dumps(payload_data)
        return f"<tool_response>\n{payload}\n</tool_response>"
