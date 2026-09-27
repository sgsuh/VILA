"""Prompt-based tool calling in the Hermes format used by Qwen2.5 chat templates.

Tool definitions are injected into the system prompt, the model answers with
`<tool_call>{"name": ..., "arguments": ...}</tool_call>` blocks, and tool results are fed
back in `<tool_response>` blocks.
"""

import json
import re
import uuid
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

__all__ = [
    "TOOL_RESPONSE_TAG",
    "build_tools_system_prompt",
    "format_tool_call",
    "format_tool_response",
    "parse_tool_calls",
]

TOOL_RESPONSE_TAG = "<tool_response>"
# A missing closing tag is tolerated (e.g. generation cut off by max_tokens).
TOOL_CALL_PATTERN = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.DOTALL)
CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def build_tools_system_prompt(tools: Sequence[Dict[str, Any]], required: Union[bool, str] = False) -> str:
    """`required` is True to force some tool call, or a function name to force that function."""
    lines = [
        "# Tools",
        "",
        "You may call one or more functions to assist with the user query.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
        *(json.dumps(tool, ensure_ascii=False) for tool in tools),
        "</tools>",
        "",
        "For each function call, return a json object with function name and arguments "
        "within <tool_call></tool_call> XML tags:",
        "<tool_call>",
        '{"name": <function-name>, "arguments": <args-json-object>}',
        "</tool_call>",
        "",
        # A concrete example markedly improves JSON validity for small / 4-bit quantized models.
        "Example:",
        "<tool_call>",
        '{"name": "function_name", "arguments": {"argument_name": "value"}}',
        "</tool_call>",
    ]
    if isinstance(required, str):
        lines += ["", f"You must call the function {required}."]
    elif required:
        lines += ["", "You must call at least one function."]
    return "\n".join(lines)


def format_tool_call(name: str, arguments: Union[str, Dict[str, Any]]) -> str:
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            pass
    return "<tool_call>\n" + json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False) + "\n</tool_call>"


def format_tool_response(content: str) -> str:
    return f"{TOOL_RESPONSE_TAG}\n{content}\n</tool_response>"


def _loads_lenient(text: str) -> Any:
    """json.loads that also accepts objects missing a few closing braces, a common small-model slip."""
    for missing in range(3):
        try:
            return json.loads(text + "}" * missing)
        except json.JSONDecodeError:
            continue
    return None


def _parse_call(text: str, tool_names: Sequence[str]) -> Optional[Dict[str, Any]]:
    match = CODE_FENCE_PATTERN.match(text.strip())
    if match:
        text = match.group(1)
    call = _loads_lenient(text.strip())
    if not isinstance(call, dict) or call.get("name") not in tool_names:
        return None
    arguments = call.get("arguments", call.get("parameters", {}))
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments, ensure_ascii=False)
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {"name": call["name"], "arguments": arguments},
    }


def parse_tool_calls(text: str, tool_names: Sequence[str]) -> Tuple[str, List[Dict[str, Any]]]:
    """Split model output into (remaining text content, OpenAI-style tool calls)."""
    calls = [call for m in TOOL_CALL_PATTERN.finditer(text) if (call := _parse_call(m.group(1), tool_names))]
    if calls:
        return TOOL_CALL_PATTERN.sub("", text).strip(), calls
    # Small models sometimes emit the bare JSON object without the tags.
    call = _parse_call(text, tool_names)
    if call is not None:
        return "", [call]
    return text, []
