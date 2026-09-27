import json

from llava.utils.tool_calls import build_tools_system_prompt, format_tool_call, parse_tool_calls

TOOLS = ["get_weather", "add"]


def test_parse_tagged_calls():
    text = 'Let me check.\n<tool_call>\n{"name": "get_weather", "arguments": {"city": "Seoul"}}\n</tool_call>'
    content, calls = parse_tool_calls(text, TOOLS)
    assert content == "Let me check."
    assert len(calls) == 1
    assert calls[0]["type"] == "function"
    assert calls[0]["id"].startswith("call_")
    assert calls[0]["function"]["name"] == "get_weather"
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Seoul"}


def test_parse_multiple_and_unclosed_calls():
    text = (
        '<tool_call>{"name": "add", "arguments": {"a": 1, "b": 2}}</tool_call>\n'
        '<tool_call>{"name": "get_weather", "arguments": {"city": "Busan"}}'
    )
    _, calls = parse_tool_calls(text, TOOLS)
    assert [c["function"]["name"] for c in calls] == ["add", "get_weather"]


def test_parse_bare_json_call():
    _, calls = parse_tool_calls('```json\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n```', TOOLS)
    assert calls[0]["function"]["name"] == "add"


def test_unknown_tool_or_plain_text_is_content():
    assert parse_tool_calls('{"name": "delete_all", "arguments": {}}', TOOLS)[1] == []
    assert parse_tool_calls("The weather is sunny.", TOOLS) == ("The weather is sunny.", [])


def test_format_round_trip():
    text = format_tool_call("add", '{"a": 1, "b": 2}')
    _, calls = parse_tool_calls(text, TOOLS)
    assert json.loads(calls[0]["function"]["arguments"]) == {"a": 1, "b": 2}


def test_system_prompt_mentions_tools_and_requirement():
    tool = {"type": "function", "function": {"name": "add", "parameters": {"type": "object"}}}
    assert '"name": "add"' in build_tools_system_prompt([tool])
    assert "You must call the function add." in build_tools_system_prompt([tool], "add")
    assert "You must call at least one function." in build_tools_system_prompt([tool], True)


def test_parse_call_missing_closing_brace():
    text = '<tool_call>\n{"name": "get_weather",\n"arguments": {"city": "Seoul"}\n</tool_call>'
    _, calls = parse_tool_calls(text, TOOLS)
    assert json.loads(calls[0]["function"]["arguments"]) == {"city": "Seoul"}
