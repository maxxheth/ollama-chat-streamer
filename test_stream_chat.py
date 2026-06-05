import io
import time
from contextlib import redirect_stdout

import stream_chat


def test_render_stream_text_smoke_no_delay(monkeypatch):
    output = io.StringIO()
    log_file = io.StringIO()

    with redirect_stdout(output):
        stream_chat.render_stream_text("hello", log_file, flush_interval=2, delay=0.0)

    assert output.getvalue() == "hello"
    assert log_file.getvalue() == "hello"


def test_render_stream_text_smoke_with_delay(monkeypatch):
    output = io.StringIO()
    log_file = io.StringIO()
    sleep_calls = []

    def fake_sleep(value):
        sleep_calls.append(value)

    monkeypatch.setattr(time, "sleep", fake_sleep)

    with redirect_stdout(output):
        stream_chat.render_stream_text("hi", log_file, flush_interval=1, delay=0.01)

    assert output.getvalue() == "hi"
    assert log_file.getvalue() == "hi"
    assert sleep_calls == [0.01, 0.01]


class _ToolFunction:
    def __init__(self, name: str, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCallNoId:
    def __init__(self, name: str, arguments):
        self.function = _ToolFunction(name, arguments)


class _ToolResponse:
    def __init__(self, tool_calls):
        self.message = type("Msg", (), {"content": "", "tool_calls": tool_calls})()


def test_chat_with_tools_handles_tool_calls_without_id(monkeypatch):
    output = io.StringIO()
    log_file = io.StringIO()
    messages = []

    tool_calls = [_ToolCallNoId("web_search", {"query": "test"})]
    response = _ToolResponse(tool_calls)

    def fake_chat(*args, **kwargs):
        return response

    monkeypatch.setattr(stream_chat.ollama, "chat", fake_chat)
    monkeypatch.setattr(stream_chat, "execute_tool_call", lambda call: "tool-result")
    monkeypatch.setattr(stream_chat, "_stream_chat_with_retry", lambda **kwargs: iter([]))

    with redirect_stdout(output):
        result = stream_chat.chat_with_tools(
            model="test-model",
            messages=messages,
            tools=[],
            file_handle=log_file,
            retry_config=stream_chat.RetryConfig(1, 0.0, 0.0, 1.0, 0.0),
            render_delay=0.0,
        )

    assert result == ""
    assert any(msg.get("role") == "tool" for msg in messages)


def test_write_output_file_appends(tmp_path):
    tool_call = {
        "function": {
            "name": "write_output_file",
            "arguments": {
                "file_path": "note.txt",
                "content": "Hello"
            }
        }
    }

    result = stream_chat.execute_tool_call(tool_call, output_dir=str(tmp_path))
    assert "File written" in result

    result = stream_chat.execute_tool_call({
        "function": {
            "name": "write_output_file",
            "arguments": {
                "file_path": "note.txt",
                "content": "World"
            }
        }
    }, output_dir=str(tmp_path))
    assert "File written" in result

    assert (tmp_path / "note.txt").read_text(encoding="utf-8") == "HelloWorld"


def test_write_output_file_rejects_outside_dir(tmp_path):
    tool_call = {
        "function": {
            "name": "write_output_file",
            "arguments": {
                "file_path": "../escape.txt",
                "content": "Nope"
            }
        }
    }

    result = stream_chat.execute_tool_call(tool_call, output_dir=str(tmp_path))
    assert "outside the allowed output directory" in result
