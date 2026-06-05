import ollama
import sys
import os
import argparse
import json
import csv
import glob
import time
import random
import threading
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, TextIO, Any, Optional, Callable, Iterable, Tuple
from dotenv import load_dotenv
import textwrap

# Optional nice interactive menu library. If unavailable we fall back to a simple prompt.
try:
    import questionary  # type: ignore
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

# Load environment variables from .env file
load_dotenv()

# ==============================
# DATABASE INTEGRATION SECTION
# ==============================
# Import database module
try:
    from db import get_database_manager, Conversation
    HAS_DB = True
except ImportError:
    HAS_DB = False

# Attempt to import DuckDuckGo for web search
try:
    from ddgs import DDGS
    HAS_DDG = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDG = True
    except ImportError:
        HAS_DDG = False

# Optional YAML config support
try:
    import yaml  # type: ignore
    HAS_YAML = True
except ImportError:
    HAS_YAML = False


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int
    initial_delay: float
    max_delay: float
    multiplier: float
    jitter: float


def _parse_env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _parse_env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _parse_comma_list(value: str) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _build_model_list(primary: str, fallbacks: List[str]) -> List[str]:
    seen = set()
    ordered = []
    for model in [primary, *fallbacks]:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered


def _backoff_delay(base_delay: float, jitter: float) -> float:
    if jitter <= 0:
        return base_delay
    return base_delay + random.uniform(0, base_delay * jitter)


def _retry_call(
    action: Callable[[], Any],
    config: RetryConfig,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None
) -> Any:
    attempts = max(1, config.max_attempts)
    delay = max(0.0, config.initial_delay)
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as exc:
            if attempt >= attempts:
                raise
            sleep_time = _backoff_delay(min(config.max_delay, delay), config.jitter)
            if on_retry:
                on_retry(attempt, exc, sleep_time)
            time.sleep(sleep_time)
            delay = min(config.max_delay, delay * max(1.0, config.multiplier))


def _stream_chat_with_retry(
    model: str,
    messages: List[Dict[str, str]],
    config: RetryConfig,
    **kwargs: Any
) -> Iterable[Dict[str, Any]]:
    attempts = max(1, config.max_attempts)
    delay = max(0.0, config.initial_delay)
    for attempt in range(1, attempts + 1):
        try:
            stream = _stream_ollama_chat_with_timeouts(
                model=model,
                messages=messages,
                start_timeout=kwargs.pop("start_timeout", None),
                idle_timeout=kwargs.pop("idle_timeout", None),
                spinner_enabled=kwargs.pop("spinner_enabled", True),
                spinner_style=kwargs.pop("spinner_style", "line"),
                spinner_interval=kwargs.pop("spinner_interval", 0.1),
                spinner_stall_delay=kwargs.pop("spinner_stall_delay", 1.5),
                **kwargs
            )
            for chunk in stream:
                yield chunk
            return
        except Exception as exc:
            if attempt >= attempts:
                raise
            sleep_time = _backoff_delay(min(config.max_delay, delay), config.jitter)
            print(f"\n[Retry {attempt}/{attempts} after error: {exc}] Waiting {sleep_time:.2f}s...")
            time.sleep(sleep_time)
            delay = min(config.max_delay, delay * max(1.0, config.multiplier))


def _run_with_timeout(action: Callable[[], Any], timeout_s: Optional[float], timeout_message: str) -> Any:
    if timeout_s is None or timeout_s <= 0:
        return action()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(action)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            raise TimeoutError(timeout_message)


class Spinner:
    def __init__(
        self,
        message: str,
        enabled: bool = True,
        style: str = "line",
        interval: float = 0.1,
        stream: Optional[TextIO] = None
    ) -> None:
        self.message = message
        self.enabled = enabled
        self.style = style
        self.interval = interval
        self.stream = stream or sys.stderr
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _frames(self) -> List[str]:
        if self.style == "dots":
            return [".", "..", "..."]
        return ["|", "/", "-", "\\"]

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()

        def run() -> None:
            frames = self._frames()
            idx = 0
            while not self._stop_event.is_set():
                frame = frames[idx % len(frames)]
                self.stream.write(f"\r{self.message} {frame}")
                self.stream.flush()
                idx += 1
                time.sleep(self.interval)
            self.stream.write("\r\033[K")
            self.stream.flush()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.2)


def _stream_ollama_chat_with_timeouts(
    model: str,
    messages: List[Dict[str, str]],
    start_timeout: Optional[float],
    idle_timeout: Optional[float],
    spinner_enabled: bool,
    spinner_style: str,
    spinner_interval: float,
    spinner_stall_delay: float,
    **kwargs: Any
) -> Iterable[Dict[str, Any]]:
    q: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
    stop_event = threading.Event()

    def worker() -> None:
        try:
            for chunk in ollama.chat(model=model, messages=messages, stream=True, **kwargs):
                if stop_event.is_set():
                    break
                q.put(("chunk", chunk))
            q.put(("done", None))
        except Exception as exc:
            q.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    first_token = True
    start_time = time.monotonic()
    spinner = Spinner(
        message="Waiting for response",
        enabled=spinner_enabled,
        style=spinner_style,
        interval=spinner_interval
    )

    def start_spinner_after(delay: float) -> Optional[threading.Timer]:
        if not spinner_enabled:
            return None
        if delay <= 0:
            spinner.start()
            return None
        timer = threading.Timer(delay, spinner.start)
        timer.daemon = True
        timer.start()
        return timer

    while True:
        timeout: Optional[float] = None
        if first_token and start_timeout and start_timeout > 0:
            elapsed = time.monotonic() - start_time
            timeout = max(0.0, start_timeout - elapsed)
        elif idle_timeout and idle_timeout > 0:
            timeout = idle_timeout

        spinner_timer = None
        if first_token:
            spinner_timer = start_spinner_after(0.0)
        else:
            spinner_timer = start_spinner_after(spinner_stall_delay)

        try:
            kind, payload = q.get(timeout=timeout)
        except queue.Empty:
            stop_event.set()
            if spinner_timer:
                spinner_timer.cancel()
            spinner.stop()
            if first_token and start_timeout and start_timeout > 0:
                raise TimeoutError("Ollama response start timed out")
            raise TimeoutError("Ollama response stalled")

        if kind == "chunk":
            if spinner_timer:
                spinner_timer.cancel()
            spinner.stop()
            if first_token:
                first_token = False
            yield payload
        elif kind == "done":
            if spinner_timer:
                spinner_timer.cancel()
            spinner.stop()
            return
        elif kind == "error":
            if spinner_timer:
                spinner_timer.cancel()
            spinner.stop()
            raise payload


def _load_yaml_config(path: str) -> Dict[str, Any]:
    if not path:
        return {}
    if not os.path.exists(path):
        print(f"[Config Warning] YAML config file not found: {path}")
        return {}
    if not HAS_YAML:
        print("[Config Warning] PyYAML not installed. Install with: pip install pyyaml")
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            if isinstance(data, dict):
                return data
    except Exception as exc:
        print(f"[Config Warning] Failed to load YAML config: {exc}")
    return {}


def _yaml_get(data: Dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _resolve_float(
    cli_value: Optional[float],
    env_name: str,
    yaml_data: Dict[str, Any],
    yaml_keys: Tuple[str, ...],
    default: float
) -> float:
    if cli_value is not None:
        return cli_value
    env_value = os.environ.get(env_name)
    if env_value:
        try:
            return float(env_value)
        except ValueError:
            return default
    yaml_value = _yaml_get(yaml_data, *yaml_keys)
    if yaml_value is not None:
        try:
            return float(yaml_value)
        except (ValueError, TypeError):
            return default
    return default


def _resolve_bool(
    cli_true: bool,
    cli_false: bool,
    env_name: str,
    yaml_data: Dict[str, Any],
    yaml_keys: Tuple[str, ...],
    default: bool
) -> bool:
    if cli_true:
        return True
    if cli_false:
        return False
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value.lower() in {"1", "true", "yes", "on"}
    yaml_value = _yaml_get(yaml_data, *yaml_keys)
    if yaml_value is not None:
        return bool(yaml_value)
    return default


def _resolve_str(
    cli_value: Optional[str],
    env_name: str,
    yaml_data: Dict[str, Any],
    yaml_keys: Tuple[str, ...],
    default: str
) -> str:
    if cli_value:
        return cli_value
    env_value = os.environ.get(env_name)
    if env_value:
        return env_value
    yaml_value = _yaml_get(yaml_data, *yaml_keys)
    if yaml_value is not None:
        return str(yaml_value)
    return default


def _ensure_model_available(model: str, config: RetryConfig) -> None:
    def show_model() -> Any:
        return ollama.show(model)

    try:
        _retry_call(show_model, config)
        return
    except ollama.ResponseError:
        pass

    print(f"Model {model} not found. Pulling...")

    def pull_model() -> Any:
        return ollama.pull(model)

    _retry_call(pull_model, config)


# Tool schema for Ollama
def get_tools() -> List[Dict[str, Any]]:
    """Returns the list of available tools for the model."""
    return [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information, news, facts, or data that may not be in the model's training data. Use this when the user asks about current events, recent news, specific facts you're unsure about, or time‑sensitive information.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query to look up on the web"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read the contents of a text file. Supports any text-based file format (txt, md, py, json, yaml, etc.). Use this when the user asks you to read or examine a file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to read"
                        },
                        "max_lines": {
                            "type": "integer",
                            "description": "Maximum number of lines to read (default: 1000, use -1 for all)",
                            "default": 1000
                        },
                        "start_line": {
                            "type": "integer",
                            "description": "Starting line number (1-indexed, default: 1)",
                            "default": 1
                        }
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "read_json_file",
                "description": "Read and parse a JSON file efficiently without loading everything into memory at once. Supports newline‑delimited JSON (NDJSON) and regular JSON with safeguards for large files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Path to the JSON file to read"},
                        "max_entries": {"type": "integer", "description": "Maximum number of entries to read (default: 100, use -1 for all)", "default": 100},
                        "query_filter": {"type": "string", "description": "Optional dot/array path filter (e.g., 'conversations[*].messages[*].content')"},
                        "return_summary": {"type": "boolean", "description": "If true, return a summary instead of full data", "default": False}
                    },
                    "required": ["file_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "write_file",
                "description": "Write content to a file. Creates the file if it doesn't exist, or overwrites it if it does. Use this when the user asks you to create or save a file with specific content.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "File path to write to"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to write to the file"
                        },
                        "create_dirs": {
                            "type": "boolean",
                            "description": "If true, create parent directories if they don't exist (default: true)",
                            "default": True
                        }
                    },
                    "required": ["file_path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "append_file",
                "description": "Append content to an existing file. Creates the file if it doesn't exist. Use this when the user asks you to add to a file without overwriting it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "File path to append to"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to append to the file"
                        },
                        "add_newline": {
                            "type": "boolean",
                            "description": "If true, add a newline before appending (default: true)",
                            "default": True
                        }
                    },
                    "required": ["file_path", "content"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "list_directory",
                "description": "List files and directories in a given path. Use this when the user asks what files are in a directory or needs to explore the file system.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": "Path to the directory to list"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Optional glob pattern to filter results (e.g., '*.py', '*.md')",
                            "default": "*"
                        },
                        "recursive": {
                            "type": "boolean",
                            "description": "If true, list files recursively (default: false)",
                            "default": False
                        }
                    },
                    "required": ["directory_path"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "Run a shell command and return its output. Use this to execute commands, install packages, run scripts, check system state, or inspect processes. Always show the user what command you're about to run before executing it.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The shell command to execute"
                        },
                        "timeout": {
                            "type": "integer",
                            "description": "Timeout in seconds (default: 30)",
                            "default": 30
                        },
                        "workdir": {
                            "type": "string",
                            "description": "Working directory for the command (default: current directory)"
                        }
                    },
                    "required": ["command"]
                }
            }
        }
    ]


def execute_tool_call(
    tool_call: Dict[str, Any],
    tool_timeout: Optional[float] = None,
    web_search_timeout: Optional[float] = None,
    output_dir: Optional[str] = None
) -> str:
    """Execute a tool call and return the result."""
    function_name = tool_call.get("function", {}).get("name")
    raw_arguments = tool_call.get("function", {}).get("arguments", "{}")
    if isinstance(raw_arguments, dict):
        arguments = raw_arguments
    else:
        arguments = json.loads(raw_arguments)
    
    if function_name == "web_search":
        query = arguments.get("query", "")
        timeout_s = web_search_timeout if web_search_timeout is not None else tool_timeout
        try:
            return _run_with_timeout(
                lambda: perform_web_search(query),
                timeout_s,
                "web_search timed out"
            )
        except TimeoutError:
            return "[Web Search Timeout: exceeded configured limit]"

    if function_name == "read_file":
        try:
            return _run_with_timeout(
                lambda: read_file(
                    file_path=arguments.get("file_path", ""),
                    max_lines=int(arguments.get("max_lines", 1000)),
                    start_line=int(arguments.get("start_line", 1)),
                ),
                tool_timeout,
                "read_file timed out"
            )
        except TimeoutError:
            return "[Tool Timeout: read_file exceeded configured limit]"

    if function_name == "read_json_file":
        try:
            return _run_with_timeout(
                lambda: read_json_file(
                    file_path=arguments.get("file_path", ""),
                    max_entries=int(arguments.get("max_entries", 100)),
                    query_filter=arguments.get("query_filter", ""),
                    return_summary=bool(arguments.get("return_summary", False))
                ),
                tool_timeout,
                "read_json_file timed out"
            )
        except TimeoutError:
            return "[Tool Timeout: read_json_file exceeded configured limit]"

    if function_name == "write_file":
        return write_file(
            file_path=arguments.get("file_path", ""),
            content=arguments.get("content", ""),
            create_dirs=bool(arguments.get("create_dirs", True)),
        )

    if function_name == "append_file":
        return append_file(
            file_path=arguments.get("file_path", ""),
            content=arguments.get("content", ""),
            add_newline=bool(arguments.get("add_newline", True)),
        )

    if function_name == "list_directory":
        return list_directory(
            directory_path=arguments.get("directory_path", ""),
            pattern=arguments.get("pattern", "*"),
            recursive=bool(arguments.get("recursive", False)),
        )

    if function_name == "run_shell":
        try:
            return _run_with_timeout(
                lambda: run_shell(
                    command=arguments.get("command", ""),
                    timeout=int(arguments.get("timeout", 30)),
                    workdir=arguments.get("workdir"),
                ),
                tool_timeout,
                "run_shell timed out"
            )
        except TimeoutError:
            return "[Tool Timeout: run_shell exceeded configured limit]"
    if function_name == "write_output_file":
        file_path = arguments.get("file_path", "")
        content = arguments.get("content", "")
        return write_output_file(
            file_path=file_path,
            content=content,
            output_dir=output_dir
        )
    
    return f"[Error: Unknown tool '{function_name}']"
    return f"[Error: Unknown tool '{function_name}']"


def chat_with_tools(
    model: str,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    file_handle: TextIO,
    retry_config: RetryConfig,
    render_delay: float = 0.0,
    tool_timeout: Optional[float] = None,
    web_search_timeout: Optional[float] = None,
    ollama_timeout: Optional[float] = None,
    ollama_first_token_timeout: Optional[float] = None,
    ollama_stream_idle_timeout: Optional[float] = None,
    tool_output_dir: Optional[str] = None,
    spinner_enabled: bool = True,
    spinner_style: str = "line",
    spinner_interval: float = 0.1,
    spinner_stall_delay: float = 1.5
) -> str:
    """
    Chat with the model, handling tool calls automatically.
    Returns the final assistant response.
    """
    # Inject a system prompt so small models know to use their tools
    tool_names = [t["function"]["name"] for t in tools if t.get("type") == "function"]
    if tool_names:
        tool_instruction = (
            "You are a helpful assistant with access to tools. "
            "You MUST use your tools when they can help answer the question — "
            "do NOT say you cannot help when a tool would provide the answer. "
            f"Available tools: {', '.join(tool_names)}. "
            "For real-time or current information, use web_search or run_shell. "
            "For file operations, use read_file, write_file, or list_directory. "
            "For shell commands, use run_shell."
        )
        has_system = any(msg.get("role") == "system" for msg in messages)
        if has_system:
            for msg in messages:
                if msg.get("role") == "system":
                    msg["content"] = f"{tool_instruction}\n\n{msg.get('content', '')}"
                    break
        else:
            messages.insert(0, {"role": "system", "content": tool_instruction})

    # First call - let the model decide if it needs tools
    selection_spinner = Spinner(
        message="Waiting for model",
        enabled=spinner_enabled,
        style=spinner_style,
        interval=spinner_interval
    )
    selection_spinner.start()
    try:
        def chat_call():
            kwargs = {}
            # LFM2.5 works best with think=False for tool calling
            if _supports_lfm2_tool_format(model):
                kwargs['think'] = False
            return ollama.chat(model=model, messages=messages, tools=tools, **kwargs)
        
        response = _retry_call(
            lambda: _run_with_timeout(chat_call, ollama_timeout, "Ollama tool selection call timed out"),
            retry_config
        )
    finally:
        selection_spinner.stop()
    
    message = response.message
    
    # Check if the model wants to use tools
    if hasattr(message, 'tool_calls') and message.tool_calls:
        tool_calls_info = []
        for tc in message.tool_calls:
            tc_function = getattr(tc, "function", None)
            if isinstance(tc, dict):
                tc_function = tc.get("function")

            tc_name = getattr(tc_function, "name", None)
            tc_args = getattr(tc_function, "arguments", None)
            if isinstance(tc_function, dict):
                tc_name = tc_function.get("name")
                tc_args = tc_function.get("arguments")

            tc_id = getattr(tc, "id", None)
            if tc_id is None and isinstance(tc, dict):
                tc_id = tc.get("id")

            tool_calls_info.append({
                "id": tc_id,
                "name": tc_name,
                "arguments": tc_args,
            })

        tool_names = [info["name"] for info in tool_calls_info if info.get("name")]
        print(f"\n[Tool calls detected: {tool_names}]")

        # Add the assistant's tool call request to history
        tool_calls_payload = []
        for info in tool_calls_info:
            payload = {
                "type": "function",
                "function": {
                    "name": info["name"],
                    "arguments": info["arguments"],
                },
            }
            if info.get("id") is not None:
                payload["id"] = info["id"]
            tool_calls_payload.append(payload)

        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": tool_calls_payload,
        })

        # Execute each tool call and add results
        for info in tool_calls_info:
            tool_name = info.get("name") or "tool"
            spinner = Spinner(
                message=f"Running tool '{tool_name}'",
                enabled=spinner_enabled,
                style=spinner_style,
                interval=spinner_interval
            )
            spinner.start()
            try:
                tool_kwargs: Dict[str, Optional[float]] = {}
                if tool_timeout is not None:
                    tool_kwargs["tool_timeout"] = tool_timeout
                if web_search_timeout is not None:
                    tool_kwargs["web_search_timeout"] = web_search_timeout
                if tool_output_dir:
                    tool_kwargs["output_dir"] = tool_output_dir
                result = execute_tool_call({
                    "function": {
                        "name": info["name"],
                        "arguments": info["arguments"],
                    }
                }, **tool_kwargs)
            finally:
                spinner.stop()

            tool_message = {
                "role": "tool",
                "content": result,
            }
            if info.get("id") is not None:
                tool_message["tool_call_id"] = info["id"]
            messages.append(tool_message)

            if info.get("name"):
                log_to_file(file_handle, f"\n[Tool '{info['name']}' used]\n")
        
        # Second call - get the final response with tool results
        print(f"{model}: ", end="", flush=True)
        log_to_file(file_handle, f"{model}: ")
        
        full_response = ""
        stream = _stream_chat_with_retry(
            model=model,
            messages=messages,
            config=retry_config,
            start_timeout=ollama_first_token_timeout,
            idle_timeout=ollama_stream_idle_timeout,
            spinner_enabled=spinner_enabled,
            spinner_style=spinner_style,
            spinner_interval=spinner_interval,
            spinner_stall_delay=spinner_stall_delay
        )
        
        for chunk in stream:
            part = chunk['message']['content']
            render_stream_text(part, file_handle, delay=render_delay)
            full_response += part
        
        return full_response
    else:
        # No tool calls - just stream the response
        print(f"{model}: ", end="", flush=True)
        log_to_file(file_handle, f"{model}: ")
        
        full_response = ""
        # Stream the content if available
        if message.content:
            render_stream_text(message.content, file_handle, delay=render_delay)
            full_response = message.content
        
        return full_response

def parse_arguments() -> argparse.Namespace:
    """
    Parses CLI arguments, allowing environment variables to serve as defaults.
    Precedence: CLI Flag > Environment Variable > Default Value
    """
    parser = argparse.ArgumentParser(description="Stream chat with Ollama models.")

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (env CHAT_CONFIG)."
    )

    parser.add_argument(
        "--model",
        type=str,
        default=os.environ.get("OLLAMA_MODEL", "llama3"),
        help="The Ollama model to use (default: llama3 or env OLLAMA_MODEL)"
    )

    parser.add_argument(
        "--model-fallbacks",
        type=str,
        default=os.environ.get("OLLAMA_MODEL_FALLBACKS", ""),
        help="Comma-separated list of fallback models (env OLLAMA_MODEL_FALLBACKS)"
    )

    parser.add_argument(
        "--dest",
        type=str,
        default=os.environ.get("CHAT_LOG_DEST", "chat_log.txt"),
        help="Path to the log file (default: chat_log.txt or env CHAT_LOG_DEST)"
    )

    # Boolean flags
    # We check env vars for 'true' string to set default to True, otherwise False
    default_experimental = os.environ.get("EXPERIMENTAL", "false").lower() == "true"
    parser.add_argument(
        "--experimental",
        action="store_true",
        default=default_experimental,
        help="Enable experimental features/modes"
    )

    default_websearch = os.environ.get("EXPERIMENTAL_WEBSEARCH", "false").lower() == "true"
    parser.add_argument(
        "--experimental-websearch",
        action="store_true",
        default=default_websearch,
        help="Enable experimental web search integration (requires duckduckgo-search)"
    )

    parser.add_argument(
        "--context",
        type=str,
        default=os.environ.get("CONTEXT_PATH", ""),
        help="Path to a directory or file to load as historical context for the LLM. Use 'db' to load from database."
    )

    parser.add_argument(
        "--persist-to-db",
        action="store_true",
        default=os.environ.get("PERSIST_TO_DB", "false").lower() == "true",
        help="Enable saving conversations to PostgreSQL database"
    )

    # -----------------------------------------------------------------
    # New session handling / export options
    # -----------------------------------------------------------------
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="List saved conversation sessions and exit"
    )
    parser.add_argument(
        "--select-session",
        action="store_true",
        help="Interactively select a saved session to continue"
    )
    parser.add_argument(
        "--export-session",
        type=int,
        metavar="ID",
        help="Export the conversation with the given ID. Use --format to choose output format."
    )
    parser.add_argument(
        "--format",
        choices=["sql", "json", "csv", "text"],
        default="json",
        help="Export format when using --export-session"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="File path to write exported data. If omitted, prints to stdout."
    )

    parser.add_argument(
        "--context-grep",
        type=str,
        default=os.environ.get("CONTEXT_GREP", "txt,log"),
        help="Comma-separated list of file extensions to include when loading context from a directory (default: txt,log)"
    )

    parser.add_argument(
        "--tool-output-dir",
        type=str,
        default=os.environ.get("TOOL_OUTPUT_DIR", "sessions"),
        help="Base directory for tool-generated output files (default: ./sessions or env TOOL_OUTPUT_DIR)"
    )

    parser.add_argument(
        "--render-delay",
        type=float,
        default=_parse_env_float("RENDER_DELAY", 0.0),
        help="Delay in seconds between rendered characters (default: 0.0 or env RENDER_DELAY)"
    )

    parser.add_argument(
        "--tool-timeout",
        type=float,
        default=None,
        help="Timeout in seconds for any tool call (env TOOL_TIMEOUT, yaml timeouts.tool)"
    )

    parser.add_argument(
        "--web-search-timeout",
        type=float,
        default=None,
        help="Timeout in seconds for web_search tool (env WEB_SEARCH_TIMEOUT, yaml timeouts.web_search)"
    )

    parser.add_argument(
        "--ollama-timeout",
        type=float,
        default=None,
        help="Timeout in seconds for non-stream Ollama calls (env OLLAMA_TIMEOUT, yaml timeouts.ollama)"
    )

    parser.add_argument(
        "--ollama-first-token-timeout",
        type=float,
        default=None,
        help="Timeout in seconds waiting for first token (env OLLAMA_FIRST_TOKEN_TIMEOUT, yaml timeouts.ollama_first_token)"
    )

    parser.add_argument(
        "--ollama-stream-idle-timeout",
        type=float,
        default=None,
        help="Timeout in seconds for stalled streams (env OLLAMA_STREAM_IDLE_TIMEOUT, yaml timeouts.ollama_stream_idle)"
    )

    spinner_group = parser.add_mutually_exclusive_group()
    spinner_group.add_argument(
        "--spinner",
        action="store_true",
        help="Enable terminal spinner indicators (env SPINNER_ENABLED, yaml ui.spinner)"
    )
    spinner_group.add_argument(
        "--no-spinner",
        action="store_true",
        help="Disable terminal spinner indicators"
    )

    parser.add_argument(
        "--spinner-style",
        type=str,
        default=None,
        choices=["line", "dots"],
        help="Spinner style: line or dots (env SPINNER_STYLE, yaml ui.spinner_style)"
    )

    parser.add_argument(
        "--spinner-interval",
        type=float,
        default=None,
        help="Spinner frame interval seconds (env SPINNER_INTERVAL, yaml ui.spinner_interval)"
    )

    parser.add_argument(
        "--spinner-stall-delay",
        type=float,
        default=None,
        help="Seconds of no tokens before showing spinner (env SPINNER_STALL_DELAY, yaml ui.spinner_stall_delay)"
    )

    parser.add_argument(
        "--retry-max-attempts",
        type=int,
        default=_parse_env_int("RETRY_MAX_ATTEMPTS", 3),
        help="Max retry attempts for Ollama calls (default: 3 or env RETRY_MAX_ATTEMPTS)"
    )

    parser.add_argument(
        "--retry-initial-delay",
        type=float,
        default=_parse_env_float("RETRY_INITIAL_DELAY", 0.5),
        help="Initial backoff delay in seconds (default: 0.5 or env RETRY_INITIAL_DELAY)"
    )

    parser.add_argument(
        "--retry-max-delay",
        type=float,
        default=_parse_env_float("RETRY_MAX_DELAY", 8.0),
        help="Max backoff delay in seconds (default: 8.0 or env RETRY_MAX_DELAY)"
    )

    parser.add_argument(
        "--retry-multiplier",
        type=float,
        default=_parse_env_float("RETRY_MULTIPLIER", 2.0),
        help="Backoff multiplier (default: 2.0 or env RETRY_MULTIPLIER)"
    )

    parser.add_argument(
        "--retry-jitter",
        type=float,
        default=_parse_env_float("RETRY_JITTER", 0.1),
        help="Jitter ratio applied to backoff delay (default: 0.1 or env RETRY_JITTER)"
    )

    return parser.parse_args()

def log_to_file(file_handle: TextIO, text: str) -> None:
    """Writes text to file and forces a flush to ensure it's saved immediately."""
    file_handle.write(text)
    file_handle.flush()

def render_stream_text(
    text: str,
    file_handle: TextIO,
    flush_interval: int = 20,
    delay: float = 0.0
) -> None:
    """Render text progressively to stdout while logging to file."""
    if not text:
        return
    for idx, char in enumerate(text, start=1):
        sys.stdout.write(char)
        if flush_interval > 0 and idx % flush_interval == 0:
            sys.stdout.flush()
        if delay > 0:
            time.sleep(delay)
    sys.stdout.flush()
    file_handle.write(text)
    file_handle.flush()

def perform_web_search(query: str) -> str:
    """
    Performs a simple web search using DuckDuckGo and returns a context string.
    """
    if not HAS_DDG:
        return "[System Error: duckduckgo-search library not installed, cannot search web.]"
    
    print(f"\n[Searching web for: '{query}'...]")
    try:
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "No web results found."
        
        context_parts = ["Web Search Results:"]
        for res in results:
            title = res.get('title', 'No Title')
            body = res.get('body', 'No Content')
            href = res.get('href', '')
            context_parts.append(f"- Source: {title} ({href})\n  Content: {body}")
        
        return "\n\n".join(context_parts)
    except Exception as e:
        return f"[Web Search Error: {str(e)}]"


def read_file(file_path: str, max_lines: int = 1000, start_line: int = 1) -> str:
    """Read a text file with optional line limits."""
    path = Path(file_path)
    if not file_path:
        return "[Error: file_path is required]"
    if not path.exists():
        return f"[Error: File not found: {file_path}]"
    if not path.is_file():
        return f"[Error: Path is not a file: {file_path}]"
    
    print(f"\n[Reading file: {file_path}]")
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        start_idx = max(0, start_line - 1)
        if max_lines > 0:
            end_idx = start_idx + max_lines
            selected_lines = lines[start_idx:end_idx]
        else:
            selected_lines = lines[start_idx:]
        
        content = ''.join(selected_lines)
        total_lines = len(lines)
        shown_lines = len(selected_lines)
        
        return f"[File: {file_path} (showing {shown_lines} of {total_lines} lines)]\n\n{content}"
    except Exception as e:
        return f"[Error reading file: {str(e)}]"


def write_file(file_path: str, content: str, create_dirs: bool = True) -> str:
    """Write content to a file, creating directories if needed."""
    if not file_path:
        return "[Error: file_path is required]"
    
    path = Path(file_path)
    try:
        if create_dirs:
            path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return f"[File written: {path}] ({len(content)} bytes)"
    except Exception as e:
        return f"[Error writing file: {str(e)}]"


def append_file(file_path: str, content: str, add_newline: bool = True) -> str:
    """Append content to a file."""
    if not file_path:
        return "[Error: file_path is required]"
    
    path = Path(file_path)
    try:
        if path.exists():
            with open(path, 'rb') as f:
                f.seek(0, 2)
                if f.tell() > 0:
                    f.seek(-1, 2)
                    last_char = f.read(1)
                    needs_newline = add_newline and last_char != b'\n'
                else:
                    needs_newline = False
        else:
            needs_newline = False
        
        with open(path, 'a', encoding='utf-8') as f:
            if needs_newline:
                f.write('\n')
            f.write(content)
        
        return f"[Content appended to: {path}] ({len(content)} bytes)"
    except Exception as e:
        return f"[Error appending to file: {str(e)}]"


def list_directory(directory_path: str, pattern: str = "*", recursive: bool = False) -> str:
    """List files and directories in a path."""
    if not directory_path:
        return "[Error: directory_path is required]"
    
    path = Path(directory_path)
    if not path.exists():
        return f"[Error: Directory not found: {directory_path}]"
    if not path.is_dir():
        return f"[Error: Path is not a directory: {directory_path}]"
    
    print(f"\n[Listing directory: {directory_path}]")
    try:
        if recursive:
            items = list(path.rglob(pattern))
        else:
            items = list(path.glob(pattern))
        
        items.sort(key=lambda x: (not x.is_dir(), x.name))
        
        if not items:
            return f"[Directory is empty: {directory_path}]"
        
        lines = [f"[Directory: {directory_path}]"]
        for item in items:
            if item.is_dir():
                lines.append(f"📁 {item.name}/")
            else:
                size = item.stat().st_size
                size_str = f"{size:,} B" if size < 1024 else f"{size/1024:.1f} KB" if size < 1024*1024 else f"{size/1024/1024:.1f} MB"
                lines.append(f"📄 {item.name} ({size_str})")
        
        return "\n".join(lines)
    except Exception as e:
        return f"[Error listing directory: {str(e)}]"


def run_shell(command: str, timeout: int = 30, workdir: Optional[str] = None) -> str:
    """Run a shell command and return its stdout, stderr, and exit code."""
    if not command:
        return "[Error: command is required]"
    
    print(f"\n[Running: {command}]")
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )
        parts = []
        if result.stdout:
            parts.append(result.stdout.rstrip())
        if result.stderr:
            parts.append(f"[stderr]\n{result.stderr.rstrip()}")
        parts.append(f"[exit code: {result.returncode}]")
        output = "\n".join(parts)
        if not result.stdout and not result.stderr:
            return f"[Command completed with exit code {result.returncode}]"
        return output
    except subprocess.TimeoutExpired:
        return f"[Error: Command timed out after {timeout}s]"
    except Exception as e:
        return f"[Error running command: {str(e)}]"


def read_json_file(
    file_path: str,
    max_entries: int = 100,
    query_filter: str = "",
    return_summary: bool = False
) -> str:
    """
    Read a JSON file efficiently with streaming support.
    Supports NDJSON (newline-delimited JSON) and regular JSON.
    """
    path = Path(file_path)
    if not file_path:
        return "[Error: file_path is required]"
    if not path.exists():
        return f"[Error: File not found: {file_path}]"
    if not path.is_file():
        return f"[Error: Path is not a file: {file_path}]"

    print(f"\n[Reading JSON file: {file_path}]")

    try:
        is_ndjson = _detect_ndjson(path)
        if is_ndjson:
            return _read_ndjson_streaming(path, max_entries, query_filter, return_summary)
        return _read_regular_json_streaming(path, max_entries, query_filter, return_summary)
    except json.JSONDecodeError as e:
        return f"[Error: Invalid JSON - {str(e)}]"
    except Exception as e:
        return f"[Error reading file: {str(e)}]"


def _resolve_output_path(file_path: str, output_dir: Optional[str]) -> Tuple[Optional[Path], Optional[str]]:
    if not file_path:
        return None, "[Error: file_path is required]"

    base_dir = Path(output_dir or "sessions").expanduser()
    if not base_dir.is_absolute():
        base_dir = (Path.cwd() / base_dir).resolve()
    else:
        base_dir = base_dir.resolve()

    target_path = Path(file_path).expanduser()
    if not target_path.is_absolute():
        target_path = (base_dir / target_path).resolve()
    else:
        target_path = target_path.resolve()

    if base_dir != target_path and base_dir not in target_path.parents:
        return None, "[Error: file_path is outside the allowed output directory]"

    return target_path, None


def write_output_file(file_path: str, content: str, output_dir: Optional[str] = None) -> str:
    target_path, error = _resolve_output_path(file_path, output_dir)
    if error:
        return error
    if target_path is None:
        return "[Error: Invalid output path]"

    try:
        target_path.parent.mkdir(parents=True, exist_ok=True)
        with open(target_path, "a", encoding="utf-8") as handle:
            handle.write(content)
        return f"[File written: {target_path}]"
    except Exception as exc:
        return f"[Error writing file: {exc}]"


def _detect_ndjson(path: Path) -> bool:
    """Detect if file is newline-delimited JSON by validating the first two lines."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first_line = f.readline().strip()
            if not first_line:
                return False
            json.loads(first_line)
            second_line = f.readline().strip()
            if second_line:
                json.loads(second_line)
            return True
    except Exception:
        return False


def _read_ndjson_streaming(
    path: Path,
    max_entries: int,
    query_filter: str,
    return_summary: bool
) -> str:
    results: List[Any] = []
    count = 0
    limit = max_entries if max_entries is not None else 100

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if limit > 0 and count >= limit:
                break
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            if query_filter:
                results.append(_extract_fields(entry, query_filter))
            else:
                results.append(entry)
            count += 1

    if return_summary:
        summary = {
            "format": "ndjson",
            "entries_read": count,
            "sample_structure": _get_structure(results[0]) if results else None,
            "file_path": str(path)
        }
        return json.dumps(summary, indent=2)

    return json.dumps(results, indent=2)


def _read_regular_json_streaming(
    path: Path,
    max_entries: int,
    query_filter: str,
    return_summary: bool
) -> str:
    file_size = path.stat().st_size
    print(f"  File size: {file_size / 1024 / 1024:.2f} MB")

    if file_size > 100 * 1024 * 1024:
        return (
            f"[Warning: File is large ({file_size / 1024 / 1024:.1f} MB). "
            "Consider NDJSON for streaming or increase limits cautiously.]"
        )

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        data = json.load(f)

    if isinstance(data, list):
        if max_entries and max_entries > 0:
            entries = data[:max_entries]
        else:
            entries = data

        if query_filter:
            entries = [_extract_fields(entry, query_filter) for entry in entries]

        if return_summary:
            summary = {
                "format": "json",
                "total_entries": len(data),
                "entries_returned": len(entries),
                "sample_structure": _get_structure(data[0]) if data else None,
                "file_path": str(path)
            }
            return json.dumps(summary, indent=2)

        return json.dumps(entries, indent=2)

    if return_summary:
        summary = {
            "format": "json",
            "type": "object",
            "structure": _get_structure(data),
            "file_path": str(path)
        }
        return json.dumps(summary, indent=2)

    return json.dumps(data, indent=2)


def _extract_fields(data: Any, filter_path: str) -> Any:
    """Extract fields using a simple dot/array path like conversations[*].messages[*].content."""
    if not filter_path:
        return data

    parts = filter_path.split(".")
    result: Any = data

    for part in parts:
        if "[" in part and part.endswith("]"):
            key = part.split("[")[0]
            index = part.split("[")[1].rstrip("]")

            if key:
                if isinstance(result, dict):
                    result = result.get(key, [])
                else:
                    return None

            if isinstance(result, list):
                if index == "*":
                    return result
                try:
                    result = result[int(index)]
                except (ValueError, IndexError):
                    return None
            else:
                return None
        else:
            if isinstance(result, dict):
                result = result.get(part)
            else:
                return None

    return result


def _get_structure(data: Any, max_depth: int = 3) -> Dict[str, Any]:
    if max_depth <= 0:
        return {"type": type(data).__name__}
    if isinstance(data, dict):
        return {
            k: _get_structure(v, max_depth - 1)
            for k, v in list(data.items())[:10]
        }
    if isinstance(data, list):
        if data:
            return {
                "type": "array",
                "length": len(data),
                "item_structure": _get_structure(data[0], max_depth - 1)
            }
        return {"type": "array", "length": 0}
    return {"type": type(data).__name__}

def load_context_files(context_path: str, extensions: List[str]) -> str:
    """
    Load context from a file or directory.
    If directory, recursively find all files with specified extensions.
    Returns concatenated content of all matching files.
    """
    if not context_path or not os.path.exists(context_path):
        return ""
    
    context_parts = []
    
    # Normalize extensions (remove dots if present)
    extensions = [ext.lstrip('.').lower() for ext in extensions]
    
    if os.path.isfile(context_path):
        # Single file
        try:
            with open(context_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                context_parts.append(f"=== File: {context_path} ===\n{content}\n")
        except Exception as e:
            context_parts.append(f"=== Error reading {context_path}: {e} ===\n")
    
    elif os.path.isdir(context_path):
        # Directory - recursively find files with matching extensions
        for root, dirs, files in os.walk(context_path):
            for file in files:
                file_ext = file.split('.')[-1].lower() if '.' in file else ''
                if file_ext in extensions or '*' in extensions:
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()
                            rel_path = os.path.relpath(file_path, context_path)
                            context_parts.append(f"=== File: {rel_path} ===\n{content}\n")
                    except Exception as e:
                        context_parts.append(f"=== Error reading {file_path}: {e} ===\n")
    
    return "\n".join(context_parts)


def load_context_from_database(
    db_manager, 
    additional_paths: Optional[List[str]] = None,
    limit: Optional[int] = None,
    extensions: Optional[List[str]] = None
) -> str:
    """
    Load context from database and optionally from additional file paths.
    
    Args:
        db_manager: Database manager instance
        additional_paths: Additional file/directory paths to load context from
        limit: Maximum number of conversations to load from database
        
    Returns:
        Concatenated context from database and files
    """
    context_parts = []
    
    # Load from database
    try:
        conversations = db_manager.get_all_conversations(limit=limit)
        if conversations:
            context_parts.append("=== Database Conversations ===\n")
            for conv in conversations:
                context_parts.append(f"--- Conversation {conv.id} (Model: {conv.model}, Created: {conv.created_at}) ---\n")
                for msg in conv.messages:
                    if msg.get('role') in ['user', 'assistant']:
                        role = msg.get('role', 'unknown').upper()
                        content = msg.get('content', '')
                        context_parts.append(f"{role}: {content}\n")
                context_parts.append("\n")
    except Exception as e:
        context_parts.append(f"=== Error loading from database: {e} ===\n")
    
    # Load from additional paths
    if additional_paths:
        load_extensions = extensions or ['txt', 'log']
        for path in additional_paths:
            if path and path != 'db':
                content = load_context_files(path, load_extensions)
                if content:
                    context_parts.append(f"=== File Context: {path} ===\n{content}\n")
    
    return "\n".join(context_parts)


def parse_context_arg(context_arg: str) -> Dict[str, Any]:
    """
    Parse the --context argument to extract database and file paths.
    
    Args:
        context_arg: The context argument value
        
    Returns:
        Dictionary with 'use_db' boolean and 'paths' list of additional paths
    """
    if not context_arg:
        return {'use_db': False, 'paths': []}
    
    # Split by comma to handle multiple sources
    parts = [part.strip() for part in context_arg.split(',')]
    
    use_db = False
    paths = []
    
    for part in parts:
        if part.lower() == 'db':
            use_db = True
        elif part:
            paths.append(part)
    
    return {'use_db': use_db, 'paths': paths}


def _get_retry_config_from_args(args: argparse.Namespace) -> RetryConfig:
    return RetryConfig(
        max_attempts=max(1, args.retry_max_attempts),
        initial_delay=max(0.0, args.retry_initial_delay),
        max_delay=max(0.0, args.retry_max_delay),
        multiplier=max(1.0, args.retry_multiplier),
        jitter=max(0.0, args.retry_jitter),
    )


def _apply_runtime_config(args: argparse.Namespace) -> Dict[str, Any]:
    config_path = args.config or os.environ.get("CHAT_CONFIG", "")
    yaml_data = _load_yaml_config(config_path)

    args.tool_timeout = _resolve_float(
        args.tool_timeout,
        "TOOL_TIMEOUT",
        yaml_data,
        ("timeouts", "tool"),
        30.0
    )

    args.web_search_timeout = _resolve_float(
        args.web_search_timeout,
        "WEB_SEARCH_TIMEOUT",
        yaml_data,
        ("timeouts", "web_search"),
        args.tool_timeout
    )

    args.ollama_timeout = _resolve_float(
        args.ollama_timeout,
        "OLLAMA_TIMEOUT",
        yaml_data,
        ("timeouts", "ollama"),
        60.0
    )

    args.ollama_first_token_timeout = _resolve_float(
        args.ollama_first_token_timeout,
        "OLLAMA_FIRST_TOKEN_TIMEOUT",
        yaml_data,
        ("timeouts", "ollama_first_token"),
        20.0
    )

    args.ollama_stream_idle_timeout = _resolve_float(
        args.ollama_stream_idle_timeout,
        "OLLAMA_STREAM_IDLE_TIMEOUT",
        yaml_data,
        ("timeouts", "ollama_stream_idle"),
        15.0
    )

    args.spinner_enabled = _resolve_bool(
        args.spinner,
        args.no_spinner,
        "SPINNER_ENABLED",
        yaml_data,
        ("ui", "spinner"),
        True
    )

    args.spinner_style = _resolve_str(
        args.spinner_style,
        "SPINNER_STYLE",
        yaml_data,
        ("ui", "spinner_style"),
        "line"
    )

    args.spinner_interval = _resolve_float(
        args.spinner_interval,
        "SPINNER_INTERVAL",
        yaml_data,
        ("ui", "spinner_interval"),
        0.1
    )

    args.spinner_stall_delay = _resolve_float(
        args.spinner_stall_delay,
        "SPINNER_STALL_DELAY",
        yaml_data,
        ("ui", "spinner_stall_delay"),
        1.5
    )

    return yaml_data


def _get_model_specific_timeout(model: str, base_timeout: Optional[float]) -> Optional[float]:
    """Get adjusted timeout for models that need more time for tool selection."""
    if base_timeout is not None:
        return base_timeout
    if model.lower().startswith("lfm2"):
        return 120.0
    return None


def _supports_lfm2_tool_format(model: str) -> bool:
    """Check if model is LFM2.x which prefers tools in system prompt."""
    return model.lower().startswith("lfm2")


def _format_tools_for_lfm2(tools: List[Dict[str, Any]]) -> str:
    """Format tools as JSON string for LFM2.5 system prompt."""
    simplified_tools = []
    for tool in tools:
        if tool.get("type") == "function":
            func = tool.get("function", {})
            simplified_tools.append({
                "name": func.get("name", ""),
                "description": func.get("description", ""),
                "parameters": func.get("parameters", {})
            })
    return json.dumps(simplified_tools, indent=2)


def _prepare_messages_for_lfm2(
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]]
) -> List[Dict[str, str]]:
    """Prepare messages with tools in system prompt for LFM2.5."""
    tools_json = _format_tools_for_lfm2(tools)
    system_content = f"List of tools: {tools_json}"
    
    # Check if there's already a system message
    has_system = any(msg.get("role") == "system" for msg in messages)
    
    if has_system:
        # Append tools to existing system message
        new_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                existing_content = msg.get("content", "")
                msg = msg.copy()
                msg["content"] = f"{existing_content}\n\n{system_content}"
            new_messages.append(msg)
        return new_messages
    else:
        # Prepend new system message
        return [{"role": "system", "content": system_content}] + messages


def _run_chat_turn(
    model: str,
    messages: List[Dict[str, str]],
    args: argparse.Namespace,
    file_handle: TextIO,
    retry_config: RetryConfig
) -> str:
    if args.experimental_websearch and HAS_DDG:
        ollama_timeout = _get_model_specific_timeout(model, args.ollama_timeout)
        tools = get_tools()
        
        return chat_with_tools(
            model=model,
            messages=messages,
            tools=tools,
            file_handle=file_handle,
            retry_config=retry_config,
            render_delay=args.render_delay,
            tool_timeout=args.tool_timeout,
            web_search_timeout=args.web_search_timeout,
            ollama_timeout=ollama_timeout,
            ollama_first_token_timeout=args.ollama_first_token_timeout,
            ollama_stream_idle_timeout=args.ollama_stream_idle_timeout,
            tool_output_dir=args.tool_output_dir,
            spinner_enabled=args.spinner_enabled,
            spinner_style=args.spinner_style,
            spinner_interval=args.spinner_interval,
            spinner_stall_delay=args.spinner_stall_delay
        )

    print(f"{model}: ", end="", flush=True)
    log_to_file(file_handle, f"{model}: ")

    full_response = ""
    stream = _stream_chat_with_retry(
        model=model,
        messages=messages,
        config=retry_config,
        start_timeout=args.ollama_first_token_timeout,
        idle_timeout=args.ollama_stream_idle_timeout,
        spinner_enabled=args.spinner_enabled,
        spinner_style=args.spinner_style,
        spinner_interval=args.spinner_interval,
        spinner_stall_delay=args.spinner_stall_delay
    )

    for chunk in stream:
        part = chunk['message']['content']
        render_stream_text(part, file_handle, delay=args.render_delay)
        full_response += part

    return full_response


def _respond_with_fallbacks(
    model_candidates: List[str],
    messages: List[Dict[str, str]],
    args: argparse.Namespace,
    file_handle: TextIO,
    retry_config: RetryConfig
) -> Dict[str, str]:
    last_error: Optional[Exception] = None
    for model in model_candidates:
        try:
            response = _run_chat_turn(model, messages, args, file_handle, retry_config)
            return {"model": model, "response": response}
        except Exception as exc:
            last_error = exc
            warning = f"[Warning] Model '{model}' failed after retries: {exc}"
            print(f"\n{warning}")
            log_to_file(file_handle, f"\n{warning}\n")
            if model != model_candidates[-1]:
                next_model = model_candidates[model_candidates.index(model) + 1]
                switch_msg = f"[Failover] Switching to fallback model: {next_model}"
                print(switch_msg)
                log_to_file(file_handle, f"{switch_msg}\n")

    if last_error:
        raise last_error
    raise RuntimeError("No available models to respond.")

def main() -> None:
    args = parse_arguments()
    _apply_runtime_config(args)

    # ------------------------------------------------------------
    # Session handling commands (list, select, export)
    # ------------------------------------------------------------
    if args.list_sessions:
        dbm = get_database_manager(sync=True)
        dbm.create_tables()
        sessions = dbm.get_all_conversations()
        for conv in sessions:
            print(f"ID: {conv.id} | Model: {conv.model} | Created: {conv.created_at}")
        return

    if args.select_session:
        dbm = get_database_manager(sync=True)
        dbm.create_tables()
        sessions = dbm.get_all_conversations()
        if not sessions:
            print("No saved sessions found.")
            return
        choices = [f"{c.id}: {c.model} ({c.created_at})" for c in sessions]
        if HAS_QUESTIONARY:
            answer = questionary.select("Select a conversation to continue:", choices=choices).ask()
        else:
            print("Select a conversation to continue:")
            for i, choice in enumerate(choices, 1):
                print(f"{i}) {choice}")
            sel = input("Enter number: ")
            try:
                idx = int(sel) - 1
                answer = choices[idx]
            except Exception:
                print("Invalid selection.")
                return
        selected_id = int(answer.split(":")[0])
        conv = dbm.get_conversation(selected_id)
        if conv is None:
            print(f"Conversation {selected_id} not found.")
            return
        # Pre‑populate messages with saved conversation and continue
        messages = conv.messages
        print(f"Resuming conversation ID {selected_id} (model={conv.model})")
        # Skip the rest of the initialization that would create a new DB manager
        db_manager = dbm
        # Continue to the chat loop with pre‑loaded messages
        # (the rest of main() will use the `messages` variable defined later)
        # We'll set a flag to indicate we already loaded messages.
        _preloaded_messages = messages
        # Jump to after the DB init block by using a guard later.
        pass

    if args.export_session is not None:
        dbm = get_database_manager(sync=True)
        dbm.create_tables()
        conv = dbm.get_conversation(args.export_session)
        if conv is None:
            print(f"Conversation {args.export_session} not found.")
            return
        out_target = sys.stdout if args.output is None else args.output.open("w", encoding="utf-8")
        if args.format == "json":
            json.dump({
                "id": conv.id,
                "model": conv.model,
                "flags": conv.flags,
                "messages": conv.messages,
                "created_at": conv.created_at.isoformat() if conv.created_at else None,
                "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
            }, out_target, indent=2)
        elif args.format == "csv":
            writer = csv.writer(out_target)
            writer.writerow(["role", "content"])
            for msg in conv.messages:
                writer.writerow([msg.get("role", ""), msg.get("content", "")])
        elif args.format == "text":
            for msg in conv.messages:
                out_target.write(f"{msg.get('role', '')}: {msg.get('content', '')}\n\n")
        elif args.format == "sql":
            sql = textwrap.dedent(f"""
                INSERT INTO conversations (id, model, flags, messages, created_at, updated_at)
                VALUES ({conv.id}, '{conv.model}', '{json.dumps(conv.flags)}', '{json.dumps(conv.messages)}',
                '{conv.created_at.isoformat() if conv.created_at else None}',
                '{conv.updated_at.isoformat() if conv.updated_at else None}');
            """)
            out_target.write(sql)
        if args.output is not None:
            out_target.close()
        print(f"Conversation {args.export_session} exported as {args.format}.")
        return

    retry_config = _get_retry_config_from_args(args)
    model_fallbacks = _parse_comma_list(args.model_fallbacks)
    model_candidates = _build_model_list(args.model, model_fallbacks)

    # Ensure models exist
    available_models: List[str] = []
    for model in model_candidates:
        try:
            _ensure_model_available(model, retry_config)
            available_models.append(model)
        except Exception as exc:
            print(f"Warning: Unable to load model '{model}': {exc}")

    if not available_models:
        print("Error: No models could be loaded. Exiting.")
        return

    # Initialize database manager if needed
    db_manager = None
    if args.persist_to_db:
        if not HAS_DB:
            print("Error: Database dependencies not installed. Run: pip install psycopg2-binary asyncpg")
            return
    
    try:
        db_manager = get_database_manager(sync=True)
        # Create tables if they don't exist
        db_manager.create_tables()
        print("[Database] Tables created/verified successfully")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Tip: Ensure PostgreSQL is running and DATABASE_URL is set correctly")
        return

    # Initialize chat history (use preloaded messages if a session was selected)
    try:
        # _preloaded_messages is set only when --select-session was used
        messages: List[Dict[str, str]] = _preloaded_messages  # type: ignore
    except NameError:
        messages: List[Dict[str, str]] = []
    
    # Parse context argument
    context_config = parse_context_arg(args.context)
    use_db_context = context_config.get("use_db", False)
    
    # Load context if provided
    context_content = ""
    if args.context:
        if context_config['use_db']:
            # Load from database
            if db_manager is None:
                print("Error: Cannot load from database without --persist-to-db flag")
                return
            
            try:
                extensions = [ext.strip() for ext in args.context_grep.split(',')]
                context_content = load_context_from_database(
                    db_manager, 
                    additional_paths=context_config['paths'],
                    extensions=extensions
                )
                if context_content:
                    # Add context as a system message
                    messages.append({
                        "role": "system",
                        "content": f"You have access to the following context:\n\n{context_content}"
                    })
                    print(f"[Loaded context from database]")
                    if context_config['paths']:
                        print(f"[Also loaded context from: {', '.join(context_config['paths'])}]")
            except Exception as e:
                print(f"Error loading context from database: {e}")
        else:
            # Load from files only
            extensions = [ext.strip() for ext in args.context_grep.split(',')]
            context_content = load_context_files(args.context, extensions)
            if context_content:
                # Add context as a system message
                messages.append({
                    "role": "system",
                    "content": f"You have access to the following context files:\n\n{context_content}"
                })
                print(f"[Loaded context from: {args.context}]")
                print(f"[File extensions: {', '.join(extensions)}]")
    
    print(f"Starting chat with {available_models[0]}.")
    if len(available_models) > 1:
        print(f"Fallback models: {', '.join(available_models[1:])}")
    print(f"Experimental Mode: {'ON' if args.experimental else 'OFF'}")
    print(f"Intelligent Web Search: {'ON (LLM decides when to search)' if args.experimental_websearch else 'OFF'}")
    if args.context:
        print(f"Context: {args.context}")
    print(f"Database Persistence: {'ON' if args.persist_to_db else 'OFF'}")
    if use_db_context:
        print("Conversation logging: OFF (context includes 'db')\n")
    else:
        print(f"Conversation is being saved to: {os.path.abspath(args.dest)}\n")
    print("Type 'exit' or 'quit' to stop.\n")

    # Open file in append mode (or /dev/null when context includes db)
    log_path = os.devnull if use_db_context else args.dest
    log_mode = "w" if use_db_context else "a"
    try:
        with open(log_path, log_mode, encoding="utf-8") as f:
            if not use_db_context:
                # Write session header
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                header = (
                    f"\n\n{'='*30}\n"
                    f"SESSION START: {timestamp}\n"
                    f"MODEL: {available_models[0]}\n"
                    f"FLAGS: exp={args.experimental}, web={args.experimental_websearch}\n"
                )
                if len(available_models) > 1:
                    header += f"FALLBACKS: {', '.join(available_models[1:])}\n"
                if args.context:
                    header += f"CONTEXT: {args.context}\n"
                if args.persist_to_db:
                    header += f"DB_PERSIST: true\n"
                header += f"{'='*30}\n"
                log_to_file(f, header)

            current_model_index = 0
            conversation_id = None  # Track database conversation ID

            while True:
                try:
                    # Get user input
                    user_input = input("You: ")
                    if user_input.lower() in ["exit", "quit"]:
                        print("Exiting...")
                        break
                    
                    # Store original query for logging
                    original_query = user_input
                    
                    # Add user message to history
                    messages.append({"role": "user", "content": user_input})
                    log_to_file(f, f"\nUser: {original_query}\n")

                    # Print divider for visual separation
                    print(f"\n{'─' * 50}")

                    response_info = _respond_with_fallbacks(
                        available_models[current_model_index:],
                        messages,
                        args,
                        f,
                        retry_config
                    )

                    full_response = response_info["response"]
                    used_model = response_info["model"]

                    if used_model in available_models:
                        current_model_index = available_models.index(used_model)
                    
                    # Add spacing after response
                    print(f"\n{'─' * 50}\n")
                    
                    # Add final newline to file and history
                    log_to_file(f, "\n")
                    messages.append({"role": "assistant", "content": full_response})
                    
                    # Save to database if enabled
                    if args.persist_to_db and db_manager:
                        try:
                            flags = {
                                'experimental': args.experimental,
                                'experimental_websearch': args.experimental_websearch,
                                'model_fallbacks': model_fallbacks,
                                'context': args.context
                            }
                            
                            if conversation_id is None:
                                # Save new conversation
                                conversation_id = db_manager.save_conversation(
                                    model=used_model,
                                    messages=messages.copy(),
                                    flags=flags
                                )
                                print(f"[Saved conversation to database with ID: {conversation_id}]")
                            else:
                                # Update existing conversation
                                db_manager.update_conversation(conversation_id, messages.copy())
                        except Exception as e:
                            print(f"[Warning] Failed to save to database: {e}")

                except KeyboardInterrupt:
                    print("\n\nChat interrupted by user.")
                    break
                except Exception as e:
                    print(f"\nError: {e}")
                    # If it's a connection error, it might be due to Docker host issues
                    if "Connection refused" in str(e):
                        print("Tip: If running in Docker, ensure OLLAMA_HOST is set correctly to reach your host machine.")
                    break

    except IOError as e:
        print(f"Error opening log file {args.dest}: {e}")
    
    finally:
        # Close database connection if used
        if db_manager:
            try:
                db_manager.close()
            except:
                pass

def entry_point():
    """Entry point for uv run and direct execution."""
    main()


if __name__ == "__main__":
    entry_point()
