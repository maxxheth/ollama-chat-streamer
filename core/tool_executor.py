"""Tool execution utilities for Ollama Chat Streamer.

Provides the tool schema (get_tools), the dispatch function
(execute_tool_call), and all built-in tool implementations.
"""

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError

from .retry_handler import RetryConfig, _run_with_timeout

# DuckDuckGo web search tool (optional)
try:
    from ddgs import DDGS
    HAS_DDG = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDG = True
    except ImportError:
        HAS_DDG = False


def get_tools(max_subagent_depth: int = 1) -> List[Dict[str, Any]]:
    """Return the list of available tools for the model."""
    tools = [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web for current information, news, facts, or data that may not be in the model's training data. Use this when the user asks about current events, recent news, specific facts you're unsure about, or time-sensitive information.",
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
                "description": "Read the contents of a text file. Files over 512 KB are rejected — use start_line/max_lines to read in chunks. Default returns up to 200 lines.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the file to read"
                        },
                        "max_lines": {
                            "type": "integer",
                            "description": "Maximum number of lines to read (default: 200)",
                            "default": 200
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
                "description": "Read and filter JSON or NDJSON files. Supports JMESPath-like filtering.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the JSON file to read"
                        },
                        "max_entries": {
                            "type": "integer",
                            "description": "Maximum number of entries to return (default: 100)",
                            "default": 100
                        },
                        "query_filter": {
                            "type": "string",
                            "description": "JMESPath-like filter expression"
                        },
                        "return_summary": {
                            "type": "boolean",
                            "description": "If true, return a summary instead of full data (default: false)",
                            "default": False
                        }
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
                "description": "Append content to a file. Creates the file if it doesn't exist.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "File path to append to"
                        },
                        "content": {
                            "type": "string",
                            "description": "The content to append"
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
                "description": "List files and directories in a path. Supports glob pattern filtering and recursive listing.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "directory_path": {
                            "type": "string",
                            "description": "Path to the directory to list"
                        },
                        "pattern": {
                            "type": "string",
                            "description": "Glob pattern to filter files (default: *)",
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
        },
    ]

    if max_subagent_depth > 0:
        tools.append({
            "type": "function",
            "function": {
                "name": "spawn_agent",
                "description": "Spawn a subagent to handle a task autonomously. The subagent can use read_file, write_file, append_file, list_directory, run_shell, and web_search to complete the task. It works through the task step by step and returns a final answer. Use this for complex multi-step tasks like analyzing files, running investigation sequences, or parallel research.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "task": {
                            "type": "string",
                            "description": "Clear description of what the subagent should accomplish"
                        },
                        "context": {
                            "type": "string",
                            "description": "Optional additional context: file paths, constraints, or hints to help the subagent"
                        },
                    },
                    "required": ["task"],
                },
            },
        })

    return tools


def execute_tool_call(
    tool_call: Dict[str, Any],
    tool_timeout: Optional[float] = None,
    web_search_timeout: Optional[float] = None,
    output_dir: Optional[str] = None,
    read_file_max_bytes: int = 512_000,
    read_file_max_lines: int = 200,
    read_file_max_content: int = 64_000,
    subagent_model: Optional[str] = None,
    subagent_retry_config: Optional[RetryConfig] = None,
    subagent_max_rounds: int = 5,
    subagent_current_depth: int = 1,
    subagent_max_depth: int = 1,
    subagent_tools: Optional[List[Dict[str, Any]]] = None,
    subagent_ollama_timeout: Optional[float] = None,
    subagent_first_token_timeout: Optional[float] = None,
    subagent_stream_idle_timeout: Optional[float] = None,
    subagent_tool_timeout: Optional[float] = None,
    subagent_web_search_timeout: Optional[float] = None,
    subagent_tool_output_dir: Optional[str] = None,
    subagent_think_setting: str = "auto",
) -> str:
    """Dispatch a tool call to the appropriate implementation."""
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
                    max_lines=int(arguments.get("max_lines", read_file_max_lines)),
                    start_line=int(arguments.get("start_line", 1)),
                    max_bytes=read_file_max_bytes,
                    max_content=read_file_max_content,
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
                "run_shell timed out",
            )
        except TimeoutError:
            return "[Tool Timeout: run_shell exceeded configured limit]"

    if function_name == "write_output_file":
        file_path = arguments.get("file_path", "")
        content = arguments.get("content", "")
        return write_output_file(
            file_path=file_path, content=content, output_dir=output_dir
        )

    if function_name == "spawn_agent":
        from .subagent import run_subagent
        task = arguments.get("task", "")
        context = arguments.get("context", "")
        if not task:
            return "[Error: task is required for spawn_agent]"
        if not subagent_model:
            return "[Error: subagent model not configured]"
        if not subagent_tools:
            subagent_tools = get_tools(max_subagent_depth=max(0, subagent_max_depth))
        if not subagent_retry_config:
            subagent_retry_config = RetryConfig(max_attempts=2, initial_delay=1.0, max_delay=10.0, multiplier=2.0, jitter=0.1)
        try:
            return _run_with_timeout(
                lambda: run_subagent(
                    task=task,
                    context=context,
                    model=subagent_model,
                    tools=subagent_tools,
                    retry_config=subagent_retry_config,
                    max_rounds=subagent_max_rounds,
                    current_depth=subagent_current_depth,
                    max_depth=subagent_max_depth,
                    tool_timeout=subagent_tool_timeout or tool_timeout,
                    web_search_timeout=subagent_web_search_timeout or web_search_timeout,
                    ollama_timeout=subagent_ollama_timeout,
                    ollama_first_token_timeout=subagent_first_token_timeout,
                    ollama_stream_idle_timeout=subagent_stream_idle_timeout,
                    tool_output_dir=subagent_tool_output_dir or output_dir,
                    read_file_max_bytes=read_file_max_bytes,
                    read_file_max_lines=read_file_max_lines,
                    read_file_max_content=read_file_max_content,
                    think_setting=subagent_think_setting,
                ),
                tool_timeout,
                "spawn_agent timed out"
            )
        except TimeoutError:
            return "[Tool Timeout: spawn_agent exceeded configured limit]"

    return f"[Error: Unknown tool '{function_name}']"


# ─── Tool implementations ────────────────────────────────────────────────


def perform_web_search(query: str) -> str:
    """Performs a web search using DuckDuckGo and returns formatted results."""
    if not HAS_DDG:
        return "[Error: DuckDuckGo search is not available. Install ddgs or duckduckgo-search.]"

    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
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


def read_file(file_path: str, max_lines: int = 200, start_line: int = 1, max_bytes: int = 512_000, max_content: int = 64_000) -> str:
    """Read a text file with optional line limits and size cap."""
    path = Path(file_path)
    if not file_path:
        return "[Error: file_path is required]"
    if not path.exists():
        return f"[Error: File not found: {file_path}]"
    if not path.is_file():
        return f"[Error: Path is not a file: {file_path}]"

    try:
        size = path.stat().st_size
        if size > max_bytes:
            return f"[Error: File too large ({size:,} bytes, limit {max_bytes:,}). Use start_line/max_lines to read in chunks.]"
    except OSError:
        pass

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

        if len(content) > max_content:
            content = content[:max_content] + f"\n\n... [truncated, {total_lines} total lines]"
            shown_lines = content.count('\n')

        return f"[File: {file_path} (showing {shown_lines} of {total_lines} lines)]\n\n{content}"
    except Exception as e:
        return f"[Error reading file: {str(e)}]"


def _decode_escapes(s: str) -> str:
    if '\\' not in s:
        return s
    s = s.replace('\\n', '\n')
    s = s.replace('\\t', '\t')
    s = s.replace('\\r', '\r')
    return s


def write_file(file_path: str, content: str, create_dirs: bool = True) -> str:
    """Write content to a file, creating directories if needed."""
    if not file_path:
        return "[Error: file_path is required]"

    content = _decode_escapes(content)
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

    content = _decode_escapes(content)
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

        return f"[File appended: {path}] ({len(content)} bytes)"
    except Exception as e:
        return f"[Error appending to file: {str(e)}]"


def list_directory(directory_path: str, pattern: str = "*", recursive: bool = False) -> str:
    """List files and directories in a directory with optional glob filtering."""
    if not directory_path:
        return "[Error: directory_path is required]"

    path = Path(directory_path)
    if not path.exists():
        return f"[Error: Directory not found: {directory_path}]"
    if not path.is_dir():
        return f"[Error: Path is not a directory: {directory_path}]"

    try:
        if recursive:
            items = sorted(path.rglob(pattern))
        else:
            items = sorted(path.glob(pattern))

        lines = []
        for item in items:
            try:
                rel = item.relative_to(path)
                if item.is_dir():
                    lines.append(f"  {rel}/")
                else:
                    size = item.stat().st_size
                    lines.append(f"  {rel} ({size} bytes)")
            except (OSError, ValueError):
                continue

        if not lines:
            return f"[No items found in {directory_path} matching '{pattern}']"

        return f"[Directory: {directory_path} ({len(lines)} items matching '{pattern}')]\n" + "\n".join(lines)
    except Exception as e:
        return f"[Error listing directory: {str(e)}]"


def run_shell(command: str, timeout: int = 30, workdir: Optional[str] = None) -> str:
    """Execute a shell command and return its output."""
    if not command:
        return "[Error: command is required]"

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=workdir,
        )

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")

        output = "\n".join(output_parts) if output_parts else "[No output]"

        if result.returncode != 0:
            output = f"[Exit code: {result.returncode}]\n{output}"

        return output
    except subprocess.TimeoutExpired:
        return f"[Error: Command timed out after {timeout}s]"
    except Exception as e:
        return f"[Error running command: {str(e)}]"


def read_json_file(
    file_path: str,
    max_entries: int = 100,
    query_filter: str = "",
    return_summary: bool = False,
) -> str:
    """Read and optionally filter a JSON or NDJSON file."""
    if not file_path:
        return "[Error: file_path is required]"

    path = Path(file_path)
    if not path.exists():
        return f"[Error: File not found: {file_path}]"
    if not path.is_file():
        return f"[Error: Path is not a file: {file_path}]"

    print(f"\n[Reading JSON file: {file_path}]")
    try:
        is_ndjson = _detect_ndjson(path)

        if is_ndjson:
            entries = _read_ndjson_streaming(path, query_filter, max_entries)
        else:
            entries = _read_regular_json_streaming(path, query_filter, max_entries)

        if not entries:
            return f"[No entries found in {file_path}]"

        if return_summary:
            structure = _get_structure(entries[0] if entries else {})
            return f"[Summary: {len(entries)} entries found in {file_path}]\nStructure: {json.dumps(structure, indent=2)}"

        return json.dumps(entries, indent=2, default=str)
    except Exception as e:
        return f"[Error reading JSON file: {str(e)}]"


def _resolve_output_path(file_path: str, output_dir: Optional[str] = None) -> str:
    """Resolve an output file path, optionally within a session directory."""
    if not file_path:
        return ""

    path = Path(file_path)

    if output_dir and not path.is_absolute():
        session_dir = Path(output_dir)
        session_dir.mkdir(parents=True, exist_ok=True)
        return str(session_dir / file_path)

    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def write_output_file(file_path: str, content: str, output_dir: Optional[str] = None) -> str:
    """Write content to a file within the output directory."""
    resolved = _resolve_output_path(file_path, output_dir)
    if not resolved:
        return "[Error: file_path is required]"
    return write_file(resolved, content)


def _detect_ndjson(path: Path) -> bool:
    """Detect if a file is in Newline-Delimited JSON (NDJSON) format."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline(8192).strip()
        if not first_line:
            return False
        try:
            json.loads(first_line)
            return True
        except json.JSONDecodeError:
            return False
    except Exception:
        return False


def _read_ndjson_streaming(
    path: Path, query_filter: str = "", max_entries: int = 100
) -> list:
    """Read an NDJSON file line by line with optional filtering."""
    entries = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                if query_filter:
                    filtered = _extract_fields(entry, query_filter)
                    if filtered is not None:
                        entries.append(filtered)
                else:
                    entries.append(entry)
                if len(entries) >= max_entries:
                    break
            except json.JSONDecodeError:
                continue
    return entries


def _read_regular_json_streaming(
    path: Path, query_filter: str = "", max_entries: int = 100
) -> list:
    """Read a regular JSON file with optional filtering."""
    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            data = json.load(f)

        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            if query_filter:
                return [_extract_fields(data, query_filter)]
            return [data]
        else:
            return [data]

        if query_filter:
            filtered = []
            for entry in entries:
                result = _extract_fields(entry, query_filter)
                if result is not None:
                    filtered.append(result)
                if len(filtered) >= max_entries:
                    break
            return filtered

        return entries[:max_entries]
    except Exception as e:
        return [{"error": str(e)}]


def _extract_fields(data: Any, filter_path: str) -> Any:
    """Extract fields from data using a dot-separated path with wildcard index [*] support."""
    if not filter_path:
        return data

    parts = filter_path.split('.')
    current = data

    for part in parts:
        if current is None:
            return None

        if part.endswith('[*]'):
            key = part[:-3]
            if isinstance(current, dict) and key in current:
                current = current[key]
            elif isinstance(current, dict):
                current = current.get(part, None)
                continue

            if isinstance(current, list):
                results = []
                for item in current:
                    remaining = '.'.join(parts[parts.index(part) + 1:])
                    if remaining:
                        result = _extract_fields(item, remaining)
                        if result is not None:
                            if isinstance(result, list):
                                results.extend(result)
                            else:
                                results.append(result)
                    else:
                        results.append(item)
                return results
            elif isinstance(current, dict):
                results = []
                for value in current.values():
                    remaining = '.'.join(parts[parts.index(part) + 1:])
                    if remaining:
                        result = _extract_fields(value, remaining)
                        if result is not None:
                            if isinstance(result, list):
                                results.extend(result)
                            else:
                                results.append(result)
                    else:
                        results.append(value)
                return results
            else:
                return None
        elif isinstance(current, dict):
            current = current.get(part, None)
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx] if 0 <= idx < len(current) else None
            except (ValueError, IndexError):
                current = None
        else:
            return None

    return current


def _get_structure(data: Any, max_depth: int = 3) -> Dict[str, Any]:
    """Analyze the structure of a data object."""
    if isinstance(data, dict):
        return {k: _get_structure(v, max_depth - 1) if max_depth > 0 else type(v).__name__ for k, v in data.items()}
    elif isinstance(data, list):
        if data:
            return f"list[{_get_structure(data[0], max_depth - 1) if max_depth > 0 else type(data[0]).__name__}]"
        return "list[empty]"
    else:
        return type(data).__name__