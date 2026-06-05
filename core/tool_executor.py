"""Tool execution utilities for Ollama Chat Streamer.

This module provides the tool schema, the dispatcher that executes a tool
call, and the concrete implementations for the built‑in tools:

* ``web_search`` – uses DuckDuckGo via the ``ddgs`` package.
* ``read_json_file`` – streams JSON or NDJSON files with optional filtering.

Helper functions for JSON handling are also included. The original
implementations lived in ``stream_chat.py``; they have been moved here to keep the
entry‑point minimal and to make the tool logic independently testable.
"""

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def get_tools() -> List[Dict[str, Any]]:
    """Return the list of available tools for the model."""
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


def execute_tool_call(tool_call: Dict[str, Any], output_dir: Optional[str] = None) -> str:
    """Dispatch a tool call to the appropriate implementation."""
    function_name = tool_call.get("function", {}).get("name")
    arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))

    if function_name == "web_search":
        return perform_web_search(arguments.get("query", ""))
    if function_name == "read_file":
        return read_file(
            file_path=arguments.get("file_path", ""),
            max_lines=int(arguments.get("max_lines", 1000)),
            start_line=int(arguments.get("start_line", 1)),
        )
    if function_name == "read_json_file":
        return read_json_file(
            file_path=arguments.get("file_path", ""),
            max_entries=int(arguments.get("max_entries", 100)),
            query_filter=arguments.get("query_filter", ""),
            return_summary=bool(arguments.get("return_summary", False)),
        )
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
        return run_shell(
            command=arguments.get("command", ""),
            timeout=int(arguments.get("timeout", 30)),
            workdir=arguments.get("workdir"),
        )
    if function_name == "write_output_file":
        return write_output_file(
            file_path=arguments.get("file_path", ""),
            content=arguments.get("content", ""),
            output_dir=output_dir
        )
    return f"[Error: Unknown tool '{function_name}']"


def perform_web_search(query: str) -> str:
    """Perform a simple DuckDuckGo search and return formatted results."""
    if not HAS_DDG:
        return "[System Error: duckduckgo-search library not installed, cannot search web.]"
    print(f"\n[Searching web for: '{query}'...]")
    try:
        try:
            from ddgs import DDGS  # type: ignore
        except ImportError:
            from duckduckgo_search import DDGS  # type: ignore
        results = list(DDGS().text(query, max_results=3))
        if not results:
            return "No web results found."
        parts = ["Web Search Results:"]
        for res in results:
            title = res.get('title', 'No Title')
            body = res.get('body', 'No Content')
            href = res.get('href', '')
            parts.append(f"- Source: {title} ({href})\n  Content: {body}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"[Web Search Error: {str(e)}]"


def read_file(file_path: str, max_lines: int = 1000, start_line: int = 1) -> str:
    """Read a text file with optional line limits.
    
    Args:
        file_path: Path to the file to read
        max_lines: Maximum lines to read (-1 for all)
        start_line: Starting line number (1-indexed)
    """
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
        
        # Apply line range
        start_idx = max(0, start_line - 1)
        if max_lines > 0:
            end_idx = start_idx + max_lines
            selected_lines = lines[start_idx:end_idx]
        else:
            selected_lines = lines[start_idx:]
        
        # Add line numbers
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
            # Check if file ends with newline
            with open(path, 'rb') as f:
                f.seek(0, 2)  # Go to end
                if f.tell() > 0:
                    f.seek(-1, 2)  # Go to last byte
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
        
        # Sort: directories first, then files
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
    return_summary: bool = False,
) -> str:
    """Read a JSON file with optional streaming and filtering.

    Supports NDJSON (newline‑delimited) and regular JSON. Returns either the
    full data or a summary structure depending on ``return_summary``.
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
        if _detect_ndjson(path):
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
    """Detect NDJSON by attempting to parse the first two lines."""
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            first = f.readline().strip()
            if not first:
                return False
            json.loads(first)
            second = f.readline().strip()
            if second:
                json.loads(second)
            return True
    except Exception:
        return False


def _read_ndjson_streaming(
    path: Path,
    max_entries: int,
    query_filter: str,
    return_summary: bool,
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
            "file_path": str(path),
        }
        return json.dumps(summary, indent=2)
    return json.dumps(results, indent=2)


def _read_regular_json_streaming(
    path: Path,
    max_entries: int,
    query_filter: str,
    return_summary: bool,
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
        entries = data[:max_entries] if max_entries and max_entries > 0 else data
        if query_filter:
            entries = [_extract_fields(entry, query_filter) for entry in entries]
        if return_summary:
            summary = {
                "format": "json",
                "total_entries": len(data),
                "entries_returned": len(entries),
                "sample_structure": _get_structure(data[0]) if data else None,
                "file_path": str(path),
            }
            return json.dumps(summary, indent=2)
        return json.dumps(entries, indent=2)
    if return_summary:
        summary = {
            "format": "json",
            "type": "object",
            "structure": _get_structure(data),
            "file_path": str(path),
        }
        return json.dumps(summary, indent=2)
    return json.dumps(data, indent=2)


def _extract_fields(data: Any, filter_path: str) -> Any:
    """Extract fields using a simple dot/array path like ``conversations[*].messages[*].content``."""
    if not filter_path:
        return data
    parts = filter_path.split('.')
    result: Any = data
    for part in parts:
        if "[" in part and part.endswith("]"):
            key = part.split("[")[0]
            idx = part.split("[")[1].rstrip("]")
            result = result.get(key, [])
            try:
                index = int(idx)
                if isinstance(result, list) and index < len(result):
                    result = result[index]
            except ValueError:
                pass
        else:
            if isinstance(result, dict):
                result = result.get(part)
    return result


def _get_structure(data: Any, max_depth: int = 3) -> Dict[str, Any]:
    if max_depth <= 0:
        return {"type": type(data).__name__}
    if isinstance(data, dict):
        return {k: _get_structure(v, max_depth - 1) for k, v in list(data.items())[:10]}
    if isinstance(data, list):
        if data:
            return {"type": "array", "length": len(data), "sample": _get_structure(data[0], max_depth - 1)}
        return {"type": "array", "length": 0}
    return {"type": type(data).__name__}
