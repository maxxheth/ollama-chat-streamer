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
from pathlib import Path
from typing import Any, Dict, List

# DuckDuckGo web search tool (optional)
try:
    from ddgs import DDGS
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
        }
    ]


def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """Dispatch a tool call to the appropriate implementation."""
    function_name = tool_call.get("function", {}).get("name")
    arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))

    if function_name == "web_search":
        return perform_web_search(arguments.get("query", ""))
    if function_name == "read_json_file":
        return read_json_file(
            file_path=arguments.get("file_path", ""),
            max_entries=int(arguments.get("max_entries", 100)),
            query_filter=arguments.get("query_filter", ""),
            return_summary=bool(arguments.get("return_summary", False)),
        )
    return f"[Error: Unknown tool '{function_name}']"


def perform_web_search(query: str) -> str:
    """Perform a simple DuckDuckGo search and return formatted results."""
    if not HAS_DDG:
        return "[System Error: duckduckgo-search library not installed, cannot search web.]"
    print(f"\n[Searching web for: '{query}'...]")
    try:
        # Import DDGS lazily to avoid import errors when the library is missing.
        from ddgs import DDGS  # type: ignore
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
