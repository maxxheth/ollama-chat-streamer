import ollama
import sys
import os
import argparse
import json
import glob
import time
import random
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass
from typing import List, Dict, TextIO, Any, Optional, Callable, Iterable
from dotenv import load_dotenv

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
    from ddgs import DDGS  # type: ignore
    HAS_DDG = True
except ImportError:
    HAS_DDG = False


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
            stream = ollama.chat(model=model, messages=messages, stream=True, **kwargs)
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
                "name": "read_json_file",
                "description": "Read and parse a JSON file efficiently without loading everything into memory at once. Supports newline-delimited JSON (NDJSON) and regular JSON with safeguards for large files.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Path to the JSON file to read"
                        },
                        "max_entries": {
                            "type": "integer",
                            "description": "Maximum number of entries to read (default: 100, use -1 for all)",
                            "default": 100
                        },
                        "query_filter": {
                            "type": "string",
                            "description": "Optional dot/array path filter (e.g., 'conversations[*].messages[*].content')"
                        },
                        "return_summary": {
                            "type": "boolean",
                            "description": "If true, return a summary instead of full data",
                            "default": False
                        }
                    },
                    "required": ["file_path"]
                }
            }
        }
    ]


def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """Execute a tool call and return the result."""
    function_name = tool_call.get("function", {}).get("name")
    arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
    
    if function_name == "web_search":
        query = arguments.get("query", "")
        return perform_web_search(query)

    if function_name == "read_json_file":
        return read_json_file(
            file_path=arguments.get("file_path", ""),
            max_entries=int(arguments.get("max_entries", 100)),
            query_filter=arguments.get("query_filter", ""),
            return_summary=bool(arguments.get("return_summary", False))
        )
    
    return f"[Error: Unknown tool '{function_name}']"


def chat_with_tools(
    model: str,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    file_handle: TextIO,
    retry_config: RetryConfig
) -> str:
    """
    Chat with the model, handling tool calls automatically.
    Returns the final assistant response.
    """
    # First call - let the model decide if it needs tools
    response = _retry_call(
        lambda: ollama.chat(
            model=model,
            messages=messages,
            tools=tools
        ),
        retry_config
    )
    
    message = response.message
    
    # Check if the model wants to use tools
    if hasattr(message, 'tool_calls') and message.tool_calls:
        print(f"\n[Tool calls detected: {[tc.function.name for tc in message.tool_calls]}]")
        
        # Add the assistant's tool call request to history
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments
                    }
                } for tc in message.tool_calls
            ]
        })
        
        # Execute each tool call and add results
        for tool_call in message.tool_calls:
            result = execute_tool_call({
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments
                }
            })
            
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id
            })
            log_to_file(file_handle, f"\n[Tool '{tool_call.function.name}' used]\n")
        
        # Second call - get the final response with tool results
        print(f"{model}: ", end="", flush=True)
        log_to_file(file_handle, f"{model}: ")
        
        full_response = ""
        stream = _stream_chat_with_retry(
            model=model,
            messages=messages,
            config=retry_config
        )
        
        for chunk in stream:
            part = chunk['message']['content']
            print(part, end="", flush=True)
            log_to_file(file_handle, part)
            full_response += part
        
        return full_response
    else:
        # No tool calls - just stream the response
        print(f"{model}: ", end="", flush=True)
        log_to_file(file_handle, f"{model}: ")
        
        full_response = ""
        # Stream the content if available
        if message.content:
            print(message.content, end="", flush=True)
            log_to_file(file_handle, message.content)
            full_response = message.content
        
        return full_response

def parse_arguments() -> argparse.Namespace:
    """
    Parses CLI arguments, allowing environment variables to serve as defaults.
    Precedence: CLI Flag > Environment Variable > Default Value
    """
    parser = argparse.ArgumentParser(description="Stream chat with Ollama models.")

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

    parser.add_argument(
        "--context-grep",
        type=str,
        default=os.environ.get("CONTEXT_GREP", "txt,log"),
        help="Comma-separated list of file extensions to include when loading context from a directory (default: txt,log)"
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
    limit: Optional[int] = None
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
        for path in additional_paths:
            if path and path != 'db':
                content = load_context_files(path, ['txt', 'log'])
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


def _run_chat_turn(
    model: str,
    messages: List[Dict[str, str]],
    args: argparse.Namespace,
    file_handle: TextIO,
    retry_config: RetryConfig
) -> str:
    if args.experimental_websearch and HAS_DDG:
        return chat_with_tools(
            model=model,
            messages=messages,
            tools=get_tools(),
            file_handle=file_handle,
            retry_config=retry_config
        )

    print(f"{model}: ", end="", flush=True)
    log_to_file(file_handle, f"{model}: ")

    full_response = ""
    stream = _stream_chat_with_retry(
        model=model,
        messages=messages,
        config=retry_config
    )

    for chunk in stream:
        part = chunk['message']['content']
        print(part, end="", flush=True)
        log_to_file(file_handle, part)
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

    # Initialize chat history
    messages: List[Dict[str, str]] = []
    
    # Parse context argument
    context_config = parse_context_arg(args.context)
    
    # Load context if provided
    context_content = ""
    if args.context:
        if context_config['use_db']:
            # Load from database
            if db_manager is None:
                print("Error: Cannot load from database without --persist-to-db flag")
                return
            
            try:
                context_content = load_context_from_database(
                    db_manager, 
                    additional_paths=context_config['paths']
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
    print(f"Conversation is being saved to: {os.path.abspath(args.dest)}\n")
    print("Type 'exit' or 'quit' to stop.\n")

    # Open file in append mode
    try:
        with open(args.dest, "a", encoding="utf-8") as f:
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

if __name__ == "__main__":
    main()
# Database integration complete. Conversations can now be persisted to PostgreSQL with --persist-to-db flag and loaded with --context db.
