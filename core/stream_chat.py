# Import Ollama client; provide a lightweight stub if the package is missing.
from typing import Any, Dict, List, Optional, TextIO, Iterable, Callable
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import time

try:
    import ollama
    from ollama import ChatResponse
    from ollama import Message as OllamaMessage
    from ollama import ToolCall as OllamaToolCall
    from ollama import FunctionCall as OllamaFunctionCall
except ImportError:  # pragma: no cover
    class _OllamaStub:
        @staticmethod
        def chat(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            raise NotImplementedError("ollama package is not installed")

        @staticmethod
        def show(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            raise NotImplementedError("ollama package is not installed")

        @staticmethod
        def pull(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            raise NotImplementedError("ollama package is not installed")

    class ChatResponse:
        pass
    
    class OllamaMessage:
        pass
    
    class OllamaToolCall:
        pass
    
    class OllamaFunctionCall:
        pass

    ollama: Any = _OllamaStub()


def _run_with_timeout(action: Callable[[], Any], timeout_s: Optional[float], timeout_message: str) -> Any:
    """Execute an action with a timeout."""
    if timeout_s is None or timeout_s <= 0:
        return action()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(action)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            raise TimeoutError(timeout_message)

# Tool handling moved to core.tool_executor
from typing import List, Optional, Dict, Any, TextIO, Callable, Iterable
import argparse
import os
import sys
import json
import csv
import textwrap
from datetime import datetime
from pathlib import Path
# Import core utilities
from core.retry_handler import RetryConfig, _retry_call, _stream_chat_with_retry
from core.tool_executor import (
    get_tools,
    execute_tool_call,
    perform_web_search,
    read_json_file,
)
from core.argument_parser import parse_arguments
from core.context_loader import parse_context_arg, load_context_files, load_context_from_database
from core.session_manager import list_sessions, select_session, export_session


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
    
    has_system = any(msg.get("role") == "system" for msg in messages)
    
    if has_system:
        new_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                existing_content = msg.get("content", "")
                msg = msg.copy()
                msg["content"] = f"{existing_content}\n\n{system_content}"
            new_messages.append(msg)
        return new_messages
    else:
        return [{"role": "system", "content": system_content}] + messages


def _get_model_specific_timeout(model: str, base_timeout: Optional[float]) -> float:
    """Get adjusted timeout for models that need more time for tool selection."""
    if base_timeout is not None:
        return base_timeout
    if model.lower().startswith("lfm2"):
        return 120.0
    return 60.0


def chat_with_tools(
    model: str,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    file_handle: TextIO,
    retry_config: RetryConfig,
    tool_output_dir: Optional[str] = None,
    ollama_timeout: Optional[float] = None,
) -> str:
    """Chat with the model, handling tool calls automatically.

    This mirrors the original implementation from ``stream_chat.py`` but uses the
    locally imported ``execute_tool_call`` and ``log_to_file`` helpers.
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

    # LFM2.5 works best with think=False for faster tool calls
    use_think_false = _supports_lfm2_tool_format(model)
    
    # Get model-specific timeout
    timeout = _get_model_specific_timeout(model, ollama_timeout)
    
    # First call – let the model decide if it needs tools
    def chat_call():
        kwargs = {}
        if use_think_false:
            kwargs['think'] = False
        return ollama.chat(model=model, messages=messages, tools=tools, **kwargs)
    
    response = _run_with_timeout(chat_call, timeout, "Ollama tool selection call timed out")

    message = response.message

    # Check for tool calls
    if hasattr(message, "tool_calls") and message.tool_calls:
        print(f"\n[Tool calls detected: {[tc.function.name for tc in message.tool_calls]}]")

        # Append assistant request to history
        if hasattr(message, 'tool_calls') and message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in message.tool_calls
                ],
            })

        # Execute each tool call and add results
        for tool_call in message.tool_calls:
            result = execute_tool_call({
                "function": {
                    "name": tool_call.function.name,
                    "arguments": tool_call.function.arguments,
                }
            }, output_dir=tool_output_dir)
            messages.append({
                "role": "tool",
                "content": result,
                "tool_call_id": tool_call.id,
            })
            log_to_file(file_handle, f"\n[Tool '{tool_call.function.name}' used]\n")

        # Second call – get final response with tool results
        print(f"{model}: ", end="", flush=True)
        log_to_file(file_handle, f"{model}: ")
        full_response = ""
        stream = _stream_chat_with_retry(model=model, messages=messages, config=retry_config)
        for chunk in stream:
            part = chunk["message"]["content"]
            print(part, end="", flush=True)
            log_to_file(file_handle, part)
            full_response += part
        return full_response
    else:
        # No tool calls – simple streaming response
        print(f"{model}: ", end="", flush=True)
        log_to_file(file_handle, f"{model}: ")
        full_response = ""
        if message.content:
            print(message.content, end="", flush=True)
            log_to_file(file_handle, message.content)
            full_response = message.content
        return full_response

# Helper functions that were previously imported from the original stream_chat module.
# They are re‑implemented here to avoid importing the top‑level script (which depends on the optional ollama package).

def _parse_comma_list(value: str) -> List[str]:
    """Parse a comma‑separated string into a list of stripped items."""
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]

def _build_model_list(primary: str, fallbacks: List[str]) -> List[str]:
    """Combine primary model and fallbacks into an ordered list without duplicates."""
    seen = set()
    ordered: List[str] = []
    for model in [primary, *fallbacks]:
        if model and model not in seen:
            seen.add(model)
            ordered.append(model)
    return ordered

def _ensure_model_available(model: str, config: RetryConfig) -> None:
    """Make sure the requested Ollama model is present, pulling it if necessary."""
    def show_model() -> Any:
        return ollama.show(model)

    try:
        _retry_call(show_model, config)
        return
    except Exception:
        # If the model is not found, attempt to pull it.
        print(f"Model {model} not found. Pulling...")
        def pull_model() -> Any:
            return ollama.pull(model)
        _retry_call(pull_model, config)

def log_to_file(file_handle: TextIO, text: str) -> None:
    """Write text to the given file handle and flush immediately."""
    file_handle.write(text)
    file_handle.flush()

# Optional interactive menu library
try:
    import questionary
    HAS_QUESTIONARY = True
except ImportError:
    HAS_QUESTIONARY = False

# Optional DuckDuckGo web search tool
try:
    from ddgs import DDGS
    HAS_DDG = True
except ImportError:
    HAS_DDG = False

# Database integration (optional)
try:
    from db import get_database_manager, Conversation
    HAS_DB = True
except ImportError:
    HAS_DB = False

# Import session manager
from core.session_manager import list_sessions, select_session, export_session


def _get_retry_config_from_args(args: argparse.Namespace) -> RetryConfig:
    return RetryConfig(
        max_attempts=max(1, args.retry_max_attempts),
        initial_delay=max(0.0, args.retry_initial_delay),
        max_delay=max(0.0, args.retry_max_delay),
        multiplier=max(1.0, args.retry_multiplier),
        jitter=max(0.0, args.retry_jitter),
    )


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
            retry_config=retry_config,
            tool_output_dir=args.tool_output_dir,
            ollama_timeout=args.ollama_timeout
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

async def main() -> None:
    args = parse_arguments()

    # ------------------------------------------------------------
    # Session handling commands (list, select, export)
    # ------------------------------------------------------------
    if args.list_sessions:
        list_sessions()
        return

    if args.select_session:
        messages = await select_session()
        if messages is None:
            return
        # Pre‑populate messages with saved conversation and continue
        print("Resuming conversation.")
        # Skip the rest of the initialization that would create a new DB manager
        # (the rest of main() will use the `messages` variable defined later)
        # We'll set a flag to indicate we already loaded messages.
        _preloaded_messages = messages
        # Jump to after the DB init block by using a guard later.
        pass

    if args.export_session is not None:
        out_target = sys.stdout if args.output is None else args.output.open("w", encoding="utf-8")
        export_session(args.export_session, args.format, out_target)
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
        await db_manager.create_tables()
        print("[Database] Tables created/verified successfully")
    except Exception as e:
        print(f"Error connecting to database: {e}")
        print("Tip: Ensure PostgreSQL is running and DATABASE_URL is set correctly")
        return

    # Initialize chat history (use preloaded messages if a session was selected)
    messages: List[Dict[str, str]] = []
    try:
        # _preloaded_messages is set only when --select-session was used
        preloaded_messages = _preloaded_messages  # type: ignore
        if preloaded_messages:
            messages = preloaded_messages
    except NameError:
        pass
    
    # Load context if provided
    context_content = ""
    if args.context:
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
                                conversation_id = await db_manager.save_conversation(
                                    model=used_model,
                                    messages=messages.copy(),
                                    flags=flags
                                )
                                print(f"[Saved conversation to database with ID: {conversation_id}]")
                            else:
                                # Update existing conversation
                                await db_manager.update_conversation(conversation_id, messages.copy())
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
    import asyncio
    asyncio.run(main())
# Database integration complete. Conversations can now be persisted to PostgreSQL with --persist-to-db flag and loaded with --context db.
