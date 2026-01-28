import ollama
import sys
import os
import argparse
import json
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, TextIO, Any, Optional, Callable
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Attempt to import DuckDuckGo for web search
try:
    from ddgs import DDGS  # type: ignore
    HAS_DDG = True
except ImportError:
    HAS_DDG = False


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
        }
    ]


def execute_tool_call(tool_call: Dict[str, Any]) -> str:
    """Execute a tool call and return the result."""
    function_name = tool_call.get("function", {}).get("name")
    arguments = json.loads(tool_call.get("function", {}).get("arguments", "{}"))
    
    if function_name == "web_search":
        query = arguments.get("query", "")
        return perform_web_search(query)
    
    return f"[Error: Unknown tool '{function_name}']"


def chat_with_tools(
    model: str,
    messages: List[Dict[str, str]],
    tools: List[Dict[str, Any]],
    file_handle: TextIO
) -> str:
    """
    Chat with the model, handling tool calls automatically.
    Returns the final assistant response.
    """
    # First call - let the model decide if it needs tools
    response = ollama.chat(
        model=model,
        messages=messages,
        tools=tools
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
        stream = ollama.chat(model=model, messages=messages, stream=True)
        
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
        help="Path to a directory or file to load as historical context for the LLM"
    )

    parser.add_argument(
        "--context-grep",
        type=str,
        default=os.environ.get("CONTEXT_GREP", "txt,log"),
        help="Comma-separated list of file extensions to include when loading context from a directory (default: txt,log)"
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

def main() -> None:
    args = parse_arguments()

    # Ensure the model exists
    try:
        ollama.show(args.model)
    except ollama.ResponseError:
        print(f"Model {args.model} not found. Pulling...")
        ollama.pull(args.model)

    # Initialize chat history
    messages: List[Dict[str, str]] = []
    
    # Load context if provided
    context_content = ""
    if args.context:
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
    
    print(f"Starting chat with {args.model}.")
    print(f"Experimental Mode: {'ON' if args.experimental else 'OFF'}")
    print(f"Intelligent Web Search: {'ON (LLM decides when to search)' if args.experimental_websearch else 'OFF'}")
    if args.context:
        print(f"Context Path: {args.context}")
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
                f"MODEL: {args.model}\n"
                f"FLAGS: exp={args.experimental}, web={args.experimental_websearch}\n"
            )
            if args.context:
                header += f"CONTEXT: {args.context}\n"
            header += f"{'='*30}\n"
            log_to_file(f, header)

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

                    # Use tool calling if web search is enabled
                    if args.experimental_websearch and HAS_DDG:
                        full_response = chat_with_tools(
                            model=args.model,
                            messages=messages,
                            tools=get_tools(),
                            file_handle=f
                        )
                    else:
                        # Standard streaming without tools
                        print(f"{args.model}: ", end="", flush=True)
                        log_to_file(f, f"{args.model}: ")
                        
                        full_response = ""
                        stream = ollama.chat(model=args.model, messages=messages, stream=True)
                        
                        for chunk in stream:
                            part = chunk['message']['content']
                            print(part, end="", flush=True)
                            log_to_file(f, part)
                            full_response += part
                    
                    # Add spacing after response
                    print(f"\n{'─' * 50}\n")
                    
                    # Add final newline to file and history
                    log_to_file(f, "\n")
                    messages.append({"role": "assistant", "content": full_response})

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

if __name__ == "__main__":
    main()
