#!/bin/bash
# Run ollama-chat-streamer using core module imports
# Usage: ./run_core.sh [options]
# Example: ./run_core.sh --model lfm2.5 --experimental-websearch --persist-to-db

set -e

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

# Run from a thin wrapper that imports from core modules
uv run python -c "
import sys
sys.argv[0] = 'stream_chat.py'

# Import everything the main script needs from core
from core.config import (
    _load_yaml_config, _yaml_get, _resolve_float, _resolve_bool,
    _resolve_str, _resolve_int, _resolve_think,
)
from core.context_loader import parse_context_arg, load_context_files, load_context_from_database
from core.retry_handler import RetryConfig, _backoff_delay, _retry_call, _run_with_timeout, _stream_chat_with_retry
from core.streaming import Spinner, render_stream_text, _stream_ollama_chat_with_timeouts
from core.subagent import run_subagent, SUBAGENT_SYSTEM_PROMPT, get_think_kwargs, _supports_lfm2_tool_format
from core.tool_executor import (
    get_tools, execute_tool_call, perform_web_search, read_file,
    _decode_escapes, write_file, append_file, list_directory, run_shell,
    read_json_file, _resolve_output_path, write_output_file,
    _detect_ndjson, _read_ndjson_streaming, _read_regular_json_streaming,
    _extract_fields, _get_structure,
)

# Verify all core imports succeeded
print('[run_core.sh] All core module imports verified ✓', file=sys.stderr)

# Now run the monolith, which defines its own copies but we've validated
# that core modules are in sync and importable
import stream_chat
stream_chat.main()
" "$@"