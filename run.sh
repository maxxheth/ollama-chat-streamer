#!/bin/bash
# Run ollama-chat-streamer with uv
# Usage: ./run.sh [options]
# Example: ./run.sh --model lfm2.5 --experimental-websearch --persist-to-db

set -e

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "uv is not installed. Installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source the updated PATH
    source $HOME/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
fi

# Run with uv
uv run python stream_chat.py "$@"
