# Running with `uv` (Recommended)

This guide explains how to run ollama-chat-streamer using [`uv`](https://github.com/astral-sh/uv) instead of Docker.

## Prerequisites

1. **Install uv** (if not already installed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Ensure Ollama is running**:
   ```bash
   ollama serve
   ```

3. **Pull your preferred model**:
   ```bash
   ollama pull lfm2.5
   ```

## Quick Start

### Option 1: Using the wrapper script
```bash
./run.sh --model lfm2.5 --experimental-websearch --persist-to-db
```

### Option 2: Using `uv run` directly
```bash
uv run python stream_chat.py --model lfm2.5 --experimental-websearch --persist-to-db
```

### Option 3: Install locally and use the CLI
```bash
uv pip install -e .
ollama-chat --model lfm2.5 --experimental-websearch --persist-to-db
```

## Configuration

### Environment Variables

Copy `.env.example` to `.env` and customize:

```bash
cp .env.example .env
```

Key settings:
```bash
# Model configuration
OLLAMA_MODEL=lfm2.5
OLLAMA_TIMEOUT=120  # Important for LFM2.5 tool calling

# Enable tool calling
EXPERIMENTAL_WEBSEARCH=true

# Database (optional)
PERSIST_TO_DB=true
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/chatdb
```

### Command-Line Flags

```bash
# Basic usage
uv run python stream_chat.py --model lfm2.5

# With tool calling (recommended for LFM2.5)
uv run python stream_chat.py --model lfm2.5 --experimental-websearch

# With context loading
uv run python stream_chat.py --model lfm2.5 --experimental-websearch \
  --context 'db,docs' --context-grep 'md,py,txt'

# With database persistence
uv run python stream_chat.py --model lfm2.5 --experimental-websearch \
  --persist-to-db
```

## Example Commands

### Chat with LFM2.5 and tool calling
```bash
uv run python stream_chat.py --model lfm2.5 --experimental-websearch
```

### Continue a previous session
```bash
uv run python stream_chat.py --model lfm2.5 --experimental-websearch \
  --select-session
```

### Export a conversation
```bash
uv run python stream_chat.py --export-session 1 --format json --output conversation.json
```

### Load context from files
```bash
uv run python stream_chat.py --model lfm2.5 --experimental-websearch \
  --context ./docs --context-grep 'md,txt'
```

## LFM2.5 Tool Calling

The script automatically applies optimizations for LFM2.5:
- Sets `think=False` for faster tool calling (~10-20s vs timeout)
- Uses 120-second timeout (vs 60s default)
- No manual configuration needed!

**Important**: You must use `--experimental-websearch` to enable tool calling.

## Troubleshooting

### "duckduckgo-search library not installed"
Install the dependency:
```bash
uv pip install duckduckgo-search
```

### "Ollama tool selection call timed out"
This should be fixed automatically for LFM2.5. If it persists:
1. Ensure Ollama is running: `ollama list`
2. Try increasing timeout: `OLLAMA_TIMEOUT=180`
3. Check model is available: `ollama show lfm2.5`

### Database connection errors
Ensure PostgreSQL is running:
```bash
# Using Docker
docker compose up -d postgres

# Or connect to your existing PostgreSQL instance
export DATABASE_URL=postgresql://user:pass@host:5432/dbname
```

## Development

Install development dependencies:
```bash
uv pip install -e ".[dev]"
```

Run type checking:
```bash
uv run mypy stream_chat.py
```

Run tests:
```bash
uv run pytest
```
