# Ollama Chat Streamer - AI Coding Agent Instructions

## Architecture Overview

This is a CLI chat interface for Ollama with streaming responses, intelligent web search, and PostgreSQL persistence. Core components:

- **stream_chat.py** (1160 lines) - Main application with tool-based architecture using Ollama's function calling
- **db.py** - Dual-mode database manager (async with asyncpg, sync with psycopg2-binary)
- **setup_db.py** - Database initialization and session management CLI

## Key Patterns & Conventions

### Retry/Backoff System
All Ollama API calls use `_retry_call()` with exponential backoff + jitter. Configuration via `RetryConfig` dataclass:
```python
config = RetryConfig(max_attempts=3, initial_delay=0.5, max_delay=8.0, multiplier=2.0, jitter=0.1)
```
Always wrap `ollama.chat()`, `ollama.pull()`, and streaming calls with retry logic.

### Tool-Based Architecture
The app uses Ollama's function calling for extensibility. Tools defined in `get_tools()`:
- `web_search` - DuckDuckGo integration (requires `ddgs` package)
- `read_json_file` - Streaming JSON parser with NDJSON support

Tool execution flow: `ollama.chat(tools=tools)` → detect `message.tool_calls` → `execute_tool_call()` → append tool result → re-call with results.

### Model Fallback Chain
Build model lists with `_build_model_list(primary, fallbacks)` - deduplicates and maintains order. Use `_respond_with_fallbacks()` for automatic failover on errors.

### Database Dual-Mode Pattern
Database operations support both async (asyncpg) and sync (psycopg2) modes:
```python
db_manager = get_database_manager(sync=True)  # For CLI/scripts
db_manager = get_database_manager(sync=False) # For async contexts
```
Always check `HAS_DB`, `HAS_ASYNCPG`, `HAS_PSYCOPG2` flags before using database features.

### Context Loading Strategy
Three context sources (combined via `parse_context_arg()`):
1. `--context db` - Loads all past conversations from PostgreSQL
2. `--context /path` - Single file or directory tree (filtered by extensions)
3. `--context db,/notes,/docs` - Mixed mode (database + multiple file paths)

Context loaded via system message: `messages.insert(0, {"role": "system", "content": context})`

### Session Management Commands
Special flags bypass chat loop for DB operations:
- `--list-sessions` - Display all saved conversations
- `--select-session` - Interactive picker (uses `questionary` if available, falls back to numbered list)
- `--export-session ID --format json|csv|text|sql` - Export conversation to stdout or file

## Development Workflows

### Building & Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Test database (requires running PostgreSQL)
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/chatdb"
python setup_db.py
python test_db.py

# Run locally
python stream_chat.py --model llama3

# Docker build (recommended for production)
docker-compose build
docker-compose run --rm chat
```

### Adding New Tools
1. Define tool schema in `get_tools()` with OpenAI-compatible function spec
2. Add execution logic to `execute_tool_call()` switch statement
3. Handle tool results in `chat_with_tools()` (already implements the loop)

### Database Schema
Single table `conversations`:
- `id SERIAL PRIMARY KEY`
- `model VARCHAR(255)` - Ollama model name
- `flags JSONB` - Command-line flags/metadata
- `messages JSONB` - Full message history array
- `created_at, updated_at TIMESTAMP WITH TIME ZONE`
- Indexes on `created_at DESC` and `model`

## Common Pitfalls

1. **Docker networking**: Linux uses `network_mode: host`, macOS/Windows need `host.docker.internal:11434`
2. **Environment precedence**: CLI flags → .env → defaults (check `parse_arguments()` for override logic)
3. **Streaming interrupts**: `Ctrl+C` handling in main loop catches `KeyboardInterrupt` gracefully
4. **JSON file parsing**: Use `_detect_ndjson()` to auto-detect format before choosing parser
5. **Database connections**: All methods create/close their own connections (no connection pooling)

## File Extension Conventions
- `.py` - Python source with type hints (mypy compatible)
- `.sh` - Bash scripts for Docker Compose shortcuts (e.g., `choose.sh`, `continue.sh`)
- No test framework yet - manual testing via `test_db.py` and `setup_db.py info`

## Logging Strategy
All conversations logged to file via `log_to_file()`:
- Streaming responses written character-by-character
- Tool calls logged as `[Tool 'tool_name' used]`
- Failover switches logged as `[Failover] Switching to fallback model: X`
- Logs are **appended**, never overwritten

## Environment Variables to Know
See `.env.example` for full list. Critical ones:
- `OLLAMA_HOST` - Often needs adjustment for Docker
- `DATABASE_URL` - Required for `--persist-to-db`
- `EXPERIMENTAL_WEBSEARCH` - Enables web_search tool
- `RETRY_*` - Fine-tune API resilience
