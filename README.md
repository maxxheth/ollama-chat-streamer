# Ollama Chat Streamer

A feature-rich CLI chat interface for Ollama with streaming responses, intelligent web search, and context loading capabilities.

## Features

- 🚀 **Streaming Responses** - Real-time token-by-token output from Ollama models
- 🔍 **Intelligent Web Search** - LLM decides when to search the web for current information
- 📁 **Context Loading** - Load files or entire directories as conversation context
- �️ **Database Persistence** - Optional PostgreSQL storage for conversation history
- 📝 **Chat Logging** - Automatically saves all conversations to a log file
- 🐳 **Docker Support** - Easy containerized deployment with PostgreSQL
- ⚙️ **Environment Configuration** - Flexible configuration via CLI flags or .env file

## Quick Start

### Prerequisites

- [Ollama](https://ollama.com) installed and running on your machine
- Docker and Docker Compose (optional, for containerized usage)
- Python 3.11+ (for local usage)

### Using Docker (Recommended)

1. Clone the repository and navigate to the project directory

2. Copy the example environment file and customize it:

```bash
cp .env.example .env
# Edit .env with your preferred settings
```

3. Build and run:

```bash
docker-compose build
docker-compose run --rm chat
```

### Local Installation

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the chat:

```bash
python stream_chat.py
```

## Configuration

Configuration can be done via:

- Command-line flags (highest priority)
- Environment variables in `.env` file
- Default values (lowest priority)

### Environment Variables

| Variable                 | Description                                  | Default                  |
| ------------------------ | -------------------------------------------- | ------------------------ |
| `OLLAMA_HOST`            | URL of your Ollama instance                  | `http://localhost:11434` |
| `OLLAMA_MODEL`           | Model to use (e.g., llama3, mistral)         | `llama3`                 |
| `OLLAMA_MODEL_FALLBACKS` | Comma-separated fallback models              | (empty)                  |
| `CHAT_LOG_DEST`          | Path to chat log file                        | `chat_log.txt`           |
| `EXPERIMENTAL`           | Enable experimental features                 | `false`                  |
| `EXPERIMENTAL_WEBSEARCH` | Enable intelligent web search                | `false`                  |
| `CONTEXT_PATH`           | Path to file/directory for context loading   | (empty)                  |
| `CONTEXT_GREP`           | File extensions to include (comma-separated) | `txt,log`                |
| `PERSIST_TO_DB`          | Enable database persistence                  | `false`                  |
| `DATABASE_URL`           | PostgreSQL connection URL                    | (empty)                  |
| `RETRY_MAX_ATTEMPTS`     | Max retry attempts for Ollama calls          | `3`                      |
| `RETRY_INITIAL_DELAY`    | Initial backoff delay in seconds             | `0.5`                    |
| `RETRY_MAX_DELAY`        | Max backoff delay in seconds                 | `8.0`                    |
| `RETRY_MULTIPLIER`       | Backoff multiplier                           | `2.0`                    |
| `RETRY_JITTER`           | Jitter ratio applied to backoff delay        | `0.1`                    |

### Command-Line Flags

```bash
python stream_chat.py [OPTIONS]

Options:
  --model TEXT              The Ollama model to use
  --dest TEXT               Path to the log file
  --experimental            Enable experimental features
  --experimental-websearch  Enable intelligent web search
  --context TEXT            Path to context file or directory (use 'db' for database)
  --context-grep TEXT       File extensions to include (e.g., "js,ts,py")
  --persist-to-db           Enable saving conversations to PostgreSQL database
  --model-fallbacks TEXT    Comma-separated fallback models
  --retry-max-attempts INT  Max retry attempts for Ollama calls
  --retry-initial-delay FLOAT
                            Initial backoff delay in seconds
  --retry-max-delay FLOAT   Max backoff delay in seconds
  --retry-multiplier FLOAT  Backoff multiplier
  --retry-jitter FLOAT      Jitter ratio applied to backoff delay
```

## Usage Examples

### Basic Chat

```bash
python stream_chat.py
```

### Use a Specific Model

```bash
python stream_chat.py --model codellama
```

### Use Fallback Models and Tweak Retry Settings

```bash
python stream_chat.py \
  --model llama3 \
  --model-fallbacks "mistral,codellama" \
  --retry-max-attempts 4 \
  --retry-initial-delay 0.75 \
  --retry-max-delay 10 \
  --retry-multiplier 2 \
  --retry-jitter 0.2
```

### Enable Intelligent Web Search

The LLM will automatically decide when to search the web for current information:

```bash
python stream_chat.py --experimental-websearch
```

### Load Context from Files

Load a single file:

```bash
python stream_chat.py --context ./notes.txt
```

Load all `.md` and `.txt` files from a directory:

```bash
python stream_chat.py --context ./docs --context-grep "md,txt"
```

Load all files from a project directory:

```bash
python stream_chat.py --context ./src --context-grep "js,ts,tsx"
```

### Combined Example

```bash
python stream_chat.py \
  --model mistral \
  --experimental-websearch \
  --context ./project_docs \
  --context-grep "md,txt" \
  --dest ./logs/my_chat.log
```

### Database Persistence

Enable PostgreSQL storage for your conversations:

```bash
# Start with database persistence
python stream_chat.py --persist-to-db

# Load past conversations from database
python stream_chat.py --context db

# Load from database + additional file paths
python stream_chat.py --context "db,./notes,./docs"

# Use environment variables
export PERSIST_TO_DB=true
export DATABASE_URL="postgresql://user:password@localhost:5432/chatdb"
python stream_chat.py
```

#### Database Setup

1. **Using Docker Compose (Recommended)**:

   ```bash
   docker-compose up -d postgres
   python setup_db.py
   ```

2. **Manual Setup**:

   ```bash
   # Install PostgreSQL and create database
   createdb chatdb

   # Set DATABASE_URL environment variable
   export DATABASE_URL="postgresql://postgres:password@localhost:5432/chatdb"

   # Initialize database
   python setup_db.py
   ```

3. **Check database status**:
   ```bash
   python setup_db.py info
   ```

#### Database Features

- **Automatic Conversation Saving**: When `--persist-to-db` is enabled, each conversation is saved with metadata (model, flags, timestamps)
- **Context Loading**: Use `--context db` to load all past conversations as context
- **Mixed Context**: Combine database and file sources: `--context "db,./docs,./notes"`
- **Query History**: Use `setup_db.py info` to view conversation statistics

## Docker Networking

### Linux

When using `network_mode: host` (configured in `docker-compose.yml`), the container shares the host's network stack. Make sure Ollama is listening on all interfaces:

```bash
export OLLAMA_HOST=0.0.0.0:11434
ollama serve
```

### macOS/Windows

Use `host.docker.internal` to reach the host machine:

```bash
# In your .env file:
OLLAMA_HOST=http://host.docker.internal:11434
```

## Project Structure

```
.
├── docker-compose.yml      # Docker Compose configuration with PostgreSQL
├── Dockerfile              # Docker image definition
├── requirements.txt        # Python dependencies (including database drivers)
├── stream_chat.py          # Main application
├── db.py                   # Database module for PostgreSQL operations
├── setup_db.py             # Database initialization script
├── .env.example            # Example environment configuration
├── .gitignore             # Git ignore rules
└── chat_log.txt           # Default chat log (created on first run)
```

## How It Works

### Intelligent Web Search

When enabled, the LLM receives a `web_search` tool definition. Based on your query, it can decide to:

- Answer directly from its training data (e.g., "What is Python?")
- Search the web for current information (e.g., "Who won the election yesterday?")

The search uses DuckDuckGo and returns the top 3 results as context for the LLM.

### Context Loading

When you specify a `--context` path:

- **Single file**: The entire file content is loaded as a system message
- **Directory**: All files matching the specified extensions are recursively loaded

This allows the LLM to reference your codebase, documentation, or notes during the conversation.

## Tips

- Use `--experimental-websearch` for queries about current events, recent news, or time-sensitive information
- Load project documentation as context for code-related questions
- Chat logs are appended, not overwritten, so you have a complete history
- Press `Ctrl+C` to interrupt the current response
- Type `exit` or `quit` to end the session

## License

MIT License - Feel free to use and modify as needed!

## Contributing

Contributions are welcome! Feel free to submit issues or pull requests.
