# Ollama Chat Streamer

A feature-rich CLI chat interface for Ollama with streaming responses, intelligent web search, and context loading capabilities.

## Features

- 🚀 **Streaming Responses** - Real-time token-by-token output from Ollama models
- 🔍 **Intelligent Web Search** - LLM decides when to search the web for current information
- 📁 **Context Loading** - Load files or entire directories as conversation context
- 📝 **Chat Logging** - Automatically saves all conversations to a log file
- 🐳 **Docker Support** - Easy containerized deployment
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
| `CHAT_LOG_DEST`          | Path to chat log file                        | `chat_log.txt`           |
| `EXPERIMENTAL`           | Enable experimental features                 | `false`                  |
| `EXPERIMENTAL_WEBSEARCH` | Enable intelligent web search                | `false`                  |
| `CONTEXT_PATH`           | Path to file/directory for context loading   | (empty)                  |
| `CONTEXT_GREP`           | File extensions to include (comma-separated) | `txt,log`                |

### Command-Line Flags

```bash
python stream_chat.py [OPTIONS]

Options:
  --model TEXT              The Ollama model to use
  --dest TEXT               Path to the log file
  --experimental            Enable experimental features
  --experimental-websearch  Enable intelligent web search
  --context TEXT            Path to context file or directory
  --context-grep TEXT       File extensions to include (e.g., "js,ts,py")
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
├── docker-compose.yml      # Docker Compose configuration
├── Dockerfile              # Docker image definition
├── requirements.txt        # Python dependencies
├── stream_chat.py          # Main application
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
