# Refactor Plan for Ollama Chat Streamer

## Overview

The goal is to split the monolithic `stream_chat.py` script into a clean, modular architecture under a `core/` package. This improves maintainability, testability, and future extensibility.

## Target Structure

```
ollama-chat-streamer/
├── core/
│   ├── __init__.py
│   ├── argument_parser.py
│   ├── session_manager.py
│   ├── tool_executor.py
│   ├── context_loader.py
│   ├── retry_handler.py
│   ├── model_manager.py
│   └── chat_engine.py
└── stream_chat.py   # Minimal entry point
```

## Step‑by‑Step Plan

### 1. Scaffold the `core/` package

- Create `core/__init__.py` (already present).
- Add empty module files listed above.
- Ensure the `docs/` folder exists for the plan file.

### 2. Migrate Functionality

| Module               | Source (in `stream_chat.py`)                                                                                                                  | Description                                                                        |
| -------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `argument_parser.py` | `parse_arguments()` and related helper functions (`_parse_env_int`, `_parse_env_float`, `_parse_comma_list`, `_build_model_list`)             | Handles CLI parsing and environment defaults.                                      |
| `session_manager.py` | Session‑related CLI flags (`--list-sessions`, `--select-session`, `--export-session`) and the logic that interacts with `db.py`.              | Provides functions like `list_sessions()`, `select_session()`, `export_session()`. |
| `tool_executor.py`   | `get_tools()`, `execute_tool_call()`, `chat_with_tools()`, `perform_web_search()`, `read_json_file()` and any helper functions used by tools. | Encapsulates all tool definitions and execution.                                   |
| `context_loader.py`  | `load_context_files()`, `load_context_from_database()`, `parse_context_arg()` and related helpers.                                            | Loads file‑based or DB‑based context for the model.                                |
| `retry_handler.py`   | `RetryConfig` dataclass, `_retry_call()`, `_backoff_delay()`, `_stream_chat_with_retry()`.                                                    | Centralises retry/back‑off logic.                                                  |
| `model_manager.py`   | `_ensure_model_available()`, `_get_retry_config_from_args()`, model‑fallback handling.                                                        | Manages model availability and selection.                                          |
| `chat_engine.py`     | The main chat loop (`_run_chat_turn`, `_respond_with_fallbacks`, `main()`) and any orchestration that ties the above pieces together.         | Drives the conversation using the other modules.                                   |

### 3. Update Imports

- Each new module will import only what it needs (e.g., `argparse`, `os`, `json`, `ollama`, `typing`).
- `stream_chat.py` will become a thin wrapper that imports `core.chat_engine` and calls its `main()`.
- Resolve any circular imports by moving shared utilities (e.g., `RetryConfig`) to a common sub‑module if needed.

### 4. Adjust Tests & CI

- Ensure existing tests (e.g., `test_db.py`) still import the correct symbols. Update import paths if they referenced functions that moved.
- Add a quick sanity test that imports `core.chat_engine.main` and runs it with `--help` to verify argument parsing works.

### 5. Run Validation

1. **Static checks** – `python -m pyflakes .` / `flake8` to catch unused imports.
2. **Unit tests** – Execute `pytest` (or the project's test runner) to confirm no regressions.
3. **Manual run** – `python stream_chat.py --model llama3 --experimental` and have a short chat to ensure behaviour matches the pre‑refactor version.

### 6. Documentation Update

- Update `README.md` to point to the new module layout.
- Add a short section in the docs describing the purpose of each `core/` module.

### 7. Commit & PR

- Stage all new files and modifications.
- Write a concise commit message, e.g., `refactor: split stream_chat.py into core package modules`.
- Open a PR for review.

## Timeline (Suggested)

| Day | Activity                                                           |
| --- | ------------------------------------------------------------------ |
| 1   | Scaffold package, create empty modules, add docs file (this plan). |
| 2‑3 | Migrate argument parsing and retry handling.                       |
| 4‑5 | Move tool definitions and context loading logic.                   |
| 6‑7 | Implement session manager and model manager.                       |
| 8   | Assemble `chat_engine.py` and update entry point.                  |
| 9   | Run tests, fix import issues, update documentation.                |
| 10  | Final validation, commit, and open PR.                             |

---

_This plan is intentionally detailed to serve as a checklist for the refactor. Adjust the timeline as needed based on team velocity._
