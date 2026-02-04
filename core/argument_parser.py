"""Argument parsing utilities for Ollama Chat Streamer.

This module contains helper functions for reading environment variables and
building the CLI argument parser.  Moving these out of ``stream_chat.py``
makes the entry‑point smaller and improves testability.
"""

import os
import argparse
from typing import List


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


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments, allowing environment variables as defaults.

    Precedence: CLI flag > environment variable > default value.
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

    # Session handling flags
    parser.add_argument("--list-sessions", action="store_true", help="List saved conversation sessions and exit")
    parser.add_argument("--select-session", action="store_true", help="Interactively select a saved session to continue")
    parser.add_argument(
        "--export-session",
        type=int,
        metavar="ID",
        help="Export the conversation with the given ID. Use --format to choose output format."
    )
    parser.add_argument(
        "--format",
        choices=["sql", "json", "csv", "text"],
        default="json",
        help="Export format when using --export-session",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="File path to write exported data. If omitted, prints to stdout."
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
