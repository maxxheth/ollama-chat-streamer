"""Retry handling utilities for Ollama Chat Streamer.

This module centralises the retry configuration dataclass and the helper
functions that perform exponential back‑off with jitter. The original
implementation lived in ``stream_chat.py``; it has been moved here to make the
codebase more modular and easier to test.
"""

import time
import random
from dataclasses import dataclass
from typing import Callable, Any, Optional, Iterable, Dict, List

# Import Ollama client; provide a lightweight stub if the package is missing.
from typing import Any, Dict, Iterable

try:
    import ollama
    from ollama import ChatResponse
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

    ollama: Any = _OllamaStub()


@dataclass(frozen=True)
class RetryConfig:
    """Configuration for retry behaviour.

    Attributes
    ----------
    max_attempts: int
        Maximum number of attempts before giving up.
    initial_delay: float
        Initial back‑off delay in seconds.
    max_delay: float
        Upper bound for the delay.
    multiplier: float
        Multiplicative factor applied after each retry.
    jitter: float
        Random jitter factor (0‑1) added to the delay to avoid thundering herd.
    """

    max_attempts: int
    initial_delay: float
    max_delay: float
    multiplier: float
    jitter: float


def _backoff_delay(base_delay: float, jitter: float) -> float:
    """Calculate a back‑off delay with optional jitter.

    If ``jitter`` is ``<= 0`` the base delay is returned unchanged. Otherwise a
    random value between ``0`` and ``base_delay * jitter`` is added.
    """
    if jitter <= 0:
        return base_delay
    return base_delay + random.uniform(0, base_delay * jitter)


def _retry_call(
    action: Callable[[], Any],
    config: RetryConfig,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Any:
    """Execute ``action`` with retry/back‑off logic.

    Parameters
    ----------
    action:
        Callable that performs the operation.
    config:
        ``RetryConfig`` instance controlling the behaviour.
    on_retry:
        Optional callback invoked on each retry with ``attempt``, ``exception``
        and ``sleep_time``.
    """
    attempts = max(1, config.max_attempts)
    delay = max(0.0, config.initial_delay)
    for attempt in range(1, attempts + 1):
        try:
            return action()
        except Exception as exc:
            if attempt >= attempts:
                raise
            sleep_time = _backoff_delay(min(config.max_delay, delay), config.jitter)
            if on_retry:
                on_retry(attempt, exc, sleep_time)
            time.sleep(sleep_time)
            delay = min(config.max_delay, delay * max(1.0, config.multiplier))


def _stream_chat_with_retry(
    model: str,
    messages: List[Dict[str, str]],
    config: RetryConfig,
    **kwargs: Any,
) -> Iterable[Dict[str, Any]]:
    """Stream chat responses from Ollama with retry handling.

    This mirrors the original implementation but lives in the dedicated module
    so that the chat engine can import it without pulling in unrelated code.
    """
    attempts = max(1, config.max_attempts)
    delay = max(0.0, config.initial_delay)
    for attempt in range(1, attempts + 1):
        try:
            stream = ollama.chat(model=model, messages=messages, stream=True, **kwargs)
            for chunk in stream:
                yield dict(chunk)  # type: ignore
                yield chunk
            return
        except Exception as exc:
            if attempt >= attempts:
                raise
            sleep_time = _backoff_delay(min(config.max_delay, delay), config.jitter)
            print(f"\n[Retry {attempt}/{attempts} after error: {exc}] Waiting {sleep_time:.2f}s...")
            time.sleep(sleep_time)
            delay = min(config.max_delay, delay * max(1.0, config.multiplier))
