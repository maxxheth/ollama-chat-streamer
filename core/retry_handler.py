"""Retry handling utilities for Ollama Chat Streamer.

Centralises the retry configuration dataclass and the helper
functions that perform exponential back-off with jitter.
"""

import time
import random
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional

try:
    import ollama
    from ollama import ChatResponse
except ImportError:
    class _OllamaStub:
        @staticmethod
        def chat(*args, **kwargs):
            raise NotImplementedError("ollama package is not installed")
        @staticmethod
        def show(*args, **kwargs):
            raise NotImplementedError("ollama package is not installed")
        @staticmethod
        def pull(*args, **kwargs):
            raise NotImplementedError("ollama package is not installed")

    class ChatResponse:
        pass

    ollama = _OllamaStub()


@dataclass(frozen=True)
class RetryConfig:
    max_attempts: int
    initial_delay: float
    max_delay: float
    multiplier: float
    jitter: float


def _backoff_delay(base_delay: float, jitter: float) -> float:
    if jitter <= 0:
        return base_delay
    return base_delay + random.uniform(0, base_delay * jitter)


def _retry_call(
    action: Callable[[], Any],
    config: RetryConfig,
    on_retry: Optional[Callable[[int, Exception, float], None]] = None,
) -> Any:
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


def _run_with_timeout(action: Callable[[], Any], timeout_s: Optional[float], timeout_message: str) -> Any:
    """Run an action with an optional timeout in seconds."""
    if timeout_s is None or timeout_s <= 0:
        return action()
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(action)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            raise TimeoutError(timeout_message)


def _stream_chat_with_retry(
    model: str,
    messages: List[Dict[str, str]],
    config: RetryConfig,
    start_timeout: Optional[float] = None,
    idle_timeout: Optional[float] = None,
    spinner_enabled: bool = True,
    spinner_style: str = "line",
    spinner_interval: float = 0.1,
    spinner_stall_delay: float = 1.5,
    **kwargs: Any,
) -> Iterable[Dict[str, Any]]:
    """Stream chat responses from Ollama with retry handling and timeout support."""
    from core.streaming import _stream_ollama_chat_with_timeouts

    attempts = max(1, config.max_attempts)
    delay = max(0.0, config.initial_delay)

    def on_retry(attempt, exc, sleep_time):
        print(f"\n  retry {attempt}/{attempts} ({exc}; waiting {sleep_time:.1f}s)...", end="", flush=True)

    for attempt in range(1, attempts + 1):
        try:
            stream = _stream_ollama_chat_with_timeouts(
                model=model,
                messages=messages,
                start_timeout=start_timeout,
                idle_timeout=idle_timeout,
                spinner_enabled=spinner_enabled,
                spinner_style=spinner_style,
                spinner_interval=spinner_interval,
                spinner_stall_delay=spinner_stall_delay,
                **kwargs
            )
            for chunk in stream:
                yield chunk
            return
        except Exception as exc:
            if attempt >= attempts:
                raise
            sleep_time = _backoff_delay(min(config.max_delay, delay), config.jitter)
            if on_retry:
                on_retry(attempt, exc, sleep_time)
            time.sleep(sleep_time)
            delay = min(config.max_delay, delay * max(1.0, config.multiplier))