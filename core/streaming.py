"""Streaming and UI utilities for Ollama Chat Streamer.

Provides Spinner, render_stream_text, and _stream_ollama_chat_with_timeouts.
"""

import queue
import sys
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, TextIO, Tuple

import ollama


class Spinner:
    """Thread-based terminal spinner animation."""

    def __init__(
        self,
        message: str,
        enabled: bool = True,
        style: str = "line",
        interval: float = 0.1,
        stream: Optional[TextIO] = None,
    ) -> None:
        self.message = message
        self.enabled = enabled
        self.style = style
        self.interval = interval
        self.stream = stream or sys.stderr
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def _frames(self) -> List[str]:
        if self.style == "dots":
            return [".", "..", "..."]
        return ["|", "/", "-", "\\"]

    def start(self) -> None:
        if not self.enabled:
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()

        def run() -> None:
            frames = self._frames()
            idx = 0
            while not self._stop_event.is_set():
                frame = frames[idx % len(frames)]
                self.stream.write(f"\r{self.message} {frame}")
                self.stream.flush()
                idx += 1
                time.sleep(self.interval)
            self.stream.write("\r\033[K")
            self.stream.flush()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self.enabled:
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=0.2)


def _stream_ollama_chat_with_timeouts(
    model: str,
    messages: List[Dict[str, str]],
    start_timeout: Optional[float],
    idle_timeout: Optional[float],
    spinner_enabled: bool,
    spinner_style: str,
    spinner_interval: float,
    spinner_stall_delay: float,
    **kwargs: Any,
) -> Iterable[Dict[str, Any]]:
    """Stream ollama.chat responses with start/idle timeouts and spinner."""
    q: "queue.Queue[Tuple[str, Any]]" = queue.Queue()
    stop_event = threading.Event()

    def worker() -> None:
        try:
            for chunk in ollama.chat(
                model=model, messages=messages, stream=True, **kwargs
            ):
                if stop_event.is_set():
                    break
                q.put(("chunk", chunk))
            q.put(("done", None))
        except Exception as exc:
            q.put(("error", exc))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    first_token = True
    start_time = time.monotonic()
    spinner = Spinner(
        message="Waiting for response",
        enabled=spinner_enabled,
        style=spinner_style,
        interval=spinner_interval,
    )

    def start_spinner_after(delay: float) -> Optional[threading.Timer]:
        if not spinner_enabled:
            return None
        if delay <= 0:
            spinner.start()
            return None
        timer = threading.Timer(delay, spinner.start)
        timer.daemon = True
        timer.start()
        return timer

    while True:
        timeout: Optional[float] = None
        if first_token and start_timeout and start_timeout > 0:
            elapsed = time.monotonic() - start_time
            timeout = max(0.0, start_timeout - elapsed)
        elif idle_timeout and idle_timeout > 0:
            timeout = idle_timeout

        spinner_timer = None
        if first_token:
            spinner_timer = start_spinner_after(0.0)
        else:
            spinner_timer = start_spinner_after(spinner_stall_delay)

        try:
            kind, payload = q.get(timeout=timeout)
        except queue.Empty:
            stop_event.set()
            if spinner_timer:
                spinner_timer.cancel()
            spinner.stop()
            if first_token and start_timeout and start_timeout > 0:
                raise TimeoutError("Ollama response start timed out")
            raise TimeoutError("Ollama response stalled")

        if kind == "chunk":
            if spinner_timer:
                spinner_timer.cancel()
            spinner.stop()
            if first_token:
                first_token = False
            yield payload
        elif kind == "done":
            if spinner_timer:
                spinner_timer.cancel()
            spinner.stop()
            return
        elif kind == "error":
            if spinner_timer:
                spinner_timer.cancel()
            spinner.stop()
            raise payload


def render_stream_text(
    text: str, file_handle: TextIO, flush_interval: int = 20, delay: float = 0.0
) -> None:
    """Render text progressively to stdout while logging to file."""
    if not text:
        return
    for idx, char in enumerate(text, start=1):
        sys.stdout.write(char)
        if flush_interval > 0 and idx % flush_interval == 0:
            sys.stdout.flush()
        if delay > 0:
            time.sleep(delay)
    sys.stdout.flush()
    file_handle.write(text)
    file_handle.flush()