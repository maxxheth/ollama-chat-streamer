import io
import time
from contextlib import redirect_stdout

import stream_chat


def test_render_stream_text_smoke_no_delay(monkeypatch):
    output = io.StringIO()
    log_file = io.StringIO()

    with redirect_stdout(output):
        stream_chat.render_stream_text("hello", log_file, flush_interval=2, delay=0.0)

    assert output.getvalue() == "hello"
    assert log_file.getvalue() == "hello"


def test_render_stream_text_smoke_with_delay(monkeypatch):
    output = io.StringIO()
    log_file = io.StringIO()
    sleep_calls = []

    def fake_sleep(value):
        sleep_calls.append(value)

    monkeypatch.setattr(time, "sleep", fake_sleep)

    with redirect_stdout(output):
        stream_chat.render_stream_text("hi", log_file, flush_interval=1, delay=0.01)

    assert output.getvalue() == "hi"
    assert log_file.getvalue() == "hi"
    assert sleep_calls == [0.01, 0.01]
