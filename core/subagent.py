"""Subagent agentic loop for Ollama Chat Streamer.

Provides the run_subagent function that creates an isolated agentic loop
for spawning subagents that can use tools iteratively.
"""

from typing import Any, Dict, List, Optional

import ollama

from .tool_executor import get_tools, execute_tool_call
from .retry_handler import RetryConfig

SUBAGENT_SYSTEM_PROMPT = (
    "You are a subagent working on a specific task. "
    "Use your available tools to complete the task step by step. "
    "When you have the answer, respond directly with it — do not ask follow-up questions. "
    "Be thorough but concise. If a tool call fails, try a different approach."
)


def _supports_lfm2_tool_format(model: str) -> bool:
    """Check if model is LFM2.x which prefers tools in system prompt."""
    return model.lower().startswith("lfm2")


def get_think_kwargs(model: str, think_setting: str) -> Dict[str, Any]:
    """Return kwargs dict for ollama.chat() based on the think setting.

    think_setting: 'auto' (LFM2->False, others->no kwargs),
                   'true' (always think=True),
                   'false' (always think=False)
    """
    if think_setting == "true":
        return {"think": True}
    elif think_setting == "false":
        return {"think": False}
    else:  # auto
        if _supports_lfm2_tool_format(model):
            return {"think": False}
        return {}


def run_subagent(
    task: str,
    context: str,
    model: str,
    tools: List[Dict[str, Any]],
    retry_config: RetryConfig,
    max_rounds: int = 5,
    current_depth: int = 1,
    max_depth: int = 1,
    tool_timeout: Optional[float] = None,
    web_search_timeout: Optional[float] = None,
    ollama_timeout: Optional[float] = None,
    ollama_first_token_timeout: Optional[float] = None,
    ollama_stream_idle_timeout: Optional[float] = None,
    tool_output_dir: Optional[str] = None,
    read_file_max_bytes: int = 512_000,
    read_file_max_lines: int = 200,
    read_file_max_content: int = 64_000,
    think_setting: str = "auto",
) -> str:
    """Run a subagent agentic loop to complete a task."""
    if current_depth > max_depth:
        return "[Error: Maximum subagent depth reached]"

    subagent_tools = [t for t in tools if t.get("function", {}).get("name") != "spawn_agent"]

    user_message = task
    if context:
        user_message = f"{task}\n\nAdditional context:\n{context}"

    messages: List[Dict[str, str]] = [
        {"role": "system", "content": SUBAGENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    print(f"\n  [Subagent depth={current_depth}] Starting: {task[:80]}{'...' if len(task) > 80 else ''}")

    for round_num in range(1, max_rounds + 1):
        try:
            kwargs = get_think_kwargs(model, think_setting)

            if not subagent_tools:
                resp = ollama.chat(model=model, messages=messages, **kwargs)
            else:
                resp = ollama.chat(model=model, messages=messages, tools=subagent_tools, **kwargs)
        except Exception as e:
            print(f"  [Subagent] Model error on round {round_num}: {e}")
            return f"[Subagent error on round {round_num}: {e}]"

        message = resp.message

        if not (hasattr(message, 'tool_calls') and message.tool_calls):
            final_text = message.content or ""
            if message.content:
                messages.append({"role": "assistant", "content": message.content})
            tool_count = sum(1 for m in messages if m.get("role") == "tool")
            print(f"  [Subagent depth={current_depth}] Complete ({round_num} round{'s' if round_num != 1 else ''}, {tool_count} tool call{'s' if tool_count != 1 else ''})")
            return final_text

        tool_calls_list = message.tool_calls if hasattr(message, 'tool_calls') else []

        tool_calls_payload = []
        for tc in tool_calls_list:
            tc_function = getattr(tc, "function", None)
            if isinstance(tc, dict):
                tc_function = tc.get("function")
            tc_name = getattr(tc_function, "name", None) or (tc_function.get("name") if isinstance(tc_function, dict) else None)
            tc_args = getattr(tc_function, "arguments", None) or (tc_function.get("arguments") if isinstance(tc_function, dict) else None)
            tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)
            tool_calls_payload.append({
                "type": "function",
                "function": {"name": tc_name, "arguments": tc_args},
            })
            if tc_id is not None:
                tool_calls_payload[-1]["id"] = tc_id

        assistant_msg: Dict[str, Any] = {
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": tool_calls_payload,
        }
        messages.append(assistant_msg)

        for tc in tool_calls_list:
            tc_function = getattr(tc, "function", None)
            if isinstance(tc, dict):
                tc_function = tc.get("function")
            tc_name = getattr(tc_function, "name", None) or (tc_function.get("name") if isinstance(tc_function, dict) else None)
            tc_args = getattr(tc_function, "arguments", None) or (tc_function.get("arguments") if isinstance(tc_function, dict) else None)
            tc_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else None)

            print(f"  [Subagent depth={current_depth}] Round {round_num}/{max_rounds} → {tc_name}")

            tool_result = execute_tool_call(
                {"function": {"name": tc_name, "arguments": tc_args}},
                tool_timeout=tool_timeout,
                web_search_timeout=web_search_timeout,
                output_dir=tool_output_dir,
                read_file_max_bytes=read_file_max_bytes,
                read_file_max_lines=read_file_max_lines,
                read_file_max_content=read_file_max_content,
            )

            tool_message: Dict[str, Any] = {"role": "tool", "content": tool_result}
            if tc_id is not None:
                tool_message["tool-call_id"] = tc_id
            messages.append(tool_message)

    print(f"  [Subagent depth={current_depth}] Max rounds ({max_rounds}) reached")
    last_assistant = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            last_assistant = m["content"]
            break
    return last_assistant or f"[Subagent reached max rounds ({max_rounds}) without a final answer]"