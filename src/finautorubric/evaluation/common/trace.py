"""Validate submitted agent traces and render a bounded judge input."""

import json
from typing import Any, Mapping


RENDERING_POLICY = {
    "version": "answer-and-trace/v1",
    "tool_args_chars": 1500,
    "tool_result_chars": 4000,
    "trace_chars": 150000,
    "overflow": "keep head and tail halves; append an omission marker",
    "layout": "final answer, then trace",
}


def validate_response(response: Mapping[str, Any]) -> None:
    if not isinstance(response.get("answer"), str) or not response["answer"].strip():
        raise ValueError("answer must be a nonempty string")
    trace = response.get("trace", [])
    if not isinstance(trace, list):
        raise ValueError("trace must be a list of step objects")
    previous_step = -1
    for index, step in enumerate(trace, start=1):
        where = f"trace[{index - 1}]"
        if not isinstance(step, dict):
            raise ValueError(f"{where} must be a step object")
        number = step.get("step", index)
        if type(number) is not int or number <= previous_step:
            raise ValueError(f"{where}.step must be a nonnegative, strictly increasing integer")
        previous_step = number
        text = step.get("assistant_text", "")
        calls, results = step.get("tool_calls", []), step.get("tool_results", [])
        if not isinstance(text, str) or not isinstance(calls, list) or not isinstance(results, list):
            raise ValueError(f"{where} requires string assistant_text and lists tool_calls/tool_results")
        if not text.strip() and not calls and not results:
            raise ValueError(f"{where} must contain assistant_text, tool_calls, or tool_results")
        for call in calls:
            if (not isinstance(call, dict)
                    or any(not isinstance(call.get(key), str) or not call[key].strip() for key in ("id", "name"))
                    or not isinstance(call.get("input"), dict)):
                raise ValueError(f"{where}.tool_calls entries require string id/name and object input")
        for result in results:
            if (not isinstance(result, dict)
                    or not isinstance(result.get("tool_use_id"), str) or not result["tool_use_id"].strip()
                    or not isinstance(result.get("content"), str)
                    or type(result.get("is_error", False)) is not bool):
                raise ValueError(f"{where}.tool_results entries require string tool_use_id/content and optional boolean is_error")
    # Reject non-finite numbers before preparing artifacts or opening a backend.
    try:
        json.dumps(response, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("response must contain finite JSON values") from exc


def render_trace(trace: list[dict[str, Any]]) -> str:
    """Use the research renderer's ordering and caps, with canonical argument keys.

    All limits count characters. The omission marker is additional to the global
    cap, matching the research head/tail policy. The final answer is never capped.
    """
    lines = []
    for index, step in enumerate(trace, start=1):
        lines.append(f"=== step {step.get('step', index)} ===")
        if step.get("assistant_text"):
            lines.append(f"assistant: {step['assistant_text']}")
        for call in step.get("tool_calls", []):
            args = json.dumps(call["input"], ensure_ascii=False, sort_keys=True, allow_nan=False)
            cap = RENDERING_POLICY["tool_args_chars"]
            if len(args) > cap:
                args = args[:cap] + "..."
            lines.append(f"tool_call {call['name']}({args})")
        for result in step.get("tool_results", []):
            content = result["content"]
            cap = RENDERING_POLICY["tool_result_chars"]
            if len(content) > cap:
                content = content[:cap] + "..."
            error = " [ERROR]" if result.get("is_error", False) else ""
            lines.append(f"tool_result{error}: {content}")
    text = "\n".join(lines)
    cap = RENDERING_POLICY["trace_chars"]
    if len(text) > cap:
        text = f"{text[:cap // 2]}\n\n... [trace truncated for length] ...\n\n{text[-(cap // 2):]}"
    return text or "[no trace steps]"


def render_answer(response: Mapping[str, Any]) -> str:
    validate_response(response)
    trace = render_trace(response["trace"]) if "trace" in response else "[trace not provided]"
    return (f"AGENT'S FINAL ANSWER:\n{response['answer']}\n\n"
            f"AGENT'S TRACE (assistant text, tool calls, tool results):\n{trace}")
