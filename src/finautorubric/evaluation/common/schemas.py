"""JSON-schema builders in the strict style the CLI backends expect.

Every object lists all properties as required and forbids extras, which is what
both `codex exec --output-schema` and `claude -p --json-schema` accept.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence


def obj(properties: Mapping[str, Any], *, required: Sequence[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required if required is not None else properties.keys()),
        "additionalProperties": False,
    }


def arr(items: Mapping[str, Any], *, min_items: int | None = None, max_items: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "array", "items": dict(items)}
    if min_items is not None:
        schema["minItems"] = min_items
    if max_items is not None:
        schema["maxItems"] = max_items
    return schema


def string(*, min_length: int = 0, enum: Sequence[str] | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if min_length:
        schema["minLength"] = min_length
    if enum is not None:
        schema["enum"] = list(enum)
    return schema


def integer(*, minimum: int | None = None, maximum: int | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "integer"}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


def number(*, minimum: float | None = None, maximum: float | None = None) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number"}
    if minimum is not None:
        schema["minimum"] = minimum
    if maximum is not None:
        schema["maximum"] = maximum
    return schema


BOOLEAN: dict[str, Any] = {"type": "boolean"}

# Free-form stages (EvalAgent's list-shaped prose outputs) are wrapped so that
# the structured-output contract of the backends still holds.
TEXT_OUTPUT = obj({"text": string(min_length=1)})
LINES_OUTPUT = obj({"lines": arr(string(min_length=1))})
