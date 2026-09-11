"""Fixed, privacy-safe diagnostic context for skill review validation."""
from dataclasses import dataclass

from openai import APIStatusError


REVIEW_REASONS = frozenset({
    "prepared_input_invalid", "prepared_input_oversized", "wire_body_oversized",
    "no_choices", "non_stop_finish", "missing_or_nontext_output",
    "output_invalid_encoding", "output_oversized", "malformed_json",
    "duplicate_fields", "proposal_not_object", "invalid_action_or_fields",
    "incomplete_complete_flag", "empty_fields", "invalid_name",
    "invalid_description", "incomplete_instructions", "safety_rejection",
})
FINISH_REASONS = frozenset({"stop", "length", "content_filter", "tool_calls", "function_call"})
COUNT_FIELDS = frozenset({
    "output_tokens", "observed_bytes", "limit_bytes", "usage_prompt_tokens",
    "usage_completion_tokens", "usage_total_tokens",
})
BOOLEAN_FIELDS = frozenset({"request_attempted", "response_received"})
METADATA_ORDER = (
    "request_attempted", "response_received", "finish_reason", "output_tokens",
    "usage_prompt_tokens", "usage_completion_tokens", "usage_total_tokens",
    "observed_bytes", "limit_bytes",
)
MAX_DIAGNOSTIC_COUNT = 1_000_000_000


class ReviewDiagnosticError(ValueError):
    """Host-owned reason code plus data that must be revalidated before display."""

    def __init__(self, reason, **metadata):
        message = {
            "prepared_input_invalid": "Prepared snapshot is invalid",
            "prepared_input_oversized": "Prepared snapshot exceeds input limit",
            "wire_body_oversized": "Complete SDK wire body exceeds limit",
        }.get(reason, "Skill review validation failed")
        super().__init__(message)
        self.reason = reason
        self.metadata = metadata


@dataclass(frozen=True)
class _TransportContext:
    metadata: dict


def finish_reason(value):
    return value if type(value) is str and value in FINISH_REASONS else "<invalid>"


def usage_metadata(response):
    usage = getattr(response, "usage", None)
    values = {}
    for source, target in (
        ("prompt_tokens", "usage_prompt_tokens"),
        ("completion_tokens", "usage_completion_tokens"),
        ("total_tokens", "usage_total_tokens"),
    ):
        value = getattr(usage, source, None) if usage is not None else None
        if type(value) is int and 0 <= value <= MAX_DIAGNOSTIC_COUNT:
            values[target] = value
    return values


def mark_transport_failure(exc, output_tokens, *, request_attempted=True):
    """Annotate known SDK observations without changing exception class or outcome."""
    metadata = dict(request_attempted=request_attempted, output_tokens=output_tokens)
    if not request_attempted:
        metadata["response_received"] = False
    elif isinstance(exc, APIStatusError):
        metadata["response_received"] = True
    context = _TransportContext(metadata)
    try:
        exc._skill_review_transport_context = context
    except Exception:
        pass


def _safe_metadata(metadata):
    if type(metadata) is not dict:
        return []
    parts = []
    for key in METADATA_ORDER:
        if key not in metadata:
            continue
        value = metadata[key]
        if key in BOOLEAN_FIELDS:
            if type(value) is bool:
                parts.append(f"{key}={'true' if value else 'false'}")
        elif key == "finish_reason":
            parts.append("finish_reason=" + finish_reason(value))
        elif key in COUNT_FIELDS:
            if type(value) is int and 0 <= value <= MAX_DIAGNOSTIC_COUNT:
                parts.append(f"{key}={value}")
    return parts


def safe_review_context(exc):
    """Serialize only exact host context types and fixed allowlisted values."""
    if type(exc) is ReviewDiagnosticError:
        reason = exc.reason if type(exc.reason) is str and exc.reason in REVIEW_REASONS else "<invalid>"
        return ["reason=" + reason, *_safe_metadata(exc.metadata)]
    context = getattr(exc, "_skill_review_transport_context", None)
    if type(context) is _TransportContext:
        return _safe_metadata(context.metadata)
    return []
