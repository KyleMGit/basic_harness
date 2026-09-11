"""Shared byte guards for prepared skill-review provider requests."""

DEFAULT_SELECTED_VIEW_BYTES = 256 * 1024
DEFAULT_PREPARED_INPUT_BYTES = 512 * 1024
DEFAULT_WIRE_BODY_BYTES = 1024 * 1024


def validate_byte_limit(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer byte limit")
    return value
