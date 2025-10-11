"""Registry for validators."""

from collections.abc import Callable

_validators: dict[str, Callable] = {}


def register_validator(name: str, validator_class: Callable) -> None:
    """Register a validator class."""
    _validators[name] = validator_class


def get_validator(name: str) -> Callable | None:
    """Get a validator class by name."""
    return _validators.get(name)


def list_validators() -> dict[str, Callable]:
    """List all registered validators."""
    return _validators.copy()
