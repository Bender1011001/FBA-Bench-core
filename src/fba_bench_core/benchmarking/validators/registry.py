"""Registry for validators."""

from collections.abc import Callable


class ValidatorRegistry:
    _validators: dict[str, Callable] = {}

    @classmethod
    def register(cls, name: str, validator_class: Callable) -> None:
        """Register a validator class."""
        cls._validators[name] = validator_class

    @classmethod
    def create_validator(cls, name: str, config=None) -> Callable | None:
        """Create a validator instance."""
        fn = cls._validators.get(name)
        if fn:
            return fn(config) if config else fn()
        return None

    @classmethod
    def get_validator(cls, name: str) -> Callable | None:
        """Get a validator class by name."""
        return cls._validators.get(name)

    @classmethod
    def list_validators(cls) -> list[str]:
        """List all registered validator names."""
        return list(cls._validators.keys())


# Global instance for function-based API
registry = ValidatorRegistry()


def get_validator(name: str) -> Callable:
    """Get a validator by name, raising KeyError if not found."""
    validator = registry.get_validator(name)
    if validator is None:
        raise KeyError(f"Validator '{name}' not found")
    return validator


def list_validators() -> list[str]:
    """List all registered validator names."""
    return registry.list_validators()


def register_validator(name: str):
    """Decorator to register a validator function with the given name."""

    def decorator(func: Callable) -> Callable:
        registry.register(name, func)
        return func

    return decorator
