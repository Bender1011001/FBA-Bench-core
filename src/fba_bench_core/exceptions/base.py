"""Base exceptions for fba_bench_core.

This module defines the foundational exception hierarchy used across
the fba_bench_core package. It provides a single root exception and
a couple of commonly-used subclasses to improve error handling and
exception semantics throughout the project.

Classes:
- FBABenchException: Root exception for all fba_bench_core errors.
- ConfigurationError: Raised for configuration-related problems.
- AgentError: Raised for errors during agent execution or lifecycle.
"""

class FBABenchException(Exception):
    """Base exception for fba_bench_core.

    Acts as the common ancestor for all library-specific exceptions so
    callers can catch library errors without accidentally catching other
    exceptions that inherit directly from Exception.
    """
    pass


class ConfigurationError(FBABenchException):
    """Raised when there is an issue with configuration or setup.

    Examples include invalid config values, missing required settings, or
    failure to parse configuration files.
    """
    pass


class AgentError(FBABenchException):
    """Raised for agent execution or lifecycle failures.

    Use this when an agent encounters an unrecoverable error during
    planning, execution, or coordination.
    """
    pass


__all__ = ["FBABenchException", "ConfigurationError", "AgentError"]