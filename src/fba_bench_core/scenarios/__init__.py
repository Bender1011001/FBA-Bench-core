from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml

# Mapping from scenario tier to YAML filename in src/scenarios.
_TIER_FILES: Dict[int, str] = {
    0: "tier_0_baseline.yaml",
    1: "tier_1_moderate.yaml",
    2: "tier_2_advanced.yaml",
}


def _scenarios_dir() -> Path:
    """
    Locate the scenarios directory relative to this package file.

    This file lives at: .../src/fba_bench_core/scenarios/__init__.py
    We need:            .../src/scenarios
    """
    return Path(__file__).resolve().parent.parent.parent / "scenarios"


def load_scenario(tier: int) -> Dict[str, Any]:
    """
    Load and parse a built-in scenario YAML by tier.

    Parameters:
        tier: Scenario tier (0, 1, or 2).

    Returns:
        Parsed YAML content as a dictionary.

    Raises:
        ValueError: If 'tier' is not one of 0, 1, 2.
        FileNotFoundError: If the mapped scenario file is not found.
        yaml.YAMLError: If the YAML content cannot be parsed.
    """
    if tier not in _TIER_FILES:
        raise ValueError(f"tier must be one of {sorted(_TIER_FILES)}; got {tier}")

    scenarios_dir = _scenarios_dir()
    path = scenarios_dir / _TIER_FILES[tier]

    if not path.exists():
        raise FileNotFoundError(f"Scenario file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data


__all__ = ["load_scenario"]