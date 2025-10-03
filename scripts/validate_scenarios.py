#!/usr/bin/env python3
"""
Validate FBA-Bench scenario YAML files against schemas/scenario_schema.json.

- Loads Draft-07 JSON Schema from file (no embedded schema).
- Scans src/scenarios recursively for *.yaml / *.yml.
- Prints per-file PASS/FAIL results and a summary.
- Exit code: 0 if all pass, 1 if any fail.

Requires: jsonschema, PyYAML
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml  # type: ignore
except Exception as e:
    print("Missing dependency: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

try:
    from jsonschema import Draft7Validator  # type: ignore
except Exception as e:
    print("Missing dependency: jsonschema is required. Install with: pip install jsonschema")
    sys.exit(1)


def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def repo_root_from_this_file() -> Path:
    # scripts/validate_scenarios.py -> repo root is parent of scripts
    return Path(__file__).resolve().parents[1]


def load_schema(schema_path: Path) -> Dict[str, Any]:
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path.as_posix()}")
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)
    return schema


def find_scenario_files(scenarios_dir: Path) -> List[Path]:
    if not scenarios_dir.exists():
        return []
    files = list(scenarios_dir.rglob("*.yaml")) + list(scenarios_dir.rglob("*.yml"))
    # Deterministic order
    files = sorted({p.resolve() for p in files})
    return files


def validate_document(doc: Any, validator: Draft7Validator) -> List[str]:
    errors = []
    for err in validator.iter_errors(doc):
        # Build a JSONPath-like location string
        path = "$"
        for p in err.absolute_path:
            if isinstance(p, int):
                path += f"[{p}]"
            else:
                path += f".{p}"
        errors.append(f"{path}: {err.message}")
    return errors


def validate_file(path: Path, validator: Draft7Validator) -> Tuple[bool, List[str]]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return False, [f"YAML parse error: {e}"]

    errors = validate_document(data, validator)
    ok = len(errors) == 0
    return ok, errors


def main() -> int:
    repo = repo_root_from_this_file()
    schema_path = repo / "schemas" / "scenario_schema.json"
    scenarios_dir = repo / "src" / "scenarios"

    _print_header("Scenario Schema Validation")
    print(f"Schema: {schema_path.as_posix()}")
    print(f"Scanning: {scenarios_dir.as_posix()}")

    try:
        schema = load_schema(schema_path)
    except Exception as e:
        print(f"Failed to load schema: {e}")
        return 1

    validator = Draft7Validator(schema)

    files = find_scenario_files(scenarios_dir)
    if not files:
        print("No scenario YAML files found.")
        _print_header("Summary")
        print("Checked: 0")
        print("Passed: 0")
        print("Failed: 0")
        print("Errors: 0")
        return 0

    total = len(files)
    passed = 0
    failed = 0
    total_errors = 0

    for path in files:
        ok, errors = validate_file(path, validator)
        if ok:
            print(f"PASS — {path.relative_to(repo).as_posix()}")
            passed += 1
        else:
            print(f"FAIL — {path.relative_to(repo).as_posix()}")
            for e in errors:
                print(f"  - {e}")
            failed += 1
            total_errors += len(errors)

    _print_header("Summary")
    print(f"Checked: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {total_errors}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())