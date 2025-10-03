#!/usr/bin/env python3
"""
FBA-Bench: Repository-wide validation

Validates:
- Scenarios: YAML files under src/scenarios against schemas/scenario_schema.json
- Leaderboard: site/data/leaderboard.json against schemas/leaderboard_schema.json and duplicate (team, model) pairs
- Golden masters: All *.json under golden_masters parse as valid JSON

Output:
- Clear, sectioned PASS/FAIL per file with error details
- Deterministic ordering
- Exit 0 if all pass, else 1

Requires: jsonschema, PyYAML
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

try:
    import yaml  # type: ignore
except Exception:
    print("Missing dependency: PyYAML is required. Install with: pip install pyyaml")
    sys.exit(1)

try:
    from jsonschema import Draft7Validator  # type: ignore
except Exception:
    print("Missing dependency: jsonschema is required. Install with: pip install jsonschema")
    sys.exit(1)


# ---------- Helpers ----------

def _print_header(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def repo_root_from_this_file() -> Path:
    # scripts/validate_all.py -> repo root is parent of scripts
    return Path(__file__).resolve().parents[1]


def load_schema(schemas_dir: Path, name: str) -> Dict[str, Any]:
    """
    Load a JSON schema by base name from schemas directory.

    name examples:
      - 'scenario_schema'
      - 'leaderboard_schema'
    """
    path = schemas_dir / f"{name}.json"
    if not path.exists():
        raise FileNotFoundError(f"Schema not found: {path.as_posix()}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _format_errors_from_validator(doc: Any, validator: Draft7Validator) -> List[str]:
    errors: List[str] = []
    for err in validator.iter_errors(doc):
        path = "$"
        for p in err.absolute_path:
            if isinstance(p, int):
                path += f"[{p}]"
            else:
                path += f".{p}"
        errors.append(f"{path}: {err.message}")
    return errors


# ---------- Scenarios Validation ----------

def find_scenario_files(scenarios_dir: Path) -> List[Path]:
    if not scenarios_dir.exists():
        return []
    files = list(scenarios_dir.rglob("*.yaml")) + list(scenarios_dir.rglob("*.yml"))
    files = sorted({p.resolve() for p in files})
    return files


def validate_scenarios(repo: Path, schemas_dir: Path) -> Dict[str, Any]:
    _print_header("SCENARIOS")
    scenarios_dir = repo / "src" / "scenarios"
    print(f"Schema: {(schemas_dir / 'scenario_schema.json').as_posix()}")
    print(f"Scanning: {scenarios_dir.as_posix()}")

    try:
        schema = load_schema(schemas_dir, "scenario_schema")
    except Exception as e:
        print(f"Failed to load scenario schema: {e}")
        return {"ok": False, "checked": 0, "passed": 0, "failed": 0, "errors": 1, "details": []}

    validator = Draft7Validator(schema)
    files = find_scenario_files(scenarios_dir)

    if not files:
        print("No scenario YAML files found.")
        return {"ok": True, "checked": 0, "passed": 0, "failed": 0, "errors": 0, "details": []}

    passed = 0
    failed = 0
    total_errors = 0
    details: List[Dict[str, Any]] = []

    for path in files:
        rel = path.relative_to(repo).as_posix()
        try:
            with path.open("r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            print(f"FAIL — {rel}")
            print(f"  - YAML parse error: {e}")
            failed += 1
            total_errors += 1
            details.append({"path": rel, "ok": False, "errors": [f"YAML parse error: {e}"]})
            continue

        errors = _format_errors_from_validator(data, validator)
        if errors:
            print(f"FAIL — {rel}")
            for e in errors:
                print(f"  - {e}")
            failed += 1
            total_errors += len(errors)
            details.append({"path": rel, "ok": False, "errors": errors})
        else:
            print(f"PASS — {rel}")
            passed += 1
            details.append({"path": rel, "ok": True, "errors": []})

    print(f"Checked: {len(files)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {total_errors}")

    return {
        "ok": failed == 0,
        "checked": len(files),
        "passed": passed,
        "failed": failed,
        "errors": total_errors,
        "details": details,
    }


# ---------- Leaderboard Validation ----------

def _extract_leaderboard_entries(doc: Any) -> Tuple[List[Dict[str, Any]], str]:
    """
    Supports both shapes:
      - top-level array of entries
      - object with 'entries': [...]
    Returns (entries, shape) where shape is 'array' or 'object'.
    """
    if isinstance(doc, list):
        return doc, "array"
    if isinstance(doc, dict) and isinstance(doc.get("entries"), list):
        return doc["entries"], "object"
    return [], "unknown"


def validate_leaderboard(repo: Path, schemas_dir: Path) -> Dict[str, Any]:
    _print_header("LEADERBOARD")
    data_path = repo / "site" / "data" / "leaderboard.json"
    print(f"Schema: {(schemas_dir / 'leaderboard_schema.json').as_posix()}")
    print(f"File:   {data_path.as_posix()}")

    try:
        schema = load_schema(schemas_dir, "leaderboard_schema")
    except Exception as e:
        print(f"Failed to load leaderboard schema: {e}")
        return {"ok": False, "checked": 0, "passed": 0, "failed": 1, "errors": 1, "details": []}

    validator = Draft7Validator(schema)

    if not data_path.exists():
        msg = "leaderboard.json not found"
        print(f"FAIL — site/data/leaderboard.json")
        print(f"  - {msg}")
        return {"ok": False, "checked": 0, "passed": 0, "failed": 1, "errors": 1, "details": [msg]}

    try:
        with data_path.open("r", encoding="utf-8") as f:
            doc = json.load(f)
    except Exception as e:
        print(f"FAIL — site/data/leaderboard.json")
        print(f"  - JSON parse error: {e}")
        return {"ok": False, "checked": 1, "passed": 0, "failed": 1, "errors": 1, "details": [f"JSON parse error: {e}"]}

    errors = _format_errors_from_validator(doc, validator)
    passed = 0
    failed = 0
    total_errors = 0
    details: List[str] = []

    if errors:
        print("FAIL — site/data/leaderboard.json")
        for e in errors:
            print(f"  - {e}")
        failed += 1
        total_errors += len(errors)
        details.extend(errors)
    else:
        print("PASS — site/data/leaderboard.json")
        passed += 1

    # Duplicate (team, model) pair check
    entries, shape = _extract_leaderboard_entries(doc)
    if entries:
        seen = set()
        dupes: List[str] = []
        for idx, item in enumerate(entries):
            if not isinstance(item, dict):
                # shape validated by schema; if not, it would appear above
                continue
            team = item.get("team")
            model = item.get("model")
            key = (team, model)
            if key in seen:
                dupes.append(f"Duplicate team+model at index {idx}: ({team}, {model})")
            else:
                seen.add(key)
        if dupes:
            print("FAIL — Duplicate (team, model) pairs")
            for d in dupes:
                print(f"  - {d}")
            failed += 1
            total_errors += len(dupes)
            details.extend(dupes)

    print(f"Checked: 1")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {total_errors}")

    return {
        "ok": failed == 0,
        "checked": 1,
        "passed": passed,
        "failed": failed,
        "errors": total_errors,
        "details": details,
    }


# ---------- Golden Masters Validation ----------

def validate_golden_masters(repo: Path) -> Dict[str, Any]:
    _print_header("GOLDEN MASTERS")
    gm_dir = repo / "golden_masters"
    print(f"Scanning: {gm_dir.as_posix()}")

    if not gm_dir.exists():
        print("No golden_masters directory found.")
        return {"ok": True, "checked": 0, "passed": 0, "failed": 0, "errors": 0, "details": []}

    files = sorted({p.resolve() for p in gm_dir.rglob("*.json")})
    if not files:
        print("No JSON files found in golden_masters.")
        return {"ok": True, "checked": 0, "passed": 0, "failed": 0, "errors": 0, "details": []}

    passed = 0
    failed = 0
    total_errors = 0
    details: List[Dict[str, Any]] = []

    for path in files:
        rel = path.relative_to(repo).as_posix()
        try:
            with path.open("r", encoding="utf-8") as f:
                json.load(f)
            print(f"PASS — {rel}")
            passed += 1
            details.append({"path": rel, "ok": True, "errors": []})
        except Exception as e:
            print(f"FAIL — {rel}")
            print(f"  - JSON parse error: {e}")
            failed += 1
            total_errors += 1
            details.append({"path": rel, "ok": False, "errors": [f"JSON parse error: {e}"]})

    print(f"Checked: {len(files)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Errors: {total_errors}")

    return {
        "ok": failed == 0,
        "checked": len(files),
        "passed": passed,
        "failed": failed,
        "errors": total_errors,
        "details": details,
    }


# ---------- Print summary and main ----------

def print_results(results: Dict[str, Dict[str, Any]]) -> None:
    _print_header("SUMMARY")
    total_checked = sum(section["checked"] for section in results.values())
    total_passed = sum(section["passed"] for section in results.values())
    total_failed = sum(section["failed"] for section in results.values())
    total_errors = sum(section["errors"] for section in results.values())
    overall_ok = all(section["ok"] for section in results.values())

    print(f"Sections: {', '.join(results.keys())}")
    print(f"Checked: {total_checked}")
    print(f"Passed: {total_passed}")
    print(f"Failed: {total_failed}")
    print(f"Errors: {total_errors}")
    print(f"Overall: {'PASS' if overall_ok else 'FAIL'}")


def main() -> int:
    repo = repo_root_from_this_file()
    schemas_dir = repo / "schemas"

    scenarios_res = validate_scenarios(repo, schemas_dir)
    leaderboard_res = validate_leaderboard(repo, schemas_dir)
    golden_res = validate_golden_masters(repo)

    results = {
        "scenarios": scenarios_res,
        "leaderboard": leaderboard_res,
        "golden_masters": golden_res,
    }

    print_results(results)
    overall_ok = all(section["ok"] for section in results.values())
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())