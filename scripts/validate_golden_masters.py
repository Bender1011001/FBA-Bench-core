"""
Golden Masters JSON validation script for FBA-Bench Core.

Usage (from workspace root or the core repo root):
  python repos/fba-bench-core/scripts/validate_golden_masters.py
or
  cd repos/fba-bench-core && python scripts/validate_golden_masters.py

Exit codes:
  0 = all JSON files under golden_masters/ parsed successfully or directory missing
  1 = one or more JSON files failed to parse
"""

import glob
import json
import os
import sys


def resolve_core_root() -> str:
    # Allow running from workspace root or from inside core repo
    here = os.path.abspath(os.path.dirname(__file__))
    core_root = os.path.abspath(os.path.join(here, os.pardir))
    return core_root


def main() -> int:
    core_root = resolve_core_root()
    gm_dir = os.path.join(core_root, "golden_masters")
    if not os.path.isdir(gm_dir):
        print("golden_masters/ not found; skipping")
        return 0

    fail = False
    pattern = os.path.join(gm_dir, "**", "*.json")
    for path in glob.glob(pattern, recursive=True):
        try:
            with open(path, encoding="utf-8") as f:
                json.load(f)
            print(f"[OK] {os.path.relpath(path, core_root)}")
        except Exception as e:
            print(f"[FAIL] {os.path.relpath(path, core_root)}: {e}")
            fail = True

    if fail:
        print("One or more golden master JSON files failed to parse.")
        return 1

    print("All golden master JSON parse checks succeeded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
