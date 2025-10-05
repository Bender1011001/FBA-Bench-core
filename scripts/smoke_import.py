"""
Lightweight import smoke test for FBA-Bench Core.

Usage (from repository root):
  python -c "import runpy; runpy.run_path('repos/fba-bench-core/scripts/smoke_import.py')"
Expected output: lines indicating successful imports.
"""
def main() -> None:
    ok = True
    failures = []

    def try_import(mod):
        try:
            __import__(mod)
            print(f"[OK] import {mod}")
        except Exception as e:
            print(f"[FAIL] import {mod}: {e}")
            return False
        return True

    EXPECTED_IMPORTS = [
        "fba_bench_core",
        "fba_bench_core.agents",
        "fba_bench_core.domain",
        "fba_bench_core.domain.events",
        "fba_bench_core.domain.models",
        "fba_bench_core.scenarios",
        "fba_bench_core.services",
        "fba_bench_core.exceptions",
    ]

    for t in EXPECTED_IMPORTS:
        if not try_import(t):
            failures.append(t)

    if failures:
        raise SystemExit(f"Smoke import failures: {failures}")
    print("All core smoke imports succeeded.")

if __name__ == "__main__":
    main()