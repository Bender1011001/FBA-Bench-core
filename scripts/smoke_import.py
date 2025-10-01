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

    # Core packages commonly used by tests and users
    targets = [
        "metrics",
        "agents",
        "baseline_bots",
        "constraints",
        "money",
        "fba_bench_core",  # marker package
        # Optional subsets if present in core stage:
        "scenarios",
        "benchmarking",
        "models",
        "config",
    ]

    for t in targets:
        if not try_import(t):
            failures.append(t)

    if failures:
        raise SystemExit(f"Smoke import failures: {failures}")
    print("All smoke imports succeeded.")

if __name__ == "__main__":
    main()