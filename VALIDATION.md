# FBA-Bench Core — Validation

This document describes how to validate the staged Core repository locally.

## Prerequisites
- Python 3.9+ and virtual environment recommended.

## 1) Environment setup
```bash
# from workspace root
python -m venv .venv
# Windows PowerShell
.\\.venv\\Scripts\\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -e repos/fba-bench-core pytest
```

Quick smoke import:
```bash
python -c "import metrics, agents, baseline_bots, fba_bench_core; print('core-ok')"
# Expected: prints 'core-ok' with no ImportError
```

## 2) Run Core tests (curated subset)
```bash
cd repos/fba-bench-core
pytest -q
cd ../..
```
Expected:
- All tests in the staged core subset pass (if present).
- If no tests are present (initial staging), proceed with smoke import and golden master checks below.

## 3) Golden Masters validation
Run the validator script to parse all JSON files under `golden_masters/`:
```bash
python repos/fba-bench-core/scripts/validate_golden_masters.py
```

Expected output (example):
```
[OK] golden_masters/golden_run_baseline/.../summary.json
[OK] golden_masters/golden_run_baseline/.../experiment_config.yaml (skipped unless JSON)
All golden master JSON parse checks succeeded.
```

Notes:
- YAML files are ignored by the JSON validator.
- This mirrors the lightweight validation embedded in Core CI ([.github/workflows/ci.yml](.github/workflows/ci.yml)).

## 4) Troubleshooting
- Import errors:
  - Ensure editable install succeeded: `pip install -e repos/fba-bench-core`
  - Verify smoke imports: `python -c "import metrics, agents, baseline_bots, fba_bench_core; print('core-ok')"`
- Golden masters:
  - Ensure directory exists: `repos/fba-bench-core/golden_masters`
  - Validator exit code 1 indicates at least one JSON parse failure.

## 5) Next steps
- See [README](README.md) for methodology, scenarios, and contribution guidance.
- Public CI will use the same validation pattern.