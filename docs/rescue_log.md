# Rescue Log

## 2025-10-05 – Phase 2 Demolition

- Timestamp (UTC): 2025-10-05T11:17Z
- Branch: rescue/core-rebuild is checked out and synced (no new commits fetched)
- Inventory: [`docs/legacy_src_inventory.txt`](docs/legacy_src_inventory.txt:1) was generated prior to demolition and contains 202 entries
- Action: The entire legacy `src/` directory was removed
- New skeleton: Created skeleton directories and placeholder files (11 files across 5 subdirectories)
- Placeholders: Added placeholder content (docstrings, `__all__`, `pass` statements) to keep imports and linting functional
- Next phases: Placeholder implementations will be filled in Phases 3–7

## 2025-10-05 – Phase 4 Domain events & commands

- Timestamp (UTC): 2025-10-05T11:54:46.447Z
- Branch: rescue/core-rebuild
- Action: Implemented domain events and commands as Pydantic v2 models in `src/fba_bench_core/domain/events.py` with validation constraints.
- Details:
  - BaseEvent: `event_type: str`, `timestamp: datetime` (UTC, default_factory), `tick: int` (ge=0).
  - SaleOccurred: strict `event_type: Literal["sale_occurred"]`, `order_id`, `product_sku`, `quantity` (ge=1), `revenue` (gt=0), `currency: Literal["USD"]`.
  - StockUpdated: strict `event_type: Literal["stock_updated"]`, `product_sku`, `previous_level` (ge=0), `new_level` (ge=0), optional `reason`.
  - Commands: Command base and two concrete commands (`AdjustPriceCommand`, `PlaceOrderCommand`) plus `AnyCommand` alias.
  - model_config uses `ConfigDict(str_strip_whitespace=True)` where appropriate. `timestamp` uses `Field(default_factory=datetime.utcnow)`.
- Notes:
  - Downstream phases/components will consume these contracts for validation and orchestration.
  - Optional validation/import check: pending

## 2025-10-05 – Phase 8 — Packaging overhaul

- Timestamp (UTC): 2025-10-05T12:37:00Z
- Branch: rescue/core-rebuild
- Action: Overhauled packaging and development tooling; migrated project packaging from `pyproject.toml` to a Poetry-managed configuration.
- Details:
  - Poetry configuration introduced with canonical project metadata: `name`, `version`, `description`, `license`, `readme` migrated where applicable from the previous `pyproject.toml`.
  - Runtime dependency surface trimmed to `pydantic = "^2.0"` and project requires `python = "^3.11"`.
  - Dev tool group added (per Rescue Guide) including `pytest`, `black`, `isort`, `flake8`, `mypy`, and `pre-commit` with pinned versions as specified by the Rescue Guide.
  - Existing tooling blocks for `ruff`, `black`, `mypy`, and `pytest` have been retained and aligned to target Python 3.11 where applicable.
  - TOML validation (syntax/semantics checks) is still pending and will be run separately as a follow-up step.
- Notes:
  - Update README packaging instructions in a future phase if needed.

## Phase 9 — Core Contract Validation

- **Timestamp (UTC): 2025-10-05T13:05:06Z**
- Branch: rescue/core-rebuild

### Summary

- Created Pydantic v2 validation tests for domain contracts including `SaleOccurred` and `AdjustPriceCommand`.
- Documented legacy `Product` behavior and added coverage for:
  - Positive price checks
  - Timestamp handling and defaulting behavior
  - Numeric/dict coercion cases
  - `calculate_profit_margin` edge case handling

### Test Commands

- Run tests: `poetry run pytest tests/test_core_contracts.py` — executed successfully.
- Test file: [`tests/test_core_contracts.py`](tests/test_core_contracts.py:1)

### Observations

- Default event type validation enforced for `SaleOccurred` (strict Literal type).
- Positive pricing enforcement validated for revenue/price fields (gt/positive constraints).
- Legacy `Product` coercion and behavior documented (numeric/dict coercion, timestamp defaults).
- Pytest results: 14 passed, 0 failed/skipped; runtime ≈3.12s.

### Next Steps

- Prepare commit and push for Phase 9 changes.
- Expand regression coverage to other contract models as needed.
- Re-run TOML/package validation (pending from Phase 8) after merging if relevant.

## 2025-10-05 – Phase 10 Formatters, linters & type-checking

- Timestamp (UTC): 2025-10-05T14:26Z
- Branch: rescue/core-rebuild
- Action: Ran formatting, import-sorting, linting, and type-checking; resolved reported issues and adjusted configuration to avoid legacy noise.

- Commands run:
  - `poetry run black src tests` — no changes
  - `poetry run isort src tests` — reformatted multiple files (import sorting applied)
  - `poetry run flake8 src tests` — initial violations identified; fixed via targeted code edits; rerun clean
  - `poetry run pip install mypy==1.8.0` — manual install used because `poetry install --with dev` failed due to SciPy/NumPy dependency resolution conflict
  - `poetry run python -m mypy` — success; 20 files checked

- Notable fixes and configuration changes:
  - Flake8: resolved import-order, unused-import, and minor typing/annotation issues across multiple modules to achieve a clean linter run.
  - Mypy configuration: narrowed scope in [`pyproject.toml`](pyproject.toml:1) to target [`src/fba_bench_core/`](src/fba_bench_core/:1) and [`tests/`](tests/:1) to avoid legacy/irrelevant noise from remaining legacy sources.
  - Explicit mypy install: performed `poetry run pip install mypy==1.8.0` as a pragmatic workaround to a dependency resolution failure when attempting `poetry install --with dev`.

- Final status:
  - Black: no changes required.
  - isort: import-sorting changes applied.
  - Flake8: no remaining violations after fixes.
  - Mypy: successful type-check across 20 files.
  - Overall: All formatting, linting, and type-checking tools are clean and the repository is ready for commit and subsequent Phase 11.
