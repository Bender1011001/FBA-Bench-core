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