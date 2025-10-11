# Migration Guide — Core Rebuild (Phase G′)

This document helps downstream consumers migrate their integrations to the rebuilt, contract-first core in the `rescue/core-rebuild` branch.

Summary of changes
- Core rebuilt with a contract-first approach: canonical, versioned Pydantic v2 models are the single source of truth for domain shapes and validation.
- Config objects are typed, immutable Pydantic models (use model_copy(update={...}) to derive modified copies).
- Events and Commands are expanded and typed with discriminated Literal event_type/command_type fields and strict validation (extra="forbid").
- Tooling: repository now includes pre-commit hooks and Towncrier integration to help maintain quality and changelog fragments.

High-level impact
- Consumers must import and validate against the canonical models instead of recreating local types.
- Unexpected or extra fields in inbound data will be rejected (models are strict).
- Numeric and money fields are represented as Decimal for precision — convert or coerce carefully.

Quick references (canonical implementations)
- Domain models: [`src/fba_bench_core/domain/models.py`](src/fba_bench_core/domain/models.py:1)
- Events & commands: [`src/fba_bench_core/domain/events.py`](src/fba_bench_core/domain/events.py:1)
- Typed configs: [`src/fba_bench_core/config.py`](src/fba_bench_core/config.py:1)
- Agent/service bases: [`src/fba_bench_core/agents/base.py`](src/fba_bench_core/agents/base.py:1) and [`src/fba_bench_core/services/base.py`](src/fba_bench_core/services/base.py:1)
- Pre-commit configuration: [`.pre-commit-config.yaml`](.pre-commit-config.yaml:1)
- Towncrier config: [`.towncrier.toml`](.towncrier.toml:1)

Migration steps for consumers
1) Update imports to canonical packages
- Replace any local copies of domain shapes with imports from the core package:
  - Before (local copy or legacy import)
    ```python
    # legacy or duplicated model
    from myproject.models import Product
    ```
  - After
    ```python
    from fba_bench_core.domain.models import Product
    ```
  See canonical definitions in [`src/fba_bench_core/domain/models.py`](src/fba_bench_core/domain/models.py:1).

2) Adopt typed configs and immutable patterns
- Replace dict-based configs with Pydantic config models or subclass the provided bases:
  - Example
    ```python
    from fba_bench_core.config import BaseAgentConfig

    cfg = BaseAgentConfig(agent_id="pricing-agent", poll_interval_seconds=30)
    # To change a field in an immutable model:
    new_cfg = cfg.model_copy(update={"poll_interval_seconds": 10})
    ```
- Where you previously mutated dicts, refactor to create updated copies with model_copy.

3) Handle stricter validation (extra fields & precise types)
- Incoming JSON/payloads validated by core models will now:
  - Reject unknown keys (extra="forbid")
  - Enforce Decimal and numeric constraints (use Decimal strings or coerce explicitly)
- If you parse external data (e.g., from third-party APIs) before handing to core models, sanitize or map fields explicitly:
  - Convert floats to Decimal:
    ```python
    from decimal import Decimal
    price = Decimal(str(external_data["price"]))
    ```
  - Remove unexpected keys or map them into known fields.

4) Update event/command handling interfaces
- The core defines BaseEvent/BaseCommand hierarchies and concrete subclasses with Literal discriminators (see [`src/fba_bench_core/domain/events.py`](src/fba_bench_core/domain/events.py:1)).
- Use pattern matching / isinstance checks against canonical classes instead of stringly-typed checks:
  ```python
  from fba_bench_core.domain.events import BaseEvent, SaleOccurred, AdjustPriceCommand

  def handle(ev: BaseEvent):
      if isinstance(ev, SaleOccurred):
          # ev.quantity, ev.revenue are validated & typed
          ...
  ```

5) Migrate business logic that relied on mutability
- Models (especially configs) are frozen/immutable. Replace in-place updates with model_copy or recreate instances.

6) Ensure tests validate contract behavior
- Add tests that import core models and assert ValidationError behavior for invalid inputs.
- Where previously you relied on duck-typing, write small adapter tests to pin expected shapes.

Breaking changes and deprecations
- Numeric types:
  - Price/cost/revenue fields are Decimal. Passing raw floats may be rejected or produce precision issues — prefer Decimal-strings or explicit Decimal conversion.
- Price invariants:
  - Product model enforces price >= cost. Code relying on loss-leading prices must instead model promotions or pass validated Promotion/Command objects.
- Immutable configs:
  - BaseAgentConfig and BaseServiceConfig are frozen. Do not mutate fields; use model_copy(update={...}).
- Extra fields:
  - Models use extra="forbid". Any consumer that previously relied on permissive parsing must explicitly map or strip extra keys.
- Event/Command discriminators:
  - event_type and command_type are Literal discriminators. Do not send arbitrary strings; use the provided classes or carefully constructed dicts that match those discriminators.

Developer checklist for a migration PR
- [ ] Replace duplicated domain models with imports from [`src/fba_bench_core/domain/models.py`](src/fba_bench_core/domain/models.py:1).
- [ ] Convert config dicts to Pydantic models or subclasses of [`fba_bench_core.config.BaseAgentConfig`](src/fba_bench_core/config.py:1).
- [ ] Add/adjust tests to assert ValidationError for invalid inputs and Decimal semantics.
- [ ] Run linters and formatters; the repository uses pre-commit hooks — see [` .pre-commit-config.yaml`](.pre-commit-config.yaml:1).
- [ ] Create a Towncrier fragment under [`newsfragments/`](newsfragments/:1) for any user-visible/breaking change and run `towncrier build --yes` to update `CHANGELOG.md`.

Examples — common migration patterns
- Mapping a third-party payload to SaleOccurred:
  ```python
  from decimal import Decimal
  from datetime import datetime
  from fba_bench_core.domain.events import SaleOccurred

  external = {"order_id": "o1", "sku": "sku-1", "qty": 2, "total": 19.98}
  ev = SaleOccurred(
      order_id=external["order_id"],
      product_sku=external["sku"],
      quantity=external["qty"],
      revenue=Decimal(str(external["total"])),
      currency="USD",
      timestamp=datetime.utcnow()
  )
  ```

Tooling notes
- Pre-commit:
  - Configuration is present at [`.pre-commit-config.yaml`](.pre-commit-config.yaml:1). Install and run locally:
    ```bash
    pip install pre-commit
    pre-commit install
    pre-commit run --all-files
    ```
- Towncrier:
  - Use Towncrier fragments under [`newsfragments/`](newsfragments/:1). Configuration: [`.towncrier.toml`](.towncrier.toml:1).
  - To build changelog:
    ```bash
    poetry run towncrier build --yes
    ```

Troubleshooting tips
- ValidationError on import/instantiation:
  - Inspect the raised pydantic ValidationError messages. They will indicate missing/extra fields or type mismatches.
- Decimal vs float issues:
  - Use Decimal(str(value)) instead of Decimal(value) if value is a float to avoid binary-float artifacts.
- Tests failing after migration:
  - Confirm tests import canonical models, and update fixtures to use Decimal-strings and validated shapes.

References
- Canonical models and events: [`src/fba_bench_core/domain/models.py`](src/fba_bench_core/domain/models.py:1), [`src/fba_bench_core/domain/events.py`](src/fba_bench_core/domain/events.py:1)
- Config bases: [`src/fba_bench_core/config.py`](src/fba_bench_core/config.py:1)
- Tooling: [`.pre-commit-config.yaml`](.pre-commit-config.yaml:1), [`.towncrier.toml`](.towncrier.toml:1)

Appendix — Quick migration example checklist (copy into PR description)
- Replace local model imports with core imports
- Convert numeric fields to Decimal
- Replace dict-configs with Base*Config subclasses or model_copy usage
- Strip or map extra fields before passing to core models
- Add Towncrier fragment for breaking changes
