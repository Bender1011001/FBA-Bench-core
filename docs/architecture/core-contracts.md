# Core Contracts Overview

This document describes the rebuilt, contract-first core used by FBA-Bench.
It summarizes the canonical domain models, events/commands taxonomy,
agent/service base classes, and typed configuration contracts.

Purpose

- Provide a single source of truth for domain semantics used across the repo.
- Make integration predictable: downstream projects import and validate
  against stable Pydantic v2 models.

Where to find the canonical implementations

- Domain models: [`src/fba_bench_core/domain/models.py`](src/fba_bench_core/domain/models.py:1)
- Events & commands: [`src/fba_bench_core/domain/events.py`](src/fba_bench_core/domain/events.py:1)
- Typed configs: [`src/fba_bench_core/config.py`](src/fba_bench_core/config.py:1)
- Agent/service bases: [`src/fba_bench_core/agents/base.py`](src/fba_bench_core/agents/base.py:1),
  [`src/fba_bench_core/services/base.py`](src/fba_bench_core/services/base.py:1)

Domain models (high level)

Product

- Purpose: canonical representation of a sellable SKU used by simulations and
  pricing/inventory logic.
- Key fields: product_id, sku, name, cost (Decimal), price (Decimal), stock (int),
  max_inventory (Optional[int]), fulfillment_latency (Optional[int]).
- Invariants:
  - cost and price are stored as Decimal for financial precision.
  - price >= cost (domain-level rule; loss-leading scenarios should be modeled
    as promotions or events in the simulation layer).
  - stock >= 0 and, if present, stock <= max_inventory.
- Where enforced: see model validators in
  [`src/fba_bench_core/domain/models.py`](src/fba_bench_core/domain/models.py:1)

InventorySnapshot

- Purpose: capture a point-in-time view of inventory for reconciliation and
  forecasting.
- Key fields: product_id, available_units, reserved_units, warehouse_location,
  timestamp.
- Invariants:
  - available_units >= 0, reserved_units >= 0
  - reserved_units must not exceed the total units represented by the snapshot.
- Where enforced: [`src/fba_bench_core/domain/models.py`](src/fba_bench_core/domain/models.py:1)

CompetitorListing & Competitor

- Purpose: represent external marketplace listings used by repricing logic.
- CompetitorListing fields: sku, price (Decimal), rating, fulfillment_latency,
  marketplace.
- Competitor invariants: listing SKUs must be unique when provided.

DemandProfile

- Purpose: compact stochastic demand assumptions (mean/std) per product.
- Invariants: daily_demand_mean >= 0, daily_demand_std >= 0.

Events & Commands taxonomy

Base contracts

- BaseEvent: common event metadata (event_type, timestamp, tick,
  correlation_id). Implementations use Literal event_type discriminators.
  [`src/fba_bench_core/domain/events.py`](src/fba_bench_core/domain/events.py:1)
- Command: intent-to-change contract with metadata, issued_by, reason,
  correlation_id, and a shallow metadata dict.

Representative events

- SaleOccurred: finalized sale; includes revenue, quantity, gross_margin.
- StockReplenished / StockDepleted / FulfillmentDelayed: inventory signals.
- PromotionLaunched / DemandSpiked: marketing and demand signals.
- ForecastUpdated / AnomalyDetected: analytics & monitoring events.

Representative commands

- AdjustPriceCommand: propose a new price for a product (proposed_price:
  Decimal).
- LaunchPromotionCommand: instruct system to apply discounts or promos.
- PlaceReplenishmentOrderCommand: place supplier orders with quantity, priority.

How events and commands are intended to flow

- Observers / sensors produce typed events (BaseEvent subclasses) when
  domain activity occurs (sales, competitor changes, monitoring signals).
- Agents (implementations of BaseAgent) receive events and may emit Commands
  (subclasses of Command) to express intent (reprice, replenish, launch promo).
- Commands are then executed by downstream services or orchestrators that
  translate intent into side-effects (DB writes, API calls, supplier orders).

Transactional guidance

- Prefer idempotent, small commands. Commands include correlation_id and
  issued_by to assist observability and retries.
- Validate incoming events/commands at service boundaries using the provided
  Pydantic models; reject unknown/extra fields (models are extra="forbid").

Agent & Service base classes

- BaseAgent: abstract async API with a constructor accepting a typed
  [`BaseAgentConfig`](src/fba_bench_core/config.py:1). Implementations must
  implement async decide(self, events: List[BaseEvent]) -> List[Command].
- BaseService: synchronous minimal base accepting [`BaseServiceConfig`](src/fba_bench_core/config.py:1)
  and exposing start() and config properties.

Typed configuration contracts

- BaseAgentConfig and BaseServiceConfig are frozen Pydantic models (immutable)
  that forbid extra fields and validate identifiers (slug-style).
- Downstream pattern: subclass the Base*Config types to add domain-specific
  fields and use model_copy(update={...}) for controlled modifications.

Integration examples

- Importing models:

```python
from fba_bench_core.domain.models import Product
from fba_bench_core.config import BaseAgentConfig

p = Product(product_id="sku-1", cost="1.00", price="2.00", stock=10)
cfg = BaseAgentConfig(agent_id="pricing-1", poll_interval_seconds=30)
```

- Agent decide example (pseudo):

```python
from fba_bench_core.agents.base import BaseAgent
from fba_bench_core.domain.events import BaseEvent
from fba_bench_core.domain.events import AdjustPriceCommand
from typing import List

class ExampleAgent(BaseAgent):
    async def decide(self, events: List[BaseEvent]) -> List[AdjustPriceCommand]:
        # Inspect events and produce commands (example only)
        cmds = []
        for ev in events:
            if getattr(ev, "event_type", "") == "price_changed_externally":
                # ev may contain competitor listing data used by repricing logic
                cmds.append(AdjustPriceCommand(product_id=getattr(ev, "product_id", None),
                                               proposed_price="9.99"))
        return cmds
```

Testing and where behavior is enforced

- Contract validation and invariants are exercised by:
  - [`tests/test_core_contracts.py`](tests/test_core_contracts.py:1)
  - [`tests/domain/test_models.py`](tests/domain/test_models.py:1)
  - [`tests/domain/test_events.py`](tests/domain/test_events.py:1)
  - [`tests/agents/test_config.py`](tests/agents/test_config.py:1)

- Tests demonstrate expected ValidationError behavior for invalid inputs and
  that models preserve Decimal semantics, immutability, and invariant checks.

Packaging and changelog notes

- CHANGELOG.md is maintained via Towncrier fragments in
  [`newsfragments/`](newsfragments/:1); see [`CHANGELOG.md`](CHANGELOG.md:1)
  for instructions.

Guidance for downstream implementers

- Rely on the domain models as authoritative; do not replicate
  validation logic in downstream code.
- Subclass config models to add fields; remain compatible by keeping
  agent_id/service_id slugs stable.
- Use model_copy(update={...}) instead of mutating frozen models.
- When translating Commands to side-effects, keep handlers small and idempotent.

References

- Implementation: see files listed above.
- Rescue log: [`docs/rescue_log.md`](docs/rescue_log.md:1)
- High-level architecture: [`docs/architecture.md`](docs/architecture.md:1)

Appendix: quick validation commands

```bash
# Run core contract tests
poetry run pytest tests/test_core_contracts.py -q

# Run linters and typing locally (same as CI)
poetry run black src tests
poetry run isort src tests
poetry run flake8 src tests
poetry run mypy