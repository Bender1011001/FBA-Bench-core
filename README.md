# FBA-Bench Core — Contract-first Core Library

Overview

FBA-Bench Core is a compact, contract-first library for building and
simulating Fulfillment-By-Amazon-style (FBA) business automation scenarios.
The rebuilt core emphasizes explicit, versioned contracts (Pydantic v2 models)
for domain models, events, commands, and typed configuration objects so
downstream projects can reliably validate and integrate simulation logic.

Core modules

- Domain models: [`src/fba_bench_core/domain/models.py`](src/fba_bench_core/domain/models.py:1)
  - Product, InventorySnapshot, CompetitorListing, Competitor, DemandProfile
- Events & Commands: [`src/fba_bench_core/domain/events.py`](src/fba_bench_core/domain/events.py:1)
  - Typed BaseEvent/BaseCommand hierarchy and concrete events like SaleOccurred,
    StockReplenished, PromotionLaunched; commands like AdjustPriceCommand,
    PlaceReplenishmentOrderCommand.
- Agent/service base classes: [`src/fba_bench_core/agents/base.py`](src/fba_bench_core/agents/base.py:1)
  and [`src/fba_bench_core/services/base.py`](src/fba_bench_core/services/base.py:1)
- Typed configs: [`src/fba_bench_core/config.py`](src/fba_bench_core/config.py:1)
  - BaseAgentConfig, BaseServiceConfig (immutable, extra="forbid")

Quick start (Poetry)

Clone and install with Poetry:

```bash
git clone <repo-url>
cd FBA-Bench-core
poetry install
```

Run the test suite:

```bash
poetry run pytest -q
```

Using Core from another project

1. Add FBA-Bench Core as a dependency (editable/local for development):

```bash
pip install -e /path/to/FBA-Bench-core
```

2. Import domain models and instantiate typed configs:

```python
from fba_bench_core.domain.models import Product
from fba_bench_core.config import BaseAgentConfig

p = Product(product_id="sku-1", cost="1.00", price="2.00", stock=10)
cfg = BaseAgentConfig(agent_id="pricing-agent", poll_interval_seconds=30)
```

Tests, linting and quality commands

The CI pipeline runs formatting, linting, type-checking and tests. Locally you
can run the same commands via Poetry:

```bash
poetry run black src tests
poetry run isort src tests
poetry run flake8 src tests
poetry run mypy
poetry run pytest
```

Packaging & versioning workflow

- CHANGELOG.md is generated from small towncrier fragments placed under
  [`newsfragments/`](newsfragments/:1) and built with `towncrier`.
- To add a changelog fragment: create a brief file `newsfragments/NN.description`
  then run `towncrier build --yes`.
- Version tooling: this repository includes [.bumpver.toml](.bumpver.toml:1)
  and uses dynamic versioning in the packaging pipeline. Follow project policy
  for release tagging and bumping; maintainers should prefer automated tools
  (bumpver) that integrate with CI.

Architecture and contracts

See the in-repo architecture notes for the rebuilt core:
- Core contracts: [`docs/architecture/core-contracts.md`](docs/architecture/core-contracts.md:1)
- Architecture overview: [`docs/architecture.md`](docs/architecture.md:1)

Documentation & rescue log

The rescue log contains per-phase notes about the rebuild:
- [`docs/rescue_log.md`](docs/rescue_log.md:1)
- Phase F created packaging and CI automation; Phase G documents updated
  documentation artifacts (this README and detailed core-contracts).

Contributing

Please follow Conventional Commits for PRs, include tests for contract changes,
and place towncrier fragments under `newsfragments/` for changelog entries.
See [`docs/README.md`](docs/README.md:1) for broader documentation contribution
guidelines.

License & contact

- License: MIT — see [`LICENSE`](LICENSE:1)
- Contact: project maintainers via repository issues or `press@fba-bench.ai`.
