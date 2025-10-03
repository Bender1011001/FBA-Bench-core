# FBA-Bench Core Quickstart

This guide gets you running FBA-Bench Core locally in minutes, with optional Docker and Dev Container workflows.

## Prerequisites

- Python: 3.9+ (tested on 3.9–3.12)
- Git, and a C compiler toolchain if you plan to build native wheels on your platform
- Optional: Docker 24+ and Docker Compose v2 for containerized workflows
- Optional: VS Code + Dev Containers extension for a preconfigured IDE container

## 1) Clone the repository

```bash
git clone https://github.com/your-org/FBA-Bench-core.git
cd FBA-Bench-core
```

## 2) Create and activate a virtual environment

- macOS/Linux (bash/zsh):
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```
- Windows PowerShell:
  ```powershell
  py -3 -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- Windows CMD:
  ```bat
  py -3 -m venv .venv
  .\.venv\Scripts\activate.bat
  ```

## 3) Install the package

- Minimal install:
  ```bash
  pip install -e .
  ```
- Developer tooling (tests, linting, formatting, etc.):
  ```bash
  pip install -e ".[dev]"
  ```

## 4) Sanity checks

Run the repository validation scripts:

```bash
python scripts/validate_all.py
python scripts/validate_scenarios.py
```

If these pass, your environment is ready.

## 5) Run tests

- Full test run:
  ```bash
  pytest
  ```
- With coverage (requires pytest-cov, included in ".[dev]"):
  ```bash
  pytest --cov=src
  ```

## 6) Read a built-in scenario (YAML)

The repository ships curated scenarios under `src/scenarios/`. The example below loads the Tier 0 baseline file and prints a few fields using PyYAML.

```python
# quick_example_read_scenario.py
from pathlib import Path
import yaml

scenario_path = Path("src/scenarios/tier_0_baseline.yaml")
with scenario_path.open("r", encoding="utf-8") as f:
    data = yaml.safe_load(f)

print("Scenario keys:", sorted(list(data.keys())))
print("id:", data.get("id"))
print("tier:", data.get("tier"))
```

Run it from the repository root:

```bash
python quick_example_read_scenario.py
```

You should see the scenario keys and values (including `id` and `tier`) printed to the console.

- File reference: [src/scenarios/tier_0_baseline.yaml](src/scenarios/tier_0_baseline.yaml)

## 7) Optional: Docker workflow

A ready-to-use [Dockerfile](Dockerfile) is provided to build and test in a container.

- Build the image:
  ```bash
  docker build -t fba-bench-core:dev .
  ```
- Run tests using the default CMD:
  ```bash
  docker run --rm fba-bench-core:dev
  ```

Alternatively, Docker Compose provides two services via [docker-compose.yml](docker-compose.yml):

- Build:
  ```bash
  docker compose build
  ```
- Run tests with the mounted working tree:
  ```bash
  docker compose run --rm fba-bench
  ```
- Interactive dev shell with a persistent in-container venv:
  ```bash
  docker compose run --rm dev -c "python --version"
  ```
  Inside the dev shell you may create/activate a dedicated venv:
  ```bash
  python -m venv /opt/venv
  source /opt/venv/bin/activate
  pip install -e ".[dev]"
  ```

## 8) Optional: VS Code Dev Container

Open the folder in a Dev Container leveraging [.devcontainer/devcontainer.json](.devcontainer/devcontainer.json):

- In VS Code, run: “Dev Containers: Reopen in Container”
- On first create, it will:
  - Install `-e ".[dev]"` and set up pre-commit hooks
  - Enable Black on save, Ruff linting, and organize imports
- Verify inside the container:
  ```bash
  python --version
  pip freeze | grep -E "pytest|ruff|black"
  pre-commit --version
  ```

## Troubleshooting

- If `pip install -e ".[dev]"` fails due to missing dev extras, ensure your `pyproject.toml` defines `[project.optional-dependencies].dev`. This repository’s ship plan includes reconciling dev extras and tooling configuration.
- If validations fail, inspect output from:
  - [scripts/validate_all.py](scripts/validate_all.py)
  - [scripts/validate_scenarios.py](scripts/validate_scenarios.py)

## Next steps

- Explore the architecture overview: [docs/architecture.md](docs/architecture.md)
- Top-level docs overview: [docs/README.md](docs/README.md)
- Project README: [README.md](README.md)