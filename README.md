# FBA-Bench Core — Trust Dossier

## Requirements

- Python: 3.9+ (tested on 3.9–3.12)
- Supported OS: Windows 10/11, macOS 12+ (Intel/Apple Silicon), Linux (x86_64/arm64) with Python 3.9+ available
- Tools:
  - Git
  - Optional: Docker 24+ and Docker Compose v2
  - Optional: VS Code + Dev Containers extension

### Installation

Create and activate a virtual environment:

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

Install the package:

- Minimal install:
  ```bash
  pip install -e .
  ```
- Developer tooling (tests, linting, formatting, etc.):
  ```bash
  pip install -e ".[dev]"
  ```

Verify the installation:
```bash
python -c "import fba_bench_core; print(fba_bench_core.__version__)"
```

For more detailed setup, see the Quickstart: [docs/quickstart.md](docs/quickstart.md)

FBA-Bench Core is the public, source-available subset of the FBA-Bench project. It provides:
- Reproducible benchmark scenarios (tiers 0–2)
- A metrics suite and baseline agents
- Golden master artifacts for validation

This repository is a staging output from the Phase 1 bifurcation process (local-only). Remote publishing and CI wiring will be finalized later.

## Methodology and Architecture
For an in-depth technical overview, see:
- docs: ./docs/architecture.md
- docs overview: ./docs/README.md

Key ideas:
- Scenario tiers reflect increasing complexity and operational constraints.
- Golden masters serve as a stable point of comparison to detect regressions.
- Baseline agents provide reference behaviors across tasks.

## Scenarios (Tiers 0–2)
Location (expected):
- src/scenarios/tier_0_baseline.yaml
- src/scenarios/tier_1_moderate.yaml
- src/scenarios/tier_2_advanced.yaml

Each YAML defines inputs, constraints, and evaluation hooks. Tiered progression:
- Tier 0 (Baseline): minimal constraints for sanity checks
- Tier 1 (Moderate): realistic constraints and objective signals
- Tier 2 (Advanced): complex interactions and tighter evaluation thresholds

## Metrics Suite and Baselines
- Metrics package: src/metrics/
- Agent interfaces: src/agents/
- Baseline agents: baseline_bots/

Metrics include trust, operations, finance, marketing, and stress dimensions as applicable to the core scenarios.

## Golden Master Validation
Golden masters reside under:
- golden_masters/

Validation checks (lightweight example):
```bash
# JSON parse validation similar to CI
python - <<'PY'
import os, json, glob, sys
gm_dir = os.path.join('golden_masters')
if os.path.isdir(gm_dir):
    fail = False
    for p in glob.glob(os.path.join(gm_dir, '**', '*.json'), recursive=True):
        try:
            json.load(open(p, 'r', encoding='utf-8'))
            print(f"[OK] {p}")
        except Exception as e:
            print(f"[FAIL] {p}: {e}")
            fail = True
    sys.exit(1 if fail else 0)
else:
    print("golden_masters/ not found; skipping")
    sys.exit(0)
PY
```

For methodology and expectations, see docs/quality notes (to be linked post-publication).

## Install and Run
Create a virtual environment and install Core:
```bash
python -m venv .venv
# Windows PowerShell
.\\\.venv\\Scripts\\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
pip install -e .
```

Run unit tests (subset curated for core):
```bash
pytest -q
```

Optional: smoke imports:
```bash
python -c "import metrics, agents, baseline_bots, fba_bench_core; print('ok')"
```

## Reproducibility
- Scenarios and metrics are versioned with the repository.
- Golden masters provide a snapshot for regression detection.
- See docs/architecture.md for pipeline and data assumptions.

## Press Kit
The Press Kit provides media resources for FBA-Bench Core, including logos, screenshots, and brand guidelines.

- Page: [site/press.html](site/press.html)
- Assets: [site/assets/press/](site/assets/press/)
  - Logos: logo-light.svg, logo-dark.svg, wordmark.svg (SVG formats for scalability)
  - Screenshots: screenshot-1.svg (Dashboard), screenshot-2.svg (Leaderboard) – placeholders
  - Guidelines: Basic usage rules on the press page (clear space, no alterations; subject to update)


Usage: Follow the basic brand guidelines on the press page. Assets are placeholders and should be replaced with final brand assets before launch. For inquiries: press@fba-bench.ai.
## Contributing
Contributions are welcome for the public core:
- Propose changes to scenarios, metrics, or documentation.
- Ensure tests pass and golden-master validation remains stable.
- Follow Conventional Commits if possible.


### Leaderboard

The Leaderboard page displays public, opt-in benchmark results from the FBA-Bench community, rendered from a local JSON file. It is fully static and loads data only from `./data/leaderboard.json` with no external requests.

- **Page Location**: [site/leaderboard.html](site/leaderboard.html)
- **Data Source**: [site/data/leaderboard.json](site/data/leaderboard.json)
- **JSON Schema**: Array of objects with required fields: `team` (string), `model` (string), `score` (number, 0-100), `date` (string, YYYY-MM-DD). Optional: `notes` (string). To add new entries, append objects to the array and sort manually or let the page handle sorting by score descending, then date descending. Example:
  ```json
  [
    {
      "team": "Example AI Lab",
      "model": "Example-1",
      "score": 87.2,
      "date": "2025-09-01",
      "notes": "Baseline performance on tier 1 scenarios"
    },
    {
      "team": "Sample Labs",
      "model": "Sample-Large",
      "score": 92.5,
      "date": "2025-09-15",
      "notes": "Optimized for multi-agent coordination"
    }
  ]
  ```

The page renders a semantic table with columns: Rank (auto-computed), Team/Model (combined), Score, Date, Notes (N/A if missing). It handles loading ("Loading leaderboard…"), error ("Failed to load leaderboard."), and empty ("No leaderboard entries yet.") states without console errors.

- **Local Viewing**: Open `site/leaderboard.html` directly (may require a static server for fetch in some browsers):
  ```bash
  cd site
  python -m http.server 8000
  ```
  Visit `http://localhost:8000/leaderboard.html`.

### CTAs Configuration
- Homepage CTAs are in `site/index.html`. The prominent CTA section features a headline ("Ready to Benchmark Your AI Agents?"), subtext ("Join the leading standard for e-commerce AI evaluation with reproducible scenarios and trusted metrics."), primary "Get Started" button linking to `./leaderboard.html`, and secondary "Contact" link to `mailto:press@fba-bench.ai`.
- Press page CTAs are in `site/press.html`:
  - "See benchmark results" currently anchors to `#results` (update to `leaderboard.html` when available).
  - "Request enterprise demo" links to `mailto:press@fba-bench.ai` (placeholder; replace with real endpoint).
  - "Read the paper" points to `research.html` (label as "coming soon" until Step 2 adds it).
- Edit the `<a>` tags directly in the HTML files to update destinations.

### Asset Replacement
- Replace placeholder assets in `site/assets/press/`:
  - Logos: `logo-light.svg`, `logo-dark.svg`, `logo.png` (SVGs scalable; PNG fallback 512x512 px, transparent).
  - OG Image: `og-image.png` (1200x630 px for social previews).
  - Brand docs: Update `brand-colors.md`, `brand-typography.md`, `brand-guidelines.md` with official content.
- After replacement, re-generate `press-kit.zip` using the PowerShell command documented in `site/assets/press/README.md`.
- Recommendations: Optimize images (<100KB), ensure accessibility (alt text in HTML), test on dark/light themes.

### Validation
- Validate HTML compliance using the W3C Nu validator (https://validator.w3.org/nu/): Paste contents of `site/index.html`, `site/leaderboard.html`, and `site/press.html` or provide file URLs if hosted.
- No local tooling added; focus on semantic HTML, relative paths, and accessibility (e.g., alt attributes on images).

## Community

The FBA-Bench Core community welcomes contributions to improve the benchmark's scenarios, metrics, baseline agents, and documentation. To get started, file issues on GitHub for bugs or feature requests using the provided templates, ensuring clear reproductions and acceptance criteria. For code contributions, create small, focused pull requests following Conventional Commits, including necessary tests and documentation updates. Adhere to Python style guidelines using ruff and black for linting and formatting; TypeScript/HTML follows simple conventional linting if applicable. Review the full guidelines below for triage, review processes, and governance.

- [Community Guide](docs/community/README.md)
- [Code of Conduct](docs/CODE_OF_CONDUCT.md)


## Basic Analytics (Opt-in)

The analytics configuration lives in [repos/fba-bench-core/site/config.js](repos/fba-bench-core/site/config.js).

### How to Enable

To enable analytics, edit [repos/fba-bench-core/site/config.js](repos/fba-bench-core/site/config.js) and set `enabled: true`, define a `trackingId` (e.g., a site identifier), optionally set an `endpoint` for a custom provider, and set `respectDNT: false` only if your privacy policy allows ignoring Do Not Track signals.

Example config snippet with analytics enabled:

```javascript
window.ANALYTICS_CONFIG = {
  enabled: true,
  provider: "placeholder",
  trackingId: "your-site-id",
  respectDNT: false,  // Set to true to respect DNT (blocks if browser signals DNT)
  sampleRate: 1.0,
  endpoint: "https://your-analytics-endpoint.com"  // Optional; required for non-placeholder providers
};
```

**Caveat**: The default implementation uses a console-based logging adapter (no network requests). To use a real analytics provider, the maintainer must manually insert the provider's script in the inline script section of [repos/fba-bench-core/site/index.html](repos/fba-bench-core/site/index.html), replacing the console adapter logic.

**Warning**: Do not commit real tracking IDs, endpoints, or other sensitive configuration to the repository if it is public. Use environment-specific configs or build-time injection for production deployments.

### Privacy Notes

- Analytics is disabled by default (`enabled: false`).
- Respects Do Not Track (DNT) by default (`respectDNT: true`); no tracking occurs if the browser signals DNT.
- No network requests are made by default; the adapter only logs to the browser console when enabled.
- All tracking is opt-in via configuration; no user data is sent externally without explicit setup.

## License
This repository is distributed under the MIT License. See LICENSE.

## Research
Explore FBA Bench research outputs, including white papers in `docs/paper/`. Propose contributions via GitHub issues or PRs following guidelines in `docs/paper/README.md`.

- Research overview and papers: [site/research.html](site/research.html)
- White paper guidelines: [docs/paper/README.md](docs/paper/README.md)
- Citation metadata: [CITATION.cff](CITATION.cff)

Short BibTeX example for citing FBA Bench:

Note: "ARXIV_ID" is a placeholder. See docs/paper/README.md for the update procedure once the real arXiv ID is available.

```
@misc{fba_bench_2025,
  title={FBA Bench: Benchmark for AI Agents in Financial Business Automation},
  author={Doe, John and Smith, Jane and Johnson, Alex},
  year={2025},
  publisher={arXiv},
  note={ARXIV_ID [cs.AI]},
  url={}
}
```
