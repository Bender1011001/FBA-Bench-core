# FBA-Bench Core Community Guide

## Welcome & Scope

Welcome to the FBA-Bench Core community! This repository is the public, source-available subset of the FBA-Bench project, focusing on reproducible benchmark scenarios (tiers 0–2), a metrics suite, baseline agents, and golden master artifacts for validation.

Contributions in scope include:
- Enhancing benchmark scenarios, metrics, and validation tools.
- Improving baseline agents and reproducibility features.
- Updating documentation for clarity and completeness.
- Bug fixes and performance optimizations in core components.

Out of scope:
- Enterprise-specific features (e.g., proprietary APIs, infrastructure).
- Non-benchmark related additions (e.g., full applications or unrelated tools).
- Changes that break reproducibility or introduce non-public dependencies.

We aim to keep Core lightweight and focused on open benchmarking standards.

## How to Contribute

### Filing Issues
- Use GitHub Issues for bug reports, feature requests, or questions.
- Bug reports: Include reproduction steps, expected vs. actual behavior, environment details (Python version, OS), and minimal code to replicate.
- Feature requests: Describe the problem it solves, proposed solution, and benefits to the community. Use the issue template if available.
- Search existing issues first to avoid duplicates.
- Label suggestions: `bug`, `enhancement`, `docs`, `good-first-issue`.

### Making Pull Requests
- Create a branch from `main`: `git checkout -b feat/your-feature` or `fix/your-bug`.
- Use Conventional Commits for messages: e.g., `feat(metrics): add new trust validator`, `fix(scenarios): resolve tier 0 edge case`.
- Keep PRs small and focused (one change per PR) for easier review.
- Include:
  - Clear description with motivation and changes.
  - Updated tests (run `pytest` locally).
  - Documentation updates if affected.
  - Verification steps (e.g., how to test the change).
- Python code: Follow ruff for linting and black for formatting (`make lint` and `make format-fix`).
- TypeScript/HTML (if any): Use simple conventional linting (e.g., ESLint with basic rules).
- Target: Ensure `make ci-local` passes before submitting.

## Project Setup (Quick Start)
For development, follow the installation in the [root README](../README.md#install-and-run):

```bash
# Clone and setup
git clone https://github.com/your-org/fba-bench-core.git
cd fba-bench-core
python -m venv .venv
# Activate venv (platform-specific)
pip install -e .  # Editable install
```

Run tests:
```bash
pytest -q  # Quick unit tests
make test-all  # Full suite with coverage
```

See [root README](../README.md) for smoke imports and golden master validation.

## Issue Triage Workflow
- **Labels**: Use `bug` for defects, `enhancement` for features, `docs` for documentation, `good-first-issue` for beginner-friendly tasks.
- **Reproductions**: Require minimal steps or code snippets; close un-reproducible issues after requesting more info.
- **Acceptance Criteria**: For enhancements, define success metrics (e.g., "New validator passes 95% of golden masters").
- Triage: Maintainers assign priorities (P0 critical, P1 high, etc.) and milestones within 48 hours.
- Close stale issues after 90 days of inactivity, unless labeled `help wanted`.

## Review Process
Reviewers check for:
- **Scope**: Aligns with Core's focus; no scope creep.
- **Tests**: New/changed code has coverage; runs pass.
- **Docs**: Updates if user-facing (e.g., scenarios, metrics).
- **Security**: No new vulnerabilities; scan with `make lint` (includes safety checks).
- **Reproducibility**: Golden masters unchanged or updated validly.

Aim for 1-2 reviewers; merge after approvals and CI pass. Address feedback iteratively.

## Governance (Lightweight)
- Maintainers (core team) decide on merges, roadmap, and disputes.
- Discussions happen openly in GitHub Issues and PRs.
- For major changes (e.g., new tier), use an RFC issue for community input (7-day comment period).
- Decisions follow lazy consensus: Merge if no objections; maintainers resolve ties.

## License
This project is licensed under the terms in the [LICENSE](../LICENSE) file in the root of the repository. See the root README for a summary of the source-available license.

## Community Conduct
We follow the [Code of Conduct](../CODE_OF_CONDUCT.md) to ensure a welcoming environment. All interactions must respect differing viewpoints and promote inclusive collaboration.

## Security Reporting
- Non-sensitive security issues: File via normal GitHub Issues, labeled `security`.
- Sensitive vulnerabilities: Email security@example.com with details (do not post publicly).
- We triage promptly and coordinate disclosures responsibly.

## Contact and Discussions
- Primary: GitHub Issues for technical discussions, PR comments for code review.
- General inquiries: Open an issue or comment on relevant discussions.
- No dedicated mailing list yet; use GitHub for transparency.
