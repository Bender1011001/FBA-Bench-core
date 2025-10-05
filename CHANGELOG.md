# CHANGELOG

All notable changes to this project will be documented in this file.
This file is managed by towncrier and is generated from small news fragments placed
in the newsfragments/ directory.

Unreleased
----------
- No notable changes yet.

How to add a changelog entry
- Create a file in [`newsfragments/`](newsfragments/:1) with a short name and category, e.g. `1.feature` or `2.bugfix`.
- Example fragment content:
  "Add tidy CI job (fix): enable HTML validation on site deploy."
- Run: `towncrier build --yes` to generate/update [`CHANGELOG.md`](CHANGELOG.md:1).

Notes
- The [tool.towncrier] configuration is present in [`pyproject.toml`](pyproject.toml:1).
- We will add a placeholder `.gitkeep` in [`newsfragments/`](newsfragments/:1) next to this changelog so the directory is tracked.