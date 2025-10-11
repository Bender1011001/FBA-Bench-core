# Technical White Papers for FBA Bench

## Scope and Purpose
Technical white papers for FBA Bench document advancements in AI agent benchmarking for financial business automation (FBA), particularly e-commerce scenarios. They qualify as in-depth analyses of methodologies, experimental results, metrics validation, or novel extensions to the benchmark (e.g., new tiers, trust evaluations). Papers should be factual, reproducible, and aligned with core goals: trustworthiness, scalability, and real-world applicability. Aim for 8-20 pages, suitable for arXiv, conferences, or internal reference.

## Naming Convention and Folder Layout
- Use a subfolder under `docs/paper/` named `YYYY-paper-shortname/` (e.g., `docs/paper/2025-fba-bench-core/`).
- Inside the folder:
  - `paper.pdf`: Final PDF version.
  - `abstract.md`: Concise summary (150-250 words).
  - `paper.md`: Full draft in Markdown (convertible to LaTeX/PDF).
  - `references.bib`: BibTeX entries.
  - Optional: `figures/`, `appendix.md` for supplementary data.

Ensure filenames use hyphens, lowercase, and no spaces. Keep PDFs under 10MB; use external links for large supplements.

## Suggested Paper Structure
Organize drafts in a subfolder under `docs/paper/` (e.g., `docs/paper/fba-bench-v1/`):
- `abstract.md`: Concise summary (150-250 words) of problem, contributions, and results.
- `methodology.md`: Detailed benchmark design, scenarios, metrics, and evaluation protocols.
- `results.md`: Experimental findings, tables/figures, and analysis (include golden master comparisons).
- `references.bib`: BibTeX entries for citations, including self-reference to FBA Bench.
- Optional: `figures/`, `appendix.md` for supplementary data.

Use Markdown for drafts; convert to LaTeX/PDF for submission. Ensure reproducibility: link to code, scenarios, and seeds.

## Adding a Paper to the Research Page
1. Place your paper assets in `docs/paper/YYYY-paper-shortname/`.
2. Create a BibTeX entry in `references.bib` for the paper.
3. Update `site/research.html` in the "Papers" section:
   - Add a new entry with title, authors (e.g., "John Doe, Jane Smith"), venue/year (e.g., "arXiv 2025").
   - Include a short abstract blurb.
   - Add links: PDF (`../docs/paper/YYYY-paper-shortname/paper.pdf`), DOI (placeholder like "10.48550/arXiv.ARXIV_ID"), BibTeX (anchor to reveal the block, e.g., `<details><summary>BibTeX</summary><pre>@inproceedings{...}</pre></details>`).
4. Commit and PR as per contribution flow.

## Guidance on Content
- **Authors/Affiliations**: List full names and affiliations (e.g., "John Doe (University of Example), Jane Smith (FBA Bench Project)"). Use ORCID if available.
- **Abstract**: 150-250 words summarizing motivation, methods, results, and implications. Keep it self-contained.
- **Links**:
  - PDF: Relative path to the file in `docs/paper/`.
  - DOI: Use arXiv or publisher DOI; placeholder format "10.48550/arXiv.ARXIV_ID".
  - BibTeX: Preformatted block in `<pre>`; use `@inproceedings`, `@misc`, or `@techreport` types.

## Contribution Flow
1. Fork the repository or create a feature branch (`git checkout -b feat/paper-new-benchmark`).
2. Add your draft folder under `docs/paper/` (e.g., `docs/paper/my-paper/`).
3. Update `site/research.html` with a new entry (title, status: "Draft" or "Published", link to PDF/DOI).
4. Commit with Conventional Commits (e.g., `docs(paper): add initial draft for trust metrics`).
5. Open a PR with checklist:
   - [ ] Paper follows structure and scope.
   - [ ] Includes reproducible elements (code links, seeds).
   - [ ] BibTeX updated in `references.bib` and synced with CITATION.cff.
   - [ ] No secrets or large binaries committed.
   - [ ] Local validation: Render MD files, check links.
6. After review/merge, update status in research.html to "Published" and add arXiv/DOI if applicable.

## Citation Guidance
- Reference the main FBA Bench project via the Research page: [site/research.html](../site/research.html).
- Use canonical metadata from [CITATION.cff](../CITATION.cff) for GitHub citations.
- For white papers, cite as `@misc` or `@techreport` in BibTeX; include arXiv/DOI when available.

## BibTeX Example
Matching the Research page citation for the core FBA Bench:

```
@misc{fba_bench_2025,
  title={FBA Bench: Benchmark for AI Agents in Financial Business Automation},
  author={Doe, John and Smith, Jane and Johnson, Alex},
  year={2025},
  publisher={arXiv},
  note={ARXIV_ID [cs.AI]}
}
```

### Sample Markdown Template Snippet for Paper Entry
In `paper.md`:
```markdown
# Title of the Paper

## Authors
John Doe^1^, Jane Smith^2^
^1^University of Example, ^2^FBA Bench Project

## Abstract
[Insert 150-250 word abstract here...]

## Introduction
...
```

Update placeholders post-publication. Validate with tools like JabRef. Sync across research.html, this README, and CITATION.cff to avoid inconsistencies.

## Updating the arXiv ID (Blocked until assignment)
The repository uses the literal token "ARXIV_ID" in citation snippets and links until the real arXiv ID is issued. When the ID is available (e.g., 2501.01234), update all references in a single change:
- POSIX (macOS/Linux):
  ```
  git grep -l 'ARXIV_ID' | xargs -I{} sed -i'' -e 's/ARXIV_ID/2501.01234/g' {}
  ```
- Windows PowerShell:
  ```
  git grep -l 'ARXIV_ID' | % { (Get-Content $_) -replace 'ARXIV_ID','2501.01234' | Set-Content $_ }
  ```
Scope to check after replacement:
- [`CITATION.cff`](../CITATION.cff)
- [`README.md`](../README.md)
- [`docs/paper/README.md`](README.md)
- [`site/research.html`](../site/research.html)
Validation:
- Verify there are no instances of "ARXIV_ID" left:
  ```
  git grep -n 'ARXIV_ID' || echo 'No ARXIV_ID placeholders remain'
  ```
Status:
- BLOCKED until a real arXiv identifier is provided.
