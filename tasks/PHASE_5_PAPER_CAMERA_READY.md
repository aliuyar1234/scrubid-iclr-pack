# PHASE_5_PAPER_CAMERA_READY

## Objective

Produce a camera-ready paper draft with traceable claims and clean citations.

## Dependencies

- `CANONICAL.IDS.PHASE_IDS.PHASE_4_EXPERIMENTS_AND_ANALYSIS` completed.

## Deliverables

- Updated paper sections in `paper/` consistent with the diagnostics definitions.
- Claim-to-evidence mapping consistent with `spec/19_PAPER_WRITING_PLAN.md`.
- Verified bibliography in `bib/references.bib`.

## Definition of Done

- Every paper claim that relies on data is backed by a table/figure produced by the pipeline.
- All citations in the paper resolve to an entry in `bib/references.bib`.
- `bib/citations_verified.md` records primary-source verification for each citation.

## Acceptance tests

- Citation audit:
  - No grey literature is cited as a primary source for peer-reviewed claims.
  - BibTeX author lists contain no “and others”.

