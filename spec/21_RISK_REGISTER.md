# 21_RISK_REGISTER.md

## SDR note

All thresholds and IDs referenced here are defined in `spec/00_CANONICAL.md`.

## Risks and mitigations

1. Risk: RR depends strongly on generator artifacts rather than true non-identifiability.
   - Mitigation: compute RR across multiple generators and report per-generator RR.

2. Risk: SSS flags instability caused by implementation bugs.
   - Mitigation: intervention correctness unit tests and invariant checks.

3. Risk: CC is dominated by granularity choice.
   - Mitigation: report CC under multiple granularities and define default.

4. Risk: Real-model tasks are sensitive to tokenizer details.
   - Mitigation: deterministic token filtering and explicit logging of token ids.

5. Risk: Synthetic suite is too toy and fails to predict real-model behavior.
   - Mitigation: use synthetic suite only to validate diagnostics, not to claim real-model mechanisms.
