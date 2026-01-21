# GPT Pro Review (Paste-In Workspace)

This file is a staging area so you can:

1) Copy/paste the prompt below into ChatGPT Pro (web).  
2) Attach the smallest set of files Pro needs (see “Attachments”).  
3) Paste Pro’s full reply into the placeholder section.  
4) Next time, tell me: “Read `paper/gpt_pro_review.md` and continue.”

---

## Prompt to ChatGPT Pro (copy/paste)

You are a senior mechanistic-interpretability + reproducibility reviewer and a systems/method designer. You have no repo access. Everything you need is in the attached files (paper + tables/figure + run bundles with logs/run_records/certificates).

Goal: tell me exactly what’s still missing to make this publishable (and ideally “10/10 ScrubID”), and give implementable instructions (definitions + pseudocode + acceptance tests). No vague advice.

Please do all of the following:

1) Red-team the paper for overclaiming. List every claim that is not supported by the attached artifacts or is undermined by the implementation notes, and provide exact replacement wording (sentence-level edits).

2) Scope decision (choose one, justify):
   A) Implement “true” `I_PATHPATCH` and “true” `I_CAUSAL_SCRUB` so intervention families are genuinely distinct.
   B) Scope/rename the paper/spec to “activation patching only” and remove/adjust any cross-family claims.
   Whichever you choose, provide concrete acceptance criteria for “intervention-family integrity”.

3) If A: True PATH PATCHING design. Provide:
   - Minimal formal definition
   - Pseudocode for a TransformerLens-style implementation (edge-aware contribution blocking, not node-only activation replacement)
   - Data structures for edges/components
   - Determinism requirements
   - 2–3 acceptance tests demonstrating it differs from activation patching

4) If A: True CAUSAL SCRUBBING design. Provide:
   - Formal definition consistent with the field
   - For IOI / GT-YN / induction: define causal variables and how to resample them
   - Pseudocode
   - Acceptance tests distinguishing it from actpatch

5) Non-trivial real-model identifiability ambiguity. Right now real evidence includes a certificate driven by SSS instability, but RR/CC are 0 (near-optimal set often singleton). Propose 3 concrete strategies to obtain a real case with:
   - |S_near| ≥ 2
   - RR ≥ 0.2 and/or CC ≥ 0.2
   Include exact parameter ranges to try first: `rr_near_optimal_mdl_rel_frac` (or a slack curve), epsilon/tau changes, candidate budgets, generator ideas, and how to rule out degeneracy.

6) Baselines + ablations. Give a minimum strong experimental plan including:
   - ≥10 explicit ablations (name each)
   - multi-seed protocol + CI method (resampling unit, seed counts)
   - which baselines/generators are must-have vs optional and why

7) Compute/cost transparency. Provide:
   - Complexity analysis for diagnostics (esp. CC necessity tests)
   - A cost-reduction approximation with pseudocode
   - Validation protocol (synthetic + at least one real)
   - What plots/tables constitute an accuracy↔cost frontier

8) Claim↔evidence closure artifact. Design a `paper_results_manifest.json` schema mapping every non-trivial claim → exact table/figure cells → run_ids → hashes, plus mechanical validation rules (what fails the build).

Deliver your response as:
   - A prioritized checklist (P0/P1/P2) with acceptance criteria
   - Pseudocode blocks where requested
   - Concrete parameter values to try first

---

## Attachments for ChatGPT Pro

Prefer: a small “essentials” zip (if the UI accepts zips).

If zips are not accepted, attach the following files individually (these are also enumerated in `paper.md` under “Paper artifact bundle”):

- `paper.md`
- `outputs/paper_ready_gpt2_20260119_v3/reports/report_20260119T023458Z_0000/table_T1.csv`
- `outputs/paper_ready_gpt2_20260119_v3/reports/report_20260119T023458Z_0000/table_T2.csv`
- `outputs/paper_ready_gpt2_20260119_v3/reports/report_20260119T023458Z_0000/table_T3.csv`
- `outputs/paper_ready_gpt2_20260119_v3/reports/report_20260119T023458Z_0000/table_T4.csv`
- `outputs/paper_ready_gpt2_20260119_v3/reports/report_20260119T023458Z_0000/fig_synth.png` (optional if upload limits)
- `outputs/paper_ready_gpt2_20260119_v3/runs/run_20260119T021834Z_0018/run_record.json`
- `outputs/paper_ready_gpt2_20260119_v3/runs/run_20260119T021834Z_0018/logs.jsonl`
- `outputs/paper_ready_gpt2_20260119_v3/runs/run_20260119T021834Z_0018/certificate.json`
- `outputs/paper_ready_gpt2_20260119_v3/runs/run_20260119T023302Z_0000/run_record.json`
- `outputs/paper_ready_gpt2_20260119_v3/runs/run_20260119T023302Z_0000/logs.jsonl`
- `outputs/paper_ready_gpt2_20260119_v3/runs/run_20260119T023302Z_0000/certificate.json`

---

## GPT Pro Response (paste below)

# ScrubID publishability audit + implementable fix plan (drift-proof)

This review is **fail-closed**: I only rely on what is present in the provided artifacts (paper + attached tables/figures + the referenced run bundles/certificates as described in the paper). If something is not demonstrably supported by the artifacts, I mark it as **MISSING/UNSUPPORTED** and treat it as a problem.

---

## Publishability (as-is)

**Not publishable as a strong ICLR submission yet** (expect **Weak Reject** from a systems + reproducibility + mech‑interp reviewer), due to **P0 blockers**:

1) **Intervention-family integrity is not real in this artifact pack.**  
   The paper explicitly states:
   - `I_CAUSAL_SCRUB` is implemented as the same operator as `I_ACTPATCH`.
   - `I_PATHPATCH` is a lightweight approximation (“reachability‑pruned activation patching”) and *does not implement per‑edge contribution blocking*.
   
   Therefore any “cross-family” stability/claims are **not evidence** about distinct intervention families.

2) **“Non-identifiability certificate” semantics are internally inconsistent.**  
   The paper (intro) says certificates happen when “multiple incompatible circuits survive,” but the method defines certificates emitted if **any diagnostic FAILs** (including SSS instability). This conflates “non-identifiability” with “discovery instability.”

3) **Reproducibility closure is incomplete for a top-tier venue.**  
   The paper references runtime medians computed from a larger artifact bundle directory and reports 36 real runs; if those per-run logs and their mapping to tables are not attached, the runtime paragraph is **not reproducible**. A top-tier submission needs a mechanical claim↔evidence manifest and a validator.

That said: the **core idea is solid and paper-worthy**: auditing circuit explanations via RR/SSS/CC on a synthetic identifiability suite + real case studies is a meaningful contribution, *if scoped honestly and made mechanically reproducible*.

---

# 1) Red-team for overclaiming (with exact replacement wording)

Below are the claims that are **UNSUPPORTED / OVERCLAIMED** under the current implementation notes, plus exact sentence-level replacements.

## P0-O1 — Certificate meaning is overclaimed / inconsistent
- **Evidence in paper:**  
  - Intro: “ScrubID produces a non-identifiability certificate when multiple incompatible circuits survive the same tolerance.”  
  - Method: “ScrubID emits a non-identifiability certificate if any diagnostic verdict is FAIL.”

- **Why this matters:** validity + clarity. If SSS-only instability can trigger, then the certificate is not a non-identifiability certificate in the causal sense.

- **Replace with (exact edit):**
  - Replace the intro sentence with:  
    > “ScrubID emits an **audit certificate** when any diagnostic crosses its FAIL threshold; the certificate records the reason code(s) (RR/CC non-identifiability vs SSS discovery instability) and the supporting circuit sets.”
  - Replace “non-identifiability certificate” everywhere with **“audit certificate”**, and reserve “non-identifiability” wording only when RR or CC causes FAIL.

- **Acceptance criteria:** Every emitted certificate includes `reason_codes` and the paper text uses:
  - **“non-identifiability”** only when `reason_codes` contains `REASON_RR` or `REASON_CC`,
  - **“discovery instability”** when only `REASON_SSS` is present.

## P0-O2 — “Drift-proof protocol” is an overclaim without claim↔evidence validation
- **Evidence in paper:** “The result is a drift-proof protocol …” (nearest-neighbor delta section).

- **Why this matters:** reproducibility. “Drift-proof” implies mechanical enforcement, not just intent.

- **Replace with (exact edit):**
  > “The result is a **determinism-first auditing protocol** designed to reduce implementation drift by standardizing diagnostics, artifact schemas, and deterministic tie-break rules.”

- **Acceptance criteria:** Provide a `paper_results_manifest.json` + a validator that fails CI if any claim’s evidence artifact/hash is missing (see §8).

## P0-O3 — Cross-family stability claims are not supported by genuinely distinct interventions
- **Evidence in paper:** experiments sections and tables include `I_PATHPATCH`/`I_CAUSAL_SCRUB` as if they were distinct families, and narrative references stability across intervention IDs.

- **Why this matters:** novelty + validity. As implemented, these are not distinct; reporting them as such is misleading.

- **Replace with (exact edit):**
  > “We evaluate ScrubID under **component-level activation patching**. We include additional intervention IDs in the interface for future work; they are approximations in this implementation and are not used to support cross-family claims.”

- **Acceptance criteria:** Main results/tables/figures reference only `I_ACTPATCH` for claims; other IDs appear only in limitations/future work.

## P0-O4 — Runtime claims must be supported by attached per-run logs + manifest
- **Evidence in paper:** runtime medians derived from an artifact bundle directory and 36 runs.

- **Why this matters:** reproducibility + trust. A reviewer will ask: “show me the logs and the mapping.”

- **Fix options (choose one):**
  1) Remove runtime medians from the submission, OR  
  2) Attach all run logs and provide a manifest mapping the medians to run_ids and hashes.

- **Acceptance criteria:** If runtime paragraph remains, you must include:
  - `outputs/…/runs/*/logs.jsonl` for every run included in runtime statistics, and
  - a manifest entry pointing to them with sha256 and the exact statistic computation script + config hash.

---

# 2) Scope decision: A vs B

## Decision: **B) Scope/rename to “activation patching only” (recommended for this submission)**

**Justification:** The paper itself states `I_CAUSAL_SCRUB` and `I_PATHPATCH` are approximations / identical to activation patching in the used implementation. Retaining cross-family claims will trigger reviewer distrust. You can still have an excellent ICLR paper by focusing on:
- diagnostics RR/SSS/CC,
- deterministic synthetic suite with ground-truth redundancy,
- real-model discovery instability (SSS) and its implications,
- plus a real-case RR/CC > 0 obtained via slack curves and generator diversification.

### Acceptance criteria: “intervention-family integrity” under scope B
**PASS iff all conditions hold:**
1) **Paper:** No main table/figure or claim uses `I_PATHPATCH` or `I_CAUSAL_SCRUB` as experimental conditions. They may appear only in a “future work” paragraph.
2) **Configs/CLI:** Paper reproduction configs reject non-actpatch families with a hard error message: `NOT_IMPLEMENTED_FOR_PAPER_REPRO`.
3) **Manifest:** `paper_results_manifest.json` (see §8) references only `I_ACTPATCH` runs for all main claims.
4) **Unit test:** `test_paper_allowed_interventions()` asserts allowed set is exactly `{I_ACTPATCH}`.

---

# 3) (Only if choosing A later) True PATH PATCHING design — OPTIONAL v2 upgrade

(Provided only as future work guidance; for v1 submission I recommend scope B.)

## Minimal edge-aware definition
Let components be nodes \(V\) and directed influence be edges \(E\). Path patching intervenes **per edge contribution**, not just node activation replacement.

For each receiver component \(v\), write its input as sum of parent contributions:  
\(\mathrm{inp}_v = \sum_{u\in Pa(v)} \mathrm{contrib}_{u\to v}\).

Define patched input:
\(\mathrm{inp}_v^{patched}=\sum_{u} g_{u\to v}\,\mathrm{contrib}_{u\to v}(x)+(1-g_{u\to v})\,\mathrm{contrib}_{u\to v}(x^{ref})\).

This differs from activation patching because it preserves only selected **edges** even when node set is constant.

## TransformerLens-style pseudocode sketch (edge-aware)
```python
# Requires a deterministic decomposition of resid_pre into sender contributions.
def path_patch_forward(model, x, x_ref, allowed_edges, caches_clean, caches_ref):
    def resid_pre_hook(resid_pre, hook, layer):
        patched = 0.0 * resid_pre
        for sender in ALL_SENDERS[layer]:
            for dst_site in SITES[layer]:
                use_clean = (sender, (layer, dst_site)) in allowed_edges
                patched += contrib(sender, layer, dst_site, caches_clean if use_clean else caches_ref)
        return patched

    hooks = [(f"blocks.{l}.hook_resid_pre", partial(resid_pre_hook, layer=l))
             for l in range(model.cfg.n_layers)]
    return model.run_with_hooks(x_ref, hooks=hooks)
```

## Acceptance tests (must differ from activation patching)
1) **Edge toggle test:** same node set, different edge set → different outputs (activation patching would not change).  
2) **Single-edge deletion:** deleting one edge changes output while node membership unchanged.  
3) **Synthetic two-path graph:** only edge-aware patching can separate the two redundant paths while keeping shared nodes.

---

# 4) (Only if choosing A later) True CAUSAL SCRUBBING design — OPTIONAL v2 upgrade

True causal scrubbing requires explicit causal variables \(Z\) and resampling those variables while holding others fixed.

## Formal definition
Choose causal variables \(Z\), generator \(g: Z \to X\), and a held set \(Z_C\). Causal scrubbing replaces non-circuit causes by resampling:
\(z^{ref} \sim p(z_{\neg C} \mid z_C)\), then set \(x^{ref}=g(z^{ref})\).

Then patch activations comparing \(x\) vs \(x^{ref}\) **while ensuring** the scrubbing corresponds to causal resampling, not arbitrary corruption.

## Acceptance tests distinguishing from activation patching
- **Hold-variable sanity:** held variables remain identical in the resampled prompts.  
- **Conditional sampling correctness:** empirical checks of \(p(z_{\neg C} \mid z_C)\) via deterministic constrained sampling + hashes.  
- **Failure when holding wrong variables:** if you hold the wrong causal variable, faithfulness should break in predictable ways.

---

# 5) Obtaining a real case with nontrivial identifiability ambiguity (RR/CC > 0)

Current real evidence is dominated by SSS instability because:
- `rr_near_optimal_mdl_rel_frac = 0.00` makes `S_near` typically a singleton → RR=0 and CC=0.

You need **at least one real suite** with:
- \(|S_{near}|\ge 2\),
- RR ≥ 0.2 and/or CC ≥ 0.2,
- and not dominated by trivial degeneracy (empty/full circuits).

## Strategy 1 — RR/CC slack curve (MDL slack sweep)
**Try first:**
- `rr_near_optimal_mdl_rel_frac ∈ {0.00, 0.05, 0.10, 0.20, 0.40}`
- `rr_num_circuits_set = 100`
- candidate budget: 1,000 → 5,000 candidates

**Anti-degeneracy rule:** measure how often the near set is dominated by `{full}` or `{empty}`. If >50%, mark slack region degenerate and report it as such.

**Acceptance criteria:** at some slack value, \(|S_{near}|\ge 2\) and RR or CC ≥ 0.2 on a real suite.

## Strategy 2 — Epsilon sweep + empty-circuit veto
**Try first:**
- hold `rr_near_optimal_mdl_rel_frac = 0.10`
- sweep `epsilon_rel_frac ∈ {0.05, 0.10, 0.15, 0.20, 0.30}`

**Empty-circuit veto:** `empty` must not become faithful; if it does, report “metric too weak / ref too strong” and change reference distribution.

**Acceptance criteria:** \(|F|\) increases, \(|S_{near}|\ge 2\), RR/CC > 0, while empty circuit remains unfaithful.

## Strategy 3 — Candidate diversification (non-nested circuits)
Add a deterministic stratified random generator:
- sizes `k ∈ {16, 32, 48, 64, 96}`
- 500 candidates per k (2,500 total), union with attribution-ranked candidates
- enforce layer coverage constraints to prevent all candidates being nested variants of the same superset

**Acceptance criteria:** `S_near` contains ≥2 circuits with Jaccard distance ≥0.2 *and* both are faithful.

---

# 6) Baselines + ablations (minimum strong plan)

## Must-have baselines
1) **Full circuit** (upper bound; also sanity for Δ=0)  
2) **Random-k size-matched** baseline (distributional control)  
3) **One automated discovery baseline** (adapter acceptable): ACDC-like or AtP*-style generator to show ScrubID is not “generator-specific.”

## ≥10 explicit ablations (name each)
A1. `epsilon_rel_frac` sweep  
A2. `rr_near_optimal_mdl_rel_frac` sweep (RR/CC curve)  
A3. `rr_num_circuits_set`: 20 vs 100  
A4. candidate budget: 100 vs 1k vs 5k  
A5. include-full-circuit: on/off  
A6. hookpoint variant: `hook_z` vs `hook_result` (if implemented)  
A7. granularity: head_mlp vs head-only vs mlp-only  
A8. reference assignment: index-aligned vs derangement shuffle (real suites)  
A9. SSS replicate count: R=3 vs 5 vs 10  
A10. CC tau scaling: `{0.02, 0.05, 0.10} * S0`  
A11. induction OOD corruption: two deterministic corruptions (report both)  
A12. D_eval size sensitivity (e.g., 128 vs 512 examples)

## Multi-seed + CI protocol
- resampling unit: run seed  
- N=5 seeds per config  
- 95% CI via bootstrap over seeds (10k resamples)

**Acceptance criteria:** every main metric reported with mean+CI+N; per-seed appendix CSV is attached.

---

# 7) Compute/cost transparency (diagnostics complexity + frontier)

## Complexity drivers
- Candidate evaluation: ~O(K · N · forward)  
- RR: O(|S_near|^2 · |V|) with bitsets  
- SSS: R repeats of discovery + O(R^2 · |V|)  
- CC exact necessity: O(N · Σ_{C∈S_near} |C| · forward) (dominant)

## Cost-reduction approximation for CC (two-stage early-stop)
```python
def approx_necessity(model, C, v, D_eval, tau, ref_cache, seed):
    rng = rng_from(seed, hash(C), v)
    order = deterministic_shuffle(range(len(D_eval)), rng)
    diffs = []
    for n, idx in enumerate(order, start=1):
        x = D_eval[idx]
        d_full = metric_diff(x, scrub(model, C, x, ref_cache))
        d_drop = metric_diff(x, scrub(model, C - {v}, x, ref_cache))
        diffs.append(d_drop - d_full)

        if n in {16, 32, 64}:
            mean = average(diffs)
            ci_lo, ci_hi = bootstrap_ci(diffs)
            if ci_hi < tau - 0.02*S0: return False
            if ci_lo > tau + 0.02*S0: return True
    return average(diffs) >= tau
```

## Validation protocol
- Synthetic: compare exact vs approx CC error vs cost  
- One real suite: stability of CC estimates vs subsample size

## Frontier artifacts
- CC error vs forward passes  
- RR/SSS/CC vs candidate budget K  
- per-suite cost table (wall time + GPU)  
- “accuracy ↔ cost” plots for CC approximation

---

# 8) Claim↔evidence closure artifact: paper_results_manifest.json

To be “drift-proof,” the submission must include a mechanical mapping of every non-trivial claim to exact evidence artifacts.

## Schema (strict)
```json
{
  "schema_version": 1,
  "paper_id": "scrubid",
  "paper_version": "2026-01-19",
  "claims": [
    {
      "claim_id": "C01",
      "claim_text": "RR and CC increase sharply when planted redundancy_factor >= 2.",
      "paper_location": {"file":"paper.md","section":"Abstract"},
      "evidence": [
        {
          "type": "table_cell",
          "artifact_path": "tables/table_T1.csv",
          "selector": {
            "row_key": {"setting_id":"setting_XOR_2_1"},
            "columns": ["RR_mean","CC_mean","non_identifiability_rate"]
          },
          "hashes": {"file_sha256": "…" },
          "run_ids": ["run_…"],
          "run_record_hashes": ["…"]
        }
      ]
    }
  ]
}
```

## Validation rules (build must FAIL if violated)
- every claim has ≥1 evidence item  
- every artifact exists and sha256 matches  
- every table selector resolves uniquely  
- every referenced run has required files (run_record, logs, candidates, diagnostics, best circuit, certificate if emitted)  
- certificate `reason_codes` are consistent with narrative labeling (“non-identifiability” vs “instability”)  

**Acceptance criteria:** `validate_manifest` exits 0 and prints `ALL CLAIMS VERIFIED`.

---

# Prioritized checklist (P0/P1/P2) with acceptance criteria

## P0 (must fix before “publishable”)
**P0-1 Scope lock to activation patching.**  
Done when: no main result uses non-actpatch families.

**P0-2 Certificate semantics fix:** rename to “audit certificate” + add `reason_codes`.  
Done when: SSS-only failures are not labeled “non-identifiability.”

**P0-3 Artifact completeness:** ship per-run candidates + per-run diagnostics + per-run best circuit for every run backing paper tables/figures.  
Done when: tables/figures regenerate from run dirs.

**P0-4 One real case with RR/CC nontrivial:** via slack curve + candidate diversification.  
Done when: \(|S_{near}|\ge 2\) and RR/CC ≥ 0.2 on a real suite.

**P0-5 Deterministic candidate generator spec:** deterministic seeds + deterministic candidate construction.  
Done when: candidate sets reproduce bit-exactly across machines.

**P0-6 Claim↔evidence manifest + validator.**  
Done when: claims are mechanically verifiable and CI fails on mismatch.

## P1 (major 10/10 upgrades)
- add automated discovery baseline (ACDC/AtP* adapter)
- compute/cost frontier for CC (exact vs approx)
- multi-seed CIs everywhere

## P2 (polish)
- add CI bands on plots
- explicit data license/provenance statements
- limitations: separate non-identifiability vs instability cleanly

---

# Concrete parameters to try first (to force real RR/CC > 0)

Start with **Induction OOD** (most unstable), `G_ATTR_PATCH`, `I_ACTPATCH`:
- `rr_near_optimal_mdl_rel_frac`: 0.10 then 0.20
- `rr_num_circuits_set`: 100
- candidate budget: K=5,000 (attribution + stratified random union)
- `epsilon_rel_frac`: 0.15 (with empty-circuit veto)
- CC necessity: subsample 64 with early stop; `tau_frac ∈ {0.02, 0.05, 0.10}`

---

## Bottom line

You’re close to a publishable and genuinely useful paper, but to reach “10/10” you must:
- remove cross-family claims (or implement true families),
- disambiguate certificate semantics (non-identifiability vs instability),
- ship complete evidence artifacts + manifest + validator,
- and produce at least one real case where RR/CC are nontrivial (not just SSS).

---

## Action Items Extracted (fill in after pasting)

- [ ] P0:
- [ ] P1:
- [ ] P2:
