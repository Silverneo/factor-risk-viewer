# Experiments

Self-contained records of investigations that informed (or could inform) decisions in this codebase. Each experiment lives in its own dated folder so the question, design, work, raw outputs, and conclusions stay together.

## Index

| Date | Experiment | Headline finding |
|------|------------|------------------|
| 2026-04-26 | [Covariance payload formats](2026-04-26-covariance-payload-formats/) | List-of-dict JSON hard-fails in Chrome at 6.5M cells (V8 string-length ceiling). Arrow IPC wins comprehensively on every axis at large sizes. |
| 2026-04-27 | [Charts experiment](2026-04-27-charts-experiment/) | Seven hierarchical / comparative / temporal chart variants for the same risk data, all driven by existing endpoints. Hand-rolled SVG layouts (squarify + partition) avoid AG Charts Enterprise. |
| 2026-04-27 | [On-the-fly risk over weekly history](2026-04-27-on-the-fly-risk/) | Direct `xᵀ Σ_t x` is interactive ≤ N=2000 (sub-second) but bad UX at N=4000 (~19 s, memory-bound). Rank-30 low-rank reconstruction is ~5 orders of magnitude faster at every N — the right production default. |

## Folder convention

```
experiments/
└── YYYY-MM-DD-<topic-slug>/
    ├── README.md       # entry point: question, headline result, file index, how to re-run
    ├── spec.md         # what we set out to do — design doc from brainstorming, frozen as written
    ├── plan.md         # the implementation plan that was executed (frozen as written)
    ├── report.md       # numerical report + findings + recommendation
    └── results/        # raw outputs (JSON / CSV / screenshots / etc.) — one source of truth
```

Reusable code that came out of an experiment lives in its proper home (`backend/bench/`, `frontend/src/bench/`, etc.) — not under `experiments/`. The experiment README links to it. Keep `experiments/` for documentation + raw data; don't bury runnable infra here.

### What goes in each file

- **`README.md`** — entry point. The first thing someone reads. Should answer "what was the question, what did we find, where do I look for more, how do I re-run?" in <2 minutes. Cross-links the other files.
- **`spec.md`** — frozen design. What we said we'd test, what was in scope, what we explicitly ruled out, success criteria. Don't rewrite this after results come in — it's the historical record of what we set out to do.
- **`plan.md`** — frozen execution plan. Task-by-task what was actually built. Useful for future audits ("did we actually run the test we said we'd run?") and as a template for similar experiments.
- **`report.md`** — the numerical findings. Tables of results, surprises encountered, mid-experiment decisions, and the final recommendation. This is what gets cited from product/engineering decisions.
- **`results/`** — every raw output. JSON dumps from benchmark scripts, CSV exports, screenshots of UI states, anything that the report tables were derived from. Commit these — recomputing months later is rarely cheap and the results may not be reproducible (different hardware, different library versions).

### When to make a new experiment folder

- Comparing implementation choices ("library X vs Y", "format A vs B", "approach 1 vs 2") with measurements
- Validating an instinct or assumption with numbers before committing to a direction
- Investigating a surprising production behaviour that's worth a written record
- Any work where someone in 6 months might ask "why did we end up doing it this way?" and the answer is non-obvious from the code

### When NOT to make one

- Routine bug fixes — those belong in commit messages and PRs
- Refactors with no measurement component
- Anything that's just "I had to look this up" — that's a memory or a comment, not an experiment

## Conventions for naming

`YYYY-MM-DD-<topic-slug>` where the date is the kickoff date (or the date of the headline result if there's no clear kickoff), and the slug is a noun phrase that survives skim-reading the index 6 months later. "covariance-payload-formats" is good; "json-test" is bad.
