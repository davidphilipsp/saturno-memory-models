# Saturno — memory models on the Duolingo dataset

A small, pre-registered study of **memory models for spaced repetition**, run on the public
Duolingo learning-traces dataset (Settles & Meeder, ACL 2016). It replicates the paper's baselines
exactly, then climbs a ladder of models one rung at a time, and only lets a model "win" when a
learner-paired bootstrap says so under rules written down *before* each run.

It was built as the research arm of a personal language-tutor app (whose scheduler uses a fixed
interval ladder of 1 · 3 · 7 · 16 · 35 · 75 days), to decide whether a learned model is worth
replacing that ladder with. Pure JavaScript (Node.js + TensorFlow.js, CPU backend).

Built with Claude Code as a pair programmer.

The detailed lab notebook is in Portuguese: [`reports/duolingo.md`](reports/duolingo.md).
[`reports/meta-analise-2026-10-01.md`](reports/meta-analise-2026-10-01.md) is a read-only code
audit of the tutor app's data pipeline that motivated this study (personal figures redacted).

## Model ladder

Each rung is compared **only to the previous approved rung**, and always also reported against the constant.

| Rung | Model | Code |
|---|---|---|
| B0 | Constant (train mean), Leitner, Pimsleur — no training | `src/model/baselines.js`, `b0.js` |
| B1 | Logistic regression (one neuron) | `src/model/lr.js`, `b1.js` |
| B2 | Half-life regression (HLR): p̂ = 2^(−Δ/ĥ), ĥ = 2^(Θ·x) | `src/model/hlr.js`, `b2.js` |
| B3 | MLP with a 16-d lexeme embedding (sparse gradient) + hidden layer 32 + dropout, log2 ĥ head | `src/model/mlp.js`, `embedding.js`, `b3.js` |
| B3-control | Linear HLR trained exactly like the MLP (Adam, same epochs, batch, init) | `src/model/b3c.js`, `b3c2.js` |

## Metrics

As in the paper's evaluation (`evaluation.r`), predictions rounded to 4 decimals first:

- **MAE** of predicted recall p̂ vs observed p (lower is better) — the deciding metric.
- **AUC** of p̂ vs binarised recall (R's rule).
- **COR_h**: Spearman correlation between predicted and target half-life.

## Decision rule (pre-registered 2026-10-01, amendments dated)

| Rule | Definition |
|---|---|
| Win | MAE lower than the previous rung, with the **95% CI of the MAE difference excluding zero** |
| Bootstrap | Paired by learner (`user_id`): resample learners with replacement, each with all their rows. 1000 resamples, seed 20261001, R type-7 quantiles (`src/model/bootstrap.js`) |
| "Won without discriminating" | Wins on MAE but worse on AUC → stop and decide. **Amendment 2026-10-02:** "worse" = the whole 95% CI of ΔAUC below zero. Written *after* the B2 result and re-applied to B1 (verdict unchanged) |
| Minimum effect (2026-10-02, forward-only) | Future comparisons also require ΔMAE ≥ 0.005. Not applied retroactively |
| Seeds | Trained models must win on 3 of 3 seeds |

Splits follow the reference code: 90% train / 10% test by row order; for B3, 0–81% train, 81–90% validation (early stopping), 90%+ test.

## Results on the public dataset (summary)

- **B0** reproduces the paper's Table 2 to three decimals for Leitner (MAE 0.235, AUC 0.542, COR_h −0.098) and Pimsleur (0.445, 0.510, −0.132) on the full 1,285,423-row test set. This validates loader, split, formulas and metrics. The paper's "p̄ = 0.859" label is not reproducible from the file (its MAE matches the train mean, 0.896).
- **B1 LR**: beats Leitner on MAE but ranks worse → *won without discriminating*; not approved.
- **B2 HLR**: first model to beat the constant (ΔMAE vs Leitner −0.107, CI excludes zero; ΔAUC CI includes zero) → approved under the amended rule.
- **B3 MLP**: beats HLR on 3/3 seeds, but by ΔMAE ≈ −0.0008 (0.6%) — statistically clear, practically irrelevant. Lexeme ablation: the embedding barely moves MAE but lifts COR_h (0.212 vs 0.167).
- **Matched linear control**: the MLP does **not** beat a linear HLR trained the same way (1 of 3 seeds). The B3 gain comes from training (optimizer, epochs, batch, init), not architecture.
- Partial-dependence check: Δ and number of errors behave consistently across models; the effect of past correct answers does not (selection effects in Duolingo's own scheduler are recorded as hypotheses H-sel / H-exp).

All runs used a 1M-row working sample of the 12.9M-row file (`src/data/sample.js`).

## Tests

```bash
npm ci
npm test          # node --test test/*.test.mjs — 54 tests, synthetic fixtures only
npm run smoke     # prints the TF.js backend
```

`test/fixtures/duolingo-mini.csv` is a hand-written 10-row fixture with fake users, **not** dataset rows.
Several tests were **mutation-checked**: a deliberate bug (e.g. weighting the bootstrap by learner instead of by row,
or breaking `embeddingLookup`) was introduced, the test was confirmed to fail, and the change reverted. These checks are logged in the report.

## How to run the experiments

1. Download the dataset as described in [`data/README.md`](data/README.md).
2. `node src/data/inventory.js` — streaming pass over the full file.
3. `node src/data/sample.js <total_rows> 1000000` — working sample.
4. `node src/model/b0.js`, `b1.js`, `b2.js`, `b3.js`, `b3c.js`, `b3c2.js` — each writes `data/b*-results.json`.

B3 takes a few minutes per seed on a laptop CPU.

## Data, license and citation

The dataset is **not included** and must not be redistributed from this repo. It is licensed
**CC BY-NC 4.0** (non-commercial). If you use it, cite:

> Burr Settles and Brendan Meeder. 2016. *A Trainable Spaced Repetition Model for Language Learning.*
> Proceedings of ACL 2016, pages 1848–1858.

The reference implementation (`github.com/duolingo/halflife-regression`, MIT) is cited by line in the report.

## Security / privacy

Personal study state is not included; example data only. The author's own review events
(previously `data/events.jsonl`) and the extractor that read them were removed, and personal
figures in the audit report were redacted. No credentials, tokens or endpoints are used.
