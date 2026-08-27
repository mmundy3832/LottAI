# Phase 3 Program v4 -- Karpathy Loop Spec (Stage A)

Harness = Claude. Experimenter role = whoever edits configs/*.json (human or
MiniMax in a later phase). Metric = the numbers below, computed only by
train.py + lag_null.py. This file is the editable-surface contract for
Stage A (plans/PHASE3_PLAN.md section 4, lag-0 control, H1 test).

## Metric

Primary (preregistered, plans/PHASE3_PLAN.md section 5, G5):
- Test per-position log-loss, three numbers (d0, d1, d2), vs baseline
  ln(10) = 2.302585. Lower is better. This is the H1 test.
- Lag table (lag_null.run_lag_analysis on the model's top20/consensus sets,
  lags 0..5, test and live), reported as observed / null_mean / p_value per
  cell, never as a raw hit rate. Lag 0 top20 exact-hit is the H1-adjacent
  cross-check; lag 1-5 belong to Stage B (H2), not Stage A.

Secondary, exploratory, not gated on: per-position top-1 accuracy (vs 0.10),
top-20 combo hit rate (vs 0.020), val loss (used only for early stopping,
not for the H1 claim).

## Editable surface

phase3/configs/*.json. Fields:
- model: "m1" | "m2"
- W: context window (prior draws)
- seed: int
- batch, lr, max_epochs, patience, weight_decay: AdamW / training-loop knobs

Nothing else is editable without changing the harness. Model architecture
(layers, d_model, heads, ff_dim, GRU hidden/layers), the loss definition,
the eval procedure, and the null procedure are fixed code, not config.

## Fixed harness

- phase3/model.py: M1 (decoder-only transformer) and M2 (GRU) definitions.
  Never edited by a sweep; only by an explicit architecture decision.
- phase3/train.py: config in, runs/<run_id>/metrics.jsonl (per epoch) and
  runs/<run_id>/final.json (val/test/live metrics, records, lag summaries)
  out. Loss is always mean cross-entropy over the 3 digit heads; early
  stopping is always on val loss.
- phase3/lag_null.py: 2000-permutation null, seed 42, unmodified across
  runs. Every lag-table number is observed vs this same null.
- phase3/tokenizer.py: leakage rule (context strictly before target draw,
  equipment tokens on the target masked) and vocab. A sweep cannot change
  what the model is allowed to see.

## Gaming vectors and guards (plans/PHASE3_PLAN.md section 5)

G1, multi-lag inflation. Not applicable to Stage A (lag 0 only, no
multi-lag reward). Guard carries forward unchanged into Stage B: per-lag
null subtraction, never a pooled hit rate.

G2, near-miss inflation. top20/consensus(30) combo sets are reported only
against their own null (chance ~0.02 / ~0.03), never against a raw rate.
Guarded by lag_null.run_lag_analysis, which computes null_mean per cell
from the same top20/consensus sets it was given.

G3, lookahead. tokenizer.make_windows() builds context from tokens[t-W:t]
only; target_digits is constructed from tokens[t] and never fed back into
the model as input (model.py's forward() takes only context and
target_known). Guarded by test_tokenize.py's permutation-invariance and
no-lookahead tests, which must pass before a run is trusted.

G4, slot leakage. target_known carries the target draw's slot/dow (known
pre-draw, legitimate) but MASK_ID for machine/ballset (not confirmed
pre-draw). Guarded by tokenizer.make_windows() unconditionally masking
EQUIPMENT_COLS in target_known; no config field can turn this off.

G5, run selection. Stage A is 6 runs (2 models x 3 seeds), all at the
chosen W/batch/lr/max_epochs/patience/weight_decay below -- no sweep over
those values in this stage, so no multiple-comparisons correction is owed
beyond reporting all 6 runs (not a cherry-picked subset). If a later W or
architecture sweep is added, Bonferroni over the sweep size applies before
any H1 claim.

G6, live contamination. load_live() is called only for eval (final.json's
"live" block); live windows are never in train_t or val_t, and the
optimizer never sees a live batch. Stage A trains and early-stops on
train/val history only.

## Chosen hyperparameters

batch = 128 (chosen)
lr = 3e-4 (chosen)
max_epochs = 50 (chosen)
patience = 5 (chosen)
weight_decay = 0.01 (chosen)
W = 64 (chosen)
seeds = 0, 1, 2 (chosen)

Six runs: m1 x {seed 0,1,2}, m2 x {seed 0,1,2}, all W=64. One-epoch timing
probe (m1: 4.54s/epoch, peak GPU 378.6MB; m2: 3.05s/epoch, peak GPU
235.9MB) came in well under the ~2-hour budget for all 6 runs at up to 50
epochs each, so all 6 ran at 3 seeds rather than the 1-seed fallback.
