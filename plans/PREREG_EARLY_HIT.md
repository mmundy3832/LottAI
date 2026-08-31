# Preregistered Test: Early-Hit Hypothesis on Held-Out Draws

Registered: 2026-08-31, before any prediction was generated for the held-out set.
This file is committed before the analysis runs. Any deviation from this spec
will be reported as a deviation.

## Hypothesis

H-early: predictions from the Phase 2 production predictor tend to match the
winning combo of a draw one or two draws LATER than the draw they were made
for, at a rate above chance. Origin: Mark's observation of the Telegram
prediction stream, May-June 2026, and the near-hit table of 2026-08-31
(autoresearch/results/near_hits_table.md), which motivated but cannot test
this hypothesis because it was derived from that data.

## Data

Held-out set: all draws in data/pick3all_live.csv with date strictly after
2026-06-05 (the last ledger-scored draw) through the file's current end
(2026-08-24 Night), all four slots, expected n approximately 280. These draws
were never used for model training or selection. Caveat on record: they
appeared inside evaluation windows of the Phase 3 Stage A/B and Design B
backtest analyses (all of which returned null); they are evaluation-reused,
not selection-reused.

## Predictions

Config A (exp#1566) and Config B (exp#1572) blended exactly as
autoresearch/predict_now.py does, trained once on all draws through
2026-06-05 (historical file plus live rows up to and including that date),
with the momentum-overlap feature in its production placeholder form for
target rows (the causal correction of backtest_lag.py). One top20_combos and
one consensus_combos set per held-out draw. Deviation from production: no
per-draw retraining. No parameter, seed, or config may be changed after this
registration.

## Primary test (one, decided in advance)

Statistic S: the rate, over all (prediction for draw t, actual draw t+k)
pairs with k in {+1, +2}, of best-in-top20 positional match >= 2 of 3.
Null: 2000 permutations of the actual-combo sequence, seed 42, same statistic.
Test: one-sided, S above null. Significance threshold: empirical p < 0.05.
This is the only confirmatory test. Its result decides H-early.

## Secondary (exploratory, reported but not confirmatory)

1. Exact top20 hit rate at k=+1 and k=+2, separately, same null.
2. The full lag 0..5 table (both metrics, top20 and consensus), same null.
3. Same statistics at k=-1 and k=-2 (late-hit control; H-early predicts
   asymmetry: excess at positive k, none at negative k).

## Decision rule

If the primary p < 0.05: H-early survives one preregistered test; the next
step is replication on genuinely future draws (cron re-enabled) before any
claim is made.
If the primary p >= 0.05: H-early is rejected at the effect sizes this n can
see, and the early-hit line of inquiry is closed unless new live data says
otherwise.

## Chosen values

n_perm=2000, seed=42, top20 set size 20, >=2-of-3 positional match metric,
k window {+1,+2}: all chosen to match the existing lag analysis harness
(autoresearch/lag_analysis.py) and the framing of the original observation.
No value was tuned on the held-out data.
