# LottAI Whitepaper Notes

Running log of findings, empirical results, and insights to cite in the whitepaper.
Add new entries at the bottom with a date header.

---

## High-D Random Pool Insight (core thesis)

In a 55-75D config space, a fixed random-frac of 0.10 is inadequate. The volume near corners grows exponentially with dimensions. Running 574 pure-random experiments (exp#843–1417) before resuming the GA broke a local optimum the GA was stuck at for 800+ experiments.

The random batch did NOT directly beat the ceiling. Best random was $7.68 vs. GA ceiling of $8.04. But it planted enough novel seeds that the GA immediately broke through to $8.18 on resumption.

Random exploration in high-D spaces serves as seed material, not direct optimizer. The GA needs diversity in the gene pool to escape local attractors — it can't generate that diversity itself.

## Convergence Results (as of exp#1507)

- GA ceiling before random batch: $8.04 (exp#346, held for 800+ experiments)
- Best random batch result: $7.68 (exp#880)
- New GA ceiling after random batch: $8.18 (exp#1471–1507)
- Winning architecture (locked in): et + xgb + lgb ensemble, custom_interact pipeline, 5 feature sets (basic + gaps + positional + temporal + momentum; recency dropped)
- Weight evolution: xgb getting heavier emphasis post-random (et:0.38, xgb:0.51, lgb:0.18)

## MLP Was a Dead End

74 MLP experiments in random batch. Zero in top 50 overall. MLPClassifier does not compete with tree/boosting ensembles on this tabular time-series classification problem.

## Key Design Finding

Pipeline is secondary. custom_interact alone drives top results. Heavy pipeline experiments (3-4 stages) in random batch all underperformed 1-stage custom_interact. Feature set architecture matters more than transformation pipeline.

## Connection to Prior Work

The random pool insight connects to Gagliardi and Tribble's MEGA experiment — corner seeding was foundational there too. This whitepaper will revisit that history and formalize the dimensional scaling argument: random injection rate should scale with the dimensionality of the search space, not be a fixed small fraction.

Cite exp#843–1507 arc as the empirical evidence. The null result on random-beats-ceiling but positive result on random-enables-GA-escape is the nuanced finding.

---

## Live Evaluation Results (234 unseen draws, 2026-02-14 to 2026-04-23)

### New Ceiling Cluster — Confirmed Live Leader

exp#1471, 1486, 1500, 1506, 1507 (val EV $8.18 each):
- $11.778 live EV (buy 16 tickets) across all 5 experiments
- Perfect clone cluster — identical architecture, five independent runs, same live result
- Stable: 200-draw score was $11.50, 234-draw score $11.778 — barely moved with 34 new draws
- Live beats val by $3.60, confirming the model generalizes without gaming

Architecture: et(0.38) + xgb(0.51) + lgb(0.18), custom_interact pipeline, 5 feature sets (basic + gaps + positional + temporal + momentum).

### The exp#796 Lesson — Why Stability Beats Peak

- 200-draw score: $13.50 (appeared to be live champion)
- 234-draw score: $7.231 — dropped $6.27 with 34 more draws
- Val EV was only $0.28 — the GA never selected for it; lucky random fit to the first 200 draws
- Conclusion: A single experiment's peak live score is not reliable. A cluster of 5 experiments all landing at the same score across 234 draws is structurally trustworthy.

### Full Live Leaderboard (234 draws, top known performers)

| Exp          | Val EV | Live EV (234) | Notes                                          |
|--------------|--------|---------------|------------------------------------------------|
| #1471–1507   | $8.18  | $11.778       | buy 16, confirmed live leader (5-exp cluster)  |
| #796         | $0.28  | $7.231        | buy 12, regressed from 200-draw peak of $13.50 |
| #1494/#1496  | $8.04  | $4.231        | buy 15, lower GA ceiling tier                  |

### Val/Live Divergence Finding

High val score does not guarantee high live score. Experiments with low val can have high live scores (exp#796: val $0.28, live $7.23 on 234 draws). The GA optimizes val — live is the honest metric. For any deployment, live performance is the only number that matters.

### Positive EV Baseline

Random baseline = -$0.50/draw. The live leader cluster at $11.778 over 234 draws represents a meaningful positive EV signal above baseline. Whether this is exploitable signal or noise is the central question — awaiting more draws and the 50-experiment live eval batch currently in progress.

Always lead with the 234-draw cluster result ($11.778), not the 200-draw exp#796 peak ($13.50). The cluster result is more defensible.

---

## GA Breakthrough — $9.183 New Ceiling (exp#1645, 2026-04-25)

### The Jump

After 1600+ experiments with a stable ceiling at $8.183, the GA produced exp#1645 at $9.183 val EV — a full $1.00 step above the previous ceiling. This is not a smooth gradient improvement; it's a discontinuous jump.

41 of 150 new experiments (27%) beat the old $8.18 ceiling, with a secondary plateau at $8.54 and the new peak at $9.18.

Architecture: same winning formula — et + xgb + lgb, custom_interact, 5 feature sets (basic + gaps + positional + temporal + momentum).

### The Basin Hypothesis

A $1.00 discontinuous jump after hundreds of experiments stuck at $8.18 suggests the optimization landscape has distinct basins, not a smooth gradient. The GA likely found the wall of a new basin. If so:
- The $9.18 peak may not be the ceiling of this new basin — there could be more headroom above it
- The random batch (574 experiments) was the mechanism that allowed the GA to escape Basin 1 ($8.18) and discover Basin 2 ($9.18+)
- This is direct empirical support for the corner-seeding thesis: interior-only sampling would have stayed in Basin 1 indefinitely

The basin structure is the finding, not just the score. A methodology that can escape local optima and discover new basins in a 55-75D space is the contribution. Whether Basin 2 contains exploitable lottery signal is secondary to demonstrating the escape mechanism worked.

Lead with the basin escape story. The $9.18 number is the hook; the methodology that produced it is the thesis. Live eval of exp#1645 pending — will determine if the new basin generalizes to unseen draws.

### GPU Models — Negative Result

GPU variants (xgb_gpu, lgb_gpu) returned best result of $1.20 against -$0.50 random baseline. Vastly underperformed CPU counterparts. Possible explanations: (1) GPU overhead hurts small-dataset training, (2) same hyperparameter ranges as CPU may not be optimal for GPU execution, (3) the winning architecture genuinely doesn't benefit from GPU parallelism at this data size (~10K training rows). Logged as a negative result for the whitepaper.
