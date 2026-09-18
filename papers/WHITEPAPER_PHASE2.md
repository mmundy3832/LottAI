# Genetic Search Over a High-Dimensional Model-Configuration Space: A Case Study on Texas Lottery Pick 3

Status: DRAFT, ready for review.

Author: Mark Mundy
AI collaborators: Claude Sonnet 4.6 (Anthropic) on the April 2026 experiment commits, per
their co-author trailers; the prior draft of this paper named Claude Sonnet 4.5.
Experimenter LLM: MiniMax M2.7 via Ollama.
Date: September 2026
Repository: https://github.com/mmundy3832/LottAI

---

## The short version

Phase 1 of this project found no statistically significant deviation from randomness in Texas Lottery Lotto Texas draws. Phase 2 asks a different question on a different game: independent of whether the draw process is random, can a large search over model configurations find something that predicts Texas Pick 3 well enough to matter, and does that survive contact with draws the search never saw?

We built a search that mixes three ways of proposing a model configuration: a genetic algorithm (breeding good configs from good configs), an LLM (MiniMax M2.7, prompted with the search rules and a history of past results), and pure random sampling. We ran 1,670 experiments. The validation score climbed in discontinuous steps, from $8.04 to $8.18 to $9.183 per draw, each step arriving after a large batch of random experiments broke a long plateau. We read this as evidence, not proof, that the search space has separate basins that the genetic algorithm's own operators cannot cross alone.

None of this held up live. A 121-prediction ledger of truly future draws and a lag-hit analysis across 72 statistical cells found nothing beyond chance. We also found, and fixed, a real data leak in one feature: once removed, a larger causal check on 662 live draws found zero of 72 cells significant, the cleanest null in the project so far. As in Phase 1, the honest answer on exploitable signal is no. The search methodology itself, and how thoroughly we found nothing when we checked, is what this phase adds.

---

## Background you need

### Pick 3 and expected value in dollars

Pick 3 is a lottery game where a machine draws three digits, 0-9 each, in order. A straight $1 ticket wins $500 if your three digits match the draw in the same order; any other outcome of the 1,000 possible ordered combinations, you win nothing. Buy a ticket with the digits 4-1-7: if the draw comes up 4-1-7 you get $500, if it comes up anything else you get nothing. Odds on one random guess: 1 in 1,000.

Expected value (EV) is the average profit per ticket if you played the same way many times. For one random $1 ticket: EV = $500 x (1/1,000) - $1 = -$0.50. You lose 50 cents a draw on average, the house's cut.

Phase 2's scoring metric, `optimal_ev`, extends this to a strategy that ranks all 1,000 combos by predicted probability and buys the top k of them (k from 1 to 20):

```
optimal_ev = max over k=1..20 of (top_k_hit_rate * $500 - k)
```

A validation `optimal_ev` of $9.183 means: on the validation draws, buying the model's best k ranked tickets at $1 each returned, on average, $9.183 more per draw than it cost, versus losing $0.50 on a single random guess. It is a model of a betting strategy's profitability, not a claim that any one ticket is more likely to win. [15]

### Genetic algorithms

A genetic algorithm (GA) searches for good solutions by keeping a population of candidates, breeding the best of them, and repeating over generations.

Think of breeding show dogs for a trait. You keep the best performers each generation ("fitness"), pair two of them so a puppy inherits traits from each parent ("crossover"), and occasionally a puppy shows a trait neither parent had ("mutation"). Repeat for enough generations and the population trends toward whatever trait you rewarded.

Here, one candidate is a full model configuration: which features to compute, which preprocessing steps to run, which classifier types to combine. Fitness is `optimal_ev` on validation data. Mutation (`autoresearch/ga_ops.py`) changes one part of a config at a time, weighted across five families: continuous nudges (30%), pipeline changes (20%), model swaps (20%), feature-set changes (15%), other discrete changes (15%). Crossover picks two parents by tournament among the top 15 and blends feature sets, pipeline, and models. [1][2]

### Hyperparameter and config search, and why 55-75 dimensions is hard

A hyperparameter is a setting chosen before training, not learned from data, like how many trees a forest has or how deep they go. A config search tries many combinations of settings, where each setting is one dimension you can vary. Choosing a pizza alone (5 toppings) is one dimension; choosing a pizza, a drink, a side, and a dessert, each with 5-10 options, multiplies out to thousands of combinations fast.

A Phase 2 config chooses which of 7 feature sets to turn on, up to 3 ordered pipeline stages each with their own settings, and up to 3 model types with their own hyperparameters (tree depth, learning rate, regularization, and more). Counting every one of these as a separate switch gives roughly 55 to 75 dimensions, depending on which optional stages and feature sets a given config has active. This is the size of the config itself, not the resulting feature table the model trains on (that table, for the six non-equipment feature sets combined, is a fixed 232 columns; do not conflate the two). Trying every combination is not feasible; even 60 independent yes/no choices alone is 2^60 possibilities. [3]

### Local optima and basins

A local optimum is a solution that looks best among everything nearby, but is not the best solution that exists, like reaching the top of a foothill in fog and feeling like you have summited because every direction you can see is downhill, when a taller peak sits across a valley you cannot see from where you stand.

The GA's validation score held at $8.04 for more than 800 experiments. Mutating and breeding the existing population kept landing back near the same peak. We describe this as the GA sitting on top of one basin in the config space, unable to cross the valley of worse-scoring configs that separates it from a possibly taller basin elsewhere. [4][17]

### Random sampling as an escape mechanism

Instead of only breeding from the current population, you can throw in an entirely new, unrelated candidate, the way an outside show dog can restore a lost coat color no living dog in your current breeding line still carries.

`autoresearch/config_space.py` places some configs at deliberate extremes ("corner seeds"), and a tunable fraction of all experiments, 10% by default, are generated fully at random rather than bred from anything. The 575-experiment batch discussed below (ids 843-1417) is a dedicated run of this kind, at 100% random. [1]

### Gradient-boosted trees (LightGBM, XGBoost)

A gradient-boosted tree model builds many small decision trees one after another, each new tree trained to correct the mistakes of the trees built before it, like a chain of proofreaders who each only fix errors the last one missed.

XGBoost (`xgb`) and LightGBM (`lgb`) are the two gradient-boosted-tree libraries used as per-digit classifiers here, and both appear in the winning architecture at every score ceiling reported in this paper. [5][6]

### Ensembles

An ensemble combines predictions from several different models instead of relying on one, usually by weighted averaging, the way averaging three friends' guesses at a stranger's age usually beats any one friend's guess, because their errors partly cancel out.

Phase 2's best-performing configs blend ExtraTrees (`et`), `xgb`, and `lgb` with weights around 0.38 / 0.51 / 0.18, before the three digit-position probability vectors are multiplied together into a combo ranking. [7]

### Train, validation, and test splits, and why a frozen cutoff matters

Splitting data by time into a training set (fits the model), a validation set (picks the best config), and a test set (checked rarely, to confirm the final choice) prevents a model from being graded on the same material it studied. Studying for an exam by repeatedly grading yourself against the same practice test can make you memorize that test's specific answers rather than the material; a second practice test, sealed until the end, tells you if you actually learned anything.

Phase 2 splits the 15,572-row Pick 3 history by draw date, never by random shuffle: rows 0-10,899 (10,900 rows, through 2022-05-23) train the models; rows 10,900-13,235 (2,336 rows, through 2024-04-03) are validation, the number the GA optimizes against; rows 13,236-15,571 (2,336 rows, through 2026-02-13) are test, checked rarely.[f6] A separate live file of draws from 2026-02-14 onward is never used to train a model or shown to the LLM experimenter (see "Data boundary" below); it exists only to score configs someone has already picked. [8]

### The Karpathy autoresearch loop

An autoresearch loop is an automated research cycle: a harness repeatedly asks another program or model to propose an experiment, runs it against a fixed scoring function, and records the result, so each new experiment can build on everything run before it without a person hand-writing each one. Picture a manager who does not do the work but hands a junior researcher a one-page brief of the rules and a log of every experiment run so far, asks for one new idea, and scores it against a single number when it finishes.

In Phase 2, Claude Sonnet 4.5 is the harness that runs the loop and inspects results; MiniMax M2.7, accessed through Ollama, is the experimenter. It reads `autoresearch/program_v3.md` (the config schema, valid ranges, and a "Directions Worth Exploring" section) plus the history of past configs and scores, and returns one new JSON config at a time. `optimal_ev` is the single scorable metric, and `program_v3.md` is the editable spec the harness can revise between runs, the "program.md" of the pattern. MiniMax never sees live draws or live scores. [9]

### Brier score and rank metrics

A Brier score measures how well-calibrated a probability is: if you say something is 70% likely, does it happen about 70% of the time, the way a weather forecaster who says "70% chance of rain" and is right 70% of the time is well-calibrated whether or not it rains today. A rank metric instead asks only where the correct answer landed once everything is sorted by predicted probability, closer to how a search engine is judged: how far down the results page the right answer lands.

Phase 2's per-digit models output calibrated probabilities (many configs set `calibrate: true` with isotonic calibration, a Brier-style calibration step) before being multiplied into the combo ranking. The project's own rank metric, `mean_rank`, is the average position of the actual combo in the sorted list of 1,000, tracked as a secondary metric alongside `optimal_ev` (baseline 500.5, lower is better). [10]

### Leakage

Leakage is when a feature fed to a model accidentally contains information about the answer being predicted, so the model looks accurate in testing but is cheating, the way a hospital-readmission model that accidentally includes a "discharge note mentions readmission" field, written only after the readmission happened, looks excellent in testing and fails on a real patient whose future has not been written yet.

Phase 2's momentum feature set includes a "consecutive-draw digit overlap" feature whose last computation read the digits of the very row being predicted, not just the rows before it (see "What we found," the causal backtest). Because that feature was present in both training and validation, it could inflate apparent structure without providing any real predictive power. [11]

### Multiple-comparison correction across 72 lag cells

Multiple-comparison correction adjusts your bar for "significant" when running many statistical tests at once, because some will look significant purely by chance: flip 100 fair coins 20 times each and test each one for "unusually streaky" behavior at the standard 5% threshold, and about 5 will look streaky by chance alone, not because any coin is rigged.

Phase 2's lag analysis checks 72 cells (6 match statistics x 2 prediction sets x 6 lags) for whether predictions line up with actual draws at some lag. At a 5% threshold with no real signal, about 3.6 of 72 cells are expected to look significant by chance. This is the number every lag result in this paper is measured against. [12][13]

---

## What we did

### Data and splits

Pick 3 draw data: 15,572 combined rows, 2013-09-09 to 2026-02-13, four draw times a day (Morning, Day, Evening, Night), Monday through Saturday. Splits, fixed by row and date (`autoresearch/prepare_v3.py`, `_TRAIN_END=10900`, `_VAL_END=13236`):

| Split | Rows | Count | Date range |
|---|---|---|---|
| Train | 0-10,899 | 10,900 | through 2022-05-23 |
| Validation | 10,900-13,235 | 2,336 | through 2024-04-03 |
| Test | 13,236-15,571 | 2,336 | through 2026-02-13 |
| Live | separate file, `pick3all_live.csv` | 662 | 2026-02-14 to 2026-08-26 |

Pay table: a $1 straight ticket pays $500 on an exact-order match; chance of a match is 1/1,000; random-guess EV is -$0.50/draw.[f6]

### Data boundary: why live draws never reached the experimenter

The live file is never included in training, validation, or anything shown to MiniMax. If the LLM experimenter could see live outcomes, it could steer proposals toward whatever happened to work on those specific draws, and the live evaluation would no longer tell us anything about draws the search had never encountered, just another thing being optimized against. Keeping live data sacred is what makes the live numbers in this paper a real test, not a second validation set with a different name.

### Config space

| Feature set | Dimensions | Default | Captures |
|---|---|---|---|
| basic | 11 | on | previous draw's sum, repeats, odd/even, high/low |
| recency | 121 | on | per-digit, per-position frequency over last 10/25/50/100 draws, plus combo gap |
| gaps | 40 | on | draws since each digit last appeared, per position |
| positional | 33 | on | per-position frequency over last 50 draws, plus position correlations |
| temporal | 20 | on | day-of-week (7), month (12), year (1) |
| momentum | 7 | on | rolling chi-square and entropy per position, plus consecutive-draw overlap (leak source, see below) |
| equipment | 8 | off | machine ID and ball-set IDs |
| draw_time | 4 | removed | draw-time one-hot; tested and confirmed noise, excluded from the valid config space |

The six non-equipment feature sets combined give a 232-column feature table. The "55-75 dimensional" figure quoted in this paper's abstract refers to the config dict itself (feature toggles, pipeline stage parameters, and up to three models' hyperparameters), not the feature table; the two should not be conflated.[f7]

Pipeline stages, up to 3, each used at most once: `custom_interact` (pairwise products and ratios of top-N/bottom-N features), `select` (ExtraTrees-based feature selection), `poly` (polynomial expansion, degree 2 or 3, degree 3 only after a `select` stage), `scale` (StandardScaler, RobustScaler, QuantileTransformer, or MinMaxScaler). Models, 1-3 per config: `et`, `rf`, `xgb`, `lgb`, `hgb`, `lr`, `mlp`, plus GPU variants `xgb_gpu` and `lgb_gpu`.

### Search operators and run counts

| Proposal source | Count | Share of 1,670 |
|---|---|---|
| random | 948 | 56.8% |
| ga_mutation | 358 | 21.4% |
| llm | 156 | 9.3% |
| ga_crossover | 77 | 4.6% |
| ga_crossover_mutate | 75 | 4.5% |
| bootstrap | 56 | 3.4% |
| **Total** | **1,670** | **100%** |

Defaults: `llm-frac` 0.20, `random-frac` 0.10, 900-second timeout per experiment, LLM model `minimax-m2.7:cloud`. 1,196 of 1,670 experiments completed with a scored `optimal_ev`; 474 did not, 334 of those from timing out at 900 seconds. `experiments_v4.jsonl` logs 1,670 lines but only 1,543 unique experiment IDs; 127 IDs were reused when the runner restarted its internal counter across sessions, which matters below (Section "The $9.183 jump").

### Live evaluation and the prediction ledger

`autoresearch/live_eval_v2.py` retrains a chosen config on train+validation and scores it against `pick3all_live.csv`, draws the config never saw during search. Two cohorts exist: 296 records (287 unique IDs) at 200 live draws, and 187 records at 234 live draws (2026-02-14 to 2026-04-23).

Separately, a standing prediction loop ("Council of Experts," `plans/PHASE2_PLAN.md`) retrains ten diverse configs on all data through the latest draw, has them vote, and sends a top-K and consensus recommendation four times a day, logging every prediction to `autoresearch/prediction_ledger.jsonl` before the draw and scoring it once the draw posts.

---

## What we found

### The $8.04 plateau and the random batch

We saw the GA hold a validation ceiling of $8.0394 optimal_ev for more than 800 experiments (exp#346, cloned at #358 and #390, all one mutation lineage; 13 experiments in the ledger share this exact value). A 575-experiment batch of pure-random configs then ran, in the ID range 843-1417.[f2] Its own best scored result was $5.5514, at exp#1162.

The prior draft, and `WHITEPAPER_NOTES.md`, reported the best random result as $7.68 at exp#880. That number is wrong.[f1] `experiments_v4.jsonl` has two different rows using the reused ID 880: a `ga_crossover` experiment from 2026-04-17, before the random batch ran, scoring $7.6849 (parents #196 and #122); and a `random`-mode experiment from 2026-04-20 that timed out with no score. The $7.68 figure belongs to a GA crossover, not the random batch. The true best pure-random experiment anywhere in the ledger is exp#1162 at $5.55.

### The $8.18 ceiling

We saw the GA resume after the random batch and reach a new ceiling of $8.1832 optimal_ev, first at exp#1471 (a crossover of #1420 and #1453), with 19 experiments across the ledger sharing that exact score. The architecture: `et` + `xgb` + `lgb` ensemble (weights roughly 0.38 / 0.51 / 0.18), a single `custom_interact` pipeline stage, and five feature sets (basic, gaps, positional, temporal, momentum; recency dropped).

We think this pattern, best random result below both ceilings it sits between, but the batch preceding a GA breakthrough, means random sampling supplied genetic diversity the converged GA population could not generate on its own, rather than winning as a search method in its own right.

### The $9.183 jump and the basin hypothesis

We saw a further run of experiments produce exp#1645 at $9.1832 optimal_ev on 2026-04-24, the highest score in the ledger. A secondary plateau of $8.5411 formed first (24 experiments share this value, including #1533, #1556, #1562, #1567, #1574).

One correction to the prior draft's framing: exp#1645's parent is #1585, which sits at the $8.5411 plateau, not at the $8.18 cluster. The actual local step was $8.54 to $9.18, not a single jump from $8.18 to $9.18 as previously described. The full trajectory is staged: $8.04, then (after the random batch) $8.18, then $8.54, then $9.18.

The prior draft also stated that 41 of 150 experiments following the random batch (27%) beat the old $8.18 ceiling. We could not verify this against any batch boundary in the ledger.[f_unverified] Two verifiable alternatives: of the 150 ledger records ending at #1645, 127 were scored and 23 scored strictly above $8.1832; of all 253 records with ID above 1417, 211 were scored and 27 scored strictly above $8.1832 (46 at or above).

We think the staged $8.04 to $8.18 to $8.54 to $9.18 trajectory, each level held for hundreds of experiments before a step up following a random-sampling injection, is consistent with a basin-structured search landscape: GA mutation and crossover explore efficiently within one basin but do not cross into a better one without new genetic material. This is inference, not a measurement of the landscape, and it rests on one non-replicated sequence of jumps (n=1 per step). An architecture-level reading is also possible: the winning ensemble family (et+xgb+lgb, custom_interact, the same five feature sets) held across all three ceilings, so the steps could equally be parameter refinement within one architecture, not a crossing into a structurally different basin. We did not visualize the loss landscape directly; the basin language is a plausible reading, not a proven structural claim.

### MLP: a dead end

We saw 87 MLP experiments run (all in `random` mode), with zero appearing in the top 50 scores; the best MLP result was $5.19 (exp#1193, rank about 209). The prior draft and `WHITEPAPER_NOTES.md` reported 74 MLP experiments; the ledger shows 87.[f3] `MLPClassifier` did not compete with tree- and boosting-based ensembles on this tabular problem at this data scale.

### GPU models: a negative result

We saw 26 experiments using `xgb_gpu` or `lgb_gpu`, best result $1.2038 optimal_ev (exp#1544), far below the $9.18 CPU ceiling and only modestly above the -$0.50 baseline. These experiments reused the same hyperparameter ranges tuned for CPU models, so this result does not rule out GPU models generally; it rules out this particular untuned transfer.

### Pipeline vs. feature architecture

We saw single-stage `custom_interact` pipelines outperform heavier 3-4 stage pipelines throughout the random batch. Feature-set composition mattered more than pipeline depth at every ceiling reported here.

### Live evaluation: EV timeline and the ledger

| Milestone | Experiment | Val optimal_ev | 200-draw live EV | 234-draw live EV |
|---|---|---|---|---|
| First ceiling | #346 (clones #358, #390) | $8.0394 | not evaluated | not evaluated |
| Lower tier | #1494 / #1496 | $8.0394 | $5.00 (buy 15) | $4.231 (buy 15) |
| Second ceiling | #1471 cluster (19 experiments) | $8.1832 | $11.50 (buy 16) | $11.778 (buy 16) |
| Unstable outlier | #796 | $0.28 | $13.50 (buy 19) | $7.231 (buy 12) |
| Peak validation score | #1645 | $9.1832 | not evaluated | $9.9145 (buy 20) |

exp#1645, the highest validation score in the ledger, scored $9.9145 live at 234 draws, lower than the $8.18 cluster's $11.778 on the same draws. The prior draft flagged this as "[TBD]"; it is no longer TBD.[f5] exp#796, at a validation score of only $0.28 (too low for the GA to have selected it forward), posted the single highest 200-draw live score in the leaderboard ($13.50), then dropped to $7.231 once 34 more draws were added, a $6.27 fall; the #1471 cluster's five independently-evolved experiments landed at the same score at both draw counts.

Across the 200-draw cohort, 296 records (287 unique IDs) exist; 276, 93.2%, show a positive live EV. Across the 234-draw cohort, 184 of 187, 98.4%, are positive. `BRIEFING.md` states "286 evaluations against 200 unseen draws, 95.4% positive"; the ledger shows a different count and rate.[f4]

Data-quality note: two separate 200-draw records exist for exp#796, and both carry a `val_results.optimal_ev` of $9.9145, exp#1645's live score, not exp#796's validation score. The 234-draw record has the correct value ($0.28). We flag this as a ledger bug, not a finding about exp#796.

### Production configs: post-hoc selected, not in either prior paper

Two configs, exp#1566 and exp#1572, scored higher live EV than anything reported in the prior draft: $19.3248 (buy 17) and $16.78, at 234 live draws. Neither appears in the prior Phase 2 paper or `WHITEPAPER_NOTES.md`. Both are named in the production predictor (`autoresearch/predict_now.py`, "Config A exp#1566," "Config B exp#1572") and in the Phase 3 preregistration as the models actually deployed.

Both were chosen after looking at all 483 live-evaluation records and picking the top scorers, not chosen in advance of seeing live results. This is post-hoc selection, and it invites the same failure as picking "the best" mutual fund manager after the fact: evaluate enough candidates and report only the winner, and you will find an impressive number even when none of the candidates has real skill, purely from how many chances you gave yourself. Unlike the pre-registered lag analysis or the causal backtest below, $19.32 and $16.78 are not an honest estimate of future performance; they describe what scored best among 483 evaluated candidates.

### Prediction ledger: hit rates near baseline

123 predictions were logged, 121 scored against actual draws (2 unscored at ledger close). Consensus-set hit rate: 5/121 = 4.13% (mean consensus size 29.70 of 1,000 combos; permutation-based chance baseline about 2.97%). Top-20 hit rate: 3/121 = 2.48%, against a fixed chance baseline of 2.0% (20/1,000). Mean rank of the actual combo, when in the top 20: 9.0 (ranks 7, 3, 17). Neither rate is large relative to its baseline at this sample size.

### Lag analysis: the ledger's own null

We saw a lag-hit analysis of the 121 scored ledger predictions across lags 0-5, six match statistics, and two prediction sets (top-20, consensus): 72 cells, tested against a permutation null (2,000 permutations, seed 42). Two cells scored p<.05: lag 1, consensus set, position-1 coverage (observed 0.6167 vs. null mean 0.5232, p=0.0200, n=120); and lag 4, top-20 set, mean max positions correct (observed 1.2137 vs. null mean 1.1062, p=0.0365, n=117). Two of 72 is below the roughly 3.6 expected by chance at a 5% threshold, and neither cell has support from an adjacent lag, the pattern real structure would produce. This analysis ran with the momentum leak described next still present in the production predictor; we report it as measured, unmodified.

### The causal backtest: a leaked feature, and the strongest null in the phase

This analysis does not appear in the prior draft of this paper.

We found that the momentum feature set's last feature, "consecutive-draw digit overlap," computed for the row being predicted using that same row's own actual digits, not only the rows before it. This is a textbook leak: a feature meant to summarize the past instead partly contains the answer. The bug is documented in `autoresearch/backtest_lag.py`'s docstring as pre-existing.

`autoresearch/backtest_lag.py` reruns the 72-cell lag analysis on two much larger, fixed windows, using the two production configs (exp#1566 and exp#1572) trained once: the 2,336-row test split, and the 662-row live file. Each window was scored twice, once reproducing production behavior with the leak present ("non-causal"), and once with a correction applied that removes the leaked value ("causal").

| Window | Records | p<.05 cells, non-causal (leak present) | p<.05 cells, causal (leak fixed) | Chance expectation |
|---|---|---|---|---|
| Test | 2,336 | 14 of 72 | 5 of 72 | 3.6 |
| Live | 662 | 15 of 72 | 0 of 72 | 3.6 |

We saw the non-causal count sit well above the roughly 3.6 expected by chance on both windows, consistent with the leak inflating apparent structure, and the causal, leak-corrected count drop to at or below chance on both.

We think this is the strongest null result in Phase 2. It does not rest on an absence of significant cells alone; it identifies a specific, documented mechanism for why the earlier, uncorrected numbers looked more structured than they were, then shows the structure disappears once that mechanism is removed, on the largest live sample in this paper (662 draws, versus 121 in the ledger). The production predictor and the 123-prediction ledger ran with the leak present throughout, so the ledger's reported 2/72 result stands as measured; this backtest is a separate, later check of the same question on more data.

---

## What this does not show

1. The basin-escape interpretation rests on one sequence of validation-score jumps, not a repeated pattern; it is our reading of the trajectory's shape, not a direct measurement of the landscape, and an architecture-level explanation (refinement within one ensemble family) fits the same data.
2. exp#1645 has the highest validation score but a lower live score than the $8.18 cluster; this paper does not resolve whether that gap is overfitting to validation, noise at n=234, or both.
3. The GPU result reused untuned CPU hyperparameter ranges; it rules out this specific untuned transfer, not GPU-trained models generally.
4. 121 scored ledger predictions cannot reliably detect a hit-rate lift much below roughly 30-50% relative to the 2.0-3.0% baselines reported here.
5. exp#1566 and exp#1572's live scores ($19.32, $16.78) were selected after seeing all 483 live-evaluation records and are not an unbiased forecast of future performance.
6. The causal backtest fixes one documented leak in the momentum feature set. It does not rule out other, undiscovered leakage elsewhere in the config space.
7. The lag analyses here cover lags 0-5 only; longer-range dependencies are untested.
8. The "41 of 150 (27%)" figure from the prior draft could not be reproduced from the ledger; it is reported only as unverified, alongside two differently-scoped counts that could be verified.

---

## Where the code is

- Experiment ledger: `autoresearch/results/experiments_v4.jsonl`
- GA operators (mutation, crossover, tournament selection): `autoresearch/ga_ops.py`
- Config space, valid ranges, corner seeds: `autoresearch/config_space.py`
- Runner and CLI defaults: `autoresearch/runner_v4.py`
- Feature engineering and split boundaries: `autoresearch/prepare_v3.py`
- Config-to-code template engine: `autoresearch/template_engine.py`
- LLM experimenter prompt/spec: `autoresearch/program_v3.md`
- Live evaluation: `autoresearch/live_eval_v2.py`, `autoresearch/results/live_eval_results_v2.jsonl`
- Prediction loop and ledger: `autoresearch/predict_now.py`, `autoresearch/orchestrate.py`, `autoresearch/telegram_bot.py`, `autoresearch/prediction_ledger.jsonl`
- Lag analysis (ledger): `autoresearch/lag_analysis.py`, `autoresearch/results/lag_analysis_output.json`
- Causal backtest (leak fix): `autoresearch/backtest_lag.py`, `autoresearch/results/backtest_lag_output_{test,live}[_causal].json`
- Near-hit table: `autoresearch/results/near_hits_table.md`
- Phase 2 plan: `plans/PHASE2_PLAN.md`

---

## Open questions / next phase

- Whether the $9.18 basin has headroom above it, or further random injection finds a fourth level, is untested; the ledger froze at 1,670 experiments on 2026-04-25.
- Why exp#1645 scores below the $8.18 cluster live despite a higher validation score is open.
- GPU models never got a dedicated tuning pass; the negative result is not a clean ablation.
- The "41 of 150" figure needs the original run log, if one exists.
- Whether other feature sets contain leaks similar to the momentum overlap bug has not been checked.
- The 574-vs-575 batch-count discrepancy and the exp#796 `val_results` data-quality bug in `live_eval_results_v2.jsonl` are unresolved bookkeeping issues, not results.
- A later phase (sequence models on tokenized draws, plus a preregistered test of whether Phase 2's predictions were merely early by a draw or two) is out of scope here and reported separately.

---

## Footnotes

[f1] Prior draft and `WHITEPAPER_NOTES.md`: best random result "$7.68 at exp#880." That row is a GA crossover predating the random batch. Verified best pure-random result: $5.55 at exp#1162.

[f2] `WHITEPAPER_NOTES.md`: random batch "574 experiments." Ledger: 575 `random`-mode experiments, IDs 843-1417.

[f3] `WHITEPAPER_NOTES.md`: "74 MLP experiments." Ledger: 87.

[f4] `BRIEFING.md`: "286 evaluations against 200 unseen draws, 95.4% positive." Live-evaluation file: 296 records (287 unique IDs) at 200 draws, 93.2% positive.

[f5] Prior draft: exp#1645 live score "[TBD]." Now measured: $9.9145 at 234 live draws (buy 20).

[f6] `BRIEFING.md` and the prior draft: validation split "rows 10,900-12,899." Code (`prepare_v3.py`, confirmed in `phase3/data_loader.py`): validation ends at row 13,236, giving validation rows 10,900-13,235 and test rows 13,236-15,571, 2,336 rows each.

[f7] `program_v3.md` (the LLM experimenter's spec): `equipment` feature set "4 features." Code (`prepare_v3.py FEATURE_DIMS`): 8. The experimenter worked from a spec that understated this feature set.

[f_unverified] "41 of 150 (27%) beat the old $8.18 ceiling" appears in the prior draft and `WHITEPAPER_NOTES.md`; no batch boundary in the ledger reproduces it. See "The $9.183 jump" for two verifiable, differently-scoped counts.

---

## References

[1] Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press. https://en.wikipedia.org/wiki/Genetic_algorithm

[2] Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.

[3] Wikipedia. "Hyperparameter optimization." https://en.wikipedia.org/wiki/Hyperparameter_optimization

[4] Wikipedia. "Fitness landscape." https://en.wikipedia.org/wiki/Fitness_landscape

[5] Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794. https://doi.org/10.1145/2939672.2939785

[6] Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30. https://github.com/microsoft/LightGBM

[7] scikit-learn developers. "Ensemble methods." https://scikit-learn.org/stable/modules/ensemble.html

[8] scikit-learn developers. "Cross-validation: evaluating estimator performance." https://scikit-learn.org/stable/modules/cross_validation.html

[9] Karpathy, A. `autoresearch`. https://github.com/karpathy/autoresearch

[10] Brier, G. W. (1950). Verification of forecasts expressed in terms of probability. *Monthly Weather Review*, 78(1), 1-3. https://en.wikipedia.org/wiki/Brier_score

[11] Wikipedia. "Leakage (machine learning)." https://en.wikipedia.org/wiki/Leakage_(machine_learning)

[12] Wikipedia. "Multiple comparisons problem." https://en.wikipedia.org/wiki/Multiple_comparisons_problem

[13] Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300. https://en.wikipedia.org/wiki/False_discovery_rate

[14] Ollama. Official documentation. https://ollama.com

[15] Texas Lottery Commission. Pick 3 winning numbers archive. https://www.texaslottery.com/

[16] Mundy, M. (2026). *Rigorous Statistical Analysis of Physical Random Number Generators: A Case Study of the Texas Lottery* [Phase 1 whitepaper]. LottAI project. https://github.com/mmundy3832/LottAI (AI collaborator: Claude Sonnet 4.5, Anthropic)

[17] Wikipedia. "Local optimum." https://en.wikipedia.org/wiki/Local_optimum
