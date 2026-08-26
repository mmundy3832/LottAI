# Genetic Search Over a High-Dimensional Model-Configuration Space: A Case Study on Texas Lottery Pick 3

**Authors**: Martin Mundy, Claude Sonnet 4.5 (Anthropic)
**Date**: August 2026
**Project**: LottAI
**Repository**: https://github.com/mundymar/LottAI

---

## Abstract

Phase 1 of this project found no statistically significant deviation from uniform randomness in 2,308 Texas Lottery Lotto Texas draws after multiple-testing correction. Phase 2 asks a different question: independent of whether the draw process is random, can a genetic-algorithm (GA) search over a 55-75 dimensional model-configuration space find a model that beats chance on Pick 3, and does a live prediction loop confirm any edge found in validation? We ran 1,670 experiments over a config space covering feature sets, transformation pipelines, and ensemble model types, combining GA mutation/crossover, LLM-proposed configs, and large batches of pure-random sampling. The search produced a discontinuous jump from a validation optimal EV of $8.04 to $9.183 (exp#1645) after 800+ experiments stuck at the lower ceiling, following a 575-experiment random-sampling batch. We read this as evidence for a basin-structured optimization landscape in which random sampling seeds escape from local attractors GA operators alone cannot escape. A 121-prediction live ledger and a 72-cell lag-hit analysis (2 cells p<.05, 0 surviving multiple-comparison correction) show no evidence of predictive power above chance in live, truly unseen draws. The basin-escape mechanism, not the score itself, is the primary contribution.

**Keywords**: genetic algorithm, hyperparameter search, high-dimensional optimization, corner seeding, local optima, lottery prediction, live evaluation, multiple-testing correction

---

## 1. Introduction

### 1.1 Relation to Phase 1

Phase 1 tested whether the Texas Lottery Lotto Texas draw process shows any statistical bias, using 15 analytical methods against 2,308 draws. The result was null: zero deviations from uniform randomness survived multiple-testing correction. That result stands and Phase 2 does not revisit it.

Phase 2 moves to Pick 3, a different game (three independent digit positions, pool of 10 per digit, 1,000 combinations), and asks a separate question: rather than testing for randomness directly, can a large-scale search over model configurations find and validate an exploitable signal, if one exists, that survives contact with truly unseen live draws?

### 1.2 Why Run a Config Search After a Null Result

The Phase 1 null result does not answer whether a sufficiently expressive search over model architectures, feature engineering, and ensemble weighting could still find weak, exploitable structure the direct statistical tests were not built to detect. A genetic search over a 55-75D config space is also a research question about high-dimensional exploration in its own right: does random sampling at scale unstick a genetic algorithm that mutation and crossover leave stuck at a local optimum? Pick 3 gives a cheaply scorable metric (dollar EV per draw) to test this, independent of whether lottery signal exists. A search finding nothing live-durable is still informative, about both the lottery and the methodology.

### 1.3 Objectives

1. Run a large-scale GA/LLM/random search over model configurations scored by validation optimal EV, and determine whether random sampling at scale changes GA search dynamics in a high-dimensional space.
2. Build a live prediction loop and ledger that scores predictions against truly unseen draws, independent of the validation metric driving the search.
3. Test whether any live hit-rate signal survives a lag-structure analysis with multiple-comparison correction.

---

## 2. Methods

### 2.1 Data and Metric

Pick 3 draw data: 15,572 combined rows across four daily draw times (Morning, Day, Evening, Night; Monday-Saturday, no Sunday draws), split by time order into train (rows 0-10,899), validation (rows 10,900-12,899), and a held-out test set. A separate live dataset (`pick3*_live.csv`, 2026-02-14 onward) is never included in training or LLM-visible data; it exists only for evaluation.

Primary fitness metric, `optimal_ev`:
```
optimal_ev = max over k=1..20 of (top_k_hit_rate * $500 - k)
```
A straight $1 ticket pays $500; random baseline is -$0.50/draw. `live_optimal_ev` applies the same formula to held-out live draws instead of validation.

### 2.2 Config Space

Each experiment is a config dict with three parts:
- **Feature sets**: basic (11 dims), recency (121), gaps (40), positional (33), temporal (20), momentum (7), equipment (8, gated off by default). Combined dimensionality spans roughly 55-75 dimensions depending on which feature sets and pipeline stages are active.
- **Pipeline**: ordered transformation stages, `custom_interact` (pairwise products/ratios of top-N and bottom-N features), `select` (ExtraTrees-based feature selection), `poly` (polynomial expansion, degree 3 gated behind a preceding `select` stage), `scale` (StandardScaler/RobustScaler/QuantileTransformer/MinMaxScaler).
- **Models**: per-digit classifiers (d1, d2, d3, each a 10-class problem), drawn from `et`, `rf`, `xgb`, `lgb`, `hgb`, `lr`, `mlp`, plus GPU variants `xgb_gpu`, `lgb_gpu`. Digit probability vectors multiply into a 1,000-combo probability matrix.

### 2.3 Search Operators

The GA (`ga_ops.py`) applies one of five weighted mutation types: continuous (30%), pipeline (20%), model (20%), feature_sets (15%), discrete (15%). Crossover combines two parents chosen by tournament selection over the top-15 by fitness. A default 20% of experiments are proposed by an external LLM (MiniMax M2.7, via Ollama) given only a config-and-score history; MiniMax never sees live data or metrics. A separate fraction (default 10%, raised to 25% after the basin-escape finding below) samples configs fully at random, to seed diversity the GA cannot generate on its own ("corner seeding").

### 2.4 Live Prediction Loop and Ledger

Phase 2 also runs a standing prediction loop: on each scheduled draw, a council of experiment configs retrains and votes on the next draw, producing a ranked probability matrix and a top-K ticket recommendation. Predictions are written to `prediction_ledger.jsonl` before the draw, then scored against the actual result once it posts. The ledger records the consensus and top-20 combo sets, whether either hit, and the rank of the actual outcome in the top 20.

---

## 3. Results

### 3.1 Scale and the $8.04 Plateau

1,670 total experiments are logged in `experiments_v4.jsonl`, of which 1,196 completed with a scored `optimal_ev`; 948 ran in pure-random mode overall, 575 of them in the exp#843-1417 range that forms the specific random-exploration batch discussed below. The GA held a ceiling of $8.04 optimal_ev for 800+ experiments (exp#346/358/390, identical clones from one mutation lineage) before that batch ran, to inject diversity the GA's own operators were not producing.

### 3.2 The Random Batch and the $8.18 Ceiling

The random batch did not itself beat the ceiling (best result $7.68, exp#880, val optimal_ev = 7.6849), but resuming the GA afterward produced a new ceiling of $8.18 (exp#1471-1507, optimal_ev = 8.1832): five experiments sharing one architecture, `et` + `xgb` + `lgb` ensemble (weights approximately 0.38/0.51/0.18), single-stage `custom_interact` pipeline, five feature sets (basic, gaps, positional, temporal, momentum; recency dropped).

### 3.3 The $9.183 Jump (exp#1645) and the Basin Hypothesis

After the $8.18 ceiling held again for several hundred experiments, a further batch of 150 produced exp#1645 at $9.183 optimal_ev (verified against `experiments_v4.jsonl`), a full $1.00 above the prior ceiling and the highest score in the dataset. 41 of 150 (27%) beat the old $8.18 ceiling, with a secondary plateau at $8.54 (exp#1533/1556/1562/1567/1574, optimal_ev = 8.5411) below the new peak. Architecture at the new ceiling matches the prior winning formula, suggesting the jump came from parameter-space structure within the same architecture family, not a new architecture.

A discontinuous $1.00 jump after hundreds of experiments stuck at a lower value, arriving right after a large random-sampling injection, fits a basin-structured landscape rather than a smooth gradient: GA operators explore within a basin efficiently but cannot cross the wall between basins alone. Random sampling, by covering corners the GA's local operators would not reach, supplied the seed material that let the GA find a second, higher basin. This basin-escape mechanism, not the $9.18 figure itself, is the central finding.

### 3.4 MLP: A Dead End

87 MLP experiments ran during the random-sampling batch (mode=`random`, confirmed by direct count against `experiments_v4.jsonl`). Zero appear in the top 50 by `optimal_ev`. MLPClassifier did not compete with tree- and boosting-based ensembles on this tabular time-series problem at this data scale.

### 3.5 Feature Set Architecture vs. Pipeline

Within the random batch, single-stage `custom_interact` pipelines outperformed heavier 3-4 stage pipelines across the board. Pipeline choice was secondary; feature-set composition mattered more than pipeline depth.

### 3.6 Stability vs. Peak: The exp#796 Lesson

exp#796 (feature sets basic+recency+gaps+positional+momentum, single `scale` stage, `hgb` model) had a validation optimal_ev of only $0.28, low enough that the GA never selected it forward. At the 200-draw live window it scored $13.50 live EV, the highest live score observed at that point. At the 234-draw window (34 more draws), it scored $7.231, a drop of $6.27. By contrast the $8.18 cluster (exp#1471-1507) scored $11.50 at 200 draws and $11.778 at 234, essentially unmoved. A single experiment's peak live score is not a reliable signal; a cluster of five independently-evolved experiments landing at the same score across 34 added draws is more defensible. Validation score does not predict live score: the GA optimizes validation, and validation after 1,600+ experiments of selection pressure is compromised. Live evaluation on unseen draws is the only honest number for judging any experiment or cluster.

### 3.7 GPU Models: A Negative Result

26 experiments used `xgb_gpu` or `lgb_gpu`. Best result: $1.20 optimal_ev (exp#1544), far below the $9.18 CPU ceiling and only modestly above the -$0.50 baseline. GPU variants underperformed CPU counterparts substantially at this training scale (~10,000 rows). Candidate explanations: GPU overhead not amortized at this data size; hyperparameter ranges reused from CPU tuning may not suit GPU execution; the winning architecture may not benefit from GPU parallelism at this scale at all. Logged as a negative result, not pursued further.

### 3.8 Live Ledger: Hit Rates vs. Baseline

123 predictions are tracked, 121 scored against actual draws. Consensus-set hit rate: 5/121 = 4.1% (mean consensus size 29.7 of 1,000 combos; empirical chance baseline approximately 3.0%). Top-20 hit rate: 3/121 = 2.5%, against a fixed chance baseline of 2.0% (20/1,000). Average rank of the actual combo, when in the top 20, was #9.0. Neither figure is large relative to its baseline at n=121; the lag analysis below tests this directly.

### 3.9 Lag Analysis and the Null

A lag-hit analysis tested whether accuracy shows structure across draw lags 0-5, using six statistics (exact hit rate, rate of at least 2 correct positions, mean max correct positions, per-position coverage for positions 0, 1, 2) for the top-20 and consensus sets, against a permutation null (n_perm=2,000, seed=42): 72 cells total. Two showed p<.05: lag=1, consensus set, position-1 coverage (observed 0.617 vs. null mean 0.523, p=0.020), and lag=4, top-20 set, mean max positions correct (observed 1.214 vs. null mean 1.106, p=0.036). Neither survives correction for 72 comparisons (expected false positives at alpha=0.05 with no signal: 3.6; observed: 2). No cell shows a pattern consistent across adjacent lags, the signature genuine structure would produce.

---

## 4. Discussion

### 4.1 The Basin Hypothesis Is the Primary Finding

The $8.04 to $8.18 to $9.18 trajectory, each ceiling held for hundreds of experiments before a discontinuous jump following random injection, is the strongest evidence for a basin-structured search landscape. A methodology that demonstrably escapes local optima in 55-75 dimensions is a contribution independent of whether the basins found contain exploitable signal. Whether Basin 2 has more headroom above it, or further sub-basins, remains open; this whitepaper does not claim the search has converged.

### 4.2 Random Exploration as Seed Material, Not Direct Optimizer

The random batch never itself produced a top result: its best entry ($7.68) sat below both GA ceilings it helped break through. Its value was not finding good solutions directly but supplying diverse genetic material the GA's own operators could not generate from a converged population. This distinguishes "random search" as a standalone method (worse than GA at every checkpoint) from "random injection" as a hybrid-search component (necessary for the GA to escape).

### 4.3 No Evidence of Predictive Power Above Chance

Every live-facing number here is consistent with chance once compared to its baseline or corrected for multiple comparisons. The $9.183 validation ceiling has not been live-evaluated at time of writing [TBD: live_optimal_ev for exp#1645 once evaluated against unseen draws]; the confirmed live leader remains the $8.18 cluster at $11.778 over 234 draws, positive against the -$0.50 baseline, but exp#796 (Section 3.6) shows a positive live score at one draw count is not sufficient evidence of durable signal. The forward, out-of-sample ledger and lag analysis are the more decisive evidence, and show no structure surviving correction.

---

## 5. Limitations

1. 121 scored ledger predictions is too small a sample to detect a hit-rate signal much below the ~30-50% relative lift needed to move the top-20 hit rate clearly above its 2.0% baseline.
2. `optimal_ev` diverges from live performance under GA selection pressure (Section 3.6); "ceiling" figures in Section 3 are validation numbers unless stated as live, and the $9.183 ceiling (exp#1645) has no live figure yet.
3. The basin hypothesis is inferred from the trajectory's shape, not a direct visualization of the loss landscape; it is a plausible reading, not a proven structural claim.
4. The GPU result reused CPU hyperparameter ranges with no GPU-specific tuning pass, so it should not be read as ruling out GPU models generally.
5. The lag analysis covers lags 0-5 only; longer-range dependencies are untested.

---

## 6. Conclusion

A GA/LLM/random hybrid search over a 55-75 dimensional model-configuration space, run for 1,670 experiments against Texas Lottery Pick 3 data, found a basin-structured optimization landscape: two ceilings ($8.04, $8.18) each held for hundreds of experiments before discontinuous jumps following random-sampling batches, culminating in a $9.183 validation peak at exp#1645. Random sampling did not win on its own but supplied the diversity the GA needed to escape local attractors, an empirical demonstration of the corner-seeding hypothesis this project set out to test. A live prediction ledger of 121 out-of-sample predictions and a 72-cell lag analysis (2 cells nominally significant, 0 surviving correction) show no predictive power above chance. As in Phase 1, the honest reading of the live-facing evidence is a null result on exploitable lottery signal; the contribution of Phase 2 is the search methodology itself.

---

## References

1. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, 785-794.
2. Ke, G., Meng, Q., Finley, T., et al. (2017). LightGBM: A highly efficient gradient boosting decision tree. *Advances in Neural Information Processing Systems*, 30.
3. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
4. Mundy, M., & Claude Sonnet 4.5 (2026). Rigorous Statistical Analysis of Physical Random Number Generators: A Case Study of the Texas Lottery [Phase 1 whitepaper]. LottAI project, https://github.com/mundymar/LottAI.
5. Texas Lottery Commission. (2026). Pick 3 winning numbers archive. https://www.texaslottery.com/

---

**Acknowledgments**

We thank the Claude Sonnet 4.5 AI system (Anthropic) for analytical assistance and the MiniMax M2.7 model (via Ollama) for its role as LLM-proposal experimenter.

---

**Conflict of Interest Statement**

The authors have no financial interest in lottery systems, gambling, or prediction services. This research was conducted independently for educational purposes.

---

**END OF WHITE PAPER**
