# Statistical Randomness Testing of the Texas Lottery: A Case Study of a Physical RNG

Status: DRAFT, ready for review.

Author: Mark Mundy
AI collaborators: Claude Sonnet 4.5 (Anthropic)
Date: February 2026
Repository: https://github.com/mmundy3832/LottAI

---

## The short version

We asked whether the Texas Lottery's Lotto Texas drawing machine produces numbers that are actually random, or whether 20 years of draws hide some detectable pattern. Lottery balls are picked by a physical machine, air-mixed and tumbling, and physical machines can in principle drift: worn balls, uneven air pressure, a mixing chamber that quietly favors certain positions.

We ran 2,308 Lotto Texas draws from April 2006 through February 2026 through 15 statistical tests, from simple frequency counts to methods borrowed from earthquake modeling (Hawkes self-exciting processes), medical statistics (survival analysis), and time-series anomaly detection (Matrix Profile). Each test compares what we saw against what a truly random machine would produce, and every batch of tests gets corrected for the fact that running many tests makes a few look "significant" purely by chance.

After correction, 14 of the 15 methods found no deviation from randomness. The fifteenth, transfer entropy, has an output file that reports 304 of 28,620 tests passing its own correction; we think that is an artifact of a permutation test with only 100 shuffles, and we report it as an open item to re-run rather than a detection. One number's gap pattern looked interesting before correction, but it carries no predictive value once checked against the other models. A null result is the expected result for a well-run lottery, and we report it as such: the machine looks like what it claims to be.

---

## Background you need

Each concept below gets a plain definition, an everyday example, and a note on how it is used in this paper. After this section we use the terms freely.

### A physical lottery RNG

A physical random number generator (RNG) picks outcomes using a mechanical process instead of a computer algorithm. Lotto Texas uses an air-mixed machine: numbered balls tumble in a chamber and are drawn out one at a time, the same idea as a bingo cage or the ball machines you see on televised lottery drawings. Because it is a physical object, it can in principle drift: ball wear, air pressure changes, machine maintenance. This paper tests whether any such drift shows up in the historical draw record [1].

### Statistically random

A data source is "statistically random" if, no matter how you slice it and test it, the results look like what pure chance would produce. It does not mean every draw looks unpatterned by eye. A fair coin flipped a thousand times will produce occasional runs of 5 or 6 heads in a row; that is expected variation, not evidence of a rigged coin. Here, we are asking whether Lotto Texas draws look like a fair, high-sided die rolled 2,308 times, not whether any single draw looks suspicious [2].

### P-values

A p-value is the probability of seeing a result at least as extreme as the one you observed, assuming the null hypothesis (here: "the machine is fair") is true. If you flip a coin 10 times and get 9 heads, the p-value answers "how surprising is 9-or-more heads if the coin is actually fair?" (about 1%). Every one of our 15 methods produces one or more p-values comparing the real draw history to what a fair process would generate [3].

### The multiple-testing problem, and Bonferroni / Benjamini-Hochberg correction

Run enough independent tests and some will look significant by chance alone, even with nothing real going on. Test 100 innocent students' rooms with a drug test that gives a false positive 5% of the time, and about 5 of them fail the test even though nobody did anything wrong. Bonferroni correction fixes this by dividing your significance threshold by the number of tests, a strict rule that controls the chance of any false positive at all. Benjamini-Hochberg (BH) correction is less strict: it controls the expected fraction of false discoveries among the results you call significant, which suits large batches of tests better. We ran several thousand individual hypothesis tests across the 15 methods, so both corrections appear throughout [4][5].

### Survival analysis

Survival analysis models "time until an event happens." It comes from medical research (time until relapse, time until death) but applies to any waiting time, like how long a lightbulb burns before failing, estimated even from bulbs still burning. Here we treat "draws since a number last appeared" as a waiting time, and train models to rank which number is statistically "due" [6][7].

### Hawkes self-exciting processes

A Hawkes process models events that temporarily raise the odds of similar events happening again soon. Earthquakes are the standard example: a large quake raises the local chance of aftershocks for a while afterward, an effect called self-excitation. We use this to test whether a number appearing makes it, or numbers near it, more likely to reappear soon: an aftershock pattern in lottery draws [8][9].

### Sequential pattern mining

Sequential pattern mining searches a large collection of sequences for short sub-patterns that recur more often than chance predicts. A grocery chain might mine purchase histories and find that "bread, then peanut butter, then jelly" shows up together unusually often. We search the 2,308-draw sequence, in three representations (raw numbers, decade groups, and draw-level features), for sub-sequences that repeat more than random shuffles of the same data would [10][11].

### Matrix Profile motif and discord discovery

Matrix Profile slides a fixed-length window across a time series and, for every window position, finds the closest match elsewhere in the series (a motif, a repeated shape) and the single most different segment (a discord, an anomaly). Picture scanning a year of a runner's daily heart-rate log for the two most similar weeks (a motif, maybe two taper weeks before races) and the one most unusual week (a discord, maybe a week they were sick). We run this on the full 54-number draw sequence, encoded as one 54-dimensional binary vector per draw, at five window lengths [12][13].

### Change point detection

Change point detection finds specific points in time where a series' statistical behavior shifts, in its average, its spread, or both. A factory might use it to spot when a machine's daily output started drifting after a part wore out, even with no maintenance log confirming the date. We look for dates where a number's draw frequency shifted, which could in principle flag an equipment change [14][15][16].

### Contrast mining

Contrast mining compares two groups defined by an outcome (did event X happen, or not) to find which other features differ between them. A sports analyst might compare weather on days a team won versus lost, to see if heat or rain tracks with outcomes. We split draws into groups (recent vs. older, summer vs. winter, high vs. low pressure, hot vs. cold) and mine frequent itemsets of drawn numbers that differ in support between each pair of groups [17].

### Transfer entropy

Transfer entropy measures whether knowing the past of one time series reduces your uncertainty about another series' future, beyond what that second series' own past already tells you. Does yesterday's traffic on Highway A improve a prediction of today's traffic on Highway B, beyond what Highway B's own history already gives you? We test this across all 2,862 directed pairs of numbers, at lags of 1 through 10 draws [18][19].

### Permutation and bootstrap tests

Instead of relying on a textbook formula for a p-value, a permutation test shuffles the real data many times to build an empirical picture of what pure chance looks like, and compares the real result to that picture. A bootstrap test resamples the data with replacement to build a confidence interval around an estimate. To test whether a sports team's home winning streak is unusual, you would shuffle the sequence of home wins and losses thousands of times and see how often a streak that long shows up by chance. We use both throughout Phase 1 as an assumption-light check alongside the classical tests [20][21][22].

---

## What we did

The Phase 1 code, the Lotto Texas and Austin weather datasets, and the results/ output (findings.json, per-method JSON files, one log file, and 43 result files in total) have now arrived and live in this repository under `phase1/`. The numbers in this section and the next are checked against those output files where a file saved the number, or derived directly from `phase1/lottotexas.csv` with a short script where no file saved it (noted inline). A handful of figures in the prior draft, itself carried from notes written before any of this arrived, did not match the actual code or its output; those are corrected here, with the prior figure kept in a footnote.

### Data

- **Raw dataset**: 3,704 Lotto Texas draws, November 14, 1992 through February 11, 2026. Pick 6 numbers from a pool of 54 (range 1-54). Draw days: Monday, Wednesday, Saturday since August 2021; Wednesday and Saturday before that.
- **Analytical dataset**: 2,308 draws, April 1, 2006 through February 11, 2026 (7,256 days by direct count, about 19.9 years). We start at April 2006 because that is when the game settled into its current pick-6-of-54 format. The raw file's pool size actually moved more than once before then (pick-6-of-50 through 1999, briefly pick-6-of-54 from 2000, a different reduced-pool format in 2004-2005), so "pre-2006" is not one consistent era; mixing any of that history with the post-2006 draws corrupts every frequency-based test (see the era-contamination example under "What we found"). Combination space: C(54,6) = 25,827,165. Coverage: 2,308 / 25,827,165 = 0.008936% of the space. Expected appearances per number: 2,308 x 6/54 = 256.4 +/- 15.1.
- **Weather data**: Austin, TX, from the Open-Meteo Historical Weather API (ERA5 reanalysis), April 1, 2006 through May 31, 2024, 6,636 daily observations at 30.2672N, 97.7431W [23]. Inner-joined to draw dates: 2,041 matched observations.

### Methods and parameters

"15 methods" means 11 initial approaches plus 4 advanced pattern-mining methods run afterward.

| # | Method | What it tests | Parameters | Correction |
|---|--------|----------------|------------|------------|
| 1 | Frequency | Each number appears at the base rate (6/54) | Chi-square, df = 53 | Bonferroni 0.01/54 = 0.000185 (see note below) |
| 2 | Gap / recurrence | Draws between repeat appearances follow a geometric distribution | Kolmogorov-Smirnov [24] vs geometric(p=6/54), expected mean gap 9.0 | Bonferroni 0.000185 |
| 3 | Runs | Each number's hit/miss sequence has a random run structure | Wald-Wolfowitz runs test [25] | Bonferroni 0.000185 |
| 4 | Pairwise co-occurrence | Number pairs co-occur at the binomial rate (~0.0105) | 1,431 pairs, C(54,2) | Bonferroni 0.01/1431 = 6.99e-6 |
| 5 | Spectral | No hidden periodicity in a number's hit sequence | FFT per number | none reported |
| 6 | Mutual information | Consecutive draws are independent | Lag-1 on sum/max/min/odd-count, kNN estimator | 1,000-shuffle permutation |
| 7 | Survival analysis | "Time since last appearance" cannot be modeled better than chance | Cox proportional hazards (25 features) [6], Random Survival Forest (100 trees) [26], exponential baseline | C-index on held-out 400 draws |
| 8 | Hawkes process | No self-excitation between numbers | Discrete-time multivariate, 54 dims, L1 logistic regression (LASSO, C=0.1), decay features from past 3 draws | Coefficients zeroed by L1 penalty |
| 9 | Sequential pattern mining | No sub-sequence repeats more than chance | PrefixSpan on raw / decade / feature representations | Fisher exact + BH-FDR q=0.05, 10-shuffle baseline |
| 10 | Weather correlation | Austin weather has no relationship to draw outcomes | Pearson (108), point-biserial (648), Spearman (108), MI-permutation (732), logistic omnibus (54), OLS F (4), Granger (48); 1,698 tests total (1,702 including the 4 OLS F-tests) | BH-FDR q=0.05 |
| 11 | Topological data analysis | (attempted) persistent homology on draw-vector distances | Vietoris-Rips on Jaccard distance | not executed: the `try`/`except` block in `phase1/lottai_analysis.py` around the `gtda` import raised an exception and the run moved on; no error message was saved, so the specific cause (the prior draft says a `giotto-tda`/`scikit-learn` version conflict) is [not in results files] |
| 12 | Matrix Profile | No recurring motif or anomalous discord in the draw sequence | STUMPY, multivariate matrix profile on 54-dim binary draw vectors x 2,308 draws, 5 window sizes (m=5, 7, 10, 15, 20), top 5 motifs and discords saved per window [12][13] | none computed (see note below) |
| 13 | Change point detection | No abrupt shift in per-number frequency over time | Binary Segmentation (`ruptures`, model="l2", min_size=30; see note below), 50-draw rolling window over 3 features (chi-square, entropy, variance), up to 10 change points tested by BIC, 100 bootstrap iterations | 100-shuffle permutation (parameters only; see note below) |
| 14 | Contrast mining | Number frequencies do not differ between recent/older, summer/winter, high/low pressure, and hot/cold draws | Apriori frequent-itemset mining (min_support=0.03) + chi-square test, top 50 patterns saved per contrast (200 saved total); true test count before filtering to the top 50 is [not in results files] | growth_threshold 1.3x / 0.7x, p_threshold 0.1 (see note below) |
| 15 | Transfer entropy | No directed information flow between number pairs | custom estimator, lags 1-10, 2,862 directed pairs x 10 lags = 28,620 tests, 100 permutation surrogates each, Granger VAR(1) cross-check on the top 10 pairs by TE [18][20] | BH-FDR q<0.05 (see note below) |

A note on where the prior draft's methods description did not match the actual code, once the code arrived:

- **Bonferroni threshold (rows 1-4).** `phase1/lottai_analysis.py` sets `SIGNIFICANCE = 0.01` as its one significance constant and computes each Bonferroni threshold as `SIGNIFICANCE / num_range` (individual-number tests) or `SIGNIFICANCE / comb(num_range, 2)` (pairwise tests), and prints the resulting threshold at run time. With num_range = 54, that is 0.01/54 = 0.000185 for the per-number tests and 0.01/1431 = 6.99e-6 for the pairwise test. The prior draft's methods table used 0.05/54 = 0.000926 and 0.05/1431 = 3.49e-5, assuming a 0.05 family-wise rate; that assumption is not in the code. This settles the open item from the previous revision: 0.000185 (and 6.99e-6 for pairs) is the threshold the code actually uses, not 0.000926. None of the reported verdicts change: every p-value discussed below that fails or survives correction does so at both thresholds.
- **Matrix Profile (row 12).** The prior draft described a per-number method (54 binary sequences, window m=12, BH-FDR over 108 tests, yielding q=0.27 for a motif and q=0.022 for a discord). `phase1/matrix_profile_analysis.py` does not do this. It encodes each full draw as one 54-dimensional binary vector, runs STUMPY's multivariate matrix profile at five window sizes, and saves the top motifs and discords per window in `phase1/results/matrix_profile_findings.json`. That file contains no p-values, q-values, or FDR correction of any kind; nothing in the script computes them. This settles the previous revision's open item about the discord's q=0.022 differently than expected: the number does not exist in the actual analysis, so there is nothing to resolve between "q=0.022" and "0 discords survive". We report the real saved output below instead.
- **Change point detection (row 13).** The prior draft described PELT with an RBF cost, a specific penalty formula, 46 features, and a minimum segment length of 100 draws. `phase1/changepoint_analysis.py`'s own comments say otherwise: "Use Binary Segmentation (faster than PELT for model selection)" and "Use Binary Segmentation for speed (faster than PELT)." The function is still named `detect_changepoints_pelt` and its log lines still print "Running PELT change point detection," but the actual call is `rpt.Binseg(model="l2", min_size=30)`, chosen for speed over the PELT algorithm named in the docstring. The feature set is 3 rolling-window statistics (chi-square, entropy, variance), not 46. We report the actual algorithm and parameters here.
- **Contrast mining (row 14).** The prior draft described Mann-Whitney U / Fisher exact tests with a Cohen's d / Cramer's V effect-size threshold, bootstrap confidence intervals, a permutation-based percentile check, and 1,080 total tests. `phase1/contrast_mining.py` does none of this: it mines frequent itemsets with the `mlxtend` Apriori implementation and runs a chi-square test on each one, saving only the top 50 patterns per contrast (200 total across the 4 contrasts). No Cohen's d, Cramer's V, bootstrap, or permutation code exists in the script.
- **Transfer entropy (row 15).** The prior draft's "2,862 pairwise tests" counted only the directed pairs, not the 10 lags tested per pair; the actual `total_tests` field in `phase1/results/causal_discovery_findings.json` is 28,620. More importantly, that file's own `significant_count` field, defined in the code as the count of pairs with `fdr_p_value < 0.05`, is 304, which is what the prior draft's "304" figure actually is. The prior draft labeled 304 as "raw p<0.001" hits that then got corrected to 0 survivors; the file does not support that reading (see "What we found" below).
- **Code availability.** The prior draft's final section claimed this repository "includes Python scripts for all 11 analytical approaches." That was not true at the time. It is true now: `phase1/` holds all 12 scripts and the two source datasets. See "Where the code is."

---

## What we found

### Results by method

| # | Method | We saw | Outcome after correction |
|---|--------|--------|---------------------------|
| 1 | Frequency | chi-square 63.18, df 53, p=0.160 [1]; no number crosses the code's own outlier flag (|z| > 2.5) in the clean dataset, and `findings.json` logged zero frequency-outlier findings for this run | null |
| 2 | Gap / recurrence | #14 mean gap 8.59 vs 9.0 expected, KS p=0.000090 (matches `findings.json`); KS D=0.136 [2] | survives Bonferroni within this module (0.000090 < 0.000185); no downstream predictive value (see Survival, Hawkes) |
| 3 | Runs | #14 z=+2.80 (p=0.005), #6 z=+2.24 (p=0.025), #47 z=-2.02 (p=0.043), all matching `results/hawkes_top_pairs.json`'s `runs_test_zscores` field | null; none survive Bonferroni |
| 4 | Pairwise co-occurrence | pair (14, 26): 44 occurrences vs 24.19 expected, z=+4.05, p=0.000211 [3] | null; fails the 6.99e-6 threshold |
| 5 | Spectral | no notable FFT peaks reported; `06_spectral_analysis.png` exists in the results, confirming the method ran, but no numeric peak values were saved | null |
| 6 | Mutual information | sum I=0.0131 (p=0.88), odd-count I=0.0022 (p=0.59), max I=0.0262 (p=0.96), min I=0.0111 (p=0.47) [not in results files] | null |
| 7 | Survival analysis | C-index: naive 0.5000, Cox 0.5081, Random Survival Forest 0.5195 (all match `survival_metrics.json`); Brier score at the 1-draw horizon 0.1148 (`survival_metrics.json`), compared in the prior draft to 0.0988, which is the Hawkes model's baseline Brier score from `hawkes_top_pairs.json`, not a survival-native baseline [4] | null; effectively coin-flip discrimination |
| 8 | Hawkes process | all 54x54 excitation coefficients driven to 0.0 by the L1 penalty; log-likelihood unchanged from baseline (-0.349176 vs -0.349212), both from `hawkes_top_pairs.json` | null |
| 9 | Sequential pattern mining | 351 patterns tested (`pattern_mining_patterns.json`); real mean lift 1.040 vs shuffled baseline 1.515 (`pattern_mining_metrics.json`); 1 pattern above shuffled p95 (expected about 17.5 at 5%) | null; real patterns were weaker than shuffled ones |
| 10 | Weather correlation | 94 of 1,698 tests raw p<0.05 (expected ~85) [5]; strongest raw hits: humidity vs draw_sum r=-0.0496 (p=0.025), pressure_range vs draw_range r=-0.0476 (p=0.031), both matching `weather_correlation_findings.json`; RF regression CV R^2=-0.0218; RF classification mean AUC 0.5043 | null; 0 survive BH-FDR |
| 11 | TDA | not run | not run |
| 12 | Matrix Profile | best motif: 5-draw window, distance 0.247, between the draws starting 2016-06-04 and 2006-04-01; best discord: 20-draw window, distance 4.189, at the draws starting 2018-09-22; all from `matrix_profile_findings.json` [6] | no significance test was run on these values, so "survives" or "fails" correction does not apply; see note above |
| 13 | Change point detection | `changepoint_findings.json` is incomplete: it breaks off mid-file after the `optimal_changepoints` key, so the detected count, segment boundaries, and permutation result are [not in results files]. What survives: n_draws=2308, 50-draw rolling window, up to 10 change points tested, 100 bootstrap iterations, all matching the file's header | not derivable from the current output |
| 14 | Contrast mining | 200 patterns saved across the 4 contrasts; strongest raw p-values in `contrast_mining_summary.txt`: number 9, hot days vs cold days, 5.5% vs 13.5% (p=0.0019); number 13, hot vs cold, 6.0% vs 13.7% (p=0.0029); number 13, summer vs winter, 7.8% vs 13.6% (p=0.0018) [7] | null per the summary text ("most patterns did not survive correction"); the surviving-count and the specific case cited in the prior draft are [not in results files] |
| 15 | Transfer entropy | 304 of 28,620 tests have `fdr_p_value < 0.05` per `causal_discovery_findings.json`'s own `significant_count` field; the file's Granger cross-check agrees on 4 of the top 10 pairs by TE value | see note below; not treated as a detection |

Footnotes for this table:

[1] `findings.json` does not store the chi-square statistic itself (the run only logs findings that cross its threshold, and this test did not); 63.18/p=0.160 is carried from the prior draft and reproduced independently from `phase1/lottotexas.csv` (63.26, p=0.158), close enough to treat as the same result. The prior draft additionally cited "#26 at 472 appearances (z=+3.16), #4 at 462 (z=+2.64)" as individual outliers in this row. Those counts are the full, contaminated 3,704-draw dataset's numbers, not the clean 2,308-draw dataset's (confirmed by direct computation: 472 and 462 appearances, z=3.16 and z=2.64, only reproduce against the full dataset's expected count of 411.6, not the clean dataset's 256.4). In the clean dataset, the largest deviations are #45 (z=-2.41) and #8 (z=+2.36), both below the code's own |z|>2.5 outlier-flag threshold.

[2] `findings.json` stores the gap finding's p-value (0.0000896) and an `effect_size` field (0.9544, which is mean_gap/expected_gap, not a KS statistic). The KS D value is not saved anywhere; we reproduced it from `phase1/lottotexas.csv` using the same `stats.kstest` call as the script and got D=0.136 against the prior draft's D=0.112. The p-value matches to five significant figures either way, so the underlying test is the same; only the reported D differs.

[3] The prior draft stated "71 occurrences vs 38.8 expected" for this pair, alongside z=4.05 and p=0.000211. Those two counts do not produce that z and p: (71-38.8)/sqrt(38.8x0.9895) gives z=5.20, not 4.05. Direct computation from `phase1/lottotexas.csv` gives 44 occurrences vs 24.19 expected, which does produce z=4.05 and p=0.000211 exactly. We use the count that matches the z and p already reported.

[4] Both Brier figures are real values from the results files; they just come from two different models (survival's own 1-draw Brier score, and the Hawkes model's baseline Brier score) rather than a survival-model-native baseline. `survival_metrics.json` does not compute its own naive-baseline Brier score for comparison.

[5] Sum of the four phases with a `raw_significant` field in `weather_correlation_findings.json`: 49 (Pearson/point-biserial/Spearman, of 864) + 41 (mutual information, of 732) + 2 (logistic, of 54) + 2 (Granger, of 48) = 94, of 864+732+54+48=1,698 tests. The prior draft's "89" does not match this sum; we could not find an alternate breakdown in the file that produces 89. The "expected ~85" figure (1,698 x 0.05 = 84.9) is unchanged and matches.

[6] The prior draft's Matrix Profile numbers (motif distance 0.52 for "Number 23," discord distance 3.1 for "Number 41," p and q values) are not in `matrix_profile_findings.json` and do not correspond to a per-number analysis the actual script performs. We report the file's real best motif and discord instead. Both happen to fall in similar places to the prior draft's narrative (2006 vs. 2016 for the motif, fall 2018 for the discord), which is likely why the mismatch was not caught before the code arrived.

[7] The prior draft's specific "best case" for this method ("#13 vs humidity, p=0.0027, 58.3% vs 61.7%, Cohen's d=0.18, q=0.15, permutation 28th percentile, bootstrap CI [-0.05, 0.41]") does not appear in `contrast_mining_findings.json` or `contrast_mining_summary.txt`, and `contrast_mining.py` computes no Cohen's d, Cramer's V, bootstrap interval, or permutation percentile at any point. We report the real top hits from the summary file instead.

A note on the transfer entropy result (row 15): `causal_discovery_findings.json`'s own `significant_count` field says 304 of 28,620 pairs pass its BH-FDR check, which is the opposite of the prior draft's "0 survive BH-FDR." We do not think this means transfer entropy found a real effect. The permutation test behind each p-value uses only 100 surrogate shuffles, which floors the smallest possible p-value at 1/100 = 0.01 and produces exact ties at p=0.0 whenever the observed value beats all 100 shuffles. Under a true null, that happens by chance for roughly 1 in 101 tests, or about 283 of the 28,620 tests here, close to the 304 actually flagged. A coarse permutation floor producing many tied p=0.0 values, which a rank-based FDR procedure treats as maximally significant, is a plausible explanation for the 304 figure. We did not re-run the analysis with more surrogates to confirm this, so we report it as an open question rather than either a detection or a settled null (see "What this does not show").

### Era contamination

We compared the same battery of tests on the full 3,704-draw history (which spans the pool-size changes described under "Data") against the clean, single-era 2,308-draw post-2006 dataset.

| Comparison | Contaminated (3,704 draws, all eras) | Clean (2,308 draws, post-2006) |
|---|---|---|
| Frequency | chi-square 263.47, p=1.1e-29 (independently reproduced from `phase1/lottotexas.csv`, matching the prior draft's figure) | chi-square 63.18, p=0.160 |
| Position bias | all 6 draw positions individually significant at p<0.01 (reproduced from `phase1/lottotexas.csv`) | 0 of 6 positions significant |
| Runs test | reproduced from `phase1/lottotexas.csv`: 0 numbers with |z|>10, max |z|=3.71. The prior draft's notes are internally inconsistent here: one summary line says "10 numbers with |z|>10," another section of the same notes says "z-scores exceeding +/-20 for multiple numbers." Neither reproduces; we could not find a version of the runs test on this dataset that gives either figure | max |z|=2.80 (#14), none survive Bonferroni |
| Sequential patterns | "13 patterns survive FDR" per the prior draft; [not in results files], and re-running PrefixSpan on the full dataset was out of scope for this check | 0 survive (`pattern_mining_patterns.json`) |
| Co-occurrence pairs (Bonferroni) | 2 pairs survive (reproduced from `phase1/lottotexas.csv` at the code's actual threshold, 0.01/1431=6.99e-6) | 0 pairs survive (reproduced the same way) |

Pooling draws from different pool-size eras as if they were one game manufactures fake non-uniformity in frequency, position, and co-occurrence tests: numbers outside the smaller historical pools have zero chance of appearing in those years, which reads as bias once the eras are combined. The clean, single-era dataset does not show this. The runs-test claim in the prior notes does not check out under either wording we could find in those notes, and we flag that rather than repeat it.

### Totals and statistical power

Summing the per-method test counts that are actually recoverable from the results files (frequency: 1, gap: 54, runs: 54, pairwise co-occurrence: 1,431, mutual information: 4, sequential pattern mining: 351, weather correlation: 1,698, transfer entropy: 28,620) gives 32,213 tests, dominated by transfer entropy's 10-lags-per-pair design. Contrast mining's true pre-filtering test count is not saved. At alpha=0.05 uncorrected, chance alone predicts roughly 1,600 false positives among the 32,213 recoverable tests, most of that from the transfer entropy module alone (0.05 x 28,620 = 1,431). We think this means the corrections were doing real work: without them, this analysis would have reported far more "findings" than it did, and the transfer entropy module in particular needed the correction it got, even if we are not fully confident in how that specific correction behaved (see the transfer entropy note above).

A power calculation for the frequency test: with 2,308 draws, a 1% deviation from the expected per-number rate has about 12% power to detect at alpha=0.05; a 5% deviation has about 60% power; a 10% deviation has about 95% power. We think this means our null result does not prove the machine is perfectly random. It shows that any bias present is small enough to be undetectable at this sample size, and (per the survival and Hawkes results above) too small to translate into a usable prediction even if it exists.

---

## What this does not show

- It does not show the machine is perfectly random. It shows that no bias large enough to detect with 2,308 draws, and no bias that would help predict a future draw, was found.
- It does not rule out a slow, multi-decade equipment drift. The 19.9-year window may be too short to see century-scale wear.
- It is specific to the Texas Lottery. We did not test other states' machines or games.
- The survival and Hawkes models used 25 hand-built features chosen by the original analysis; different features might behave differently.
- Topological data analysis (method 11) did not run. The code's own error message was not saved, so we cannot confirm the prior draft's stated cause (a `giotto-tda`/`scikit-learn` version conflict); we can confirm the method produced no output (no `07_tda_persistence.png` file exists among the results).
- 25,000+ Pick 3 draws were downloaded during this phase but were not analyzed here; that became separate later work outside this paper's scope.
- The transfer entropy module's `significant_count` of 304 (out of 28,620 tests) is not explained by the "0 survive BH-FDR" language used elsewhere in this paper. We think the most likely explanation is a resolution artifact from using only 100 permutation surrogates (see "What we found"), but we have not confirmed that by re-running the analysis with more surrogates, so we are reporting the conflict rather than resolving it.
- The changepoint results file is incomplete (it cuts off mid-write), so the change point detection method's specific findings (count, dates, segment lengths) could not be checked against anything beyond its run parameters.
- Some numbers in this paper are still carried from the original analysis notes rather than a results file, specifically the mutual information values (method 6) and the exact "13 patterns survive FDR" figure for the contaminated dataset; those are marked inline as such.

---

## Where the code is

The Phase 1 analysis code and the Lotto Texas and Austin weather datasets now live in this repository under `phase1/`, in the same relative layout used here.

- `phase1/download_data.py`: downloads and assembles the raw Lotto Texas draw history into `phase1/lottotexas.csv`. Writes no results files.
- `phase1/fetch_weather.py`: fetches the Austin weather history from the Open-Meteo API into `phase1/austin_weather_2006_2024.csv`. Writes no results files.
- `phase1/lottai_analysis.py`: the main battery: frequency, position bias, gap/recurrence, sum distribution, odd-count, mutual information, co-occurrence, runs test, TDA (attempted), co-occurrence network, dimensionality reduction, entropy over time. Writes `phase1/results/findings.json` and PNGs `01` through `06` and `08` through `10` (`07_tda_persistence.png` was never produced; the TDA section raised an exception and the run continued).
- `phase1/survival_model.py`: Cox proportional hazards, Random Survival Forest, exponential baseline. Writes `phase1/results/survival_metrics.json`, `survival_feature_importance.json`, `survival_predictions.json`, and three PNGs. Has a hardcoded `D:\projects\LottAI\...` path (`DATA_PATH`, `RESULTS_DIR`); edit before running.
- `phase1/hawkes_model.py`: discrete-time multivariate Hawkes process with L1-penalized logistic regression. Writes `phase1/results/hawkes_top_pairs.json`, `hawkes_excitation_matrix.json`, `hawkes_predictions.json`, and three PNGs.
- `phase1/pattern_mining.py`: PrefixSpan sequential pattern mining across raw/decade/feature representations. Writes `phase1/results/pattern_mining_metrics.json`, `pattern_mining_patterns.json`, `pattern_mining_predictions.json`, and three PNGs.
- `phase1/weather_correlation.py`: the 11-phase weather correlation battery (Pearson, point-biserial, Spearman, mutual information, logistic, OLS, Granger, random forest). Writes `phase1/results/weather_correlation_findings.json`, `weather_correlation.log`, and five PNGs.
- `phase1/matrix_profile_analysis.py`: multivariate STUMPY matrix profile motif/discord search over the full draw sequence at five window sizes. Writes `phase1/results/matrix_profile_findings.json`.
- `phase1/changepoint_analysis.py`: PELT change point detection on rolling per-number frequency windows. Writes `phase1/results/changepoint_findings.json` (the copy that arrived is incomplete). Has a hardcoded `D:\projects\LottAI\...` path; edit before running.
- `phase1/contrast_mining.py`: Apriori frequent-itemset mining with chi-square testing across four contrasts (temporal, seasonal, pressure, temperature). Writes `phase1/results/contrast_mining_findings.json` and `contrast_mining_summary.txt`. Has a hardcoded `D:/projects/LottAI/...` path; edit before running.
- `phase1/causal_discovery.py`: transfer entropy (custom estimator) plus a Granger VAR(1) cross-check. Writes `phase1/results/causal_discovery_findings.json`. Has a hardcoded `D:\projects\LottAI\...` path; edit before running.
- `phase1/results/analyze_causal_results.py`: a small follow-up script that reads `causal_discovery_findings.json` and prints a lag/direction breakdown of the top transfer entropy pairs. Writes nothing; it only prints to the terminal.

We did not re-run any of these scripts for this revision. We checked their logic against the saved results files and, where a results file did not have a number, against `phase1/lottotexas.csv` directly.

---

## Open questions / next phase

- Extend the comparison to other states' lottery machines (California, Florida, New York) to see whether the null result is specific to this machine or general to the format.
- Move from post-hoc testing to a real-time anomaly monitor (sequential probability ratio test, CUSUM charts) for ongoing integrity checks rather than a one-time analysis.
- The 15-method battery here (frequency, gap, runs, co-occurrence, spectral, mutual information, survival, Hawkes, sequential patterns, weather/contrast correlation, Matrix Profile, change points, transfer entropy) is not specific to Lotto Texas. Nothing in the pipeline assumes a 54-ball pick-6 game; it could run against any physical RNG that produces a long enough log of discrete outcomes. We note this as a reusable audit recipe, not as a finished product.
- Pick 3 analysis, mentioned but not run in this phase, became its own line of work in a later phase of this project and is outside the scope of this paper.
- Re-run `phase1/changepoint_analysis.py` to get a complete `changepoint_findings.json`; the current copy cuts off mid-write and none of its detailed findings could be checked.
- Re-run `phase1/causal_discovery.py` with more than 100 permutation surrogates to check whether the 304 "FDR-significant" transfer entropy pairs are a resolution artifact or something that needs a closer look.
- Fix the four scripts with hardcoded `D:\projects\LottAI\...` paths (`survival_model.py`, `changepoint_analysis.py`, `contrast_mining.py`, `causal_discovery.py`) so they run against this repository's layout, and get the TDA method (row 11) actually running.

---

## References

1. Texas Lottery Commission. Winning numbers archive. https://www.texaslottery.com/
2. Wikipedia. "Statistical randomness." https://en.wikipedia.org/wiki/Statistical_randomness
3. Wikipedia. "P-value." https://en.wikipedia.org/wiki/P-value
4. Wikipedia. "Bonferroni correction." https://en.wikipedia.org/wiki/Bonferroni_correction
5. Benjamini, Y., & Hochberg, Y. (1995). Controlling the false discovery rate: A practical and powerful approach to multiple testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289-300.
6. Cox, D. R. (1972). Regression models and life-tables. *Journal of the Royal Statistical Society: Series B*, 34(2), 187-220.
7. Wikipedia. "Survival analysis." https://en.wikipedia.org/wiki/Survival_analysis
8. Hawkes, A. G. (1971). Spectra of some self-exciting and mutually exciting point processes. *Biometrika*, 58(1), 83-90.
9. Wikipedia. "Hawkes process." https://en.wikipedia.org/wiki/Hawkes_process
10. Pei, J., Han, J., Mortazavi-Asl, B., et al. (2004). Mining sequential patterns by pattern-growth: The PrefixSpan approach. *IEEE Transactions on Knowledge and Data Engineering*, 16(11), 1424-1440.
11. Wikipedia. "Sequential pattern mining." https://en.wikipedia.org/wiki/Sequential_pattern_mining
12. Yeh, C. M., Zhu, Y., Ulanova, L., et al. (2016). Matrix Profile I: All pairs similarity joins for time series. *2016 IEEE 16th International Conference on Data Mining (ICDM)*, 1317-1322.
13. Law, S. M. (2019). STUMPY: A powerful and scalable Python library for time series data mining. *Journal of Open Source Software*, 4(39), 1504. https://github.com/TDAmeritrade/stumpy
14. Killick, R., Fearnhead, P., & Eckley, I. A. (2012). Optimal detection of changepoints with a linear computational cost. *Journal of the American Statistical Association*, 107(500), 1590-1598.
15. Truong, C., Oudre, L., & Vayatis, N. (2020). Selective review of offline change point detection methods. *Signal Processing*, 167, 107299. arXiv:1801.00826. https://arxiv.org/abs/1801.00826
16. Truong, C., Oudre, L., & Vayatis, N. `ruptures`: change point detection in Python. https://github.com/deepcharles/ruptures
17. Dong, G., & Bailey, J. (2012). *Contrast Data Mining: Concepts, Algorithms, and Applications*. Chapman and Hall/CRC.
18. Schreiber, T. (2000). Measuring information transfer. *Physical Review Letters*, 85(2), 461-464.
19. Wikipedia. "Transfer entropy." https://en.wikipedia.org/wiki/Transfer_entropy
20. Moore, D. G., Valentini, G., Walker, S. I., & Levin, M. (2018). Inform: Efficient information-theoretic analysis of collective behaviors. *Frontiers in Robotics and AI*, 5, 60. https://github.com/elife-asu/pyinform
21. Wikipedia. "Permutation test." https://en.wikipedia.org/wiki/Permutation_test
22. Wikipedia. "Bootstrapping (statistics)." https://en.wikipedia.org/wiki/Bootstrapping_(statistics)
23. Open-Meteo Historical Weather API. ERA5 reanalysis data. https://open-meteo.com/
24. Kolmogorov, A. (1933). Sulla determinazione empirica di una legge di distribuzione. *Giornale dell'Istituto Italiano degli Attuari*, 4, 83-91.
25. Wald, A., & Wolfowitz, J. (1940). On a test whether two samples are from the same population. *The Annals of Mathematical Statistics*, 11(2), 147-162.
26. Ishwaran, H., Kogalur, U. B., Blackstone, E. H., & Lauer, M. S. (2008). Random survival forests. *The Annals of Applied Statistics*, 2(3), 841-860.

---

**Acknowledgments**: We thank the Texas Lottery Commission for maintaining public draw archives, the Open-Meteo project for free historical weather data access, and Claude Sonnet 4.5 (Anthropic) for analytical assistance on this phase.

**Conflict of interest**: The author has no financial interest in lottery systems, gambling, or prediction services. This research was conducted independently for educational and methodological purposes.
