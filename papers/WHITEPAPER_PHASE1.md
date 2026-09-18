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

After correction, none of the 15 methods found a real deviation from randomness. One number's gap pattern looked interesting before correction, but it carries no predictive value once checked against the other models. A null result is the expected result for a well-run lottery, and we report it as such: the machine looks like what it claims to be.

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

Run enough independent tests and some will look significant by chance alone, even with nothing real going on. Test 100 innocent students' rooms with a drug test that gives a false positive 5% of the time, and about 5 of them fail the test even though nobody did anything wrong. Bonferroni correction fixes this by dividing your significance threshold by the number of tests, a strict rule that controls the chance of any false positive at all. Benjamini-Hochberg (BH) correction is less strict: it controls the expected fraction of false discoveries among the results you call significant, which suits large batches of tests better. We ran roughly 3,650 individual hypothesis tests across the 15 methods, so both corrections appear throughout [4][5].

### Survival analysis

Survival analysis models "time until an event happens." It comes from medical research (time until relapse, time until death) but applies to any waiting time, like how long a lightbulb burns before failing, estimated even from bulbs still burning. Here we treat "draws since a number last appeared" as a waiting time, and train models to rank which number is statistically "due" [6][7].

### Hawkes self-exciting processes

A Hawkes process models events that temporarily raise the odds of similar events happening again soon. Earthquakes are the standard example: a large quake raises the local chance of aftershocks for a while afterward, an effect called self-excitation. We use this to test whether a number appearing makes it, or numbers near it, more likely to reappear soon: an aftershock pattern in lottery draws [8][9].

### Sequential pattern mining

Sequential pattern mining searches a large collection of sequences for short sub-patterns that recur more often than chance predicts. A grocery chain might mine purchase histories and find that "bread, then peanut butter, then jelly" shows up together unusually often. We search the 2,308-draw sequence, in three representations (raw numbers, decade groups, and draw-level features), for sub-sequences that repeat more than random shuffles of the same data would [10][11].

### Matrix Profile motif and discord discovery

Matrix Profile slides a fixed-length window across a time series and, for every window position, finds the closest match elsewhere in the series (a motif, a repeated shape) and the single most different segment (a discord, an anomaly). Picture scanning a year of a runner's daily heart-rate log for the two most similar weeks (a motif, maybe two taper weeks before races) and the one most unusual week (a discord, maybe a week they were sick). We run this on each number's 2,308-draw hit-or-miss sequence with a 12-draw window [12][13].

### Change point detection

Change point detection finds specific points in time where a series' statistical behavior shifts, in its average, its spread, or both. A factory might use it to spot when a machine's daily output started drifting after a part wore out, even with no maintenance log confirming the date. We look for dates where a number's draw frequency shifted, which could in principle flag an equipment change [14][15][16].

### Contrast mining

Contrast mining compares two groups defined by an outcome (did event X happen, or not) to find which other features differ between them. A sports analyst might compare weather on days a team won versus lost, to see if heat or rain tracks with outcomes. We split draws into "number X appeared" versus "did not" and compare weather and calendar features across the split [17].

### Transfer entropy

Transfer entropy measures whether knowing the past of one time series reduces your uncertainty about another series' future, beyond what that second series' own past already tells you. Does yesterday's traffic on Highway A improve a prediction of today's traffic on Highway B, beyond what Highway B's own history already gives you? We test this across all 2,862 directed pairs of numbers [18][19].

### Permutation and bootstrap tests

Instead of relying on a textbook formula for a p-value, a permutation test shuffles the real data many times to build an empirical picture of what pure chance looks like, and compares the real result to that picture. A bootstrap test resamples the data with replacement to build a confidence interval around an estimate. To test whether a sports team's home winning streak is unusual, you would shuffle the sequence of home wins and losses thousands of times and see how often a streak that long shows up by chance. We use both throughout Phase 1 as an assumption-light check alongside the classical tests [20][21][22].

---

## What we did

The figures in this section and the next are carried over from the original Phase 1 analysis notes, written before this repository existed. We could not check them against code or data files in this repository (see "Where the code is"), and we will re-verify every number once the Phase 1 code is added here.

### Data

- **Raw dataset**: 3,704 Lotto Texas draws, November 14, 1992 through February 11, 2026. Pick 6 numbers from a pool of 54 (range 1-54). Draw days: Monday, Wednesday, Saturday since August 2021; Wednesday and Saturday before that.
- **Analytical dataset**: 2,308 draws, April 1, 2006 through February 11, 2026 (7,252 days, about 19.9 years). We start at April 2006 because that is when the game switched to its current pick-6-of-54 format; mixing it with the earlier pick-6-of-50 era corrupts every frequency-based test (see the era-contamination example under "What we found"). Combination space: C(54,6) = 25,827,165. Coverage: 2,308 / 25,827,165 = 0.008936% of the space. Expected appearances per number: 2,308 x 6/54 = 256.9 +/- 15.1.
- **Weather data**: Austin, TX, from the Open-Meteo Historical Weather API (ERA5 reanalysis), April 1, 2006 through May 31, 2024, 6,636 daily observations at 30.2672N, 97.7431W [23]. Inner-joined to draw dates: 2,041 matched observations.

### Methods and parameters

"15 methods" means 11 initial approaches plus 4 advanced pattern-mining methods run afterward.

| # | Method | What it tests | Parameters | Correction |
|---|--------|----------------|------------|------------|
| 1 | Frequency | Each number appears at the base rate (6/54) | Chi-square, df = 53 | Bonferroni 0.05/54 = 0.000926 |
| 2 | Gap / recurrence | Draws between repeat appearances follow a geometric distribution | Kolmogorov-Smirnov [24] vs geometric(p=6/54), expected mean gap 9.0 | Bonferroni 0.000926 |
| 3 | Runs | Each number's hit/miss sequence has a random run structure | Wald-Wolfowitz runs test [25] | Bonferroni 0.000926 |
| 4 | Pairwise co-occurrence | Number pairs co-occur at the binomial rate (~0.0343) | 1,431 pairs, C(54,2) | Bonferroni 0.05/1431 = 3.49e-5 |
| 5 | Spectral | No hidden periodicity in a number's hit sequence | FFT per number | none reported |
| 6 | Mutual information | Consecutive draws are independent | Lag-1 on sum/max/min/odd-count, kNN estimator | 1,000-shuffle permutation |
| 7 | Survival analysis | "Time since last appearance" cannot be modeled better than chance | Cox proportional hazards (25 features) [6], Random Survival Forest (100 trees) [26], exponential baseline | C-index on held-out 400 draws |
| 8 | Hawkes process | No self-excitation between numbers | Discrete-time multivariate, 54 dims, L1 logistic regression (LASSO, C=0.1), decay features from past 3 draws | Coefficients zeroed by L1 penalty |
| 9 | Sequential pattern mining | No sub-sequence repeats more than chance | PrefixSpan on raw / decade / feature representations | Fisher exact + BH-FDR q=0.05, 10-shuffle baseline |
| 10 | Weather correlation | Austin weather has no relationship to draw outcomes | Pearson (108), point-biserial (648), Spearman (108), MI-permutation (732), logistic omnibus (54), OLS F (4), Granger (48); ~1,700 tests total | BH-FDR q=0.05 |
| 11 | Topological data analysis | (attempted) persistent homology on draw-vector distances | Vietoris-Rips on Jaccard distance | not executed: `giotto-tda` was incompatible with the installed `scikit-learn` |
| 12 | Matrix Profile | No recurring motif or anomalous discord in a number's hit sequence | STUMPY, 54 binary sequences x 2,308 draws, window m=12, z-normalized Euclidean distance, exclusion zone m/2 [12][13] | BH q=0.05 over 108 tests |
| 13 | Change point detection | No abrupt shift in per-number frequency over time | PELT (`ruptures`), 46 features x 2,258 sliding 50-draw windows, RBF cost, penalty beta = log(n)*d*sigma^2 (d=46, multiplier 2.0), min segment 100 draws [14][15][16] | 100-shuffle permutation |
| 14 | Contrast mining | Weather / calendar features do not differ between "number appeared" and "did not" | Mann-Whitney U / Fisher exact, 12 weather + 8 temporal features, min support 100, effect threshold Cohen's d > 0.2 or Cramer's V > 0.1, 1,080 tests [17] | BH-FDR q=0.05 |
| 15 | Transfer entropy | No directed information flow between number pairs | `pyinform`, lag k=1, 2,862 pairwise tests, Granger VAR(1) cross-check [18][20] | alpha=0.001, then BH-FDR q<0.05 |

---

## What we found

### Results by method

| # | Method | We saw | Outcome after correction |
|---|--------|--------|---------------------------|
| 1 | Frequency | chi-square 63.18, df 53, p=0.160; #26 at 472 appearances (z=+3.16), #4 at 462 (z=+2.64) | null; neither outlier survives Bonferroni |
| 2 | Gap / recurrence | #14 mean gap 8.6 vs 9.0 expected, KS D=0.112, p=0.000090 | survives Bonferroni within this module (0.000090 < 0.000926); no downstream predictive value (see Survival, Hawkes) |
| 3 | Runs | #14 z=+2.80 (p=0.005), #6 z=+2.24 (p=0.025), #47 z=-2.02 (p=0.043) | null; none survive Bonferroni |
| 4 | Pairwise co-occurrence | pair (14, 26): 71 occurrences vs 38.8 expected, z=+4.05, p=0.000211 | null; fails the 3.49e-5 threshold |
| 5 | Spectral | no notable FFT peaks reported | null |
| 6 | Mutual information | sum I=0.0131 (p=0.88), odd-count I=0.0022 (p=0.59), max I=0.0262 (p=0.96), min I=0.0111 (p=0.47) | null |
| 7 | Survival analysis | C-index: naive 0.5000, Cox 0.5081, Random Survival Forest 0.5195; Brier score worse than baseline (0.1148 vs 0.0988) | null; effectively coin-flip discrimination |
| 8 | Hawkes process | all 54x54 excitation coefficients driven to 0.0 by the L1 penalty; log-likelihood unchanged from baseline (-0.3492) | null |
| 9 | Sequential pattern mining | 351 patterns tested; real mean lift 1.040 vs shuffled baseline 1.515; 1 pattern above shuffled p95 (expected ~17.5) | null; real patterns were weaker than shuffled ones |
| 10 | Weather correlation | 89 of 1,698 tests raw p<0.05 (expected ~85); strongest raw hits: humidity vs draw_sum r=-0.0496 (p=0.025), pressure_range vs draw_range r=-0.0476 (p=0.031); RF regression CV R^2=-0.0218; RF classification mean AUC 0.5043 | null; 0 survive BH-FDR |
| 11 | TDA | not run | not run |
| 12 | Matrix Profile | best motif: #23, 2006 vs 2016 windows (2,080 draws apart), distance 0.52, p=0.005 raw, q=0.27. Best discord: #41, Fall 2018, 47-draw absence vs ~9 expected, distance 3.1, p=0.00004 raw, q=0.022 | motif fails BH-FDR. The discord's q-value (0.022) is below the 0.05 cutoff, not above it (see the note below). |
| 13 | Change point detection | 10 change points detected (2007 through 2025, the last flagged as a boundary artifact); longest segment 2012-04-18 to 2018-02-07 (900 draws), within-segment chi-square p=0.56, between-segment chi-square p=0.13; permutation: real 10 vs shuffled 8.2 +/- 3.1, p=0.28 | null; 10 change points is unremarkable |
| 14 | Contrast mining | 62 of 1,080 tests raw p<0.05 (expected ~54); best case #13 vs humidity, p=0.0027, 58.3% vs 61.7%, Cohen's d=0.18, q=0.15; permutation placed it at the 28th percentile; bootstrap CI for d spans [-0.05, 0.41] | null; 0 survive BH-FDR |
| 15 | Transfer entropy | 304 of 2,862 pairs raw p<0.001 (expected ~2.9); Granger agreement 122/304 (40%); top TE values 0.0034-0.0042 bits vs shuffled 0.0031 +/- 0.0018 bits (t-test p=0.83) | null; 0 survive BH-FDR |

A note on three places where the prior draft was internally inconsistent, and how we resolved them here:

- **Bonferroni threshold for gap and runs tests.** The methods section of the prior draft states the threshold as 0.05/54 = 0.000926, and that is the formula actually used (54 individual number-level tests, family-wise alpha 0.05). The results section of the prior draft instead applied 0.000185 in two places (gap test, runs test) without deriving it. We use 0.000926 throughout, per the stated formula, and flag the 0.000185 figure as the prior draft's unexplained substitution. The practical conclusions do not change either way: #14's gap p-value (0.000090) is below both thresholds, and the runs test z-scores (p >= 0.005) fail both.
- **Matrix Profile discord q-value.** The prior draft described the #41 discord's q-value of 0.022 as "barely above 0.05." That is wrong: 0.022 is below 0.05. Read at face value, a q-value below 0.05 would nominally pass a standard BH-FDR cutoff, which conflicts with the prior draft's summary that "0 discords survive." We do not have the original code to determine which statement is correct, so we are not resolving it here; we flag it as an open item under "What this does not show" and "Open questions."
- **Code availability.** The prior draft's final section claimed this repository "includes Python scripts for all 11 analytical approaches." That is not true of this repository as it exists. See "Where the code is."

### Era contamination

Before restricting to the post-April-2006 dataset, we ran the frequency and runs tests on the full 3,704-draw history, which mixes the pre-2006 pick-6-of-50 format with the post-2006 pick-6-of-54 format. That contaminated run produced chi-square 263.47 (p near 0) on the frequency test and runs z-scores beyond +/-20. Those numbers look dramatic, but they are an artifact: pooling two games with different odds (1-in-50 vs 1-in-54 per position) as if they were one game manufactures a huge, fake "non-uniformity" signal. Restricting the analysis to the single post-2006 era removes this and gives the honest null result reported above. Treat this as a general caution for any longitudinal dataset that spans a procedural change: check for era boundaries before running frequency-based tests across the whole history.

### Totals and statistical power

We ran roughly 3,650 individual hypothesis tests across the 15 methods (the per-method counts as listed sum closer to 3,696; we have not reconciled the discrepancy). At alpha=0.05 uncorrected, chance alone predicts about 183 false positives among that many tests. We saw roughly that: 2 raw hits in frequency (expected ~2.7), 89 in weather (expected ~85), 62 in contrast mining (expected ~54), 304 in transfer entropy (expected ~2.9, at the stricter alpha=0.001 used there). We think this means the corrections were doing real work: without them, this analysis would have reported dozens of fake findings.

A power calculation for the frequency test: with 2,308 draws, a 1% deviation from the expected per-number rate has about 12% power to detect at alpha=0.05; a 5% deviation has about 60% power; a 10% deviation has about 95% power. We think this means our null result does not prove the machine is perfectly random. It shows that any bias present is small enough to be undetectable at this sample size, and (per the survival and Hawkes results above) too small to translate into a usable prediction even if it exists.

---

## What this does not show

- It does not show the machine is perfectly random. It shows that no bias large enough to detect with 2,308 draws, and no bias that would help predict a future draw, was found.
- It does not rule out a slow, multi-decade equipment drift. The 19.9-year window may be too short to see century-scale wear.
- It is specific to the Texas Lottery. We did not test other states' machines or games.
- The survival and Hawkes models used 25 hand-built features chosen by the original analysis; different features might behave differently. We have no way to check that choice against code right now.
- Topological data analysis (method 11) did not run, due to a library version conflict, so that method contributes nothing either way.
- 25,000+ Pick 3 draws were downloaded during this phase but were not analyzed here; that became separate later work outside this paper's scope.
- The Matrix Profile discord's q=0.022 vs. the "0 discords survive" summary is an unresolved inconsistency in the original notes (see "What we found"). We are reporting it, not resolving it.
- Every number in this paper is carried from the original analysis notes, not re-derived from code or data in this repository. Treat all of it as provisional until the code is added and re-run (see "Where the code is").

---

## Where the code is

The Phase 1 analysis code and the Lotto Texas and Austin weather datasets are not in this repository. They live on the laptop where the original analysis was run and have not been copied over.

We checked this repository directly before writing this section. Searching all tracked files for terms tied to the methods above (Hawkes, Matrix Profile, STUMPY, transfer entropy, survival analysis, Kaplan-Meier, Cox proportional hazards) turns up only this paper itself; no Python source matches. The repository's `data/` directory contains Pick 3 draw files used by later phases of this project, and no Lotto Texas file of any kind. The only Python scripts in the repository outside the later-phase directories are two small CSV joiners for Pick 3 data, unrelated to Phase 1.

The prior draft of this paper stated that the repository "includes Python scripts for all 11 analytical approaches." That statement is false as the repository is currently tracked, and we removed it in this rewrite.

---

## Open questions / next phase

- Add the Phase 1 code and raw datasets (Lotto Texas draws, Austin weather) to this repository, so every number in this paper can be checked against files instead of carried from notes.
- Once the code is available, resolve the Matrix Profile discord labeling (q=0.022) against the "0 discords survive" summary; right now we do not know which statement the original analysis actually intended.
- Fix the `giotto-tda` / `scikit-learn` version conflict and run the topological data analysis that method 11 could not complete.
- Extend the comparison to other states' lottery machines (California, Florida, New York) to see whether the null result is specific to this machine or general to the format.
- Move from post-hoc testing to a real-time anomaly monitor (sequential probability ratio test, CUSUM charts) for ongoing integrity checks rather than a one-time analysis.
- The 15-method battery here (frequency, gap, runs, co-occurrence, spectral, mutual information, survival, Hawkes, sequential patterns, weather/contrast correlation, Matrix Profile, change points, transfer entropy) is not specific to Lotto Texas. Nothing in the pipeline assumes a 54-ball pick-6 game; it could run against any physical RNG that produces a long enough log of discrete outcomes. We note this as a reusable audit recipe, not as a finished product.
- Pick 3 analysis, mentioned but not run in this phase, became its own line of work in a later phase of this project and is outside the scope of this paper.

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
