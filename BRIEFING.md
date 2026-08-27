# LottAI Project Briefing
**Date:** 2026-04-20
**Machine:** beastmode (Linux, /mnt/beastmode/lottai)
**Repo:** https://github.com/mmundy3832/LottAI
**Read this before touching any code.**

---

## What This Project Is

Texas Lottery Pick 3 draw data as a test bed for a 25-year-old hypothesis about high-dimensional search space exploration. This is not primarily a lottery project — it is a rigorous test of corner-seeding methodology in a 55-75D config space, using Pick 3 as a domain with a real scorable metric.

The core bet: **straight $1 ticket pays $500**. If we can push the true hit rate above 1/500 = 0.2% per draw, we have positive EV. The question is whether any exploitable signal exists and whether the methodology finds it.

**Expected result: null.** The lottery is expected to be pure random. A well-designed null result is a valid scientific finding.

---

## Physical Draw Facts

- **4 draws per day**: Morning (10am), Day (12:27pm), Evening (6pm), Night (10:12pm) — Mon-Sat, NO Sundays
- **Single studio**: George H.W. Bush Building, 1801 N. Congress Ave., Austin TX
- **Same machine pool**: Primary + alternate machine, rotated monthly on a fixed schedule
- **Ball set selection rule**: Prior day's results determine which ball sets load the next day — a deterministic, known mapping
- **Three independent chambers**: One per digit position. Mechanical bias in chamber N affects only digit N.
- **Pre-test data published**: Texas Lottery publishes machine ID and ball set IDs for every draw

---

## Project Structure

```
/mnt/beastmode/lottai/
├── data/
│   ├── pick3morning.csv       training data, frozen at 2026-02-13
│   ├── pick3day.csv
│   ├── pick3evening.csv
│   ├── pick3night.csv
│   ├── pick3_combined.csv     all 4 draw times interleaved, 15,572 rows
│   ├── pick3*_live.csv        LIVE DATA -- truly unseen, never in models, update before live_eval
│   └── pretest/               machine + ball set assignment CSVs (one per draw time)
│
│   *** LIVE FILE UPDATE ***
│   Update all four _live files before running live_eval or predict.
│   Download from Texas Lottery and extract rows after 2026-02-13.
│     Morning: https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3morning.csv
│     Day:     https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3day.csv
│     Evening: https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3evening.csv
│     Night:   https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3night.csv
│
├── papers/                    whitepaper.md, WHITEPAPER_PHASE2.md, WHITEPAPER_NOTES.md
├── plans/                     PHASE2_PLAN.md, PHASE3_PLAN.md, phase2-feature-list.json
├── phase3/                    Phase 3 work (empty, self-contained by design -- copy not import)
├── requirements.txt
├── .env                       OLLAMA_API_KEY (never committed)
│
└── autoresearch/
    ├── prepare_v3.py          DATA MODULE -- combined dataset, all 4 draw times, 15,572 rows
    ├── template_engine.py     renders config dicts to Python experiment code
    ├── config_space.py        valid config parameter ranges + bootstrap seeds
    ├── config_validator.py    validates configs before running
    ├── ga_ops.py              mutation, crossover, LLM exploration operators
    ├── runner_v4.py           MAIN RUNNER (current) -- GA + LLM + random search loop
    ├── live_eval_v2.py        evaluate experiments against unseen live data
    └── results/
        ├── experiments_v4.jsonl        ALL experiment results (1000+ entries, append-only ledger)
        ├── lag_analysis_output.json
        └── live_eval_results_v2.jsonl  live evaluation scores
```

---

## The Autoresearch Framework (Karpathy Loop)

LottAI is a direct implementation of the Karpathy Loop pattern:
- **Claude** = harness/meta-agent (selects experiments, analyzes results)
- **MiniMax M2.7 via Ollama** = experimenter (proposes novel configs)
- **program_v3.md** = program.md (LLM prompt template)
- **optimal_ev** = the single scorable metric
- **900s timeout** = fixed time budget per experiment

Each experiment is a config dict specifying:
1. **feature_sets**: which feature groups to use
2. **pipeline**: ordered transformation stages
3. **models**: per-digit classifiers with hyperparameters and weights

The template engine renders the config to Python code. Code trains per-digit models (d1, d2, d3 as 10-class classifiers). Three digit probability vectors multiply to form a 1000-combo probability matrix, evaluated against the held-out val set.

### How Predictions Work
```
P(combo "xyz") = P(d1=x) * P(d2=y) * P(d3=z)
```
Sort descending, buy top-K tickets.

---

## Feature Sets (in prepare_v3.py)

| Name | Dims | What it captures |
|------|------|-----------------|
| basic | 11 | Properties of previous draw (sum, repeating digits, odd/even, high/low) |
| recency | 121 | Per-digit per-position frequency in last 10/25/50/100 draws + combo gap |
| gaps | 40 | Draws since each digit last appeared in each position |
| positional | 33 | Per-position digit frequency over last 50 draws + position correlations |
| temporal | 20 | Day-of-week one-hot (7) + month one-hot (12) + year normalized (1) |
| momentum | 7 | Rolling chi-squared and entropy per position + consecutive overlap |
| equipment | 8 | Machine ID + ball set IDs (gated behind --equipment flag, default off) |

**draw_time removed** — confirmed noise, no signal above equipment wear features.
**equipment off by default** — signal tested and found to be noise at current data size.

---

## Pipeline Stages

- **custom_interact**: pairwise products/ratios of top-N and bottom-N features
- **select**: ExtraTreesClassifier-based feature selection (SelectFromModel); zero-feature fallback to top-10 by importance if threshold is too aggressive
- **poly**: PolynomialFeatures expansion, keep top-K by variance; degree>2 requires select stage before it
- **scale**: StandardScaler / RobustScaler / QuantileTransformer / MinMaxScaler

## Model Types
`et` (ExtraTrees), `rf` (RandomForest), `xgb` (XGBoost), `lgb` (LightGBM), `hgb` (HistGradientBoosting), `lr` (LogisticRegression)

---

## The Metric System

### Primary Metric: `optimal_ev`
```
optimal_ev = max over k=1..20 of: (top_k_hit_rate * $500 - k)
```
Random baseline: $500 * 0.001 - 1 = -$0.50/draw.

### Secondary Metrics
- `mean_rank`: average rank of actual combo in sorted predictions (baseline = 500.5, lower is better)
- `box_optimal_ev`: same but for box bets
- `live_optimal_ev`: optimal_ev on truly unseen live draws (the honest number)

### Data Splits (fixed, time-ordered, no shuffling)
```
Train: rows 0-10899   (2013-09-09 to ~2022-05)   model training
Val:   rows 10900-12899                            GA fitness function
Test:  rows 12900-14571                            held out, used rarely
Live:  pick3*_live.csv, 2026-02-14 to present     truly unseen, sacred
```
Combined dataset: 15,572 rows across all 4 draw times.

**Live data is sacred. Never routes to MiniMax or any external LLM. Used only for evaluation.**

---

## Runner Commands

```bash
# Activate venv first -- always
source /mnt/beastmode/lottai/.venv/bin/activate
cd /mnt/beastmode/lottai/autoresearch

# Resume GA with defaults (llm-frac 0.20, random-frac 0.10)
python -u runner_v4.py --resume 2>&1 | tee -a autoresearch_v4.log

# Pure random exploration (escape local optima)
python -u runner_v4.py --resume --llm-frac 0.0 --random-frac 1.0 --max N 2>&1 | tee -a autoresearch_v4.log

# Resume GA with elevated random injection (recommended post-random-batch)
python -u runner_v4.py --resume --random-frac 0.25 2>&1 | tee -a autoresearch_v4.log
```

Key flags:
- `--resume`: load experiments_v4.jsonl and continue from current count
- `--max N`: stop at N total experiments
- `--llm-frac F`: fraction of experiments using MiniMax (default 0.20)
- `--random-frac F`: fraction fully random (default 0.10)
- `--equipment`: enable equipment features (off by default)

Model: `DEFAULT_MODEL = "minimax-m2.7:cloud"` in runner_v4.py line 49.

---

## Current State (2026-04-20)

### 1000+ Experiments, GA Ceiling at $8.04

- **GA ceiling**: $8.04 optimal_ev, experiments #346/#358/#390 (clones, ga_mutation lineage)
- **Fresh crossover**: exp#880 at $7.69 from bootstrap lineage #196 × #122 (different gene pool)
- **Random batch running**: 500 pure-random experiments in progress to break out of local attractor

### Karpathy Loop Insight
The val/live divergence observed (GA gaming val metric) matches the metric gaming failure mode described in the Karpathy Loop literature — arrived at the right defense by reasoning from first principles before understanding the theoretical framing.

### High-D Random Pool Insight (April 2026)
In 55-75D space, a fixed random-frac of 0.10 is inadequate. The volume near corners grows exponentially with dimensions — random exploration needs to scale with dimensionality. Plan: run 500 pure-random experiments, then resume GA with random-frac 0.25.

### Live Eval (from live_eval_v2, 286 evaluations against 200 unseen draws)
- Top live performer: exp#796, +$13.50 live EV (buy 19)
- 95.4% of experiments show positive live EV

---

## Key Design Decisions (Don't Revisit Without New Evidence)

- **Per-digit models, not combo model**: factored representation generalizes better
- **Straight bets, not box**: model has real positional signal, box discards it
- **optimal_ev as fitness**: more useful than mean_rank or top_k_hit_rate alone
- **Live eval as truth**: val scores are compromised by 1000+ experiments of GA optimization
- **draw_time removed**: tested and confirmed noise
- **Equipment off by default**: tested and confirmed noise at current data size
- **No machine-specific models**: not enough data per machine to train reliably
- **Corner seeding mandatory**: random injection into gene pool is not optional — it is the hypothesis being tested
- **random-frac 0.10 → 0.25**: higher random injection defensible in high-D spaces
- **ET calibration = noise**: calibrating ExtraTrees does not improve live performance
- **VALID_POLY_DEGREES = [2, 3]**: degree=3 allowed only when select precedes poly in pipeline

---

## Environment

```bash
# Setup (already done on beastmode)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Ollama (already installed, model already pulled)
# OLLAMA_API_KEY in .env at project root

# Feature cache auto-builds on first run (~5 min)
# Cache: autoresearch/.feature_cache_v3/ (gitignored)
```

---

*Updated 2026-04-20*
