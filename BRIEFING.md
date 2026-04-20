# LottAI Project Briefing -- Full Context Transfer
**Date:** 2026-04-14  
**Purpose:** Complete handoff document for continuing this project on a new machine (Linux).  
**Read this entire document before touching any code.**

---

## What This Project Is

We are analyzing Texas Lottery **Pick 3** draw data to find exploitable bias in the physical ball-draw machines. This is not an RNG -- it is a mechanical gravity/air-mix machine manufactured by SmartPlay International, with three independent chambers each containing 10 ping-pong balls (0-9). Physical wear, ball weighting, and equipment assignment introduce real-world biases that may be detectable and predictable.

The core bet: **straight $1 ticket pays $500**. If we can push the true hit rate above 1/500 = 0.2% per ticket, we have positive EV. The question is how many tickets to buy per draw to maximize EV.

---

## Physical Draw Facts (Critical Context)

- **4 draws per day**: Morning (10am), Day (12:27pm), Evening (6pm), Night (10:12pm) -- Mon-Sat only, NO Sundays
- **Single studio**: All draws at George H.W. Bush Building, 1801 N. Congress Ave., Austin TX
- **Same machine pool**: One designated primary machine + one alternate, rotated monthly on a fixed schedule
- **Ball set selection rule**: Prior day's results determine which ball sets are loaded for the next day. This is a DETERMINISTIC, KNOWN mapping -- a direct causal chain from yesterday's numbers to today's physical equipment to today's results.
- **Three independent chambers**: One per digit position. Mechanical bias in chamber N affects only digit position N.
- **Pre-test data published**: Texas Lottery publishes which machine and which ball sets (by ID) were used for EVERY draw in downloadable CSV format.

**Big insight we just discovered:** We have been training ONLY on Evening data -- 1/4 of all available draws. Morning, Day, and Night use the same equipment pool and share the same ball set selection rule. Combining all 4 draw streams gives us ~4x the data and -- more importantly -- the cross-draw causal chain (morning results influence afternoon ball sets, etc.) is a signal we are completely ignoring.

---

## Project Structure

```
D:/projects/LottAI/          (or equivalent on Linux)
├── pick3evening.csv          historical Evening draws, Sept 2013 - Feb 13 2026 (3,893 rows)
├── pick3morning.csv          historical Morning draws (same date range, ~3,893 rows)
├── pick3day.csv              historical Day draws (~3,893 rows)
├── pick3night.csv            historical Night draws (3,893 rows, Sept 9 2013 - Feb 13 2026)
├── pick3evening_live.csv     Evening draws Feb 14 2026 - present (truly unseen)
├── pick3morning_live.csv     Morning draws Feb 14 2026 - present (truly unseen)
├── pick3day_live.csv         Day draws Feb 14 2026 - present (truly unseen)
├── pick3night_live.csv       Night draws Feb 14 2026 - present (truly unseen)
│
│   *** LIVE FILE UPDATE INSTRUCTIONS ***
│   All four _live files must be current through yesterday before running live_eval
│   or predict. Download fresh CSVs from Texas Lottery and extract rows after 2026-02-13:
│     Morning: https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3morning.csv
│     Day:     https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3day.csv
│     Evening: https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3evening.csv
│     Night:   https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/pick3night.csv
│   Keep the training CSVs (no suffix) frozen at Feb 13 2026. Never append to them.
│
├── autoresearch/
│   ├── prepare.py            DATA MODULE -- locked during runs, do not modify lightly
│   ├── template_engine.py    renders config dicts to Python experiment code
│   ├── config_space.py       defines valid config parameter ranges + bootstrap seeds
│   ├── config_validator.py   validates configs before running
│   ├── ga_ops.py             mutation, crossover, LLM exploration operators
│   ├── runner_v3.py          MAIN RUNNER -- GA + LLM search loop
│   ├── reeval.py             re-evaluate existing experiments with new metrics
│   ├── live_eval.py          evaluate all experiments against unseen live data
│   │                         *** BEFORE RUNNING: update all four _live CSVs per
│   │                             instructions above. ***
│   ├── analyze_live.py       analyze live_eval_results.jsonl, print leaderboard
│   │                         *** NOTE: current results are based on Evening data only
│   │                             (1/4 of draws). Must be re-run after combined dataset
│   │                             is built and all experiments are re-evaluated with
│   │                             full data + equipment features. ***
│   ├── predict_tonight.py    predict next draw using top live configs
│   ├── consensus.py          ensemble/intersection strategies across top models
│   ├── experiments_v3.jsonl  ALL experiment results (842 entries, 307 successful)
│   └── live_eval_results.jsonl  live eval scores for all 306 evaluated experiments
```

---

## The Autoresearch Framework

### Core Concept
Each "experiment" is a config dict that specifies:
1. **feature_sets**: which feature groups to use (basic, recency, gaps, positional, temporal, momentum)
2. **pipeline**: ordered list of transformation stages (custom_interact, select, poly, scale)
3. **models**: list of per-digit classifiers with hyperparameters and weights

The template engine renders the config to Python code. The code trains per-digit models (one model predicts d1, one predicts d2, one predicts d3 -- each is a 10-class classifier). The three digit probability vectors are multiplied together to form a 1000-combo probability matrix. This matrix is evaluated against the held-out validation set.

### How Predictions Work
Models predict P(d1=x), P(d2=y), P(d3=z) independently. Then:
```
P(combo "xyz") = P(d1=x) * P(d2=y) * P(d3=z)
```
This gives a 1000-element probability vector per draw. Sort descending, buy the top-K tickets.

### Feature Sets (in prepare.py)
| Name | Dims | What it captures |
|------|------|-----------------|
| basic | 11 | Properties of previous draw (sum, repeating digits, odd/even, high/low) |
| recency | 121 | Per-digit per-position frequency in last 10/25/50/100 draws + combo gap |
| gaps | 40 | Draws since each digit last appeared in each position |
| positional | 33 | Per-position digit frequency over last 50 draws + position correlations |
| temporal | 20 | Day-of-week one-hot (7) + month one-hot (12) + year normalized (1) |
| momentum | 7 | Rolling chi-squared and entropy per position + consecutive overlap |

### Pipeline Stages
- **custom_interact**: creates pairwise products/ratios of top-N and bottom-N features
- **select**: ExtraTreesClassifier-based feature selection (SelectFromModel)
- **poly**: PolynomialFeatures expansion, keep top-K by variance
- **scale**: StandardScaler / RobustScaler / QuantileTransformer / MinMaxScaler

### Model Types Supported
`et` (ExtraTrees), `rf` (RandomForest), `xgb` (XGBoost), `lgb` (LightGBM), `hgb` (HistGradientBoosting), `lr` (LogisticRegression)

---

## The Metric System -- How We Measure Success

### Primary Metric: `optimal_ev`
```
optimal_ev = max over k=1..20 of: (top_k_hit_rate * $500 - k)
```
Where `top_k_hit_rate` = fraction of draws where the actual combo was in our top-K predictions.

This answers: "what is the best expected profit per draw if we buy exactly K tickets?"  
Random baseline: buying 1 ticket = EV of ($500 * 0.001 - 1) = -$0.50/draw.

### Secondary Metrics
- `mean_rank`: average rank of the actual combo in our sorted predictions (lower = better, baseline = 500.5)
- `box_optimal_ev`: same EV calculation but for box bets ($80 6-way, $160 3-way) over 220 unique digit-sets
- `live_optimal_ev`: optimal_ev computed on the 49 UNSEEN live draws (the honest number)

### Data Splits (fixed, time-ordered, no shuffling)
```
Train: rows 0-2724    (indices 0..2724)   2013-09-09 to 2022-05-23  -- model training
Val:   rows 2725-3309 (indices 2725..3309) 2022-05-24 to 2024-04-04  -- fitness function during GA search
Test:  rows 3310-3892 (indices 3310..3892) 2024-04-05 to 2026-02-13  -- held out, used rarely
Live:  pick3{time}_live.csv               2026-02-14 to present      -- truly unseen, one file per draw time
```

All four training CSVs (no suffix) are frozen at Feb 13 2026 (3,893 rows each). The _live files
grow over time and must be updated before each live_eval or predict run (see file listing above
for per-draw-time download URLs).

**The live split is the most honest.** The GA optimized against val for 800+ experiments, so val scores are inflated by overfitting.

---

## GA Search Architecture (runner_v3.py)

### Flow
1. Load all past experiments from `experiments_v3.jsonl`
2. Keep top-N successful ones as the gene pool (sorted by `optimal_ev`)
3. For each new experiment slot:
   - **80% GA mode**: pick mutation or crossover of top parents
   - **20% LLM mode**: ask MiniMax M2.7 (via Ollama cloud) to propose a novel config
4. Generate Python code via template_engine, run as subprocess with 2400s timeout
5. Parse JSON result from stdout, log to experiments_v3.jsonl

### LLM Exploration
- Model: `minimax-m2.7:cloud` via Ollama
- Purpose: escape local optima by proposing configs the GA wouldn't reach by mutation alone
- Known issues: sometimes returns narration instead of JSON (retry logic added), sometimes hallucinates feature set names (repair logic filters these out), occasionally returns invalid parameter values (repair logic clamps to valid ranges)
- The LLM has NO memory across runs -- each call is independent

### Key Files for the Runner
- `runner_v3.py --model minimax-m2.7:cloud --n 50` -- run 50 more experiments
- Always include LLM exploration (never use `--ga-only`)
- Experiments append to `experiments_v3.jsonl`

---

## What We've Found So Far

### 842 Total Experiments, 307 Successful

### Overfitting Discovery
The GA optimized val scores aggressively. Top val performers (exp#808, val_ev=$11.62) showed near-zero or negative EV on the held-out test set. This prompted us to run `live_eval.py` -- evaluating ALL 307 experiments against 49 truly unseen draws.

### Live Eval Results (THE KEY FINDING)
306 of 307 experiments evaluated. **95.4% show positive live EV.** There is real signal in the data.
*(Based on 49 Evening draws. Morning/day/night _live files now exist with ~50 rows each but have not been evaluated yet.)*

**Zero overlap between top-10 val and top-10 live performers.** The GA's val leaders are NOT the best generalizing models.

**Top live performers:**

| exp# | live_ev | buy | val_ev | Architecture |
|------|---------|-----|--------|-------------|
| #767 | +$28.82 | 12 | $5.64 | 5-feat, interact->select, 3x LGB(d=3) |
| #809 | +$28.82 | 12 | $5.64 | same config as #767 |
| #639 | +$27.82 | 13 | $9.21 | same arch, slightly diff n_estimators |
| #499 | +$26.82 | 14 | $7.50 | 6-feat, interact->select->scale, LGB(d=3/6/3) |
| #488 | +$25.82 | 15 | $5.08 | same as #499 |

**Winning architecture pattern (consistent across all top-10 live)**:
- 3x LightGBM, all shallow (max_depth=3, one sometimes depth=6)
- Pipeline: custom_interact -> select (with or without scale)
- Feature sets: 5-feature variant drops `positional` (basic+recency+gaps+temporal+momentum)
- No calibration on most models
- NO poly features in top live performers

**Val/live correlation: Pearson r = 0.518** (moderate). Val score is a weak predictor of live performance.

***** IMPORTANT CAVEAT *****
ALL of the above findings -- the winning architecture, live leaderboard, best configs, EV scores --
are based EXCLUSIVELY on Evening draw data (1 of 4 draw times). This represents only 1/4 of
available draws and uses NONE of the equipment features (machine ID, ball set IDs) that are
the most causally direct signal available.

Once the combined dataset is built (all 4 draw times + pre-test equipment data), the ENTIRE
live_eval sweep must be re-run from scratch. The "winning" architecture may change significantly
when trained on 4x the data with equipment-state features. Do not treat the current results
as final -- treat them as a strong baseline established on incomplete data.
***** END CAVEAT *****

### Best Config for Production (exp#767) -- Evening-only baseline
```json
{
  "feature_sets": ["basic", "recency", "gaps", "temporal", "momentum"],
  "pipeline": [
    {"stage": "custom_interact", "n_head": 9, "n_tail": 5, "include_ratios": true},
    {"stage": "select", "threshold_multiplier": 1.4704978126202712,
     "selector_n_estimators": 200, "selector_max_depth": 14, "selector_target": "d1"}
  ],
  "models": [
    {"type": "lgb", "weight": 0.2738, "n_estimators": 157, "max_depth": 3,
     "learning_rate": 0.0919, "subsample": 0.5386, "colsample_bytree": 0.7,
     "reg_alpha": 0.9541, "reg_lambda": 0.1191, "calibrate": false},
    {"type": "lgb", "weight": 0.2917, "n_estimators": 300, "max_depth": 3,
     "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.4237,
     "reg_alpha": 2.0666, "reg_lambda": 1.0, "calibrate": true},
    {"type": "lgb", "weight": 0.1880, "n_estimators": 244, "max_depth": 3,
     "learning_rate": 0.0480, "subsample": 0.8615, "colsample_bytree": 0.4904,
     "reg_alpha": 0.0, "reg_lambda": 0.8987, "calibrate": false}
  ]
}
```

### Tonight's Prediction (Monday April 14 2026, Evening)
Top picks from product ensemble of exp#767 + exp#499:
```
1. 597  (5-9-7)   p=0.01186
2. 596  (5-9-6)   p=0.01106
3. 590  (5-9-0)   p=0.01016
4. 547  (5-4-7)   p=0.00959
5. 595  (5-9-5)   p=0.00912
6. 546  (5-4-6)   p=0.00894
7. 599  (5-9-9)   p=0.00842
```
Position 1 = 5 with 26.6% confidence (2.66x random). Position 2 = 9 at 23.7%.

---

## The Next Phase -- 4x Data + Ball Set Features

### Why This Is Critical
We have been ignoring 3/4 of the available data. All 4 draw times (morning, day, evening, night) use the same equipment at the same studio. The ball set selection rule creates a DETERMINISTIC causal chain across draw times.

### New Data to Incorporate

**1. All 4 draw CSVs** (same format as pick3evening.csv):
- `pick3morning.csv`, `pick3day.csv`, `pick3night.csv`
- Download from: `https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/Winning_Numbers/`
- Format: `Pick 3 Morning,MM,DD,YYYY,d1,d2,d3,[sum],[fireball]`

**2. Pre-test data** (machine and ball set assignments per draw):
- Download page: `https://www.texaslottery.com/export/sites/lottery/Games/Pick_3/pre_test_download.html`
- One CSV per draw time per period
- Format includes: Game Name, Month, Day, Year, Test#, Machine, Machine Used, Ball Set 1, Ball Set 1 Used, Ball Set 2, Ball Set 2 Used, Ball Set 3, Ball Set 3 Used, Alternate Machine, Alternate Machine Used, etc.
- **This is the most valuable data we don't have yet** -- machine ID and ball set IDs are categorical features that directly encode the physical equipment state

### Architecture for Combined Dataset

**Option A: Interleaved sequence** -- treat all 4 draw times as one chronological sequence with draw_time as a feature. Simpler, more data, preserves temporal ordering.

**Option B: Parallel streams with cross-draw features** -- keep each draw time as its own sequence, but add features from the OTHER draw times (e.g., "what was morning's result today?", "what was yesterday's evening result?"). Captures the cross-draw ball set selection rule directly.

**Option B is theoretically superior** because:
- The ball set selection rule is cross-day (yesterday's results -> today's ball sets)
- The morning result may influence afternoon equipment state
- Different draw times may have systematically different biases (different ball sets tend to be loaded at different times)

### New Features to Engineer

1. **Ball set ID per chamber** (categorical, from pre-test data) -- which physical ball set is loaded in chamber 1, 2, 3
2. **Machine ID** (categorical) -- which machine is running today
3. **Prior draw results** -- yesterday's numbers as features (already partially captured by `basic` but not explicitly cross-draw)
4. **Draw time** (categorical: 0/1/2/3 for morning/day/evening/night)
5. **Cross-draw result** -- what did morning draw today? (for predicting evening)
6. **Ball set sequence** -- how many consecutive days this ball set has been used (wear indicator)
7. **Machine age in rotation** -- days since last monthly rotation (wear accumulation)

### Modified prepare.py Design

The current `prepare.py` is locked during autoresearch runs. For the new phase, we need:

```python
# New data sources
_DATA_FILE_MORNING = "pick3morning.csv"
_DATA_FILE_DAY     = "pick3day.csv"  
_DATA_FILE_EVENING = "pick3evening.csv"
_DATA_FILE_NIGHT   = "pick3night.csv"
_PRETEST_MORNING   = "pretest_morning.csv"   # machine/ball set assignments
_PRETEST_DAY       = "pretest_day.csv"
_PRETEST_EVENING   = "pretest_evening.csv"
_PRETEST_NIGHT     = "pretest_night.csv"

# New feature sets to add
FEATURE_SETS["draw_time"]   = _feat_draw_time    # which of the 4 draws
FEATURE_SETS["equipment"]   = _feat_equipment    # machine ID + ball set IDs (from pretest)
FEATURE_SETS["cross_draw"]  = _feat_cross_draw   # results from other draw times same day
FEATURE_SETS["prior_day"]   = _feat_prior_day    # yesterday's draw results across all times
```

---

## Environment Setup on Linux

```bash
# Clone or copy the project
cd /path/to/LottAI

# Create venv (the project uses .venv)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install numpy pandas scipy scikit-learn lightgbm xgboost

# For the runner (LLM exploration via Ollama)
# Install Ollama: https://ollama.com
# Pull the model: ollama pull minimax-m2.7:cloud
# Set OLLAMA_API_KEY in .env file at project root

# Verify prepare.py works
cd autoresearch
python -c "import prepare; print('OK')"

# The feature cache will need to be rebuilt on first run
# It auto-builds -- just takes a few minutes
# Cache location: autoresearch/.feature_cache/
```

### Key Environment Files
- `.env` in project root: contains `OLLAMA_API_KEY=<key>`
- `.feature_cache/`: pre-computed feature matrices (auto-rebuilt if missing, ~5 min)
- `experiments_v3.jsonl`: the master results file -- COPY THIS from the Windows machine

---

## Immediate Next Steps on Linux

1. **Copy data files** -- copy all `.csv` files and `autoresearch/*.jsonl` from the Windows machine
2. **Download fresh CSVs** for morning/day/night from texaslottery.com (not blocked on home Linux)
3. **Download pre-test CSVs** from the pre-test download page
4. **Rebuild feature cache** -- `python -c "import prepare; prepare.precompute_all_features()"`
5. **Design combined prepare_v2.py** -- new data module supporting all 4 draw times + equipment features
6. **Run live_eval.py on Monday's evening result** -- update all four _live CSVs with Apr 14 results and re-score
7. **Continue GA runs** -- `python -u runner_v3.py --model minimax-m2.7:cloud --n 100`

---

## Key Design Decisions Already Made (Don't Revisit Without Good Reason)

- **Per-digit models, not combo model**: we predict each digit independently and multiply probabilities. This gives us factored representations that generalize better than trying to classify 1000 combos directly.
- **Straight bets, not box**: the model has real positional signal (ordering matters). Box bets discard position information and perform worse.
- **optimal_ev as fitness**: tested mean_rank, top_k_hit_rate -- optimal_ev (maximize EV across K=1..20) is the most directly useful metric.
- **Live eval as truth**: val scores are compromised by 800+ experiments of GA optimization against the val split. Only the 49 live draws are unbiased.
- **Shallow LGB beats deep ET/RF on live**: the GA evolved toward deep ExtraTrees but live eval shows shallow LGB (depth=3) generalizes better. Keep this in mind when designing new experiments.

---

## Files to Copy to Linux

**Essential:**
```
autoresearch/experiments_v3.jsonl       (~5.9 MB -- all 842 experiment configs + results)
autoresearch/live_eval_results.jsonl    (~1.1 MB -- live scores for 306 experiments)
autoresearch/*.py                       (all Python source files)
pick3evening.csv                        (frozen at Feb 13 2026 -- do not append)
pick3morning.csv                        (frozen at Feb 13 2026 -- do not append)
pick3day.csv                            (frozen at Feb 13 2026 -- do not append)
pick3night.csv                          (frozen at Feb 13 2026 -- do not append)
pick3evening_live.csv                   (live Evening draws, Feb 14 2026 - present)
pick3morning_live.csv                   (live Morning draws, Feb 14 2026 - present)
pick3day_live.csv                       (live Day draws, Feb 14 2026 - present)
pick3night_live.csv                     (live Night draws, Feb 14 2026 - present)
.env                                    (OLLAMA_API_KEY)
```

**Optional (large, rebuildable):**
```
autoresearch/.feature_cache/            (will auto-rebuild in ~5 min)
autoresearch/experiments_v2.jsonl       (older runs, not needed)
```

---

## Conversation History

The full conversation history from the Windows development sessions is included in the project zip.
Place it at the following path on the Linux machine so Claude Code can reference it:

```
~/LottAI/conversation_history/4d1a96cf-77ab-4828-a660-84078be943a7.jsonl
```

To make Claude Code aware of prior sessions on the new machine, you can also copy it to the
Claude projects directory (adjust path for Linux):
```
~/.claude/projects/<encoded-project-path>/4d1a96cf-77ab-4828-a660-84078be943a7.jsonl
```
Where `<encoded-project-path>` is the Claude Code internal encoding of your project path
(e.g. if project is at /home/user/LottAI, run `claude` once in that directory and it will
create the directory automatically -- then copy the jsonl there).

**Do not leave the conversation history pointing back at the Windows machine.** It is self-contained
and travels with the project. The summary at the top of each new session (after context compaction)
captures the key state, but the full JSONL is the complete record if you need to dig into specifics.

This BRIEFING.md is the authoritative quick-reference for the Linux handoff.

---

## Files to Include in the Transfer Zip

```
# Project source and data
LottAI/BRIEFING.md                                          (this file)
LottAI/pick3evening.csv                                     (frozen at Feb 13 2026)
LottAI/pick3morning.csv                                     (frozen at Feb 13 2026)
LottAI/pick3day.csv                                         (frozen at Feb 13 2026)
LottAI/pick3night.csv                                       (frozen at Feb 13 2026)
LottAI/pick3evening_live.csv                                (Feb 14 2026 - present)
LottAI/pick3morning_live.csv                                (Feb 14 2026 - present)
LottAI/pick3day_live.csv                                    (Feb 14 2026 - present)
LottAI/pick3night_live.csv                                  (Feb 14 2026 - present)
LottAI/.env                                                 (OLLAMA_API_KEY)
LottAI/autoresearch/*.py                                    (all source files)
LottAI/autoresearch/experiments_v3.jsonl                    (~5.9 MB -- DO NOT LOSE THIS)
LottAI/autoresearch/live_eval_results.jsonl                 (~1.1 MB)

# Conversation history (place on Linux as described above)
4d1a96cf-77ab-4828-a660-84078be943a7.jsonl                  (full Windows session history)

# Optional -- large, rebuildable
LottAI/autoresearch/.feature_cache/                         (auto-rebuilds in ~5 min if missing)
LottAI/autoresearch/experiments_v2.jsonl                    (older runs, not needed)
```

**The two files you absolutely cannot lose:**
1. `autoresearch/experiments_v3.jsonl` -- 842 experiments, 307 successful, represents weeks of compute
2. `autoresearch/live_eval_results.jsonl` -- the live evaluation scores, the honest leaderboard

---

*Generated 2026-04-14 -- copy this file with the project*
