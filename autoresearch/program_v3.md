# LottAI AutoResearch v4 -- Config Explorer

## Mission

You are searching for the best machine learning strategy to predict Texas Pick 3 lottery draws.
The data is 15,572 draws across all 4 draw times (Morning/Day/Evening/Night), split into
train (10,900) / val (2,336) / test (2,336). Each draw has winning digits plus the physical
machine ID and ball set IDs used -- these encode real-world mechanical bias directly.
**Baseline mean_rank = 500.5 (random). Lower is better.**

**Fresh start -- no established best yet. The previous best on Evening-only data was
3x shallow LightGBM (max_depth=3) with custom_interact->select. That may or may not
hold on the full combined dataset with equipment features.**

- optimal_ev = top-K hit rate * $500 payout - K tickets. Higher is better.

The signal is real -- top strategies generalized to unseen live data on the previous run.

Your job: propose a **JSON config** for a new experiment. Be diverse -- explore architectures
that are structurally different from 3x shallow LGB, especially ones that use the new
`equipment` and `draw_time` feature sets.

---

## JSON Config Schema

Output a single JSON object with exactly these top-level keys:

### `description`
String. Brief description of what makes this config unique.

### `feature_sets`
List of feature set names. Any non-empty subset of:
`["basic", "recency", "gaps", "positional", "temporal", "momentum", "draw_time", "equipment"]`

Feature set meanings:
- `basic` (11 features): previous draw digits, sum, repeat/triple/sequential flags
- `recency` (121 features): per-digit/position frequency in windows [10, 25, 50, 100] + combo gap
- `gaps` (40 features): per-digit gap since last appearance
- `positional` (33 features): per-position digit frequencies + position pair correlations
- `temporal` (20 features): day-of-week, month, year -- **often HURTS generalization, try removing it**
- `momentum` (7 features): chi-squared per position, entropy per position, consecutive overlap
- `draw_time` (4 features): one-hot encoding of draw time (morning/day/evening/night) -- **NEW, try including**
- `equipment` (4 features): label-encoded machine ID + ball set IDs for all 3 chambers -- **NEW, high-value signal, always try including**

### `pipeline`
Ordered list of preprocessing stages applied before model training. Can be empty `[]`.
Each stage appears **at most once**. Max 3 stages.

Valid stage objects:

```json
{ "stage": "select",
  "threshold_multiplier": 1.5,
  "selector_n_estimators": 200,
  "selector_max_depth": 10,
  "selector_target": "d1" }
```
`threshold_multiplier`: 0.5-3.0. `selector_n_estimators`: 50-500. `selector_max_depth`: 5-20.
`selector_target`: "d1" | "d2" | "d3" | "mean" (which digit to train the selector on; "mean" uses all three)

```json
{ "stage": "poly",
  "degree": 2,
  "interaction_only": false,
  "top_k_variance": 250 }
```
`degree`: 2 or 3. `top_k_variance`: 50-500 (how many polynomial features to keep by variance).

```json
{ "stage": "custom_interact",
  "n_head": 5,
  "n_tail": 5,
  "include_ratios": true }
```
`n_head`/`n_tail`: 2-15. `include_ratios`: true/false (whether to include f1/f2 ratio features).

```json
{ "stage": "scale",
  "method": "standard" }
```
`method`: "standard" | "robust" | "quantile" | "minmax"

**Pipeline ordering matters!** `[select, poly]` vs `[poly, select]` are qualitatively different strategies.

### `models`
List of 1-3 model configs. Weights are relative (they get normalized to sum=1).

Model types and their parameters:

**`et`** (ExtraTrees):
```json
{ "type": "et", "weight": 0.6,
  "n_estimators": 600, "max_depth": 18,
  "min_samples_leaf": 3, "min_samples_split": 8, "max_features": 0.6,
  "calibrate": true, "calibrate_method": "isotonic", "calibrate_cv": 5 }
```
Ranges: n_estimators 50-1000, max_depth 3-30, min_samples_leaf 1-20, max_features 0.1-1.0

**`xgb`** (XGBoost):
```json
{ "type": "xgb", "weight": 0.4,
  "n_estimators": 400, "max_depth": 6, "learning_rate": 0.05,
  "subsample": 0.8, "colsample_bytree": 0.7,
  "reg_alpha": 0.0, "reg_lambda": 1.0,
  "calibrate": false, "calibrate_method": "isotonic", "calibrate_cv": 5 }
```

**`lgb`** (LightGBM -- same params as xgb):
```json
{ "type": "lgb", "weight": 0.5, "n_estimators": 400, "max_depth": 6,
  "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
  "reg_alpha": 0.0, "reg_lambda": 1.0,
  "calibrate": true, "calibrate_method": "isotonic", "calibrate_cv": 5 }
```

**`rf`** (RandomForest -- same params as et):

**`hgb`** (HistGradientBoosting -- use `max_iter` instead of `n_estimators`):
```json
{ "type": "hgb", "weight": 0.5, "max_iter": 300, "max_depth": 6,
  "learning_rate": 0.05,
  "calibrate": false, "calibrate_method": "isotonic", "calibrate_cv": 5 }
```

**`lr`** (LogisticRegression):
```json
{ "type": "lr", "weight": 0.3, "C": 1.0,
  "calibrate": false, "calibrate_method": "isotonic", "calibrate_cv": 5 }
```

`calibrate_method`: "isotonic" | "sigmoid". `calibrate_cv`: 3, 5, 7, or 10.

---

## Directions Worth Exploring

This is a fresh search on new data. Prioritize diversity over convergence:

- **Always include `equipment`** -- machine ID and ball set IDs are direct causal signals; this is the most promising new feature set
- **Include `draw_time`** -- morning/day/evening/night may have systematically different equipment bias profiles
- **Drop `temporal`** -- day/month/year features have historically hurt generalization; try without
- **Mix model types** -- try `lgb + et`, `lgb + xgb`, or `xgb + rf` instead of 3x lgb
- **Single strong model** -- try 1x lgb or 1x xgb, heavily regularized, no ensemble overhead
- **Deeper models with equipment** -- equipment features are low-cardinality categoricals; deeper trees (max_depth=6-10) may capture interactions better
- **Scale + equipment** -- try `[scale, select]` with equipment features; scaling may help with label-encoded categoricals
- **Selector target "mean"** -- try "mean" to select features jointly across all three digit positions
- **HGB** -- HistGradientBoosting handles mixed feature types well; good candidate for equipment features
- **LR baseline** -- logistic regression with `select` only; useful as a diversity anchor
- **No pipeline** -- try raw features with `equipment` + `basic` only, single model
- **Positional + equipment only** -- test whether equipment alone without recency/gaps changes the picture

---

## Output Instructions

Output **ONLY** a JSON config between triple backticks. Nothing else.

```json
{
  "description": "your description here",
  "feature_sets": [...],
  "pipeline": [...],
  "models": [...]
}
```
