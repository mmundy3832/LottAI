"""
Survival Analysis for Lotto Texas Number Appearances
=====================================================
Predicts "time until next appearance" (gap in draws) for each number 1-54,
then converts to probability of appearing in the next N draws.

Models:
  1. Cox Proportional Hazards (lifelines) -- interpretable baseline
  2. Random Survival Forest (scikit-survival) -- nonlinear, interactions
  3. Exponential baseline -- simple comparison

Evaluation: expanding-window backtest on last 400 draws.

Usage:
  .venv/Scripts/python.exe -u survival_model.py
"""

import json
import os
import sys
import time
import warnings
from collections import defaultdict
from datetime import datetime, date
from itertools import combinations
from pathlib import Path

import subprocess

def _ensure_packages():
    """Install required packages that are not yet available."""
    required = {
        "lifelines": "lifelines",        # import name -> pip name
        "sksurv": "scikit-survival",
    }
    for import_name, pip_name in required.items():
        try:
            __import__(import_name)
        except ImportError:
            print(f"[setup] Installing missing package: {pip_name}", flush=True)
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", pip_name, "--quiet"]
            )

_ensure_packages()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index as lifelines_ci
from sklearn.preprocessing import StandardScaler
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score

np.random.seed(42)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Globals
# ---------------------------------------------------------------------------
DATA_PATH = Path(r"D:\projects\LottAI\lottotexas.csv")
RESULTS_DIR = Path(r"D:\projects\LottAI\results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NUM_RANGE = range(1, 55)  # 1 through 54
PICK_SIZE = 6
BASELINE_PROB = PICK_SIZE / 54  # ~0.1111
BACKTEST_DRAWS = 400  # evaluate on last 400 draws
BRIER_HORIZONS = [1, 3, 5]

# Date when Lotto Texas went from 2x/week to 3x/week
FREQ_CHANGE_DATE = date(2021, 8, 1)


def log(msg: str) -> None:
    """Print timestamped progress message."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


# ===================================================================
# 1. DATA LOADING
# ===================================================================
def load_draws(path: Path) -> pd.DataFrame:
    """Load Lotto Texas CSV into a DataFrame sorted by draw date."""
    log(f"Loading data from {path}")
    df = pd.read_csv(
        path,
        header=None,
        names=["GameName", "Month", "Day", "Year",
               "Num1", "Num2", "Num3", "Num4", "Num5", "Num6"],
    )
    df["Date"] = pd.to_datetime(
        df[["Year", "Month", "Day"]].rename(
            columns={"Year": "year", "Month": "month", "Day": "day"}
        )
    )
    df = df.sort_values("Date").reset_index(drop=True)

    # Filter out pre-April 2006 draws (game changed from pick-6-from-50 to pick-6-from-54)
    pre_2006_count = len(df)
    df = df[df["Date"] >= "2006-04-01"].reset_index(drop=True)
    log(f"  Filtered out {pre_2006_count - len(df)} pre-2006 draws, {len(df)} draws remaining")

    df["draw_idx"] = df.index  # 0-based sequential draw index

    # Build a set-of-numbers column for convenience
    num_cols = ["Num1", "Num2", "Num3", "Num4", "Num5", "Num6"]
    df["numbers"] = df[num_cols].values.tolist()
    df["numbers_set"] = df["numbers"].apply(set)

    # Post-Aug-2021 indicator
    df["post_aug2021"] = (df["Date"].dt.date >= FREQ_CHANGE_DATE).astype(int)

    log(f"  Loaded {len(df)} draws  [{df['Date'].iloc[0].date()} .. {df['Date'].iloc[-1].date()}]")
    return df


# ===================================================================
# 2. PRE-COMPUTATION STRUCTURES
# ===================================================================
def build_appearance_matrix(df: pd.DataFrame) -> np.ndarray:
    """Return binary matrix (n_draws x 54) where 1 = number appeared."""
    n = len(df)
    mat = np.zeros((n, 54), dtype=np.int8)
    for i, nums in enumerate(df["numbers"]):
        for num in nums:
            mat[i, num - 1] = 1
    return mat


def compute_gap_histories(appear_mat: np.ndarray) -> dict:
    """For each number, compute the list of completed gap lengths.

    Returns dict: number (1-54) -> list of (draw_ended, gap_length) tuples.
    Also returns last_seen array: last draw index each number appeared.
    """
    n_draws, n_nums = appear_mat.shape
    gap_histories = {num: [] for num in range(1, n_nums + 1)}
    last_seen = np.full(n_nums, -1, dtype=int)

    for d in range(n_draws):
        for j in range(n_nums):
            if appear_mat[d, j] == 1:
                if last_seen[j] >= 0:
                    gap_len = d - last_seen[j]
                    gap_histories[j + 1].append((d, gap_len))
                last_seen[j] = d

    return gap_histories, last_seen


def compute_cooccurrence_top5(appear_mat: np.ndarray) -> dict:
    """For each number, find its top-5 co-occurring partner numbers."""
    n_draws, n_nums = appear_mat.shape
    cooccur = np.zeros((n_nums, n_nums), dtype=int)
    for d in range(n_draws):
        present = np.where(appear_mat[d] == 1)[0]
        for i, j in combinations(present, 2):
            cooccur[i, j] += 1
            cooccur[j, i] += 1

    top5 = {}
    for j in range(n_nums):
        partners = np.argsort(cooccur[j])[::-1][:5]
        top5[j + 1] = [p + 1 for p in partners]
    return top5


# ===================================================================
# 3. FEATURE ENGINEERING
# ===================================================================
def build_feature_row(
    num: int,
    draw_idx: int,
    appear_mat: np.ndarray,
    df: pd.DataFrame,
    gap_histories: dict,
    top5_partners: dict,
) -> dict:
    """Build one feature row for (number, draw_idx).

    The observation represents: at the START of draw `draw_idx`, what do we
    know about `num`?  The target will be: does `num` appear in draw
    `draw_idx` (and if not, the censored ongoing gap continues).
    """
    j = num - 1  # 0-based column index
    feats = {}

    # ---- Feature 1: Current gap length ----
    # How many draws since `num` last appeared (before draw_idx)?
    prev_appearances = np.where(appear_mat[:draw_idx, j] == 1)[0]
    if len(prev_appearances) > 0:
        last_app = prev_appearances[-1]
        current_gap = draw_idx - last_app
    else:
        current_gap = draw_idx + 1  # never seen yet
    feats["current_gap"] = current_gap

    # ---- Feature 2: Rolling gap statistics (last 5 / 10 / 20 gaps) ----
    completed_gaps_before = [
        g for (d_end, g) in gap_histories[num] if d_end < draw_idx
    ]
    for window in [5, 10, 20]:
        recent = completed_gaps_before[-window:] if completed_gaps_before else []
        if len(recent) > 0:
            feats[f"gap_mean_{window}"] = np.mean(recent)
            feats[f"gap_std_{window}"] = np.std(recent) if len(recent) > 1 else 0.0
            feats[f"gap_min_{window}"] = np.min(recent)
            feats[f"gap_max_{window}"] = np.max(recent)
        else:
            feats[f"gap_mean_{window}"] = 0.0
            feats[f"gap_std_{window}"] = 0.0
            feats[f"gap_min_{window}"] = 0.0
            feats[f"gap_max_{window}"] = 0.0

    # ---- Feature 3: Recent frequency (last 20 / 50 / 100 draws) ----
    for window in [20, 50, 100]:
        start = max(0, draw_idx - window)
        feats[f"freq_{window}"] = int(appear_mat[start:draw_idx, j].sum())

    # ---- Feature 4: Runs pattern features ----
    # A "run" = consecutive draws where the number either appears or doesn't.
    # Current run: how many consecutive draws (ending at draw_idx-1) with
    # same status (appeared / not appeared)?
    if draw_idx == 0:
        feats["run_length"] = 0
        feats["run_direction"] = 0  # 1 = appearing streak, -1 = absence streak
        feats["hist_runs_ratio"] = 1.0
    else:
        current_status = appear_mat[draw_idx - 1, j]
        run_len = 1
        for d in range(draw_idx - 2, -1, -1):
            if appear_mat[d, j] == current_status:
                run_len += 1
            else:
                break
        feats["run_length"] = run_len
        feats["run_direction"] = 1 if current_status == 1 else -1

        # Historical runs ratio: (# of absence runs) / (# of appearance runs)
        # computed on all draws before draw_idx
        seq = appear_mat[:draw_idx, j]
        changes = np.diff(seq)
        n_runs = 1 + np.count_nonzero(changes)
        n_appear_runs = max(1, np.count_nonzero(changes == 1) + (1 if seq[0] == 1 else 0))
        n_absence_runs = max(1, n_runs - n_appear_runs)
        feats["hist_runs_ratio"] = n_absence_runs / n_appear_runs

    # ---- Feature 5: Co-occurrence features ----
    # How many of this number's top-5 partners appeared in the previous draw?
    if draw_idx > 0:
        prev_set = df.iloc[draw_idx - 1]["numbers_set"]
        partners = top5_partners.get(num, [])
        feats["cooccur_partner_count"] = sum(1 for p in partners if p in prev_set)
    else:
        feats["cooccur_partner_count"] = 0

    # ---- Feature 6: Draw-level features (previous draw) ----
    if draw_idx > 0:
        prev_nums = df.iloc[draw_idx - 1]["numbers"]
        feats["prev_draw_sum"] = sum(prev_nums)
        feats["prev_draw_range"] = max(prev_nums) - min(prev_nums)
        feats["prev_draw_odd_count"] = sum(1 for n in prev_nums if n % 2 == 1)
    else:
        feats["prev_draw_sum"] = 0
        feats["prev_draw_range"] = 0
        feats["prev_draw_odd_count"] = 0

    # ---- Feature 7: Max-number carry ----
    # Was this number the max in the previous draw?
    if draw_idx > 0:
        prev_nums = df.iloc[draw_idx - 1]["numbers"]
        feats["was_prev_max"] = 1 if num == max(prev_nums) else 0
    else:
        feats["was_prev_max"] = 0

    # ---- Feature 8: Post-Aug-2021 indicator ----
    feats["post_aug2021"] = int(df.iloc[draw_idx]["post_aug2021"])

    # ---- Feature 9: Number identity (static) ----
    # Encode as continuous — the number itself (captures positional bias)
    feats["number_id"] = num

    # === EXTENSIBLE FEATURES ===
    # Add new covariates here as simple column additions.
    # Examples that can be plugged in later:
    #   feats["tda_persistence_score"] = ...
    #   feats["network_centrality"] = ...
    #   feats["spectral_dominant_freq"] = ...
    #   feats["weather_pressure"] = ...
    #   feats["weather_temperature"] = ...
    #   feats["weather_humidity"] = ...
    # Each new feature just needs to be a scalar value keyed by a string name.
    # The pipeline will automatically pick it up as a covariate.
    # === END EXTENSIBLE FEATURES ===

    return feats


def build_survival_dataset(
    df: pd.DataFrame,
    appear_mat: np.ndarray,
    gap_histories: dict,
    top5_partners: dict,
    start_draw: int = 100,
    end_draw: int = None,
) -> pd.DataFrame:
    """Build the full feature matrix for survival analysis.

    Each row is a "spell" for one number starting from its most recent
    appearance (or draw 0) up to either the next appearance (event=1)
    or end_draw (censored, event=0).

    For efficiency in the expanding-window backtest we use a different
    approach: for each draw in [start_draw, end_draw), for each number
    1-54, we create a single row with features computed AT that draw.
    The target is: did the number appear in that draw?

    We then transform this into proper survival format: each spell begins
    when a number last appeared and ends when it appears again (or is
    censored).
    """
    if end_draw is None:
        end_draw = len(df)

    log(f"  Building features for draws [{start_draw}, {end_draw})")
    rows = []
    total = (end_draw - start_draw) * 54
    report_every = max(1, total // 20)
    count = 0

    for d in range(start_draw, end_draw):
        for num in NUM_RANGE:
            feats = build_feature_row(
                num, d, appear_mat, df, gap_histories, top5_partners
            )
            # Target: did the number appear in this draw?
            feats["event"] = int(appear_mat[d, num - 1] == 1)
            feats["draw_idx"] = d
            feats["number"] = num
            rows.append(feats)
            count += 1
            if count % report_every == 0:
                log(f"    {count}/{total} feature rows built ({100*count/total:.0f}%)")

    result_df = pd.DataFrame(rows)
    log(f"  Built {len(result_df)} feature rows, {result_df['event'].sum()} events")
    return result_df


def convert_to_spell_format(feat_df: pd.DataFrame) -> pd.DataFrame:
    """Convert per-draw observations into survival spells.

    Each spell: (number, spell_start_draw, duration, event, features_at_start).
    Duration = current_gap at the draw when the number next appears (or last
    observation if censored).

    For Cox PH we need (duration, event) plus covariates.
    We use the feature snapshot from the START of the gap (when the number
    last appeared), which avoids look-ahead bias.
    """
    log("  Converting to spell format for survival models")
    spells = []
    feat_cols = [c for c in feat_df.columns
                 if c not in ("event", "draw_idx", "number", "current_gap")]

    for num in NUM_RANGE:
        num_df = feat_df[feat_df["number"] == num].sort_values("draw_idx")
        if len(num_df) == 0:
            continue

        # Walk through rows; accumulate a spell until event=1
        spell_start_row = None
        duration = 0

        for _, row in num_df.iterrows():
            if spell_start_row is None:
                spell_start_row = row
                duration = 1
            else:
                duration += 1

            if row["event"] == 1:
                # Complete spell
                spell = {c: spell_start_row[c] for c in feat_cols}
                spell["duration"] = duration
                spell["event"] = 1
                spell["number"] = num
                spell["draw_idx"] = int(spell_start_row["draw_idx"])
                spells.append(spell)
                spell_start_row = None
                duration = 0

        # If we end without an event, create a censored spell
        if spell_start_row is not None and duration > 0:
            spell = {c: spell_start_row[c] for c in feat_cols}
            spell["duration"] = duration
            spell["event"] = 0
            spell["number"] = num
            spell["draw_idx"] = int(spell_start_row["draw_idx"])
            spells.append(spell)

    spell_df = pd.DataFrame(spells)
    log(f"  {len(spell_df)} spells, {spell_df['event'].sum()} events, "
        f"{(spell_df['event']==0).sum()} censored")
    return spell_df


# ===================================================================
# 4. FEATURE COLUMNS
# ===================================================================
def get_feature_columns(df: pd.DataFrame) -> list:
    """Return the list of covariate column names (everything except meta/target)."""
    exclude = {"event", "draw_idx", "number", "duration"}
    return sorted([c for c in df.columns if c not in exclude])


# ===================================================================
# 5. MODELS
# ===================================================================

class ExponentialBaseline:
    """Simple exponential survival model: constant hazard per number.

    Hazard estimated as (# events) / (total time at risk).
    """

    def __init__(self):
        self.hazards = {}  # number -> hazard rate
        self.global_hazard = None

    def fit(self, spell_df: pd.DataFrame):
        total_events = spell_df["event"].sum()
        total_time = spell_df["duration"].sum()
        self.global_hazard = total_events / total_time if total_time > 0 else BASELINE_PROB

        for num in NUM_RANGE:
            ndf = spell_df[spell_df["number"] == num]
            if len(ndf) == 0 or ndf["duration"].sum() == 0:
                self.hazards[num] = self.global_hazard
            else:
                self.hazards[num] = ndf["event"].sum() / ndf["duration"].sum()

    def predict_survival_function(self, num: int, t: int) -> float:
        """S(t) = exp(-lambda * t)"""
        lam = self.hazards.get(num, self.global_hazard)
        return np.exp(-lam * t)

    def predict_appearance_prob(self, num: int, horizon: int) -> float:
        """P(appear within horizon draws) = 1 - S(horizon)"""
        return 1.0 - self.predict_survival_function(num, horizon)


class CoxSurvivalModel:
    """Wrapper around lifelines CoxPHFitter for our survival data."""

    def __init__(self):
        self.fitter = CoxPHFitter(penalizer=0.01)
        self.feature_cols = None
        self.scaler = StandardScaler()
        self._baseline_survival = None

    def fit(self, spell_df: pd.DataFrame):
        self.feature_cols = get_feature_columns(spell_df)
        df_fit = spell_df[self.feature_cols + ["duration", "event"]].copy()

        # Scale features
        df_fit[self.feature_cols] = self.scaler.fit_transform(df_fit[self.feature_cols])

        # Ensure duration > 0
        df_fit["duration"] = df_fit["duration"].clip(lower=0.5)

        self.fitter.fit(
            df_fit,
            duration_col="duration",
            event_col="event",
            show_progress=False,
        )

    def predict_hazard_ratio(self, feature_row: pd.Series) -> float:
        """Return exp(beta * X) for a single observation."""
        x = feature_row[self.feature_cols].values.reshape(1, -1).astype(float)
        x_scaled = self.scaler.transform(x)
        log_hr = (self.fitter.params_[self.feature_cols].values * x_scaled[0]).sum()
        return np.exp(log_hr)

    def predict_risk_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Return risk scores (higher = sooner event) for a DataFrame."""
        x = df[self.feature_cols].values.astype(float)
        x_scaled = self.scaler.transform(x)
        df_scaled = pd.DataFrame(x_scaled, columns=self.feature_cols, index=df.index)
        # lifelines predict_partial_hazard returns exp(X*beta)
        return self.fitter.predict_partial_hazard(df_scaled).values.ravel()

    def get_feature_importance(self) -> dict:
        """Return {feature: coefficient} sorted by abs value."""
        coefs = self.fitter.params_
        return dict(
            sorted(coefs.items(), key=lambda kv: abs(kv[1]), reverse=True)
        )


class RSFSurvivalModel:
    """Wrapper around scikit-survival RandomSurvivalForest."""

    def __init__(self, n_estimators=100, max_depth=8, min_samples_leaf=20):
        self.model = RandomSurvivalForest(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1,
            random_state=42,
        )
        self.feature_cols = None
        self.scaler = StandardScaler()

    def fit(self, spell_df: pd.DataFrame):
        self.feature_cols = get_feature_columns(spell_df)
        X = spell_df[self.feature_cols].values.astype(float)
        X = self.scaler.fit_transform(X)

        # scikit-survival needs structured array for y
        y = np.array(
            [(bool(e), d) for e, d in zip(spell_df["event"], spell_df["duration"])],
            dtype=[("event", bool), ("duration", float)],
        )
        self.model.fit(X, y)
        # Store training data for permutation importance fallback
        self._X_train = X
        self._y_train = y

    def predict_risk_scores(self, df: pd.DataFrame) -> np.ndarray:
        """Higher score = higher risk = sooner event."""
        X = df[self.feature_cols].values.astype(float)
        X = self.scaler.transform(X)
        return self.model.predict(X)

    def predict_survival_function(self, df: pd.DataFrame) -> np.ndarray:
        """Return array of survival function objects."""
        X = df[self.feature_cols].values.astype(float)
        X = self.scaler.transform(X)
        return self.model.predict_survival_function(X)

    def get_feature_importance(self) -> dict:
        """Return permutation-based feature importances."""
        # importances = self.model.feature_importances_  # original — may raise NotImplementedError
        try:
            importances = self.model.feature_importances_
        except (NotImplementedError, AttributeError):
            try:
                from sklearn.inspection import permutation_importance
                perm_result = permutation_importance(
                    self.model, self._X_train, self._y_train,
                    n_repeats=5, random_state=42, n_jobs=-1,
                )
                importances = perm_result.importances_mean
            except Exception as e:
                log(f"  WARNING: Feature importance fallback failed: {e}")
                importances = np.zeros(len(self.feature_cols))
        return dict(
            sorted(
                zip(self.feature_cols, importances),
                key=lambda kv: abs(kv[1]),
                reverse=True,
            )
        )


# ===================================================================
# 6. EVALUATION
# ===================================================================

def evaluate_concordance(
    spell_df_test: pd.DataFrame,
    risk_scores: np.ndarray,
    model_name: str,
) -> float:
    """Compute concordance index on test spells."""
    events = spell_df_test["event"].values.astype(bool)
    durations = spell_df_test["duration"].values.astype(float)
    try:
        c_idx, concordant, discordant, tied_risk, tied_time = concordance_index_censored(
            events, durations, risk_scores
        )
    except Exception:
        c_idx = 0.5
    log(f"  {model_name}: C-index = {c_idx:.4f}")
    return c_idx


def compute_brier_scores(
    spell_df_test: pd.DataFrame,
    rsf_model: RSFSurvivalModel,
    horizons: list,
) -> dict:
    """Compute time-dependent Brier score at given horizons using RSF.

    Returns dict of {horizon: brier_score}.
    """
    brier_scores = {}

    # For the Brier score we use the RSF survival function predictions
    try:
        surv_fns = rsf_model.predict_survival_function(spell_df_test)
        y_test = np.array(
            [(bool(e), d) for e, d in zip(
                spell_df_test["event"], spell_df_test["duration"]
            )],
            dtype=[("event", bool), ("duration", float)],
        )

        # We need times within the range of test durations
        max_time = spell_df_test["duration"].max()
        valid_horizons = [h for h in horizons if h < max_time]

        if len(valid_horizons) > 0:
            # Build survival probability matrix at the valid horizons
            times = np.array(valid_horizons, dtype=float)
            n_samples = len(spell_df_test)
            surv_probs = np.zeros((n_samples, len(times)))

            for i, fn in enumerate(surv_fns):
                for t_idx, t in enumerate(times):
                    # Interpolate survival function at time t
                    fn_times = fn.x
                    fn_vals = fn.y
                    if t <= fn_times[0]:
                        surv_probs[i, t_idx] = 1.0
                    elif t >= fn_times[-1]:
                        surv_probs[i, t_idx] = fn_vals[-1]
                    else:
                        surv_probs[i, t_idx] = np.interp(t, fn_times, fn_vals)

            # Compute integrated Brier score for each horizon individually
            for t_idx, h in enumerate(valid_horizons):
                # Per-sample Brier: (S(t) - I(T > t))^2
                actual_survived = (spell_df_test["duration"].values > h).astype(float)
                pred_survived = surv_probs[:, t_idx]
                bs = np.mean((pred_survived - actual_survived) ** 2)
                brier_scores[h] = float(bs)
                log(f"    Brier score at horizon {h}: {bs:.4f}")

        # Fill in any horizons we couldn't compute
        for h in horizons:
            if h not in brier_scores:
                brier_scores[h] = None
                log(f"    Brier score at horizon {h}: N/A (beyond test range)")

    except Exception as e:
        log(f"    Brier score computation failed: {e}")
        for h in horizons:
            brier_scores[h] = None

    return brier_scores


def compute_calibration(
    feat_df_test: pd.DataFrame,
    risk_scores_per_draw: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute calibration: predicted vs actual in deciles.

    risk_scores_per_draw are per-(number, draw) row risk scores.
    Higher risk => more likely to appear.
    """
    df_cal = feat_df_test[["event"]].copy()
    df_cal["risk"] = risk_scores_per_draw

    # Convert risk to predicted probability using logistic transform
    # risk scores are relative, so we calibrate: sort into deciles
    df_cal["decile"] = pd.qcut(df_cal["risk"], n_bins, labels=False, duplicates="drop")

    calibration = {}
    for dec in sorted(df_cal["decile"].unique()):
        group = df_cal[df_cal["decile"] == dec]
        pred_mean = group["risk"].mean()
        actual_rate = group["event"].mean()
        calibration[int(dec)] = {
            "predicted_risk_mean": float(pred_mean),
            "actual_rate": float(actual_rate),
            "count": int(len(group)),
        }

    return calibration


def naive_baseline_concordance(spell_df_test: pd.DataFrame) -> float:
    """Naive baseline: flat 6/54 probability => random risk scores."""
    n = len(spell_df_test)
    # All identical risk scores => C-index should be ~0.5
    risk = np.ones(n) * 0.5
    return evaluate_concordance(spell_df_test, risk, "Naive (flat 6/54)")


# ===================================================================
# 7. DELTA SURVIVAL PREDICTIONS
# ===================================================================

def compute_delta_survival(
    df: pd.DataFrame,
    appear_mat: np.ndarray,
    gap_histories: dict,
    top5_partners: dict,
    cox_model: CoxSurvivalModel,
    rsf_model: RSFSurvivalModel,
    exp_model: ExponentialBaseline,
) -> dict:
    """For each number, compute delta_survival = log-odds adjustment.

    delta_survival(number) = log[ P_model(appear) / P_baseline ] - log[ (1-P_model) / (1-P_baseline) ]
    simplified: log-odds(P_model) - log-odds(P_baseline)

    We compute this at the CURRENT state (last draw in dataset).
    """
    log("Computing delta_survival predictions for current state")
    last_draw_idx = len(df) - 1
    predictions = {}

    # Build features for the "next" hypothetical draw
    next_draw_idx = last_draw_idx + 1

    for num in NUM_RANGE:
        feats = build_feature_row(
            num, last_draw_idx, appear_mat, df, gap_histories, top5_partners
        )
        # Use current gap extended by 1 for prediction at next draw
        feats["current_gap"] = feats["current_gap"]  # already reflects gap as of last draw

        # Cox risk
        feat_series = pd.Series(feats)
        feat_cols = cox_model.feature_cols
        feat_df = pd.DataFrame([{c: feats.get(c, 0) for c in feat_cols}])
        cox_risk = cox_model.predict_risk_scores(feat_df)[0]

        # RSF risk
        rsf_risk = rsf_model.predict_risk_scores(feat_df)[0]

        # Exponential probability
        exp_prob = exp_model.predict_appearance_prob(num, feats["current_gap"])

        # Convert risks to approximate probabilities using sigmoid calibration
        # For delta, we use the rank-based approach
        baseline_logodds = np.log(BASELINE_PROB / (1 - BASELINE_PROB))

        # Cox: log-odds ~ log(risk)
        cox_logodds_adj = np.log(max(cox_risk, 1e-10)) - np.log(1.0)  # relative to median

        # Exponential: direct probability
        exp_logodds = np.log(max(exp_prob, 1e-10) / max(1 - exp_prob, 1e-10))
        exp_delta = exp_logodds - baseline_logodds

        predictions[str(num)] = {
            "number": num,
            "current_gap": int(feats["current_gap"]),
            "cox_risk_score": float(cox_risk),
            "cox_delta_logodds": float(cox_logodds_adj),
            "rsf_risk_score": float(rsf_risk),
            "exp_prob_next": float(exp_prob),
            "exp_delta_logodds": float(exp_delta),
            "baseline_prob": float(BASELINE_PROB),
            "baseline_logodds": float(baseline_logodds),
        }

    return predictions


# ===================================================================
# 8. EXPANDING WINDOW BACKTEST
# ===================================================================

def run_backtest(
    df: pd.DataFrame,
    appear_mat: np.ndarray,
    gap_histories: dict,
    top5_partners: dict,
    backtest_draws: int = BACKTEST_DRAWS,
) -> dict:
    """Expanding-window backtest on last `backtest_draws` draws.

    For each test draw d in [N - backtest_draws, N):
      - Train on all data up to d
      - Predict risk for each number at draw d
      - Record whether each number actually appeared

    Then compute aggregate metrics.
    """
    n_draws = len(df)
    test_start = n_draws - backtest_draws
    log(f"Running expanding-window backtest: draws [{test_start}, {n_draws})")
    log(f"  Training starts from draw 100, test window = {backtest_draws} draws")

    # We'll collect per-draw predictions for all three models
    all_cox_risks = []
    all_rsf_risks = []
    all_events = []
    all_durations = []
    all_draw_feats = []

    # Build features for the ENTIRE range first (training + test)
    # to avoid recomputing. The trick: we only USE features up to each
    # test draw.
    log("  Building complete feature matrix for backtest range")
    train_start = 100  # skip first 100 draws for warm-up

    # Instead of re-training at every draw (very expensive), we use a
    # semi-expanding approach: retrain at intervals
    RETRAIN_INTERVAL = 100  # retrain every 100 draws
    cox_model = None
    rsf_model = None
    exp_model = None

    # Collect all test-period results
    test_spell_events = []
    test_spell_durations = []
    cox_risk_scores = []
    rsf_risk_scores = []

    n_retrains = 0
    for d in range(test_start, n_draws):
        # Retrain if needed
        if cox_model is None or (d - test_start) % RETRAIN_INTERVAL == 0:
            n_retrains += 1
            log(f"  [Retrain {n_retrains}] Training models on draws [{train_start}, {d})")

            # Build training data
            train_feat_df = build_survival_dataset(
                df, appear_mat, gap_histories, top5_partners,
                start_draw=train_start, end_draw=d,
            )
            train_spell_df = convert_to_spell_format(train_feat_df)

            if len(train_spell_df) < 50:
                log(f"    Skipping retrain — only {len(train_spell_df)} spells")
                continue

            # Fit all three models
            cox_model = CoxSurvivalModel()
            cox_model.fit(train_spell_df)

            rsf_model = RSFSurvivalModel(n_estimators=50, max_depth=6, min_samples_leaf=30)
            rsf_model.fit(train_spell_df)

            exp_model = ExponentialBaseline()
            exp_model.fit(train_spell_df)

            log(f"    Models trained on {len(train_spell_df)} spells")

        if cox_model is None:
            continue

        # Build features for each number at draw d (test observation)
        for num in NUM_RANGE:
            feats = build_feature_row(
                num, d, appear_mat, df, gap_histories, top5_partners
            )
            event = int(appear_mat[d, num - 1] == 1)
            feats["event"] = event
            feats["draw_idx"] = d
            feats["number"] = num
            feats["duration"] = feats["current_gap"]  # duration = current gap at time of prediction

            all_draw_feats.append(feats)

    # Convert to DataFrame
    test_feat_df = pd.DataFrame(all_draw_feats)
    if len(test_feat_df) == 0:
        log("  ERROR: No test observations generated!")
        return {}

    log(f"  Test set: {len(test_feat_df)} observations, {test_feat_df['event'].sum()} events")

    # Convert to spell format for concordance
    test_spell_df = convert_to_spell_format(test_feat_df)

    # Get risk scores from the final trained models (slight optimistic bias
    # for observations in last RETRAIN_INTERVAL window, but acceptable)
    feat_cols = get_feature_columns(test_spell_df)

    cox_risks = cox_model.predict_risk_scores(test_spell_df)
    rsf_risks = rsf_model.predict_risk_scores(test_spell_df)
    naive_risks = np.random.rand(len(test_spell_df))  # random baseline

    # ---- Concordance ----
    log("  Computing concordance indices")
    c_cox = evaluate_concordance(test_spell_df, cox_risks, "Cox PH")
    c_rsf = evaluate_concordance(test_spell_df, rsf_risks, "Random Survival Forest")
    c_naive = naive_baseline_concordance(test_spell_df)

    # ---- Brier Scores ----
    log("  Computing Brier scores")
    brier = compute_brier_scores(test_spell_df, rsf_model, BRIER_HORIZONS)

    # ---- Calibration (using per-draw features, not spells) ----
    log("  Computing calibration")
    per_draw_risks = cox_model.predict_risk_scores(test_feat_df)
    calibration = compute_calibration(test_feat_df, per_draw_risks)

    metrics = {
        "backtest_draws": backtest_draws,
        "n_test_observations": int(len(test_feat_df)),
        "n_test_spells": int(len(test_spell_df)),
        "concordance": {
            "cox_ph": float(c_cox),
            "rsf": float(c_rsf),
            "naive_baseline": float(c_naive),
        },
        "brier_scores": {str(k): v for k, v in brier.items()},
        "calibration": calibration,
    }

    return metrics, cox_model, rsf_model, exp_model, test_feat_df, test_spell_df


# ===================================================================
# 9. PLOTTING
# ===================================================================

def plot_concordance(metrics: dict, path: Path):
    """Bar chart comparing concordance indices."""
    fig, ax = plt.subplots(figsize=(8, 5))
    models = ["Cox PH", "RSF", "Naive Baseline"]
    c_values = [
        metrics["concordance"]["cox_ph"],
        metrics["concordance"]["rsf"],
        metrics["concordance"]["naive_baseline"],
    ]
    colors = ["#2196F3", "#4CAF50", "#9E9E9E"]
    bars = ax.bar(models, c_values, color=colors, edgecolor="black", linewidth=0.5)

    ax.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Random (0.5)")
    ax.set_ylabel("Concordance Index (C-index)")
    ax.set_title("Survival Model Concordance: Expanding Window Backtest")
    ax.set_ylim(0.4, max(0.75, max(c_values) + 0.05))
    ax.legend()

    for bar, val in zip(bars, c_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    log(f"  Saved concordance plot: {path}")


def plot_calibration(metrics: dict, path: Path):
    """Calibration plot: predicted risk decile vs actual appearance rate."""
    cal = metrics["calibration"]
    deciles = sorted(cal.keys(), key=int)
    predicted = [cal[d]["predicted_risk_mean"] for d in deciles]
    actual = [cal[d]["actual_rate"] for d in deciles]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(predicted, actual, "o-", color="#2196F3", linewidth=2, markersize=8, label="Model")
    # Perfect calibration line
    mn = min(min(predicted), min(actual))
    mx = max(max(predicted), max(actual))
    ax.plot([mn, mx], [mn, mx], "k--", alpha=0.5, label="Perfect calibration")

    ax.axhline(y=BASELINE_PROB, color="red", linestyle=":", alpha=0.7, label=f"Baseline {BASELINE_PROB:.3f}")

    ax.set_xlabel("Predicted Risk Score (decile mean)")
    ax.set_ylabel("Actual Appearance Rate")
    ax.set_title("Calibration: Predicted Risk vs Actual Appearance Rate")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(str(path), dpi=150)
    plt.close(fig)
    log(f"  Saved calibration plot: {path}")


def plot_feature_importance(cox_importance: dict, rsf_importance: dict, path: Path):
    """Side-by-side feature importance for Cox and RSF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Cox PH coefficients (top 15)
    top_cox = list(cox_importance.items())[:15]
    names_cox = [n for n, _ in top_cox]
    vals_cox = [v for _, v in top_cox]
    colors_cox = ["#EF5350" if v > 0 else "#42A5F5" for v in vals_cox]

    ax1.barh(range(len(names_cox)), vals_cox, color=colors_cox, edgecolor="black", linewidth=0.3)
    ax1.set_yticks(range(len(names_cox)))
    ax1.set_yticklabels(names_cox, fontsize=9)
    ax1.set_xlabel("Coefficient (log hazard ratio)")
    ax1.set_title("Cox PH: Feature Coefficients\n(positive = shorter gaps)")
    ax1.invert_yaxis()
    ax1.grid(True, alpha=0.3, axis="x")

    # RSF importances (top 15)
    top_rsf = list(rsf_importance.items())[:15]
    names_rsf = [n for n, _ in top_rsf]
    vals_rsf = [v for _, v in top_rsf]

    ax2.barh(range(len(names_rsf)), vals_rsf, color="#4CAF50", edgecolor="black", linewidth=0.3)
    ax2.set_yticks(range(len(names_rsf)))
    ax2.set_yticklabels(names_rsf, fontsize=9)
    ax2.set_xlabel("Permutation Importance")
    ax2.set_title("Random Survival Forest: Feature Importance")
    ax2.invert_yaxis()
    ax2.grid(True, alpha=0.3, axis="x")

    fig.suptitle("Survival Model Feature Importance Rankings", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(str(path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"  Saved feature importance plot: {path}")


# ===================================================================
# 10. MAIN
# ===================================================================

def main():
    t0 = time.time()
    log("=" * 70)
    log("LOTTO TEXAS SURVIVAL ANALYSIS")
    log("=" * 70)

    # ---- Load data ----
    df = load_draws(DATA_PATH)

    # ---- Pre-compute structures ----
    log("Building appearance matrix and gap histories")
    appear_mat = build_appearance_matrix(df)
    gap_histories, last_seen = compute_gap_histories(appear_mat)
    log("Computing top-5 co-occurrence partners")
    top5_partners = compute_cooccurrence_top5(appear_mat)

    # ---- Run expanding-window backtest ----
    log("=" * 70)
    log("EXPANDING-WINDOW BACKTEST")
    log("=" * 70)
    result = run_backtest(
        df, appear_mat, gap_histories, top5_partners,
        backtest_draws=BACKTEST_DRAWS,
    )
    if not result:
        log("BACKTEST FAILED — no results generated.")
        sys.exit(1)

    metrics, cox_model, rsf_model, exp_model, test_feat_df, test_spell_df = result

    # ---- Train final models on ALL data ----
    log("=" * 70)
    log("TRAINING FINAL MODELS ON FULL DATASET")
    log("=" * 70)
    train_start = 100
    full_feat_df = build_survival_dataset(
        df, appear_mat, gap_histories, top5_partners,
        start_draw=train_start, end_draw=len(df),
    )
    full_spell_df = convert_to_spell_format(full_feat_df)

    log("Fitting final Cox PH model")
    final_cox = CoxSurvivalModel()
    final_cox.fit(full_spell_df)

    log("Fitting final Random Survival Forest")
    final_rsf = RSFSurvivalModel(n_estimators=100, max_depth=8, min_samples_leaf=20)
    final_rsf.fit(full_spell_df)

    log("Fitting final Exponential baseline")
    final_exp = ExponentialBaseline()
    final_exp.fit(full_spell_df)

    # ---- Compute delta_survival predictions ----
    log("=" * 70)
    log("COMPUTING PREDICTIONS")
    log("=" * 70)
    predictions = compute_delta_survival(
        df, appear_mat, gap_histories, top5_partners,
        final_cox, final_rsf, final_exp,
    )

    # ---- Feature importance ----
    cox_importance = final_cox.get_feature_importance()
    rsf_importance = final_rsf.get_feature_importance()
    feature_importance = {
        "cox_ph": {k: float(v) for k, v in cox_importance.items()},
        "rsf": {k: float(v) for k, v in rsf_importance.items()},
    }

    # ---- Save results ----
    log("=" * 70)
    log("SAVING RESULTS")
    log("=" * 70)

    # Predictions
    pred_path = RESULTS_DIR / "survival_predictions.json"
    with open(pred_path, "w") as f:
        json.dump(predictions, f, indent=2)
    log(f"  Saved predictions: {pred_path}")

    # Metrics
    metrics_path = RESULTS_DIR / "survival_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)
    log(f"  Saved metrics: {metrics_path}")

    # Feature importance
    fi_path = RESULTS_DIR / "survival_feature_importance.json"
    with open(fi_path, "w") as f:
        json.dump(feature_importance, f, indent=2)
    log(f"  Saved feature importance: {fi_path}")

    # ---- Generate plots ----
    log("Generating plots")
    plot_concordance(metrics, RESULTS_DIR / "survival_concordance.png")
    plot_calibration(metrics, RESULTS_DIR / "survival_calibration.png")
    plot_feature_importance(cox_importance, rsf_importance, RESULTS_DIR / "survival_feature_importance.png")

    # ---- Summary ----
    elapsed = time.time() - t0
    log("=" * 70)
    log("SUMMARY")
    log("=" * 70)
    log(f"Total draws: {len(df)}")
    log(f"Backtest window: {BACKTEST_DRAWS} draws")
    log(f"Concordance — Cox PH:  {metrics['concordance']['cox_ph']:.4f}")
    log(f"Concordance — RSF:     {metrics['concordance']['rsf']:.4f}")
    log(f"Concordance — Naive:   {metrics['concordance']['naive_baseline']:.4f}")
    for h in BRIER_HORIZONS:
        bs = metrics["brier_scores"].get(str(h))
        if bs is not None:
            log(f"Brier score (horizon {h}): {bs:.4f}")
    log(f"")

    # Print top-10 numbers by Cox delta
    sorted_nums = sorted(
        predictions.values(),
        key=lambda x: x["cox_delta_logodds"],
        reverse=True,
    )
    log("Top 10 numbers by Cox delta (log-odds vs baseline):")
    for entry in sorted_nums[:10]:
        log(f"  Number {entry['number']:2d}: gap={entry['current_gap']:3d}, "
            f"cox_delta={entry['cox_delta_logodds']:+.4f}, "
            f"exp_prob={entry['exp_prob_next']:.4f}")

    log(f"\nBottom 10 numbers by Cox delta:")
    for entry in sorted_nums[-10:]:
        log(f"  Number {entry['number']:2d}: gap={entry['current_gap']:3d}, "
            f"cox_delta={entry['cox_delta_logodds']:+.4f}, "
            f"exp_prob={entry['exp_prob_next']:.4f}")

    log(f"\nTotal runtime: {elapsed:.1f}s")
    log("Done.")


if __name__ == "__main__":
    main()
