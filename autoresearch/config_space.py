"""
config_space.py -- Canonical search space definition for runner_v3.

Single source of truth for:
  - Valid values for all categorical parameters
  - Ranges for all numeric parameters
  - Default config (encodes experiment #269, best-generalizing v2 result)
  - Bootstrap configs (8 seeds covering diverse strategies)
"""

import json

# ---------------------------------------------------------------------------
# Valid categorical values
# ---------------------------------------------------------------------------

VALID_FEATURE_SETS = ["basic", "recency", "gaps", "positional", "temporal", "momentum", "equipment"]
# draw_time removed -- confirmed noise, no signal above equipment wear features
# equipment kept in validator list for backward compat with existing experiments,
# but excluded from random/LLM proposals by default (use --equipment flag to enable)
VALID_STAGES = ["select", "poly", "custom_interact", "scale"]
VALID_MODEL_TYPES = ["et", "xgb", "lgb", "rf", "hgb", "lr"]
VALID_SCALE_METHODS = ["standard", "robust", "quantile", "minmax"]
VALID_CALIB_METHODS = ["isotonic", "sigmoid"]
VALID_SELECTOR_TARGETS = ["d1", "d2", "d3", "mean"]
VALID_CALIB_CVS = [3, 5, 7, 10]
VALID_POLY_DEGREES = [2, 3]

# ---------------------------------------------------------------------------
# Parameter ranges: (type, min, max)
# type is "float" or "int"
# ---------------------------------------------------------------------------

PARAM_RANGES = {
    # select stage
    "select.threshold_multiplier":   ("float", 0.5,  3.0),
    "select.selector_n_estimators":  ("int",   50,   500),
    "select.selector_max_depth":     ("int",   5,    20),

    # poly stage
    "poly.top_k_variance":           ("int",   50,   500),

    # custom_interact stage
    "custom_interact.n_head":        ("int",   2,    15),
    "custom_interact.n_tail":        ("int",   2,    15),

    # model -- shared
    "model.n_estimators":            ("int",   50,   1000),
    "model.max_depth":               ("int",   3,    30),
    "model.min_samples_leaf":        ("int",   1,    20),
    "model.min_samples_split":       ("int",   2,    30),
    "model.max_features":            ("float", 0.1,  1.0),
    "model.weight":                  ("float", 0.05, 1.0),

    # model -- xgb / lgb
    "model.learning_rate":           ("float", 0.005, 0.3),
    "model.subsample":               ("float", 0.4,   1.0),
    "model.colsample_bytree":        ("float", 0.4,   1.0),
    "model.reg_alpha":               ("float", 0.0,   10.0),
    "model.reg_lambda":              ("float", 0.1,   10.0),

    # model -- lr
    "model.C":                       ("float", 0.001, 100.0),

    # model -- hgb
    "model.learning_rate_hgb":       ("float", 0.005, 0.3),
    "model.max_iter":                ("int",   50,   1000),
}

# ---------------------------------------------------------------------------
# Default stage configs
# ---------------------------------------------------------------------------

def default_stage(stage_type):
    """Return a default config dict for a given stage type."""
    if stage_type == "select":
        return {
            "stage": "select",
            "threshold_multiplier": 1.5,
            "selector_n_estimators": 200,
            "selector_max_depth": 10,
            "selector_target": "d1",
        }
    elif stage_type == "poly":
        return {
            "stage": "poly",
            "degree": 2,
            "interaction_only": False,
            "top_k_variance": 250,
        }
    elif stage_type == "custom_interact":
        return {
            "stage": "custom_interact",
            "n_head": 5,
            "n_tail": 5,
            "include_ratios": True,
        }
    elif stage_type == "scale":
        return {
            "stage": "scale",
            "method": "standard",
        }
    raise ValueError(f"Unknown stage type: {stage_type}")


# ---------------------------------------------------------------------------
# Default model configs (per type)
# ---------------------------------------------------------------------------

def default_model(model_type, weight=1.0):
    """Return a default model config dict for a given model type."""
    base = {"type": model_type, "weight": weight,
            "calibrate": False, "calibrate_method": "isotonic", "calibrate_cv": 5}
    if model_type in ("et", "rf"):
        return {**base, "n_estimators": 300, "max_depth": 15,
                "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.6}
    elif model_type == "xgb":
        return {**base, "n_estimators": 300, "max_depth": 6,
                "learning_rate": 0.05, "subsample": 0.8,
                "colsample_bytree": 0.7, "reg_alpha": 0.0, "reg_lambda": 1.0}
    elif model_type == "lgb":
        return {**base, "n_estimators": 300, "max_depth": 6,
                "learning_rate": 0.05, "subsample": 0.8,
                "colsample_bytree": 0.7, "reg_alpha": 0.0, "reg_lambda": 1.0}
    elif model_type == "hgb":
        return {**base, "max_iter": 300, "max_depth": 6, "learning_rate": 0.05}
    elif model_type == "lr":
        return {**base, "C": 1.0}
    raise ValueError(f"Unknown model type: {model_type}")


# ---------------------------------------------------------------------------
# Default config (encodes experiment #269 -- best-generalizing v2 result)
# mean_rank 414.80 on val, 415.03 on test (+0.24 delta -- near-perfect generalization)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "description": "Default: #269 encoding -- select->poly->custom_interact, et+xgb isotonic",
    "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
    "pipeline": [
        {
            "stage": "select",
            "threshold_multiplier": 1.5,
            "selector_n_estimators": 200,
            "selector_max_depth": 10,
            "selector_target": "d1",
        },
        {
            "stage": "poly",
            "degree": 2,
            "interaction_only": False,
            "top_k_variance": 250,
        },
        {
            "stage": "custom_interact",
            "n_head": 5,
            "n_tail": 5,
            "include_ratios": True,
        },
    ],
    "models": [
        {
            "type": "et",
            "weight": 0.6,
            "n_estimators": 600,
            "max_depth": 18,
            "min_samples_leaf": 3,
            "min_samples_split": 8,
            "max_features": 0.6,
            "calibrate": True,
            "calibrate_method": "isotonic",
            "calibrate_cv": 5,
        },
        {
            "type": "xgb",
            "weight": 0.4,
            "n_estimators": 400,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.7,
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "calibrate": False,
            "calibrate_method": "isotonic",
            "calibrate_cv": 5,
        },
    ],
}

# ---------------------------------------------------------------------------
# Bootstrap seeds (8 configs covering diverse strategies)
# ---------------------------------------------------------------------------

BOOTSTRAP_CONFIGS = [
    # Seed 1: #269 encoding (best generalizer, val=414.80, test=415.03)
    DEFAULT_CONFIG,

    # Seed 2: #323 approximation (best val=404.90) -- both ET and XGB calibrated
    # Reduced n_estimators and cv=3 to stay under 600s timeout
    {
        "description": "Bootstrap #323 approx -- both ET+XGB calibrated isotonic cv=3, select->poly->custom_interact",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "pipeline": [
            {"stage": "select", "threshold_multiplier": 1.5, "selector_n_estimators": 200,
             "selector_max_depth": 10, "selector_target": "d1"},
            {"stage": "poly", "degree": 2, "interaction_only": False, "top_k_variance": 250},
            {"stage": "custom_interact", "n_head": 5, "n_tail": 5, "include_ratios": True},
        ],
        "models": [
            {"type": "et", "weight": 0.6, "n_estimators": 300, "max_depth": 18,
             "min_samples_leaf": 3, "min_samples_split": 8, "max_features": 0.6,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 3},
            {"type": "xgb", "weight": 0.4, "n_estimators": 300, "max_depth": 6,
             "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 1.0,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 3},
        ],
    },

    # Seed 3: Minimal baseline -- no pipeline, single ET
    {
        "description": "Bootstrap minimal -- no pipeline, single ET(200)",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "pipeline": [],
        "models": [
            {"type": "et", "weight": 1.0, "n_estimators": 200, "max_depth": 15,
             "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.6,
             "calibrate": False, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Seed 4: Select-only pipeline, ET calibrated
    {
        "description": "Bootstrap select-only -- feature selection, ET calibrated isotonic",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "pipeline": [
            {"stage": "select", "threshold_multiplier": 1.5, "selector_n_estimators": 200,
             "selector_max_depth": 10, "selector_target": "d1"},
        ],
        "models": [
            {"type": "et", "weight": 1.0, "n_estimators": 400, "max_depth": 18,
             "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.6,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Seed 5: Scale-first -- novel ordering never tried in v2
    {
        "description": "Bootstrap scale-first -- robust scale -> select -> poly, XGB calib sigmoid",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "pipeline": [
            {"stage": "scale", "method": "robust"},
            {"stage": "select", "threshold_multiplier": 1.5, "selector_n_estimators": 200,
             "selector_max_depth": 10, "selector_target": "d1"},
            {"stage": "poly", "degree": 2, "interaction_only": False, "top_k_variance": 250},
        ],
        "models": [
            {"type": "xgb", "weight": 1.0, "n_estimators": 300, "max_depth": 6,
             "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 1.0,
             "calibrate": True, "calibrate_method": "sigmoid", "calibrate_cv": 5},
        ],
    },

    # Seed 6: LGB variant
    {
        "description": "Bootstrap LGB -- select->poly, LightGBM calibrated",
        "feature_sets": ["basic", "recency", "gaps", "positional", "momentum"],
        "pipeline": [
            {"stage": "select", "threshold_multiplier": 1.5, "selector_n_estimators": 200,
             "selector_max_depth": 10, "selector_target": "d1"},
            {"stage": "poly", "degree": 2, "interaction_only": False, "top_k_variance": 200},
        ],
        "models": [
            {"type": "lgb", "weight": 1.0, "n_estimators": 400, "max_depth": 6,
             "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 1.0,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Seed 7: Three-model ensemble with select gate (no pipeline hurt badly without feature selection)
    {
        "description": "Bootstrap 3-model ensemble -- select + ET + XGB + LGB calibrated",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "pipeline": [
            {"stage": "select", "threshold_multiplier": 1.5, "selector_n_estimators": 200,
             "selector_max_depth": 10, "selector_target": "mean"},
        ],
        "models": [
            {"type": "et", "weight": 0.4, "n_estimators": 300, "max_depth": 15,
             "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.6,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
            {"type": "xgb", "weight": 0.35, "n_estimators": 300, "max_depth": 6,
             "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 1.0,
             "calibrate": False, "calibrate_method": "isotonic", "calibrate_cv": 5},
            {"type": "lgb", "weight": 0.25, "n_estimators": 300, "max_depth": 6,
             "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 1.0,
             "calibrate": False, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Seed 8: Poly-then-select (novel ordering -- v2 always did select->poly)
    {
        "description": "Bootstrap poly-then-select -- novel ordering: poly(deg=2) -> select(1.0x), ET calib",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "pipeline": [
            {"stage": "poly", "degree": 2, "interaction_only": False, "top_k_variance": 300},
            {"stage": "select", "threshold_multiplier": 1.0, "selector_n_estimators": 200,
             "selector_max_depth": 10, "selector_target": "d1"},
        ],
        "models": [
            {"type": "et", "weight": 1.0, "n_estimators": 500, "max_depth": 18,
             "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.6,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # -------------------------------------------------------------------------
    # CORNER SEEDS -- systematically cover unexplored extremes of the search
    # space. The GA collapsed into one basin early (et+xgb+lgb + select + 6
    # features). These seeds place parents at the true corners so crossover
    # and mutation have the full space to work from.
    # -------------------------------------------------------------------------

    # Corner A: LR baseline -- the simplest possible model (never tried)
    {
        "description": "CORNER: LR baseline -- single logistic regression, no pipeline, basic only",
        "feature_sets": ["basic"],
        "pipeline": [],
        "models": [
            {"type": "lr", "weight": 1.0, "C": 1.0,
             "calibrate": False, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner B: LR + scale + select -- LR needs scaling; full-feature linear model
    {
        "description": "CORNER: LR scaled -- scale(standard)->select(mean), all 6 features",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "pipeline": [
            {"stage": "scale", "method": "standard"},
            {"stage": "select", "threshold_multiplier": 1.5, "selector_n_estimators": 200,
             "selector_max_depth": 10, "selector_target": "mean"},
        ],
        "models": [
            {"type": "lr", "weight": 1.0, "C": 0.1,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner C: RF-only -- model type with only 1 experiment ever (none successful)
    {
        "description": "CORNER: RF-only -- no pipeline, 6 features, deep forest",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "pipeline": [],
        "models": [
            {"type": "rf", "weight": 1.0, "n_estimators": 500, "max_depth": 20,
             "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.5,
             "calibrate": False, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner D: RF + select -- give RF the same feature gating that helped ET
    {
        "description": "CORNER: RF + select(mean) -- 6 features, calibrated isotonic",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "pipeline": [
            {"stage": "select", "threshold_multiplier": 1.5, "selector_n_estimators": 200,
             "selector_max_depth": 10, "selector_target": "mean"},
        ],
        "models": [
            {"type": "rf", "weight": 1.0, "n_estimators": 500, "max_depth": 20,
             "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.5,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner E: HGB alone, no pipeline -- HGB handles mixed types natively
    {
        "description": "CORNER: HGB-only no pipeline -- raw 6 features, HistGradientBoosting",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "pipeline": [],
        "models": [
            {"type": "hgb", "weight": 1.0, "max_iter": 400, "max_depth": 8,
             "learning_rate": 0.05,
             "calibrate": False, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner F: Minimal features -- basic only, deep single XGB
    {
        "description": "CORNER: minimal features -- basic(11) only, single XGB deep, no pipeline",
        "feature_sets": ["basic"],
        "pipeline": [],
        "models": [
            {"type": "xgb", "weight": 1.0, "n_estimators": 500, "max_depth": 8,
             "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8,
             "reg_alpha": 0.1, "reg_lambda": 1.0,
             "calibrate": False, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner G: basic + recency only -- positional/gaps/momentum stripped out
    {
        "description": "CORNER: basic+recency only -- stripped features, LGB+ET ensemble, select",
        "feature_sets": ["basic", "recency"],
        "pipeline": [
            {"stage": "select", "threshold_multiplier": 1.0, "selector_n_estimators": 200,
             "selector_max_depth": 10, "selector_target": "mean"},
        ],
        "models": [
            {"type": "lgb", "weight": 0.5, "n_estimators": 400, "max_depth": 6,
             "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 1.0,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
            {"type": "et", "weight": 0.5, "n_estimators": 400, "max_depth": 15,
             "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.6,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner H: equipment + basic only -- pure equipment signal isolation
    {
        "description": "CORNER: equipment+basic only -- no pipeline, ET, pure wear signal test",
        "feature_sets": ["basic", "equipment"],
        "pipeline": [],
        "models": [
            {"type": "et", "weight": 1.0, "n_estimators": 400, "max_depth": 15,
             "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.6,
             "calibrate": False, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner I: equipment + full 5-set -- equipment in context, no temporal
    {
        "description": "CORNER: equipment+5sets -- basic+recency+gaps+positional+momentum+equipment, select, ET+LGB",
        "feature_sets": ["basic", "recency", "gaps", "positional", "momentum", "equipment"],
        "pipeline": [
            {"stage": "select", "threshold_multiplier": 1.5, "selector_n_estimators": 200,
             "selector_max_depth": 10, "selector_target": "mean"},
        ],
        "models": [
            {"type": "et", "weight": 0.5, "n_estimators": 400, "max_depth": 15,
             "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.6,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
            {"type": "lgb", "weight": 0.5, "n_estimators": 400, "max_depth": 6,
             "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 1.0,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner J: scale-only pipeline (no select) -- almost never tried
    {
        "description": "CORNER: scale-only pipeline -- quantile scale, no select, ET+XGB",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "pipeline": [
            {"stage": "scale", "method": "quantile"},
        ],
        "models": [
            {"type": "et", "weight": 0.6, "n_estimators": 400, "max_depth": 15,
             "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.6,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
            {"type": "xgb", "weight": 0.4, "n_estimators": 400, "max_depth": 6,
             "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 1.0,
             "calibrate": False, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner K: custom_interact-only (no select) -- interaction features without pruning
    {
        "description": "CORNER: custom_interact only (no select) -- interaction+ratios, ET+LGB",
        "feature_sets": ["basic", "recency", "gaps", "positional", "momentum"],
        "pipeline": [
            {"stage": "custom_interact", "n_head": 8, "n_tail": 8, "include_ratios": True},
        ],
        "models": [
            {"type": "et", "weight": 0.5, "n_estimators": 400, "max_depth": 15,
             "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.6,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
            {"type": "lgb", "weight": 0.5, "n_estimators": 400, "max_depth": 6,
             "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 1.0,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner L: HGB + LR ensemble -- maximum architectural distance from et+xgb+lgb
    {
        "description": "CORNER: HGB+LR ensemble -- max diversity, scale->select, 6 features",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "pipeline": [
            {"stage": "scale", "method": "standard"},
            {"stage": "select", "threshold_multiplier": 1.5, "selector_n_estimators": 200,
             "selector_max_depth": 10, "selector_target": "mean"},
        ],
        "models": [
            {"type": "hgb", "weight": 0.7, "max_iter": 400, "max_depth": 8,
             "learning_rate": 0.05,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
            {"type": "lr", "weight": 0.3, "C": 1.0,
             "calibrate": False, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner M: RF + LGB -- tree diversity combo never tried
    {
        "description": "CORNER: RF+LGB ensemble -- select(mean), 6 features, calibrated",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "pipeline": [
            {"stage": "select", "threshold_multiplier": 1.5, "selector_n_estimators": 200,
             "selector_max_depth": 10, "selector_target": "mean"},
        ],
        "models": [
            {"type": "rf", "weight": 0.5, "n_estimators": 400, "max_depth": 20,
             "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.5,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
            {"type": "lgb", "weight": 0.5, "n_estimators": 400, "max_depth": 6,
             "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 1.0,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner N: single XGB, deep, heavily regularized, no pipeline -- raw power test
    {
        "description": "CORNER: single deep XGB no pipeline -- max_depth=10, heavy reg, 6 features",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum"],
        "pipeline": [],
        "models": [
            {"type": "xgb", "weight": 1.0, "n_estimators": 600, "max_depth": 10,
             "learning_rate": 0.03, "subsample": 0.7, "colsample_bytree": 0.6,
             "reg_alpha": 1.0, "reg_lambda": 5.0,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner O: all 8 features -- kitchen sink including equipment+temporal
    {
        "description": "CORNER: all 8 feature sets -- kitchen sink, select(mean), ET+XGB+LGB",
        "feature_sets": ["basic", "recency", "gaps", "positional", "temporal", "momentum", "equipment"],
        "pipeline": [
            {"stage": "select", "threshold_multiplier": 1.5, "selector_n_estimators": 200,
             "selector_max_depth": 10, "selector_target": "mean"},
        ],
        "models": [
            {"type": "et", "weight": 0.4, "n_estimators": 300, "max_depth": 15,
             "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.6,
             "calibrate": True, "calibrate_method": "isotonic", "calibrate_cv": 5},
            {"type": "xgb", "weight": 0.35, "n_estimators": 300, "max_depth": 6,
             "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 1.0,
             "calibrate": False, "calibrate_method": "isotonic", "calibrate_cv": 5},
            {"type": "lgb", "weight": 0.25, "n_estimators": 300, "max_depth": 6,
             "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.7,
             "reg_alpha": 0.0, "reg_lambda": 1.0,
             "calibrate": False, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },

    # Corner P: recency-only -- isolate recency signal, no pipeline, single ET
    {
        "description": "CORNER: recency-only features -- no pipeline, single ET, isolate frequency signal",
        "feature_sets": ["recency"],
        "pipeline": [],
        "models": [
            {"type": "et", "weight": 1.0, "n_estimators": 400, "max_depth": 15,
             "min_samples_leaf": 3, "min_samples_split": 6, "max_features": 0.6,
             "calibrate": False, "calibrate_method": "isotonic", "calibrate_cv": 5},
        ],
    },
]


# ---------------------------------------------------------------------------
# Random config generator -- for seeding diverse initial populations
# ---------------------------------------------------------------------------

import random as _random


def _rand_float(lo, hi):
    return round(_random.uniform(lo, hi), 4)


def _rand_int(lo, hi):
    return _random.randint(lo, hi)


def _rand_stage(stage_type):
    """Return a randomly parameterized stage config."""
    if stage_type == "select":
        return {
            "stage": "select",
            "threshold_multiplier": _rand_float(0.5, 3.0),
            "selector_n_estimators": _rand_int(50, 500),
            "selector_max_depth": _rand_int(5, 20),
            "selector_target": _random.choice(VALID_SELECTOR_TARGETS),
        }
    elif stage_type == "poly":
        return {
            "stage": "poly",
            "degree": _random.choice(VALID_POLY_DEGREES),
            "interaction_only": _random.choice([True, False]),
            "top_k_variance": _rand_int(50, 500),
        }
    elif stage_type == "custom_interact":
        return {
            "stage": "custom_interact",
            "n_head": _rand_int(2, 15),
            "n_tail": _rand_int(2, 15),
            "include_ratios": _random.choice([True, False]),
        }
    elif stage_type == "scale":
        return {
            "stage": "scale",
            "method": _random.choice(VALID_SCALE_METHODS),
        }
    raise ValueError(f"Unknown stage: {stage_type}")


def _rand_model(model_type):
    """Return a randomly parameterized model config."""
    weight = _rand_float(0.05, 1.0)
    calibrate = _random.choice([True, False])
    base = {
        "type": model_type,
        "weight": weight,
        "calibrate": calibrate,
        "calibrate_method": _random.choice(VALID_CALIB_METHODS),
        "calibrate_cv": _random.choice(VALID_CALIB_CVS),
    }
    if model_type in ("et", "rf"):
        return {**base,
                "n_estimators": _rand_int(50, 1000),
                "max_depth": _rand_int(3, 30),
                "min_samples_leaf": _rand_int(1, 20),
                "min_samples_split": _rand_int(2, 30),
                "max_features": _rand_float(0.1, 1.0)}
    elif model_type in ("xgb", "lgb"):
        return {**base,
                "n_estimators": _rand_int(50, 1000),
                "max_depth": _rand_int(3, 15),
                "learning_rate": _rand_float(0.005, 0.3),
                "subsample": _rand_float(0.4, 1.0),
                "colsample_bytree": _rand_float(0.4, 1.0),
                "reg_alpha": _rand_float(0.0, 10.0),
                "reg_lambda": _rand_float(0.1, 10.0)}
    elif model_type == "hgb":
        return {**base,
                "max_iter": _rand_int(50, 1000),
                "max_depth": _rand_int(3, 15),
                "learning_rate": _rand_float(0.005, 0.3)}
    elif model_type == "lr":
        return {**base, "C": _rand_float(0.001, 100.0)}
    raise ValueError(f"Unknown model type: {model_type}")


def generate_random_config(allow_equipment=False):
    """
    Sample a fully random config from the valid config space.

    Used to seed diverse initial populations before the GA has enough
    experiments to work from. Every call produces a structurally different
    config -- model types, feature sets, and pipeline stages are all
    independently randomized.

    Parameters
    ----------
    allow_equipment : bool
        If True, equipment features may be included. Default False -- equipment
        is disabled by default until a better encoding strategy is confirmed.
        Enable via runner_v4.py --equipment flag.
    """
    # Feature sets: draw from core + temporal only by default
    n_sets = _random.randint(2, 6)
    core = ["basic", "recency", "gaps", "positional", "momentum"]
    pool = core + ["temporal"]
    if allow_equipment:
        pool.append("equipment")
    # Always pick at least one core set
    chosen = set(_random.sample(core, min(n_sets - 1, len(core))))
    remaining = [s for s in pool if s not in chosen]
    _random.shuffle(remaining)
    chosen.update(remaining[:max(0, n_sets - len(chosen))])
    feature_sets = [s for s in VALID_FEATURE_SETS if s in chosen]  # preserve canonical order

    # Pipeline: 0-3 stages, no duplicates, max 3
    n_stages = _random.randint(0, 3)
    stage_pool = _random.sample(VALID_STAGES, min(n_stages, len(VALID_STAGES)))
    pipeline = [_rand_stage(s) for s in stage_pool]

    # Models: 1-3, all types equally likely
    n_models = _random.randint(1, 3)
    model_types = _random.choices(VALID_MODEL_TYPES, k=n_models)
    models = [_rand_model(t) for t in model_types]

    return {
        "description": f"random: {'+'.join(feature_sets[:3])} | "
                       f"{'+'.join(s['stage'][:3] for s in pipeline) or 'no-pipe'} | "
                       f"{'+'.join(m['type'] for m in models)}",
        "feature_sets": feature_sets,
        "pipeline": pipeline,
        "models": models,
    }


# ---------------------------------------------------------------------------
# Quick sanity check when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"VALID_FEATURE_SETS: {VALID_FEATURE_SETS}")
    print(f"VALID_MODEL_TYPES:  {VALID_MODEL_TYPES}")
    print(f"PARAM_RANGES:       {len(PARAM_RANGES)} entries")
    print(f"BOOTSTRAP_CONFIGS:  {len(BOOTSTRAP_CONFIGS)} seeds")
    print("\nSample random configs:")
    for i in range(3):
        cfg = generate_random_config()
        print(f"  {i+1}: {cfg['description']}")
    print()
    print("DEFAULT_CONFIG:")
    print(json.dumps(DEFAULT_CONFIG, indent=2))
    print()
    print(f"All bootstrap config descriptions:")
    for i, cfg in enumerate(BOOTSTRAP_CONFIGS, 1):
        print(f"  {i}. {cfg['description']}")
