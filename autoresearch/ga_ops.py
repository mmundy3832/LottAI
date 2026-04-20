"""
ga_ops.py -- Genetic algorithm mutation and crossover operations on v3 configs.

All mutations:
  - Operate on deep copies (never in-place)
  - Call validate_config() before returning
  - Return the original config unchanged if the mutation produces an invalid config

Public API:
  pick_mutation(config) -> config      # top-level dispatcher
  crossover(config_a, config_b) -> config
"""

import copy
import math
import random

from config_space import (
    PARAM_RANGES, VALID_FEATURE_SETS, VALID_STAGES, VALID_MODEL_TYPES,
    VALID_SCALE_METHODS, VALID_CALIB_METHODS, VALID_SELECTOR_TARGETS,
    VALID_CALIB_CVS, VALID_POLY_DEGREES, default_stage, default_model,
)
from config_validator import validate_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp(value, lo, hi):
    return max(lo, min(hi, value))


def _nudge_float(value, lo, hi, sigma_frac):
    sigma = (hi - lo) * sigma_frac
    new_val = value + random.gauss(0, sigma)
    return _clamp(new_val, lo, hi)


def _nudge_int(value, lo, hi, sigma_frac):
    sigma = max(1.0, (hi - lo) * sigma_frac)
    new_val = value + random.gauss(0, sigma)
    return int(round(_clamp(new_val, lo, hi)))


def _apply_nudge(obj, key, range_key, sigma_frac=0.15):
    """Nudge obj[key] according to PARAM_RANGES[range_key].

    80% of the time: gaussian nudge (local exploration).
    10% of the time: snap near the minimum of the range.
    10% of the time: snap near the maximum of the range.
    """
    if range_key not in PARAM_RANGES:
        return
    ptype, lo, hi = PARAM_RANGES[range_key]
    val = obj.get(key)
    if val is None:
        return

    roll = random.random()
    if roll < 0.10:
        # Near-minimum: sample uniformly from the bottom 10% of range
        new_val = lo + (hi - lo) * random.uniform(0.0, 0.10)
    elif roll < 0.20:
        # Near-maximum: sample uniformly from the top 10% of range
        new_val = hi - (hi - lo) * random.uniform(0.0, 0.10)
    else:
        # Normal gaussian nudge
        if ptype == "float":
            obj[key] = _nudge_float(float(val), lo, hi, sigma_frac)
            return
        else:
            obj[key] = _nudge_int(int(val), lo, hi, sigma_frac)
            return

    # Apply the extreme value
    if ptype == "float":
        obj[key] = _clamp(new_val, lo, hi)
    else:
        obj[key] = int(round(_clamp(new_val, lo, hi)))


def _validated(cfg, original):
    """Return cfg if valid, else return original."""
    ok, _ = validate_config(cfg)
    return cfg if ok else original


# ---------------------------------------------------------------------------
# Feature set mutations
# ---------------------------------------------------------------------------

def mutate_feature_sets(config):
    """Add, remove, or swap one feature set."""
    cfg = copy.deepcopy(config)
    current = list(cfg["feature_sets"])
    current_set = set(current)
    all_sets = set(VALID_FEATURE_SETS)
    missing = list(all_sets - current_set)

    roll = random.random()
    if roll < 0.35 and missing:
        # Add one
        current.append(random.choice(missing))
    elif roll < 0.65 and len(current) > 2:
        # Remove one
        current.remove(random.choice(current))
    elif missing and len(current) > 1:
        # Swap one
        remove = random.choice(current)
        current.remove(remove)
        remaining_missing = list(all_sets - set(current))
        if remaining_missing:
            current.append(random.choice(remaining_missing))
    else:
        return config  # no-op

    cfg["feature_sets"] = current
    return _validated(cfg, config)


# ---------------------------------------------------------------------------
# Pipeline mutations
# ---------------------------------------------------------------------------

def _nudge_stage_param(stage_cfg):
    """Nudge a random numeric param within a stage config (in-place)."""
    s = stage_cfg["stage"]
    sigma = random.uniform(0.1, 0.25)

    if s == "select":
        choice = random.choice(["threshold_multiplier", "selector_n_estimators", "selector_max_depth"])
        key_map = {
            "threshold_multiplier": "select.threshold_multiplier",
            "selector_n_estimators": "select.selector_n_estimators",
            "selector_max_depth": "select.selector_max_depth",
        }
        _apply_nudge(stage_cfg, choice, key_map[choice], sigma)

    elif s == "poly":
        choice = random.choice(["degree", "top_k_variance"])
        if choice == "degree":
            stage_cfg["degree"] = random.choice(VALID_POLY_DEGREES)
        else:
            _apply_nudge(stage_cfg, "top_k_variance", "poly.top_k_variance", sigma)

    elif s == "custom_interact":
        choice = random.choice(["n_head", "n_tail", "include_ratios"])
        if choice == "include_ratios":
            stage_cfg["include_ratios"] = not stage_cfg.get("include_ratios", True)
        elif choice == "n_head":
            _apply_nudge(stage_cfg, "n_head", "custom_interact.n_head", sigma)
        else:
            _apply_nudge(stage_cfg, "n_tail", "custom_interact.n_tail", sigma)

    elif s == "scale":
        stage_cfg["method"] = random.choice(VALID_SCALE_METHODS)


def mutate_pipeline(config):
    """Insert, delete, swap adjacent stages, or nudge a stage param."""
    cfg = copy.deepcopy(config)
    pipeline = cfg["pipeline"]
    existing_types = {s["stage"] for s in pipeline}

    ops = ["param_nudge"]
    insertable = [s for s in VALID_STAGES if s not in existing_types]
    if insertable:
        ops.append("insert")
    if len(pipeline) > 0:
        ops.append("delete")
    if len(pipeline) >= 2:
        ops.append("reorder")
    if len(pipeline) >= 1:
        ops.append("selector_target")  # change select target if select stage exists

    op = random.choice(ops)

    if op == "insert" and insertable:
        new_stage_type = random.choice(insertable)
        new_stage = default_stage(new_stage_type)
        pos = random.randint(0, len(pipeline))
        pipeline.insert(pos, new_stage)

    elif op == "delete" and pipeline:
        idx = random.randrange(len(pipeline))
        pipeline.pop(idx)

    elif op == "reorder" and len(pipeline) >= 2:
        # Swap two adjacent stages -- select->poly vs poly->select are qualitatively different
        i = random.randrange(len(pipeline) - 1)
        pipeline[i], pipeline[i + 1] = pipeline[i + 1], pipeline[i]

    elif op == "param_nudge" and pipeline:
        idx = random.randrange(len(pipeline))
        _nudge_stage_param(pipeline[idx])

    elif op == "selector_target":
        for s in pipeline:
            if s["stage"] == "select":
                s["selector_target"] = random.choice(VALID_SELECTOR_TARGETS)
                break

    cfg["pipeline"] = pipeline
    return _validated(cfg, config)


# ---------------------------------------------------------------------------
# Model mutations
# ---------------------------------------------------------------------------

def _nudge_model_param(m):
    """Nudge a random numeric param within a model config (in-place)."""
    mtype = m["type"]
    sigma = random.uniform(0.1, 0.25)

    if mtype in ("et", "rf"):
        choice = random.choice(["n_estimators", "max_depth", "min_samples_leaf",
                                 "min_samples_split", "max_features"])
        key_map = {
            "n_estimators": "model.n_estimators",
            "max_depth": "model.max_depth",
            "min_samples_leaf": "model.min_samples_leaf",
            "min_samples_split": "model.min_samples_split",
            "max_features": "model.max_features",
        }
        _apply_nudge(m, choice, key_map[choice], sigma)

    elif mtype in ("xgb", "lgb"):
        choice = random.choice(["n_estimators", "max_depth", "learning_rate",
                                 "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"])
        key_map = {
            "n_estimators": "model.n_estimators",
            "max_depth": "model.max_depth",
            "learning_rate": "model.learning_rate",
            "subsample": "model.subsample",
            "colsample_bytree": "model.colsample_bytree",
            "reg_alpha": "model.reg_alpha",
            "reg_lambda": "model.reg_lambda",
        }
        _apply_nudge(m, choice, key_map[choice], sigma)

    elif mtype == "hgb":
        choice = random.choice(["max_iter", "max_depth", "learning_rate"])
        if choice == "max_iter":
            _apply_nudge(m, "max_iter", "model.max_iter", sigma)
        elif choice == "max_depth":
            _apply_nudge(m, "max_depth", "model.max_depth", sigma)
        else:
            _apply_nudge(m, "learning_rate", "model.learning_rate_hgb", sigma)

    elif mtype == "lr":
        _apply_nudge(m, "C", "model.C", sigma)


def mutate_model(config):
    """Add/remove a model, change type, toggle calibration, or nudge a hyperparameter."""
    cfg = copy.deepcopy(config)
    models = cfg["models"]
    existing_types = {m["type"] for m in models}

    ops = ["nudge_param", "toggle_calibration", "nudge_weight",
           "calibrate_method", "calibrate_cv"]
    if len(models) < 3:
        ops.append("add")
    if len(models) > 1:
        ops.append("remove")
    ops.append("change_type")

    op = random.choice(ops)
    idx = random.randrange(len(models))

    if op == "nudge_param":
        _nudge_model_param(models[idx])

    elif op == "toggle_calibration":
        models[idx]["calibrate"] = not models[idx].get("calibrate", False)

    elif op == "nudge_weight":
        for m in models:
            m["weight"] = max(0.05, m.get("weight", 1.0) + random.gauss(0, 0.1))

    elif op == "calibrate_method":
        models[idx]["calibrate_method"] = random.choice(VALID_CALIB_METHODS)

    elif op == "calibrate_cv":
        models[idx]["calibrate_cv"] = random.choice(VALID_CALIB_CVS)

    elif op == "add":
        new_types = [t for t in VALID_MODEL_TYPES if t not in existing_types]
        if not new_types:
            new_types = VALID_MODEL_TYPES
        new_type = random.choice(new_types)
        models.append(default_model(new_type, weight=0.3))

    elif op == "remove" and len(models) > 1:
        models.pop(idx)

    elif op == "change_type":
        new_type = random.choice(VALID_MODEL_TYPES)
        old_weight = models[idx].get("weight", 1.0)
        models[idx] = default_model(new_type, weight=old_weight)

    cfg["models"] = models
    return _validated(cfg, config)


# ---------------------------------------------------------------------------
# Discrete param mutation
# ---------------------------------------------------------------------------

def mutate_discrete(config):
    """Resample a single categorical parameter to a new valid value."""
    cfg = copy.deepcopy(config)

    # Collect all mutable categorical params with their locations
    choices = []

    # Pipeline stage categoricals
    for i, stage in enumerate(cfg.get("pipeline", [])):
        s = stage.get("stage")
        if s == "select":
            choices.append(("pipeline_stage_select_target", i))
        elif s == "poly":
            choices.append(("pipeline_stage_poly_degree", i))
        elif s == "scale":
            choices.append(("pipeline_stage_scale_method", i))
        elif s == "custom_interact":
            choices.append(("pipeline_stage_ci_ratios", i))

    # Model categoricals
    for i in range(len(cfg.get("models", []))):
        choices.append(("model_calib_method", i))
        choices.append(("model_calib_cv", i))

    if not choices:
        return config

    pick, idx = random.choice(choices)

    if pick == "pipeline_stage_select_target":
        cfg["pipeline"][idx]["selector_target"] = random.choice(VALID_SELECTOR_TARGETS)
    elif pick == "pipeline_stage_poly_degree":
        cfg["pipeline"][idx]["degree"] = random.choice(VALID_POLY_DEGREES)
    elif pick == "pipeline_stage_scale_method":
        cfg["pipeline"][idx]["method"] = random.choice(VALID_SCALE_METHODS)
    elif pick == "pipeline_stage_ci_ratios":
        cfg["pipeline"][idx]["include_ratios"] = random.choice([True, False])
    elif pick == "model_calib_method":
        cfg["models"][idx]["calibrate_method"] = random.choice(VALID_CALIB_METHODS)
    elif pick == "model_calib_cv":
        cfg["models"][idx]["calibrate_cv"] = random.choice(VALID_CALIB_CVS)

    return _validated(cfg, config)


# ---------------------------------------------------------------------------
# Continuous param mutation
# ---------------------------------------------------------------------------

def mutate_continuous(config):
    """Gaussian nudge a single numeric param chosen at random."""
    cfg = copy.deepcopy(config)
    sigma_frac = random.uniform(0.05, 0.25)

    # Collect all mutable numeric params with their locations
    numeric_params = []

    for i, stage in enumerate(cfg.get("pipeline", [])):
        s = stage.get("stage")
        if s == "select":
            for key, rkey in [
                ("threshold_multiplier", "select.threshold_multiplier"),
                ("selector_n_estimators", "select.selector_n_estimators"),
                ("selector_max_depth", "select.selector_max_depth"),
            ]:
                if key in stage:
                    numeric_params.append(("pipeline", i, key, rkey))
        elif s == "poly":
            numeric_params.append(("pipeline", i, "top_k_variance", "poly.top_k_variance"))
        elif s == "custom_interact":
            numeric_params.append(("pipeline", i, "n_head", "custom_interact.n_head"))
            numeric_params.append(("pipeline", i, "n_tail", "custom_interact.n_tail"))

    for i, m in enumerate(cfg.get("models", [])):
        mtype = m.get("type")
        if mtype in ("et", "rf"):
            for key, rkey in [
                ("n_estimators", "model.n_estimators"),
                ("max_depth", "model.max_depth"),
                ("min_samples_leaf", "model.min_samples_leaf"),
                ("max_features", "model.max_features"),
            ]:
                numeric_params.append(("model", i, key, rkey))
        elif mtype in ("xgb", "lgb"):
            for key, rkey in [
                ("n_estimators", "model.n_estimators"),
                ("max_depth", "model.max_depth"),
                ("learning_rate", "model.learning_rate"),
                ("subsample", "model.subsample"),
                ("colsample_bytree", "model.colsample_bytree"),
                ("reg_alpha", "model.reg_alpha"),
                ("reg_lambda", "model.reg_lambda"),
            ]:
                numeric_params.append(("model", i, key, rkey))
        elif mtype == "hgb":
            for key, rkey in [
                ("max_iter", "model.max_iter"),
                ("max_depth", "model.max_depth"),
                ("learning_rate", "model.learning_rate_hgb"),
            ]:
                numeric_params.append(("model", i, key, rkey))
        elif mtype == "lr":
            numeric_params.append(("model", i, "C", "model.C"))

    if not numeric_params:
        return config

    location, idx, key, rkey = random.choice(numeric_params)
    if location == "pipeline":
        _apply_nudge(cfg["pipeline"][idx], key, rkey, sigma_frac)
    else:
        _apply_nudge(cfg["models"][idx], key, rkey, sigma_frac)

    return _validated(cfg, config)


# ---------------------------------------------------------------------------
# Crossover
# ---------------------------------------------------------------------------

def crossover(config_a, config_b):
    """
    Combine components from two parent configs.

    Each of feature_sets, pipeline, models is independently
    taken from parent A or B with 50% probability.
    """
    cfg = copy.deepcopy(config_a)

    if random.random() < 0.5:
        cfg["feature_sets"] = copy.deepcopy(config_b["feature_sets"])

    if random.random() < 0.5:
        cfg["pipeline"] = copy.deepcopy(config_b["pipeline"])

    if random.random() < 0.5:
        cfg["models"] = copy.deepcopy(config_b["models"])
    elif len(config_b["models"]) > 0 and len(cfg["models"]) < 3:
        # Fine-grained: splice one model from B into A
        b_model = copy.deepcopy(random.choice(config_b["models"]))
        cfg["models"].append(b_model)

    # Build a description
    a_id = config_a.get("_experiment_id", "A")
    b_id = config_b.get("_experiment_id", "B")
    cfg["description"] = f"CROSSOVER from #{a_id} x #{b_id}"

    return _validated(cfg, config_a)


# ---------------------------------------------------------------------------
# Top-level dispatcher
# ---------------------------------------------------------------------------

# Mutation type weights (must sum to 100 for clarity)
_MUTATION_WEIGHTS = [
    ("continuous",    30),
    ("pipeline",      20),
    ("model",         20),
    ("feature_sets",  15),
    ("discrete",      15),
]
_MUTATION_TYPES  = [m for m, _ in _MUTATION_WEIGHTS]
_MUTATION_PROBS  = [w for _, w in _MUTATION_WEIGHTS]


def pick_mutation(config, mutation_type=None):
    """
    Pick a random mutation and apply it to config.

    Parameters
    ----------
    config : dict
        A valid config dict.
    mutation_type : str or None
        If provided, force this mutation type. Otherwise sampled by weight.

    Returns
    -------
    dict
        Mutated config (deep copy). Original if mutation produced invalid config.
    """
    if mutation_type is None:
        mutation_type = random.choices(_MUTATION_TYPES, weights=_MUTATION_PROBS)[0]

    if mutation_type == "continuous":
        return mutate_continuous(config)
    elif mutation_type == "pipeline":
        return mutate_pipeline(config)
    elif mutation_type == "model":
        return mutate_model(config)
    elif mutation_type == "feature_sets":
        return mutate_feature_sets(config)
    elif mutation_type == "discrete":
        return mutate_discrete(config)

    return config  # unknown type -- no-op


# ---------------------------------------------------------------------------
# Tournament selection
# ---------------------------------------------------------------------------

def select_parents(population, top_n=15):
    """
    Tournament selection from the population.

    Returns (parent_a, parent_b_or_None).
    Both are experiment log entries (dicts with 'config', 'results', etc.)
    """
    pool = population[:top_n]
    if not pool:
        return None, None

    # Parent A: tournament of 3
    tournament_a = random.sample(pool, min(3, len(pool)))
    parent_a = min(tournament_a, key=lambda e: e["results"]["mean_rank"])

    if len(pool) < 2:
        return parent_a, None

    # Parent B: tournament of 3 from remaining
    remaining = [e for e in pool if e is not parent_a]
    if not remaining:
        return parent_a, None
    tournament_b = random.sample(remaining, min(3, len(remaining)))
    parent_b = min(tournament_b, key=lambda e: e["results"]["mean_rank"])

    return parent_a, parent_b


# ---------------------------------------------------------------------------
# Self-test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config_space import DEFAULT_CONFIG, BOOTSTRAP_CONFIGS

    print("Running GA ops smoke test (100 mutations + 20 crossovers)...\n")

    all_ok = True
    cfg_a = copy.deepcopy(DEFAULT_CONFIG)
    cfg_b = copy.deepcopy(BOOTSTRAP_CONFIGS[7])  # poly-then-select (very different)

    # Test each mutation type
    for mtype in _MUTATION_TYPES:
        successes = 0
        for _ in range(20):
            mutated = pick_mutation(cfg_a, mutation_type=mtype)
            ok, errs = validate_config(mutated)
            if ok:
                successes += 1
            else:
                print(f"  [WARN] {mtype} produced invalid config: {errs[:1]}")
        print(f"  [{mtype}]: {successes}/20 valid mutations")

    # Test crossover
    print("\n  Crossover tests:")
    cross_ok = 0
    for _ in range(20):
        crossed = crossover(cfg_a, cfg_b)
        ok, errs = validate_config(crossed)
        if ok:
            cross_ok += 1
        else:
            print(f"    [WARN] crossover produced invalid config: {errs[:1]}")
    print(f"  crossover: {cross_ok}/20 valid")

    # Verify mutations don't modify original
    cfg_orig = copy.deepcopy(cfg_a)
    for _ in range(20):
        pick_mutation(cfg_a)
    if cfg_a == cfg_orig:
        print("\n  [PASS] Original config not modified by mutations")
    else:
        print("\n  [FAIL] Original config was mutated!")
        all_ok = False

    print(f"\nAll OK: {all_ok}")
