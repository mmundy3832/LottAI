"""
config_validator.py -- Validate a v3 experiment config dict.

validate_config(config) -> (bool, [error_strings])

Called before every template render, after LLM JSON parse, and after GA mutation.
All logic is purely deterministic -- no I/O, no LLM.
"""

from config_space import (
    VALID_FEATURE_SETS, VALID_STAGES, VALID_MODEL_TYPES,
    VALID_SCALE_METHODS, VALID_CALIB_METHODS, VALID_SELECTOR_TARGETS,
    VALID_CALIB_CVS, VALID_POLY_DEGREES, PARAM_RANGES,
)


def _check_range(val, key, errors, label):
    """Validate a numeric value against PARAM_RANGES."""
    if key not in PARAM_RANGES:
        return  # no range defined -- skip
    ptype, lo, hi = PARAM_RANGES[key]
    if not isinstance(val, (int, float)):
        errors.append(f"{label}: expected number, got {type(val).__name__}")
        return
    if val < lo or val > hi:
        errors.append(f"{label}: {val} out of range [{lo}, {hi}]")


def _validate_stage(stage_cfg, idx, errors):
    stype = stage_cfg.get("stage")
    prefix = f"pipeline[{idx}]"

    if stype == "select":
        _check_range(stage_cfg.get("threshold_multiplier", 1.5),
                     "select.threshold_multiplier", errors,
                     f"{prefix}.threshold_multiplier")
        _check_range(stage_cfg.get("selector_n_estimators", 200),
                     "select.selector_n_estimators", errors,
                     f"{prefix}.selector_n_estimators")
        _check_range(stage_cfg.get("selector_max_depth", 10),
                     "select.selector_max_depth", errors,
                     f"{prefix}.selector_max_depth")
        tgt = stage_cfg.get("selector_target", "d1")
        if tgt not in VALID_SELECTOR_TARGETS:
            errors.append(f"{prefix}.selector_target: '{tgt}' not in {VALID_SELECTOR_TARGETS}")

    elif stype == "poly":
        deg = stage_cfg.get("degree", 2)
        if deg not in VALID_POLY_DEGREES:
            errors.append(f"{prefix}.degree: {deg} not in {VALID_POLY_DEGREES}")
        if not isinstance(stage_cfg.get("interaction_only", False), bool):
            errors.append(f"{prefix}.interaction_only: must be bool")
        _check_range(stage_cfg.get("top_k_variance", 250),
                     "poly.top_k_variance", errors,
                     f"{prefix}.top_k_variance")

    elif stype == "custom_interact":
        _check_range(stage_cfg.get("n_head", 5),
                     "custom_interact.n_head", errors, f"{prefix}.n_head")
        _check_range(stage_cfg.get("n_tail", 5),
                     "custom_interact.n_tail", errors, f"{prefix}.n_tail")
        if not isinstance(stage_cfg.get("include_ratios", True), bool):
            errors.append(f"{prefix}.include_ratios: must be bool")

    elif stype == "scale":
        method = stage_cfg.get("method", "standard")
        if method not in VALID_SCALE_METHODS:
            errors.append(f"{prefix}.method: '{method}' not in {VALID_SCALE_METHODS}")


def _validate_model(m, idx, errors):
    prefix = f"models[{idx}]"
    mtype = m.get("type")
    if mtype not in VALID_MODEL_TYPES:
        errors.append(f"{prefix}.type: '{mtype}' not in {VALID_MODEL_TYPES}")
        return  # can't validate further without knowing the type

    _check_range(m.get("weight", 1.0), "model.weight", errors, f"{prefix}.weight")

    # Calibration fields
    calib = m.get("calibrate", False)
    if not isinstance(calib, bool):
        errors.append(f"{prefix}.calibrate: must be bool")
    if m.get("calibrate_method", "isotonic") not in VALID_CALIB_METHODS:
        errors.append(f"{prefix}.calibrate_method: must be one of {VALID_CALIB_METHODS}")
    if m.get("calibrate_cv", 5) not in VALID_CALIB_CVS:
        errors.append(f"{prefix}.calibrate_cv: must be one of {VALID_CALIB_CVS}")

    # Type-specific params
    if mtype in ("et", "rf"):
        _check_range(m.get("n_estimators", 300), "model.n_estimators", errors, f"{prefix}.n_estimators")
        _check_range(m.get("max_depth", 15), "model.max_depth", errors, f"{prefix}.max_depth")
        _check_range(m.get("min_samples_leaf", 3), "model.min_samples_leaf", errors, f"{prefix}.min_samples_leaf")
        _check_range(m.get("min_samples_split", 6), "model.min_samples_split", errors, f"{prefix}.min_samples_split")
        _check_range(m.get("max_features", 0.6), "model.max_features", errors, f"{prefix}.max_features")

    elif mtype in ("xgb", "lgb", "xgb_gpu", "lgb_gpu"):
        _check_range(m.get("n_estimators", 300), "model.n_estimators", errors, f"{prefix}.n_estimators")
        _check_range(m.get("max_depth", 6), "model.max_depth", errors, f"{prefix}.max_depth")
        _check_range(m.get("learning_rate", 0.05), "model.learning_rate", errors, f"{prefix}.learning_rate")
        _check_range(m.get("subsample", 0.8), "model.subsample", errors, f"{prefix}.subsample")
        _check_range(m.get("colsample_bytree", 0.7), "model.colsample_bytree", errors, f"{prefix}.colsample_bytree")
        _check_range(m.get("reg_alpha", 0.0), "model.reg_alpha", errors, f"{prefix}.reg_alpha")
        _check_range(m.get("reg_lambda", 1.0), "model.reg_lambda", errors, f"{prefix}.reg_lambda")

    elif mtype == "hgb":
        _check_range(m.get("max_iter", 300), "model.max_iter", errors, f"{prefix}.max_iter")
        _check_range(m.get("max_depth", 6), "model.max_depth", errors, f"{prefix}.max_depth")
        _check_range(m.get("learning_rate", 0.05), "model.learning_rate_hgb", errors, f"{prefix}.learning_rate")

    elif mtype == "lr":
        _check_range(m.get("C", 1.0), "model.C", errors, f"{prefix}.C")

    elif mtype == "mlp":
        _check_range(m.get("mlp_hidden_layers", 1), "model.mlp_hidden_layers", errors, f"{prefix}.mlp_hidden_layers")
        _check_range(m.get("mlp_layer_size", 100), "model.mlp_layer_size", errors, f"{prefix}.mlp_layer_size")
        _check_range(m.get("alpha", 0.0001), "model.alpha", errors, f"{prefix}.alpha")
        _check_range(m.get("learning_rate_init", 0.001), "model.learning_rate_init", errors, f"{prefix}.learning_rate_init")
        _check_range(m.get("max_iter", 500), "model.max_iter", errors, f"{prefix}.max_iter")


def validate_config(config):
    """
    Validate a v3 experiment config.

    Returns
    -------
    (bool, list[str])
        (True, []) if valid, (False, [error messages]) if not.
    """
    errors = []

    if not isinstance(config, dict):
        return False, ["config must be a dict"]

    # feature_sets
    fs = config.get("feature_sets")
    if not isinstance(fs, list) or len(fs) == 0:
        errors.append("feature_sets must be a non-empty list")
    else:
        for s in fs:
            if s not in VALID_FEATURE_SETS:
                errors.append(f"feature_sets: unknown set '{s}'")
        if len(fs) != len(set(fs)):
            errors.append("feature_sets: duplicate entries")

    # pipeline
    pipeline = config.get("pipeline")
    if not isinstance(pipeline, list):
        errors.append("pipeline must be a list (can be empty [])")
    else:
        if len(pipeline) > 4:
            errors.append(f"pipeline: max 4 stages, got {len(pipeline)}")
        seen_stages = []
        for i, stage_cfg in enumerate(pipeline):
            if not isinstance(stage_cfg, dict):
                errors.append(f"pipeline[{i}]: must be a dict")
                continue
            stype = stage_cfg.get("stage")
            if stype not in VALID_STAGES:
                errors.append(f"pipeline[{i}].stage: '{stype}' not in {VALID_STAGES}")
                continue
            if stype in seen_stages:
                errors.append(f"pipeline[{i}].stage: '{stype}' appears more than once")
            seen_stages.append(stype)
            _validate_stage(stage_cfg, i, errors)
        # high-degree poly requires select to run first (memory safety)
        for i, stage_cfg in enumerate(pipeline):
            if stage_cfg.get("stage") == "poly" and stage_cfg.get("degree", min(VALID_POLY_DEGREES)) > min(VALID_POLY_DEGREES):
                prior_stages = [s.get("stage") for s in pipeline[:i]]
                if "select" not in prior_stages:
                    errors.append(f"pipeline[{i}]: degree>{min(VALID_POLY_DEGREES)} poly requires a select stage before it")

    # models
    models = config.get("models")
    if not isinstance(models, list) or len(models) == 0:
        errors.append("models must be a non-empty list (1-3 entries)")
    elif len(models) > 3:
        errors.append(f"models: max 3 entries, got {len(models)}")
    else:
        for i, m in enumerate(models):
            if not isinstance(m, dict):
                errors.append(f"models[{i}]: must be a dict")
                continue
            _validate_model(m, i, errors)

    return len(errors) == 0, errors


# ---------------------------------------------------------------------------
# Quick test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config_space import DEFAULT_CONFIG, BOOTSTRAP_CONFIGS

    print("Testing DEFAULT_CONFIG...")
    ok, errs = validate_config(DEFAULT_CONFIG)
    print(f"  {'PASS' if ok else 'FAIL'}", errs if errs else "")

    print(f"\nTesting {len(BOOTSTRAP_CONFIGS)} bootstrap configs...")
    for i, cfg in enumerate(BOOTSTRAP_CONFIGS, 1):
        ok, errs = validate_config(cfg)
        status = "PASS" if ok else "FAIL"
        print(f"  [{status}] Seed {i}: {cfg['description'][:60]}")
        if errs:
            for e in errs:
                print(f"           ERROR: {e}")

    print("\nTesting intentionally broken configs...")
    bad_configs = [
        ({}, "empty dict"),
        ({"feature_sets": [], "pipeline": [], "models": []}, "empty feature_sets and models"),
        ({"feature_sets": ["basic", "bad_set"], "pipeline": [], "models": [{"type": "et", "weight": 1.0}]},
         "unknown feature set"),
        ({"feature_sets": ["basic"], "pipeline": [{"stage": "select"}, {"stage": "select"}],
          "models": [{"type": "et", "weight": 1.0}]}, "duplicate pipeline stage"),
        ({"feature_sets": ["basic"], "pipeline": [],
          "models": [{"type": "et", "weight": 1.0}, {"type": "et", "weight": 1.0},
                     {"type": "et", "weight": 1.0}, {"type": "et", "weight": 1.0}]},
         "too many models"),
    ]
    all_caught = True
    for cfg, label in bad_configs:
        ok, errs = validate_config(cfg)
        caught = not ok
        print(f"  [{'CAUGHT' if caught else 'MISSED'}] {label}: {errs[:1] if errs else '(no errors)'}")
        if not caught:
            all_caught = False

    print(f"\nAll bad configs caught: {all_caught}")
