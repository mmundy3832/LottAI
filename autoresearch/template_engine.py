"""
template_engine.py -- Render a v3 config dict to valid experiment Python code.

config_to_code(config) -> str

The generated code:
  - Uses only names pre-loaded by the v3 boilerplate (same as v2 boilerplate)
  - Chains pipeline stages via X_train_{stage} / X_val_{stage} variable naming
  - Always ends with normalization, evaluate_predictions, and print(json.dumps(results))
  - Is guaranteed syntactically valid if config passes validate_config()

No LLM is involved. This function is a pure deterministic renderer.
"""

from config_space import VALID_SCALE_METHODS


# ---------------------------------------------------------------------------
# Stage renderers
# Each appends lines to `lines` list and reads from X_train_{prev}/X_val_{prev}
# Writing to X_train_{cur}/X_val_{cur}
# ---------------------------------------------------------------------------

def _render_select(cfg, prev, cur, lines):
    tgt = cfg.get("selector_target", "d1")
    if tgt == "mean":
        fit_target = "((y_train_d1 + y_train_d2 + y_train_d3) // 3)"
    else:
        fit_target = f"y_train_{tgt}"

    n_est = int(cfg.get("selector_n_estimators", 200))
    max_d = int(cfg.get("selector_max_depth", 10))
    thresh = float(cfg.get("threshold_multiplier", 1.5))

    lines += [
        f"# -- Stage: select (threshold={thresh}x mean, target={tgt}) --",
        f"_sel_et = ExtraTreesClassifier(",
        f"    n_estimators={n_est}, max_depth={max_d},",
        f"    random_state=42, n_jobs=-1",
        f")",
        f"_sel_et.fit(X_train_{prev}, {fit_target})",
        f"from sklearn.feature_selection import SelectFromModel as _SFM",
        f"_selector = _SFM(_sel_et, prefit=True, threshold='{thresh}*mean')",
        f"X_train_{cur} = _selector.transform(X_train_{prev})",
        f"if X_train_{cur}.shape[1] == 0:",
        f"    _top10 = np.argsort(_sel_et.feature_importances_)[-10:]",
        f"    X_train_{cur} = X_train_{prev}[:, _top10]",
        f"    _selector = None",
        f"    _sel_top10_idx = _top10",
        f"    X_val_{cur} = X_val_{prev}[:, _top10]",
        f"else:",
        f"    X_val_{cur} = _selector.transform(X_val_{prev})",
        f"",
    ]


def _render_poly(cfg, prev, cur, lines):
    degree = int(cfg.get("degree", 2))
    interaction_only = bool(cfg.get("interaction_only", False))
    top_k = int(cfg.get("top_k_variance", 250))

    lines += [
        f"# -- Stage: poly (degree={degree}, interaction_only={interaction_only}, top_k={top_k}) --",
        f"_poly = PolynomialFeatures(degree={degree},",
        f"    include_bias=False, interaction_only={interaction_only})",
        f"_Xtr_poly_full = _poly.fit_transform(X_train_{prev})",
        f"_Xva_poly_full = _poly.transform(X_val_{prev})",
        f"_poly_var = np.var(_Xtr_poly_full, axis=0)",
        f"_top_k = min({top_k}, _Xtr_poly_full.shape[1])",
        f"_top_idx = np.argsort(_poly_var)[-_top_k:]",
        f"X_train_{cur} = np.hstack([X_train_{prev}, _Xtr_poly_full[:, _top_idx]])",
        f"X_val_{cur} = np.hstack([X_val_{prev}, _Xva_poly_full[:, _top_idx]])",
        f"",
    ]


def _render_custom_interact(cfg, prev, cur, lines):
    n_head = int(cfg.get("n_head", 5))
    n_tail = int(cfg.get("n_tail", 5))
    include_ratios = bool(cfg.get("include_ratios", True))

    lines += [
        f"# -- Stage: custom_interact (n_head={n_head}, n_tail={n_tail}, ratios={include_ratios}) --",
        f"_ci_n_head = min({n_head}, X_train_{prev}.shape[1])",
        f"_ci_n_tail = min({n_tail}, X_train_{prev}.shape[1])",
        f"_ci_f1_tr = X_train_{prev}[:, :_ci_n_head]",
        f"_ci_f2_tr = X_train_{prev}[:, -_ci_n_tail:]",
        f"_ci_f1_va = X_val_{prev}[:, :_ci_n_head]",
        f"_ci_f2_va = X_val_{prev}[:, -_ci_n_tail:]",
        f"_ci_parts_tr, _ci_parts_va = [], []",
        f"for _ci_i in range(_ci_n_head):",
        f"    for _ci_j in range(_ci_n_tail):",
        f"        _ci_parts_tr.append(_ci_f1_tr[:, _ci_i] * _ci_f2_tr[:, _ci_j])",
        f"        _ci_parts_va.append(_ci_f1_va[:, _ci_i] * _ci_f2_va[:, _ci_j])",
    ]
    if include_ratios:
        lines += [
            f"        _ci_parts_tr.append(_ci_f1_tr[:, _ci_i] / (_ci_f2_tr[:, _ci_j] + 1e-6))",
            f"        _ci_parts_va.append(_ci_f1_va[:, _ci_i] / (_ci_f2_va[:, _ci_j] + 1e-6))",
        ]
    lines += [
        f"_ci_tr = np.column_stack(_ci_parts_tr) if _ci_parts_tr else np.zeros((X_train_{prev}.shape[0], 1))",
        f"_ci_va = np.column_stack(_ci_parts_va) if _ci_parts_va else np.zeros((X_val_{prev}.shape[0], 1))",
        f"_ci_tr = np.nan_to_num(_ci_tr, nan=0.0, posinf=1e6, neginf=-1e6)",
        f"_ci_va = np.nan_to_num(_ci_va, nan=0.0, posinf=1e6, neginf=-1e6)",
        f"X_train_{cur} = np.hstack([X_train_{prev}, _ci_tr])",
        f"X_val_{cur} = np.hstack([X_val_{prev}, _ci_va])",
        f"",
    ]


def _render_scale(cfg, prev, cur, lines):
    method = cfg.get("method", "standard")
    scaler_map = {
        "standard": "StandardScaler()",
        "robust":   "RobustScaler()",
        "quantile": "QuantileTransformer(output_distribution='normal', random_state=42)",
        "minmax":   "MinMaxScaler()",
    }
    scaler_expr = scaler_map.get(method, "StandardScaler()")
    lines += [
        f"# -- Stage: scale ({method}) --",
        f"_scaler = {scaler_expr}",
        f"X_train_{cur} = _scaler.fit_transform(X_train_{prev})",
        f"X_val_{cur} = _scaler.transform(X_val_{prev})",
        f"",
    ]


# ---------------------------------------------------------------------------
# Pipeline renderer
# Returns the suffix name of the final X_train/X_val variables
# ---------------------------------------------------------------------------

def _render_pipeline(config, lines):
    """Emit feature loading + all pipeline stages. Returns final variable suffix."""
    feature_sets = config["feature_sets"]
    fs_repr = repr(feature_sets)

    lines += [
        f"# -- Feature Loading --",
        f"X_train_raw = load_cached_features('train', {fs_repr})",
        f"X_val_raw = load_cached_features('val', {fs_repr})",
        f"",
    ]

    prev = "raw"
    for stage_cfg in config.get("pipeline", []):
        stage = stage_cfg["stage"]
        cur = stage  # e.g. "select", "poly", "custom_interact", "scale"

        if stage == "select":
            _render_select(stage_cfg, prev, cur, lines)
        elif stage == "poly":
            _render_poly(stage_cfg, prev, cur, lines)
        elif stage == "custom_interact":
            _render_custom_interact(stage_cfg, prev, cur, lines)
        elif stage == "scale":
            _render_scale(stage_cfg, prev, cur, lines)

        prev = cur

    return prev


# ---------------------------------------------------------------------------
# Model classifier expression builders
# ---------------------------------------------------------------------------

def _clf_expr(m):
    """Build the classifier constructor expression for a model config."""
    mtype = m["type"]

    if mtype == "et":
        return (
            f"ExtraTreesClassifier("
            f"n_estimators={int(m['n_estimators'])}, "
            f"max_depth={int(m['max_depth'])}, "
            f"min_samples_leaf={int(m['min_samples_leaf'])}, "
            f"min_samples_split={int(m['min_samples_split'])}, "
            f"max_features={float(m['max_features']):.4f}, "
            f"random_state=42, n_jobs=-1)"
        )
    elif mtype == "rf":
        return (
            f"RandomForestClassifier("
            f"n_estimators={int(m['n_estimators'])}, "
            f"max_depth={int(m['max_depth'])}, "
            f"min_samples_leaf={int(m['min_samples_leaf'])}, "
            f"min_samples_split={int(m['min_samples_split'])}, "
            f"max_features={float(m['max_features']):.4f}, "
            f"random_state=42, n_jobs=-1)"
        )
    elif mtype == "xgb":
        return (
            f"xgb.XGBClassifier("
            f"objective='multi:softprob', num_class=10, "
            f"n_estimators={int(m['n_estimators'])}, "
            f"max_depth={int(m['max_depth'])}, "
            f"learning_rate={float(m['learning_rate']):.5f}, "
            f"subsample={float(m['subsample']):.4f}, "
            f"colsample_bytree={float(m['colsample_bytree']):.4f}, "
            f"reg_alpha={float(m['reg_alpha']):.4f}, "
            f"reg_lambda={float(m['reg_lambda']):.4f}, "
            f"eval_metric='mlogloss', random_state=42, n_jobs=-1, verbosity=0)"
        )
    elif mtype == "lgb":
        return (
            f"lgb.LGBMClassifier("
            f"objective='multiclass', num_class=10, "
            f"n_estimators={int(m['n_estimators'])}, "
            f"max_depth={int(m['max_depth'])}, "
            f"learning_rate={float(m['learning_rate']):.5f}, "
            f"subsample={float(m['subsample']):.4f}, "
            f"colsample_bytree={float(m['colsample_bytree']):.4f}, "
            f"reg_alpha={float(m['reg_alpha']):.4f}, "
            f"reg_lambda={float(m['reg_lambda']):.4f}, "
            f"random_state=42, n_jobs=-1, verbose=-1)"
        )
    elif mtype == "hgb":
        return (
            f"HistGradientBoostingClassifier("
            f"max_iter={int(m.get('max_iter', m.get('n_estimators', 300)))}, "
            f"max_depth={int(m['max_depth'])}, "
            f"learning_rate={float(m['learning_rate']):.5f}, "
            f"random_state=42)"
        )
    elif mtype == "lr":
        return (
            f"LogisticRegression("
            f"C={float(m.get('C', 1.0)):.4f}, "
            f"max_iter=2000, "
            f"random_state=42, n_jobs=-1)"
        )
    raise ValueError(f"Unknown model type: {mtype}")


# ---------------------------------------------------------------------------
# Model training renderer
# ---------------------------------------------------------------------------

def _render_models(config, final_suffix, lines, save_clfs=False):
    """Emit per-digit model training + weighted ensemble combination.

    Returns norm_weights list (needed by live section when save_clfs=True).
    When save_clfs=True, saves fitted clf objects in _clf_list_{idx} lists.
    """
    models = config["models"]

    # Normalize weights
    total_w = sum(float(m.get("weight", 1.0)) for m in models)
    if total_w <= 0:
        total_w = 1.0
    norm_weights = [float(m.get("weight", 1.0)) / total_w for m in models]

    lines.append("# -- Per-Digit Model Training --")
    lines.append("_digit_targets_train = [y_train_d1, y_train_d2, y_train_d3]")
    lines.append("")

    for idx, (m, w) in enumerate(zip(models, norm_weights)):
        var = f"_digit_probs_{idx}"
        mtype = m["type"]

        if save_clfs:
            lines.append(f"_clf_list_{idx} = []")

        lines += [
            f"# Model {idx}: {mtype} (weight={w:.4f})",
            f"{var} = []",
            f"for _d_idx, _y_d in enumerate(_digit_targets_train):",
        ]

        clf = _clf_expr(m)
        if m.get("calibrate", False):
            lines += [
                f"    _base = {clf}",
                f"    _clf = CalibratedClassifierCV(estimator=_base,",
                f"        method='{m.get('calibrate_method', 'isotonic')}',",
                f"        cv={int(m.get('calibrate_cv', 5))})",
            ]
        else:
            lines.append(f"    _clf = {clf}")

        lines += [
            f"    _clf.fit(X_train_{final_suffix}, _y_d)",
        ]

        if save_clfs:
            lines.append(f"    _clf_list_{idx}.append(_clf)")

        lines += [
            f"    _proba = _clf.predict_proba(X_val_{final_suffix})",
            f"    _fp = np.zeros((n_val, 10))",
            f"    for _ci, _cls in enumerate(_clf.classes_):",
            f"        _fp[:, int(_cls)] = _proba[:, _ci]",
            f"    {var}.append(_fp)",
            f"",
        ]

    # Weighted ensemble combination
    lines.append("# -- Weighted Ensemble --")
    lines.append("_digit_probs_combined = []")
    lines.append("for _d in range(3):")
    weight_expr = " + ".join(
        f"{w:.6f} * _digit_probs_{i}[_d]"
        for i, w in enumerate(norm_weights)
    )
    lines.append(f"    _digit_probs_combined.append({weight_expr})")
    lines.append("")

    return norm_weights


# ---------------------------------------------------------------------------
# Combo matrix + normalization + evaluation (always identical)
# ---------------------------------------------------------------------------

def _render_combo_and_eval(lines):
    lines += [
        "# -- Combo Probability Matrix --",
        "prob_matrix = np.zeros((n_val, NUM_COMBOS))",
        "for _combo in range(NUM_COMBOS):",
        "    _d1, _d2, _d3 = combo_to_digits(_combo)",
        "    prob_matrix[:, _combo] = (",
        "        _digit_probs_combined[0][:, _d1] *",
        "        _digit_probs_combined[1][:, _d2] *",
        "        _digit_probs_combined[2][:, _d3]",
        "    )",
        "",
        "# -- Normalize --",
        "prob_matrix = np.clip(prob_matrix, 0, None)",
        "_row_sums = prob_matrix.sum(axis=1, keepdims=True)",
        "_row_sums[_row_sums < 1e-15] = 1.0",
        "prob_matrix = prob_matrix / _row_sums",
        "",
        "results = evaluate_predictions(prob_matrix, y_val)",
        "print(json.dumps(results))",
    ]


# ---------------------------------------------------------------------------
# Live eval helpers (used when live_eval=True in config_to_code)
# ---------------------------------------------------------------------------

def _render_live_stage(stage_cfg, prev, cur, lines):
    """Apply a fitted pipeline stage transformer to X_live_{prev} -> X_live_{cur}.

    Assumes the fitted objects from the train pass are in scope:
      select -> _selector
      poly   -> _poly, _top_idx
      scale  -> _scaler
      custom_interact -> _ci_n_head, _ci_n_tail (recomputed for live)
    """
    stage = stage_cfg["stage"]
    if stage == "select":
        lines += [
            f"X_live_{cur} = X_live_{prev}[:, _sel_top10_idx] if _selector is None else _selector.transform(X_live_{prev})",
            "",
        ]
    elif stage == "poly":
        lines += [
            f"_Xlv_poly_full = _poly.transform(X_live_{prev})",
            f"X_live_{cur} = np.hstack([X_live_{prev}, _Xlv_poly_full[:, _top_idx]])",
            "",
        ]
    elif stage == "custom_interact":
        include_ratios = bool(stage_cfg.get("include_ratios", True))
        # _ci_n_head / _ci_n_tail were set during train pass; same dims apply to live
        lines += [
            f"_ci_f1_lv = X_live_{prev}[:, :_ci_n_head]",
            f"_ci_f2_lv = X_live_{prev}[:, -_ci_n_tail:]",
            f"_ci_parts_lv = []",
            f"for _ci_i in range(_ci_n_head):",
            f"    for _ci_j in range(_ci_n_tail):",
            f"        _ci_parts_lv.append(_ci_f1_lv[:, _ci_i] * _ci_f2_lv[:, _ci_j])",
        ]
        if include_ratios:
            lines += [
                f"        _ci_parts_lv.append(_ci_f1_lv[:, _ci_i] / (_ci_f2_lv[:, _ci_j] + 1e-6))",
            ]
        lines += [
            f"_ci_lv = np.column_stack(_ci_parts_lv) if _ci_parts_lv else np.zeros((X_live_{prev}.shape[0], 1))",
            f"_ci_lv = np.nan_to_num(_ci_lv, nan=0.0, posinf=1e6, neginf=-1e6)",
            f"X_live_{cur} = np.hstack([X_live_{prev}, _ci_lv])",
            "",
        ]
    elif stage == "scale":
        lines += [
            f"X_live_{cur} = _scaler.transform(X_live_{prev})",
            "",
        ]


def _render_live_section(config, final_suffix, norm_weights, lines):
    """Emit live data loading, feature building, pipeline transforms, predictions, evaluation.

    Assumes:
      - full_df is in scope (from boilerplate)
      - _LIVE_PATH is in scope (from live_eval.py boilerplate)
      - _clf_list_{i} lists are in scope (from save_clfs=True in _render_models)
      - All fitted pipeline objects (_selector, _poly, _scaler, etc.) are in scope
    """
    feature_sets = config["feature_sets"]
    fs_repr = repr(feature_sets)
    models = config["models"]

    lines += [
        "# -- Live Evaluation: Load fresh draws --",
        "import pandas as _pd",
        "_live_raw = _pd.read_csv(_LIVE_PATH, header=None,",
        "    names=['game','month','day','year','d1','d2','d3','sum_col','trailing'], dtype=str)",
        "_live_raw = _live_raw.dropna(subset=['year']).reset_index(drop=True)",
        "_live_raw['date'] = _pd.to_datetime(",
        "    _live_raw['year'].str.strip() + '-' + _live_raw['month'].str.strip() + '-' + _live_raw['day'].str.strip(),",
        "    format='%Y-%m-%d')",
        "for _c in ['d1','d2','d3']:",
        "    _live_raw[_c] = _live_raw[_c].astype(int)",
        "_live_raw['combo'] = _live_raw['d1'].apply(str) + _live_raw['d2'].apply(str) + _live_raw['d3'].apply(str)",
        "_live_raw['combo_int'] = _live_raw['d1']*100 + _live_raw['d2']*10 + _live_raw['d3']",
        "_live_raw['day_of_week'] = _live_raw['date'].dt.dayofweek",
        "_live_raw['month'] = _live_raw['date'].dt.month",
        "_live_raw['year'] = _live_raw['date'].dt.year",
        "_live_raw = _live_raw[['date','d1','d2','d3','combo','combo_int','day_of_week','month','year']].copy()",
        "_combined_df = _pd.concat([full_df, _live_raw], ignore_index=True)",
        "_live_start = len(full_df)",
        "_live_indices = list(range(_live_start, len(_combined_df)))",
        "y_live = _live_raw['combo_int'].values",
        "n_live = len(y_live)",
        f"X_live_raw = build_features(_combined_df, _live_indices, {fs_repr})",
        "",
    ]

    # Apply pipeline stages to X_live
    prev = "raw"
    for stage_cfg in config.get("pipeline", []):
        stage = stage_cfg["stage"]
        cur = stage
        _render_live_stage(stage_cfg, prev, cur, lines)
        prev = cur

    # Live model predictions using saved clf objects
    lines.append("# -- Live Model Predictions --")
    for idx in range(len(models)):
        lines += [
            f"_digit_probs_live_{idx} = []",
            f"for _d_idx in range(3):",
            f"    _clf_lv = _clf_list_{idx}[_d_idx]",
            f"    _proba_lv = _clf_lv.predict_proba(X_live_{final_suffix})",
            f"    _fp_lv = np.zeros((n_live, 10))",
            f"    for _ci, _cls in enumerate(_clf_lv.classes_):",
            f"        _fp_lv[:, int(_cls)] = _proba_lv[:, _ci]",
            f"    _digit_probs_live_{idx}.append(_fp_lv)",
            "",
        ]

    # Weighted ensemble for live
    lines.append("# -- Live Weighted Ensemble --")
    lines.append("_digit_probs_live_combined = []")
    lines.append("for _d in range(3):")
    weight_expr = " + ".join(
        f"{w:.6f} * _digit_probs_live_{i}[_d]"
        for i, w in enumerate(norm_weights)
    )
    lines.append(f"    _digit_probs_live_combined.append({weight_expr})")
    lines.append("")

    # Live combo probability matrix
    lines += [
        "# -- Live Combo Probability Matrix --",
        "prob_matrix_live = np.zeros((n_live, NUM_COMBOS))",
        "for _combo in range(NUM_COMBOS):",
        "    _d1, _d2, _d3 = combo_to_digits(_combo)",
        "    prob_matrix_live[:, _combo] = (",
        "        _digit_probs_live_combined[0][:, _d1] *",
        "        _digit_probs_live_combined[1][:, _d2] *",
        "        _digit_probs_live_combined[2][:, _d3]",
        "    )",
        "prob_matrix_live = np.clip(prob_matrix_live, 0, None)",
        "_row_sums_live = prob_matrix_live.sum(axis=1, keepdims=True)",
        "_row_sums_live[_row_sums_live < 1e-15] = 1.0",
        "prob_matrix_live = prob_matrix_live / _row_sums_live",
        "",
        "live_results = evaluate_predictions(prob_matrix_live, y_live)",
        "results['live_n_draws'] = int(n_live)",
        "results['live_mean_rank'] = live_results['mean_rank']",
        "results['live_optimal_ev'] = live_results['optimal_ev']",
        "results['live_optimal_k'] = live_results['optimal_k']",
        "results['live_box_optimal_ev'] = live_results['box_optimal_ev']",
        "results['live_box_optimal_k'] = live_results['box_optimal_k']",
        "results['live_top_5_hit'] = live_results['top_5_hit']",
        "results['live_top_10_hit'] = live_results['top_10_hit']",
        "print(json.dumps(results))",
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def config_to_code(config, live_eval=False):
    """
    Render a v3 config dict to a valid Python experiment code string.

    The output is prepended with the runner's BOILERPLATE before execution.
    It uses only names that the boilerplate pre-defines.

    Parameters
    ----------
    config : dict
        A config dict that has passed validate_config().
    live_eval : bool
        If True, generate code that also evaluates on the live draw data.
        Requires _LIVE_PATH and full_df to be defined in the boilerplate.

    Returns
    -------
    str
        Valid Python code (the experiment portion only, without boilerplate).
    """
    desc = config.get("description", "Config-driven experiment")
    lines = [
        f"# DESCRIPTION: {desc}",
        "",
    ]

    final_suffix = _render_pipeline(config, lines)
    norm_weights = _render_models(config, final_suffix, lines, save_clfs=live_eval)

    if live_eval:
        # Val combo matrix (no print yet -- live section prints at the end)
        lines += [
            "# -- Combo Probability Matrix (Val) --",
            "prob_matrix = np.zeros((n_val, NUM_COMBOS))",
            "for _combo in range(NUM_COMBOS):",
            "    _d1, _d2, _d3 = combo_to_digits(_combo)",
            "    prob_matrix[:, _combo] = (",
            "        _digit_probs_combined[0][:, _d1] *",
            "        _digit_probs_combined[1][:, _d2] *",
            "        _digit_probs_combined[2][:, _d3]",
            "    )",
            "",
            "# -- Normalize --",
            "prob_matrix = np.clip(prob_matrix, 0, None)",
            "_row_sums = prob_matrix.sum(axis=1, keepdims=True)",
            "_row_sums[_row_sums < 1e-15] = 1.0",
            "prob_matrix = prob_matrix / _row_sums",
            "",
            "results = evaluate_predictions(prob_matrix, y_val)",
            # No print here -- live section appends live metrics then prints
        ]
        _render_live_section(config, final_suffix, norm_weights, lines)
    else:
        _render_combo_and_eval(lines)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Test when run directly
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from config_space import DEFAULT_CONFIG, BOOTSTRAP_CONFIGS
    from config_validator import validate_config

    print("Rendering and compile-checking all bootstrap configs...\n")
    all_ok = True

    for i, cfg in enumerate(BOOTSTRAP_CONFIGS, 1):
        ok, errs = validate_config(cfg)
        if not ok:
            print(f"[FAIL] Seed {i}: validation errors: {errs}")
            all_ok = False
            continue

        code = config_to_code(cfg)
        try:
            compile(code, f"<seed_{i}>", "exec")
            print(f"[PASS] Seed {i}: {len(code)} chars, compiles OK -- {cfg['description'][:60]}")
        except SyntaxError as e:
            print(f"[FAIL] Seed {i}: SyntaxError: {e}")
            all_ok = False

    print(f"\nAll seeds OK: {all_ok}")

    if all_ok:
        print("\n--- Generated code for Seed 1 (DEFAULT_CONFIG) ---")
        print(config_to_code(DEFAULT_CONFIG))
