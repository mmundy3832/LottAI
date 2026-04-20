"""
consensus.py -- Multi-model consensus evaluation on the held-out test set.

Takes the top-N experiments by optimal_ev from experiments_v3.jsonl,
re-runs each config capturing per-draw probability matrices for the test set,
then evaluates two combination strategies:

  Strategy A -- Product ensemble:
    Multiply probability vectors across all models, re-rank.
    Always produces a full ranking. Equivalent to Bayesian model combination
    assuming (approximately) independent models.

  Strategy B -- Intersection filter:
    For each draw, buy only combos that appear in EVERY model's top-K.
    Fewer tickets per draw, higher confidence per ticket.
    Some draws may yield no intersection (skip those draws or fall back).

Usage:
    python -u consensus.py                # top 5 by optimal_ev
    python -u consensus.py --n 3          # top 3
    python -u consensus.py --top-k 15     # intersection threshold (default 20)
    python -u consensus.py --dry-run      # show which experiments, don't run
"""

import os
import sys
import json
import time
import argparse
import subprocess
import tempfile
import numpy as np
from collections import defaultdict

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)

from prepare import (
    evaluate_predictions, NUM_COMBOS, combo_to_digits,
    _BOX_MAP, _BOX_PAYOUT, _COMBO_TO_BOX, PICK3_PAYOUT,
    get_train_val_test
)

_LOG = os.path.join(_DIR, "consensus.log")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def log(msg):
    safe = msg.encode("ascii", errors="replace").decode("ascii")
    print(safe)
    with open(_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(path):
    entries = []
    if not os.path.exists(path):
        return entries
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return entries


def top_n_by_optimal_ev(jsonl_path, n):
    entries = load_jsonl(jsonl_path)
    good = [
        e for e in entries
        if e.get("status") == "success"
        and isinstance(e.get("results"), dict)
        and e["results"].get("optimal_ev") is not None
        and e.get("config") is not None
    ]
    good.sort(key=lambda e: e["results"]["optimal_ev"], reverse=True)
    return good[:n]


# ---------------------------------------------------------------------------
# Code generation -- extended template that handles train + val + test
# ---------------------------------------------------------------------------

def _render_pipeline_all(config, lines):
    """Load features and apply pipeline to train, val, AND test. Returns final suffix."""
    feature_sets = config["feature_sets"]
    fs_repr = repr(feature_sets)

    lines += [
        "# -- Feature Loading (train / val / test) --",
        f"X_train_raw = load_cached_features('train', {fs_repr})",
        f"X_val_raw   = load_cached_features('val',   {fs_repr})",
        f"X_test_raw  = load_cached_features('test',  {fs_repr})",
        "",
    ]

    prev = "raw"
    for stage_cfg in config.get("pipeline", []):
        stage = stage_cfg["stage"]
        cur   = stage

        if stage == "select":
            tgt   = stage_cfg.get("selector_target", "d1")
            ftgt  = "((y_train_d1 + y_train_d2 + y_train_d3) // 3)" if tgt == "mean" else f"y_train_{tgt}"
            n_est = int(stage_cfg.get("selector_n_estimators", 200))
            max_d = int(stage_cfg.get("selector_max_depth", 10))
            thresh = float(stage_cfg.get("threshold_multiplier", 1.5))
            lines += [
                f"# -- Stage: select (target={tgt}, thresh={thresh}x) --",
                f"_sel_et = ExtraTreesClassifier(n_estimators={n_est}, max_depth={max_d}, random_state=42, n_jobs=-1)",
                f"_sel_et.fit(X_train_{prev}, {ftgt})",
                f"from sklearn.feature_selection import SelectFromModel as _SFM",
                f"_selector = _SFM(_sel_et, prefit=True, threshold='{thresh}*mean')",
                f"X_train_{cur} = _selector.transform(X_train_{prev})",
                f"X_val_{cur}   = _selector.transform(X_val_{prev})",
                f"X_test_{cur}  = _selector.transform(X_test_{prev})",
                "",
            ]

        elif stage == "poly":
            degree  = int(stage_cfg.get("degree", 2))
            ionly   = bool(stage_cfg.get("interaction_only", False))
            top_k   = int(stage_cfg.get("top_k_variance", 250))
            lines += [
                f"# -- Stage: poly (degree={degree}, top_k={top_k}) --",
                f"_poly = PolynomialFeatures(degree={degree}, include_bias=False, interaction_only={ionly})",
                f"_Xtr_poly_full = _poly.fit_transform(X_train_{prev})",
                f"_Xva_poly_full = _poly.transform(X_val_{prev})",
                f"_Xte_poly_full = _poly.transform(X_test_{prev})",
                f"_poly_var = np.var(_Xtr_poly_full, axis=0)",
                f"_top_k = min({top_k}, _Xtr_poly_full.shape[1])",
                f"_top_idx = np.argsort(_poly_var)[-_top_k:]",
                f"X_train_{cur} = np.hstack([X_train_{prev}, _Xtr_poly_full[:, _top_idx]])",
                f"X_val_{cur}   = np.hstack([X_val_{prev},   _Xva_poly_full[:, _top_idx]])",
                f"X_test_{cur}  = np.hstack([X_test_{prev},  _Xte_poly_full[:, _top_idx]])",
                "",
            ]

        elif stage == "custom_interact":
            n_head = int(stage_cfg.get("n_head", 5))
            n_tail = int(stage_cfg.get("n_tail", 5))
            ratios = bool(stage_cfg.get("include_ratios", True))
            lines += [
                f"# -- Stage: custom_interact (n_head={n_head}, n_tail={n_tail}, ratios={ratios}) --",
                f"_ci_n_head = min({n_head}, X_train_{prev}.shape[1])",
                f"_ci_n_tail = min({n_tail}, X_train_{prev}.shape[1])",
            ]
            for split in ("train", "val", "test"):
                X = f"X_{split}_{prev}"
                parts = f"_ci_parts_{split}"
                lines += [
                    f"_ci_f1_{split} = {X}[:, :_ci_n_head]",
                    f"_ci_f2_{split} = {X}[:, -_ci_n_tail:]",
                    f"{parts} = []",
                    f"for _ci_i in range(_ci_n_head):",
                    f"    for _ci_j in range(_ci_n_tail):",
                    f"        {parts}.append(_ci_f1_{split}[:, _ci_i] * _ci_f2_{split}[:, _ci_j])",
                ]
                if ratios:
                    lines.append(
                        f"        {parts}.append(_ci_f1_{split}[:, _ci_i] / (_ci_f2_{split}[:, _ci_j] + 1e-6))"
                    )
                lines += [
                    f"_ci_{split} = np.column_stack({parts}) if {parts} else np.zeros(({X}.shape[0], 1))",
                    f"_ci_{split} = np.nan_to_num(_ci_{split}, nan=0.0, posinf=1e6, neginf=-1e6)",
                    f"X_{split}_{cur} = np.hstack([{X}, _ci_{split}])",
                ]
            lines.append("")

        elif stage == "scale":
            method = stage_cfg.get("method", "standard")
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
                f"X_val_{cur}   = _scaler.transform(X_val_{prev})",
                f"X_test_{cur}  = _scaler.transform(X_test_{prev})",
                "",
            ]

        prev = cur

    return prev


def _render_models_all(config, final_suffix, lines):
    """Train models, predict on val AND test, build both prob matrices."""
    models = config["models"]
    total_w = sum(float(m.get("weight", 1.0)) for m in models)
    if total_w <= 0:
        total_w = 1.0
    norm_weights = [float(m.get("weight", 1.0)) / total_w for m in models]

    from template_engine import _clf_expr

    lines.append("# -- Per-Digit Model Training (val + test) --")
    lines.append("_digit_targets_train = [y_train_d1, y_train_d2, y_train_d3]")
    lines.append("n_test_ = len(test_df)")
    lines.append("")

    for idx, (m, w) in enumerate(zip(models, norm_weights)):
        var_v = f"_dprobs_val_{idx}"
        var_t = f"_dprobs_tst_{idx}"
        mtype = m["type"]
        clf_expr = _clf_expr(m)
        lines += [
            f"# Model {idx}: {mtype} (w={w:.4f})",
            f"{var_v} = []",
            f"{var_t} = []",
            f"for _d_idx, _y_d in enumerate(_digit_targets_train):",
        ]
        if m.get("calibrate", False):
            lines += [
                f"    _base = {clf_expr}",
                f"    _clf = CalibratedClassifierCV(estimator=_base,",
                f"        method='{m.get('calibrate_method', 'isotonic')}',",
                f"        cv={int(m.get('calibrate_cv', 5))})",
            ]
        else:
            lines.append(f"    _clf = {clf_expr}")
        lines += [
            f"    _clf.fit(X_train_{final_suffix}, _y_d)",
            # val
            f"    _pv = _clf.predict_proba(X_val_{final_suffix})",
            f"    _fv = np.zeros((n_val, 10))",
            f"    for _ci, _cls in enumerate(_clf.classes_):",
            f"        _fv[:, int(_cls)] = _pv[:, _ci]",
            f"    {var_v}.append(_fv)",
            # test
            f"    _pt = _clf.predict_proba(X_test_{final_suffix})",
            f"    _ft = np.zeros((n_test_, 10))",
            f"    for _ci, _cls in enumerate(_clf.classes_):",
            f"        _ft[:, int(_cls)] = _pt[:, _ci]",
            f"    {var_t}.append(_ft)",
            "",
        ]

    # Weighted ensemble
    lines.append("# -- Weighted Ensemble --")
    lines.append("_dprobs_val_comb = []")
    lines.append("_dprobs_tst_comb = []")
    lines.append("for _d in range(3):")
    wexpr_v = " + ".join(f"{w:.6f} * _dprobs_val_{i}[_d]" for i, w in enumerate(norm_weights))
    wexpr_t = " + ".join(f"{w:.6f} * _dprobs_tst_{i}[_d]" for i, w in enumerate(norm_weights))
    lines.append(f"    _dprobs_val_comb.append({wexpr_v})")
    lines.append(f"    _dprobs_tst_comb.append({wexpr_t})")
    lines.append("")


def _render_combo_matrices(save_path, lines):
    """Build prob matrices for val and test, evaluate val, save test matrix."""
    lines += [
        "# -- Combo Probability Matrix (val) --",
        "prob_matrix_val = np.zeros((n_val, NUM_COMBOS))",
        "for _c in range(NUM_COMBOS):",
        "    _d1, _d2, _d3 = combo_to_digits(_c)",
        "    prob_matrix_val[:, _c] = _dprobs_val_comb[0][:, _d1] * _dprobs_val_comb[1][:, _d2] * _dprobs_val_comb[2][:, _d3]",
        "prob_matrix_val = np.clip(prob_matrix_val, 0, None)",
        "_rs = prob_matrix_val.sum(axis=1, keepdims=True); _rs[_rs < 1e-15] = 1.0",
        "prob_matrix_val = prob_matrix_val / _rs",
        "",
        "# -- Combo Probability Matrix (test) --",
        "prob_matrix_test = np.zeros((n_test_, NUM_COMBOS))",
        "for _c in range(NUM_COMBOS):",
        "    _d1, _d2, _d3 = combo_to_digits(_c)",
        "    prob_matrix_test[:, _c] = _dprobs_tst_comb[0][:, _d1] * _dprobs_tst_comb[1][:, _d2] * _dprobs_tst_comb[2][:, _d3]",
        "prob_matrix_test = np.clip(prob_matrix_test, 0, None)",
        "_rs = prob_matrix_test.sum(axis=1, keepdims=True); _rs[_rs < 1e-15] = 1.0",
        "prob_matrix_test = prob_matrix_test / _rs",
        "",
        "# -- Evaluate on val, print for runner --",
        "results = evaluate_predictions(prob_matrix_val, y_val)",
        "print(json.dumps(results))",
        "",
        f"# -- Save test matrix --",
        f"np.save(r'{save_path}', prob_matrix_test)",
        f"print('CONSENSUS_MATRIX_SAVED')",
    ]


BOILERPLATE = f'''import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, r'{_DIR}')  # hardcoded autoresearch dir -- scripts may run from temp
import numpy as np
import prepare
from prepare import (load_data, get_train_val_test, build_features,
                     evaluate_predictions, NUM_COMBOS, combo_to_digits,
                     digits_to_combo, FEATURE_DIMS, load_cached_features)
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler
from sklearn.calibration import CalibratedClassifierCV
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None

train_df, val_df, test_df = get_train_val_test()
y_train = train_df["combo_int"].values
y_val   = val_df["combo_int"].values
n_val   = len(val_df)
n_train = len(train_df)
y_train_d1 = y_train // 100
y_train_d2 = (y_train // 10) % 10
y_train_d3 = y_train % 10
y_val_d1 = y_val // 100
y_val_d2 = (y_val // 10) % 10
y_val_d3 = y_val % 10
'''


def generate_consensus_script(config, save_path):
    """Generate a Python script that trains the config and saves the test prob matrix."""
    lines = [f"# CONSENSUS SCRIPT: {config.get('description', '')[:80]}", ""]
    final_suffix = _render_pipeline_all(config, lines)
    _render_models_all(config, final_suffix, lines)
    _render_combo_matrices(save_path, lines)
    return BOILERPLATE + "\n\n" + "\n".join(lines)


# ---------------------------------------------------------------------------
# Run one experiment, return (val_results, test_matrix_path)
# ---------------------------------------------------------------------------

def run_config(config, exp_id, tmp_dir, timeout=2400):
    script_path = os.path.join(tmp_dir, f"consensus_{exp_id}.py")
    matrix_path = os.path.join(tmp_dir, f"matrix_{exp_id}.npy")

    code = generate_consensus_script(config, matrix_path)
    with open(script_path, "w", encoding="utf-8") as f:
        f.write(code)

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=timeout
        )
        elapsed = round(time.time() - t0, 2)
        stdout, stderr = result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return None, None, "TIMEOUT"
    except Exception as e:
        return None, None, str(e)

    # Parse val results from stdout (last JSON line)
    val_results = None
    for line in reversed(stdout.strip().split("\n")):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "mean_rank" in obj:
                val_results = obj
                break
        except json.JSONDecodeError:
            pass

    if not os.path.exists(matrix_path):
        return val_results, None, f"matrix not saved (rc={result.returncode})\n{stderr[:200]}"

    matrix = np.load(matrix_path)
    return val_results, matrix, None


# ---------------------------------------------------------------------------
# Consensus evaluation
# ---------------------------------------------------------------------------

def product_ensemble(matrices):
    """Multiply probability matrices element-wise, re-normalize."""
    combined = np.ones_like(matrices[0])
    for m in matrices:
        combined *= m
    combined = np.clip(combined, 0, None)
    row_sums = combined.sum(axis=1, keepdims=True)
    row_sums[row_sums < 1e-15] = 1.0
    return combined / row_sums


def intersection_eval(matrices, actual_combos, top_k=20):
    """
    For each draw: find combos in top-K for ALL models.
    Returns per-draw stats: intersection size, hit (True/False), tickets_bought.
    When intersection is empty, we skip the draw (don't buy).
    """
    N = matrices[0].shape[0]
    results = []
    for i in range(N):
        # Top-K sets for each model
        sets = [set(np.argsort(-m[i])[:top_k].tolist()) for m in matrices]
        common = sets[0]
        for s in sets[1:]:
            common = common & s
        hit = actual_combos[i] in common
        results.append({
            "intersection_size": len(common),
            "hit": hit,
            "skipped": len(common) == 0,
        })
    return results


def report_eval(label, prob_matrix, actual_combos, N):
    """Run evaluate_predictions and print a summary line."""
    res = evaluate_predictions(prob_matrix, actual_combos)
    oev = res.get("optimal_ev", float("nan"))
    ok_ = res.get("optimal_k", "?")
    bev = res.get("box_optimal_ev", float("nan"))
    bk_ = res.get("box_optimal_k", "?")
    mr  = res.get("mean_rank", float("nan"))
    log(f"  {label:30s}  rank={mr:>6.2f}  str_ev=${oev:>+7.3f}(buy{ok_:>2})  box_ev=${bev:>+6.3f}(buy{bk_:>2})")
    return res


def report_intersection(label, inter_results, payout=PICK3_PAYOUT):
    """Report stats for the intersection filter strategy."""
    N = len(inter_results)
    skipped   = sum(1 for r in inter_results if r["skipped"])
    played    = N - skipped
    hits      = sum(1 for r in inter_results if r["hit"])
    hit_rate  = hits / played if played > 0 else 0.0
    avg_tix   = np.mean([r["intersection_size"] for r in inter_results if not r["skipped"]]) if played > 0 else 0
    total_cost = sum(r["intersection_size"] for r in inter_results if not r["skipped"])
    total_won  = hits * payout
    net_ev_per_draw = (total_won - total_cost) / N  # EV per draw (including skipped)
    log(f"  {label:30s}  played={played}/{N}  hit_rate={hit_rate:.4f}  "
        f"avg_tix={avg_tix:.1f}  net_ev_per_draw=${net_ev_per_draw:>+7.3f}")
    return {"played": played, "hits": hits, "hit_rate": hit_rate,
            "avg_tix": avg_tix, "net_ev_per_draw": net_ev_per_draw}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Consensus evaluation on test set")
    parser.add_argument("--n",       type=int, default=5,
                        help="Number of top experiments to combine (default: 5)")
    parser.add_argument("--top-k",   type=int, default=20,
                        help="Top-K threshold for intersection filter (default: 20)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    log(f"\n{'='*70}")
    log(f"consensus.py  --n={args.n}  --top-k={args.top_k}")
    log(f"{'='*70}")

    jsonl_path = os.path.join(_DIR, "experiments_v3.jsonl")
    top_exps = top_n_by_optimal_ev(jsonl_path, args.n)

    if not top_exps:
        log("No qualifying experiments found. Exiting.")
        return

    log(f"\nTop {len(top_exps)} experiments by optimal_ev:")
    for e in top_exps:
        r = e["results"]
        log(f"  exp#{e['experiment_id']:>4}  optimal_ev=${r['optimal_ev']:.3f}  "
            f"optimal_k={r['optimal_k']}  mean_rank={r['mean_rank']:.2f}  "
            f"mode={e.get('mode','?')}")
        log(f"    {e.get('description','')[:70]}")

    if args.dry_run:
        return

    # Load test labels
    _, val_df, test_df_ = get_train_val_test()
    y_test  = test_df_["combo_int"].values
    N_test  = len(test_df_)
    log(f"\nTest set: {N_test} draws")

    # Run each experiment, collect test matrices
    tmp_dir  = tempfile.mkdtemp(prefix="lottai_consensus_")
    matrices = []
    val_scores = []

    log(f"\n{'='*70}")
    log(f"Training {len(top_exps)} models...")
    log(f"{'='*70}")

    for i, exp in enumerate(top_exps, 1):
        exp_id = exp["experiment_id"]
        desc   = exp.get("description", "")[:60]
        log(f"\n[{i}/{len(top_exps)}] exp#{exp_id}: {desc}")
        t0 = time.time()
        val_res, matrix, err = run_config(exp["config"], exp_id, tmp_dir)
        elapsed = round(time.time() - t0, 2)

        if err:
            log(f"  FAILED ({elapsed}s): {err[:120]}")
            continue

        val_mr  = val_res.get("mean_rank", float("nan")) if val_res else float("nan")
        val_oev = val_res.get("optimal_ev", float("nan")) if val_res else float("nan")
        val_ok  = val_res.get("optimal_k", "?") if val_res else "?"
        log(f"  OK ({elapsed}s)  val: rank={val_mr:.2f}  str_ev=${val_oev:.3f}(buy {val_ok})")
        matrices.append(matrix)
        val_scores.append(val_res)

    if len(matrices) < 2:
        log(f"\nNeed at least 2 successful models for consensus. Got {len(matrices)}. Exiting.")
        return

    log(f"\n{len(matrices)} models ready for consensus.")

    # -----------------------------------------------------------------------
    # Evaluate individual models on test
    # -----------------------------------------------------------------------
    log(f"\n{'='*70}")
    log(f"Individual model performance on TEST SET ({N_test} draws):")
    log(f"{'='*70}")
    for i, (m, exp) in enumerate(zip(matrices, top_exps), 1):
        report_eval(f"Model {i} (exp#{exp['experiment_id']})", m, y_test, N_test)

    # -----------------------------------------------------------------------
    # Strategy A: Product ensemble
    # -----------------------------------------------------------------------
    log(f"\n{'='*70}")
    log(f"Strategy A -- Product Ensemble ({len(matrices)} models):")
    log(f"{'='*70}")
    prod_matrix = product_ensemble(matrices)
    report_eval("Product ensemble", prod_matrix, y_test, N_test)

    # -----------------------------------------------------------------------
    # Strategy B: Intersection filter at multiple thresholds
    # -----------------------------------------------------------------------
    log(f"\n{'='*70}")
    log(f"Strategy B -- Intersection Filter ({len(matrices)} models):")
    log(f"{'='*70}")
    log(f"  (Baseline: random buy-{args.top_k} = {args.top_k/1000:.1%} hit rate per draw)")
    log("")
    for k in [5, 10, 15, 20]:
        inter = intersection_eval(matrices, y_test, top_k=k)
        report_intersection(f"Intersection top-{k:>2}", inter)

    # -----------------------------------------------------------------------
    # Strategy B with product ensemble ranking (best of both)
    # -----------------------------------------------------------------------
    log(f"\n{'='*70}")
    log(f"Strategy C -- Intersection of individual top-K, ranked by product score:")
    log(f"{'='*70}")
    log(f"  (Use intersection as a filter, rank survivors by product probability)")
    log("")
    for k in [10, 15, 20]:
        N = matrices[0].shape[0]
        results = []
        total_cost = 0
        total_won  = 0
        skipped    = 0
        sizes      = []
        for i in range(N):
            # Find intersection
            sets = [set(np.argsort(-m[i])[:k].tolist()) for m in matrices]
            common = sets[0]
            for s in sets[1:]:
                common = common & s
            if not common:
                skipped += 1
                results.append({"hit": False, "skipped": True, "size": 0})
                continue
            # Rank survivors by product probability
            prod_row = prod_matrix[i]
            survivors = sorted(common, key=lambda c: prod_row[c], reverse=True)
            hit = y_test[i] in common
            sizes.append(len(survivors))
            total_cost += len(survivors)
            if hit:
                total_won += PICK3_PAYOUT
            results.append({"hit": hit, "skipped": False, "size": len(survivors)})

        played   = N - skipped
        hits     = sum(1 for r in results if r["hit"])
        hr       = hits / played if played > 0 else 0
        avg_sz   = np.mean(sizes) if sizes else 0
        ev_draw  = (total_won - total_cost) / N
        log(f"  Intersection+product top-{k:>2}  played={played}/{N}  "
            f"hits={hits}  hit_rate={hr:.4f}  avg_tix={avg_sz:.1f}  ev_per_draw=${ev_draw:>+7.3f}")

    log(f"\nDone. Temp files in: {tmp_dir}")


if __name__ == "__main__":
    main()
