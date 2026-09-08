"""
online.py - Phase 3 Stage C: online replica vs frozen control.

plans/PHASE3_PLAN.md section 4 (D3, decided: option O5, replica only) and
section 5 (G6, live contamination rule). Reuses train.py/train_lag.py's
data/model/loss plumbing; neither file is modified.

Procedure (task instructions):
  - Base model: the Stage B run m1_lag_W64_seed0. Reload its weights if a
    checkpoint was saved; phase3/runs/m1_lag_W64_seed0/ holds only
    final.json/metrics.jsonl/records_*.json/lag_*.json -- no .pt state dict
    -- so the model is retrained deterministically from its own config
    (get_base_model() / _retrain_stage_b() below), mirroring
    train_lag.train_run_lag()'s data/model/train loop exactly (same seed,
    hyperparameters, early stopping, time budget).
  - Frozen control: that model, untouched, predicting every live draw
    walk-forward (all four slots, target equipment tokens masked as usual
    -- automatic, tokenizer.make_windows() always masks EQUIPMENT_COLS of
    the target regardless of split).
  - Replica: a copy updated after each scored live draw with one AdamW
    step (lr 1e-4) on a replay batch of the 64 newest scored live draws
    plus 64 uniformly sampled history (train-split) windows. Live draws
    enter ONLY the replica, ONLY after being scored (G6): predict, then
    score, then update, per draw, in walk-forward order.
  - Both models scored per draw: per-position log-loss, top-1, top-20
    combo hit (product ranking, same convention as train.py combo_eval).
    Frozen vs replica compared via a paired sign-flip permutation test on
    per-draw log-loss (2000 resamples, seed 42), and both models' top-20
    hit rate vs the shuffled-actuals null via lag_null.run_lag_analysis
    (lag-0 cell only -- Stage C's primary metrics, section 6, are lag-0).

DESIGN CHOICES (labeled, not specified by the plan text):
  D1. Replica update loss uses only the lag-0 heads (train.py's plain
      cross-entropy on the immediate next draw, via Lag0View), not the
      full 3*(K+1)-head lag-weighted Stage-B loss. Live replay windows for
      draws near the front of the live window don't yet have lag-1/2
      future targets available online (G6 forbids waiting for them), and
      Stage C's scored metrics (section 6) are lag-0 only, so the update
      objective matches what is measured. Gradient still flows into the
      shared backbone from this loss, only the lag-1/2 heads go unused.
  D2. History replay pool = the Stage A/B train split only (10,836 W=64
      windows), sampled uniformly with replacement each step via a fixed
      RandomState (REPLAY_HISTORY_SAMPLE_SEED), so "uniformly sampled
      history windows" is reproducible run to run.
  D3. "64 newest live draws" is "as many scored live draws as exist so
      far" until 64 accumulate (draw 1 replays with a pool of 1, etc).
  D4. Paired log-loss diff = frozen_avg_logloss - replica_avg_logloss per
      draw (avg over the 3 positions, same convention as train.py's
      compute_logits_loss scalar loss); positive means the replica has
      lower loss (is "better"). The sign-flip test is two-sided.
  D5. Replica AdamW uses weight_decay=0.0 (the plan specifies only lr).

Usage: python3 online.py
"""

import os
# Thread cap (task instruction): a Stage D TimesFM job may be running
# CPU-heavy: set OpenMP/MKL thread env before numpy/torch import so the
# BLAS backends actually respect the cap.
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import sys
import json
import time
import copy

import numpy as np
import torch
import torch.nn.functional as F

from data_loader import load_history, load_live, split_indices
from tokenizer import load_vocab, FIRST_REAL_ID

import train as stage_a
from train import (
    RUNS_DIR, NUM_DIGIT_POSITIONS, TOP20_SIZE, CONSENSUS_SIZE, _COMBO_DIGITS,
    BASELINE_LOGLOSS, BASELINE_TOP1, BASELINE_TOP20, summarize_lag0_top20,
)
import train_lag
from lag_null import run_lag_analysis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_RUN_ID = "m1_lag_W64_seed0"
BASE_CONFIG_PATH = os.path.join(_DIR, "configs", f"{BASE_RUN_ID}.json")

STAGE_C_RUN_ID = "stage_c"
STAGE_C_DIR = os.path.join(RUNS_DIR, STAGE_C_RUN_ID)

MAX_THREADS = 8  # chosen (task instruction): cap threads alongside the Stage D job

REPLICA_LR = 1e-4                 # plan section 4, D3: "chosen"
REPLAY_LIVE_SIZE = 64             # plan section 4: "the newest 64 live draws"
REPLAY_HISTORY_SIZE = 64          # plan section 4: "64 sampled from history"
REPLAY_HISTORY_SAMPLE_SEED = 0    # D2 above: chosen, not specified by the plan

PERM_N_RESAMPLES = 2000  # task instructions: paired-diff sign-flip test
PERM_SEED = 42            # task instructions


def pick_torch_threads():
    torch.set_num_threads(MAX_THREADS)


# ---------------------------------------------------------------------------
# Base model: reload checkpoint, else retrain deterministically (D3 task step 1)
# ---------------------------------------------------------------------------
def _retrain_stage_b(config, run_id, vocab, history_df, device):
    """Mirrors train_lag.train_run_lag()'s data/model/train loop exactly
    (same seed, hyperparameters, early stopping, time budget) but returns
    the trained model in memory instead of writing final.json/records --
    that run's own output files under runs/<run_id>/ already exist and are
    left untouched here."""
    stage_a.set_seed(config["seed"])
    K = config["K"]
    lam = config["lambda"]
    lag_weights = [float(np.exp(-lam * k)) for k in range(K + 1)]
    stage_a.DEVICE = device

    train_w, val_w, _test_w, _dropped = train_lag.build_split_windows_lag(history_df, config["W"], K, vocab)
    train_t = train_lag.to_tensors_lag(train_w)
    val_t = train_lag.to_tensors_lag(val_w)

    model = train_lag.build_lag_model(config["model"], vocab, config["W"], K).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_since_improve = 0
    gen = torch.Generator().manual_seed(config["seed"])
    run_start = time.time()
    epoch = 0

    for epoch in range(1, config["max_epochs"] + 1):
        model.train()
        for batch in stage_a.iterate_batches(train_t, config["batch"], shuffle=True, generator=gen):
            optimizer.zero_grad()
            _, loss, _ = train_lag.compute_lag_loss(model, batch, K, lag_weights)
            loss.backward()
            optimizer.step()

        val_loss = train_lag.evaluate_lag_total(model, val_t, K, lag_weights, batch_size=config["batch"])
        print(f"[{run_id}/retrain] epoch {epoch} val_lag_loss={val_loss:.4f}")

        if val_loss < best_val_loss - stage_a.IMPROVE_EPS:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improve = 0
        else:
            epochs_since_improve += 1
            if epochs_since_improve >= config["patience"]:
                print(f"[{run_id}/retrain] early stop at epoch {epoch}, best_epoch={best_epoch}")
                break

        if time.time() - run_start > train_lag.TIME_BUDGET_S:
            print(f"[{run_id}/retrain] TIME BUDGET exceeded at epoch {epoch}; stopping")
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    elapsed = time.time() - run_start
    return model, best_epoch, best_val_loss, elapsed


def get_base_model(config, run_id, vocab, history_df, device):
    run_dir = os.path.join(RUNS_DIR, run_id)
    ckpt_path = os.path.join(run_dir, "model_state.pt")
    if os.path.exists(ckpt_path):
        model = train_lag.build_lag_model(config["model"], vocab, config["W"], config["K"]).to(device)
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        return model, f"loaded checkpoint {ckpt_path}"

    model, best_epoch, best_val_loss, elapsed = _retrain_stage_b(config, run_id, vocab, history_df, device)
    provenance = (
        f"no checkpoint saved for {run_id} (only final.json/metrics.jsonl/records/lag_*.json exist "
        f"under runs/{run_id}/); retrained deterministically from {BASE_CONFIG_PATH} "
        f"(seed={config['seed']}) on {device} in {elapsed:.1f}s, best_epoch={best_epoch}, "
        f"best_val_lag_loss={best_val_loss:.6f}. NOTE: the original run picked CPU (low free GPU "
        f"memory at the time, see runs/{run_id}_stdout.log); this retrain may run on a different "
        f"device depending on current GPU headroom, which is a source of nondeterminism relative "
        f"to a literal re-run, though the training procedure (seed, data, hyperparameters) is identical."
    )
    return model, provenance


# ---------------------------------------------------------------------------
# Per-draw scoring (product-ranking combo sets, same convention as
# train.py's combo_eval())
# ---------------------------------------------------------------------------
def score_record(logits, actual_digits):
    """logits: (3, 10) tensor (one draw). actual_digits: (3,) int array, raw 0-9."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp().detach().cpu().numpy()
    log_probs_np = log_probs.detach().cpu().numpy()

    pos_logloss = [float(-log_probs_np[p, int(actual_digits[p])]) for p in range(NUM_DIGIT_POSITIONS)]
    pos_correct = [int(np.argmax(probs[p]) == int(actual_digits[p])) for p in range(NUM_DIGIT_POSITIONS)]

    combo_probs = probs[0, _COMBO_DIGITS[:, 0]] * probs[1, _COMBO_DIGITS[:, 1]] * probs[2, _COMBO_DIGITS[:, 2]]
    order = np.argsort(-combo_probs)
    top20_combos = [f"{c:03d}" for c in order[:TOP20_SIZE]]
    consensus_combos = [f"{c:03d}" for c in order[:CONSENSUS_SIZE]]
    actual_combo = f"{int(actual_digits[0]) * 100 + int(actual_digits[1]) * 10 + int(actual_digits[2]):03d}"

    return {
        "pos_logloss": pos_logloss,
        "avg_logloss": float(np.mean(pos_logloss)),
        "pos_correct": pos_correct,
        "top20_combos": top20_combos,
        "consensus_combos": consensus_combos,
        "actual_combo": actual_combo,
        "top20_hit": bool(actual_combo in top20_combos),
    }


def predict(view, context_np, target_known_np, device):
    ctx = torch.from_numpy(context_np[None, ...]).to(device)
    tk = torch.from_numpy(target_known_np[None, ...]).to(device)
    with torch.no_grad():
        logits = view(ctx, tk)[0]  # (3, 10)
    return logits


def replay_update(replica_view, optimizer, live_pool, hist_pool, rng, device):
    n_live_take = min(REPLAY_LIVE_SIZE, len(live_pool))
    live_sample = live_pool[-n_live_take:]
    ctx_live = np.stack([r["context"] for r in live_sample], axis=0)
    tk_live = np.stack([r["target_known"] for r in live_sample], axis=0)
    td_live = np.stack([r["target_digits"] for r in live_sample], axis=0)  # raw 0-9

    n_hist = hist_pool["context"].shape[0]
    hist_idx = rng.choice(n_hist, size=REPLAY_HISTORY_SIZE, replace=True)
    ctx_hist = hist_pool["context"][hist_idx]
    tk_hist = hist_pool["target_known"][hist_idx]
    td_hist = hist_pool["target_digits"][hist_idx] - FIRST_REAL_ID  # raw 0-9

    ctx = np.concatenate([ctx_live, ctx_hist], axis=0)
    tk = np.concatenate([tk_live, tk_hist], axis=0)
    td = np.concatenate([td_live, td_hist], axis=0)

    batch = {
        "context": torch.from_numpy(ctx).to(device),
        "target_known": torch.from_numpy(tk).to(device),
        "target_digits": torch.from_numpy(td).to(device),
    }
    replica_view.train()
    optimizer.zero_grad()
    _, loss, _ = stage_a.compute_logits_loss(replica_view, batch)
    loss.backward()
    optimizer.step()
    replica_view.eval()
    return float(loss.item())


# ---------------------------------------------------------------------------
# Paired sign-flip permutation test on per-draw log-loss
# ---------------------------------------------------------------------------
def sign_flip_test(diffs, n_perm=PERM_N_RESAMPLES, seed=PERM_SEED):
    diffs = np.asarray(diffs, dtype=float)
    observed = float(diffs.mean())
    rng = np.random.RandomState(seed)
    n = diffs.shape[0]
    null = np.empty(n_perm)
    for t in range(n_perm):
        signs = rng.choice(np.array([-1.0, 1.0]), size=n)
        null[t] = (diffs * signs).mean()
    p_value = float((1 + np.sum(np.abs(null) >= abs(observed))) / (n_perm + 1))
    return {
        "diff_convention": "frozen_avg_logloss - replica_avg_logloss (positive = replica has lower loss)",
        "observed_mean_diff": observed,
        "null_mean": float(null.mean()),
        "null_std": float(null.std()),
        "n_perm": n_perm,
        "seed": seed,
        "p_value_two_sided": p_value,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def summarize_model(records, label):
    n = len(records)
    avg_logloss = float(np.mean([r["avg_logloss"] for r in records]))
    pos_logloss = np.mean([r["pos_logloss"] for r in records], axis=0).tolist()
    pos_top1 = np.mean([r["pos_correct"] for r in records], axis=0).tolist()
    top20_hit_rate = float(np.mean([r["top20_hit"] for r in records]))

    lag_result = run_lag_analysis(records)
    top20_null = summarize_lag0_top20(lag_result)

    return {
        "label": label,
        "n": n,
        "avg_logloss": avg_logloss,
        "pos_logloss": pos_logloss,
        "pos_top1_acc": pos_top1,
        "top20_hit_rate": top20_hit_rate,
        "top20_vs_null": top20_null,
        "baseline_logloss": BASELINE_LOGLOSS,
        "baseline_top1": BASELINE_TOP1,
        "baseline_top20": BASELINE_TOP20,
    }


def main():
    pick_torch_threads()
    os.makedirs(STAGE_C_DIR, exist_ok=True)

    with open(BASE_CONFIG_PATH) as f:
        config = json.load(f)

    device, device_note = train_lag.pick_device()
    stage_a.DEVICE = device
    print(f"[stage_c] device={device_note} threads={MAX_THREADS}")

    vocab = load_vocab()
    history_df = load_history()

    t_base0 = time.time()
    base_model, provenance = get_base_model(config, BASE_RUN_ID, vocab, history_df, device)
    base_model.to(device)
    base_model.eval()
    print(f"[stage_c] base model ready in {time.time() - t_base0:.1f}s: {provenance}")

    frozen_view = train_lag.Lag0View(base_model).to(device)
    frozen_view.eval()

    replica_model = copy.deepcopy(base_model).to(device)
    replica_view = train_lag.Lag0View(replica_model).to(device)
    replica_view.eval()
    optimizer = torch.optim.AdamW(replica_model.parameters(), lr=REPLICA_LR, weight_decay=0.0)

    W = config["W"]
    live_w = stage_a.build_live_windows(history_df, W, vocab)
    n_live = live_w["context"].shape[0]

    train_w, _val_w, _test_w = stage_a.build_split_windows(history_df, W, vocab)
    hist_pool = train_w
    print(f"[stage_c] n_live={n_live} hist_pool={hist_pool['context'].shape[0]}")

    rng = np.random.RandomState(REPLAY_HISTORY_SAMPLE_SEED)

    live_pool = []
    frozen_records = []
    replica_records = []

    t_loop0 = time.time()
    for i in range(n_live):
        ctx_np = live_w["context"][i]
        tk_np = live_w["target_known"][i]
        actual_digits = live_w["target_digits"][i] - FIRST_REAL_ID  # raw 0-9

        logits_f = predict(frozen_view, ctx_np, tk_np, device)
        logits_r = predict(replica_view, ctx_np, tk_np, device)

        frozen_records.append(score_record(logits_f, actual_digits))
        replica_records.append(score_record(logits_r, actual_digits))

        live_pool.append({"context": ctx_np, "target_known": tk_np, "target_digits": actual_digits})
        replay_update(replica_view, optimizer, live_pool, hist_pool, rng, device)

        if (i + 1) % 100 == 0 or i == n_live - 1:
            print(f"[stage_c] scored {i + 1}/{n_live} live draws")

    runtime_s = time.time() - t_loop0
    total_runtime_s = time.time() - t_base0

    frozen_summary = summarize_model(frozen_records, "frozen")
    replica_summary = summarize_model(replica_records, "replica")

    diffs = np.array([frozen_records[j]["avg_logloss"] - replica_records[j]["avg_logloss"] for j in range(n_live)])
    paired_diff = sign_flip_test(diffs)

    final = {
        "run_id": STAGE_C_RUN_ID,
        "base_run_id": BASE_RUN_ID,
        "base_config": config,
        "base_provenance": provenance,
        "device": device_note,
        "max_threads": MAX_THREADS,
        "n_live": n_live,
        "loop_runtime_s": runtime_s,
        "total_runtime_s": total_runtime_s,
        "replica_lr": REPLICA_LR,
        "replay_live_size": REPLAY_LIVE_SIZE,
        "replay_history_size": REPLAY_HISTORY_SIZE,
        "replay_history_sample_seed": REPLAY_HISTORY_SAMPLE_SEED,
        "frozen": frozen_summary,
        "replica": replica_summary,
        "paired_logloss_diff": paired_diff,
        "design_choices": [
            "D1: replica update loss is lag-0-only cross-entropy (Lag0View + train.py's "
            "compute_logits_loss), not the full 3*(K+1)-head Stage-B lag loss -- see module docstring.",
            "D2: history replay pool = Stage A/B train split only, sampled uniformly with "
            f"replacement via RandomState(seed={REPLAY_HISTORY_SAMPLE_SEED}).",
            "D3: '64 newest live draws' means 'as many scored so far' until 64 accumulate.",
            "D4: paired-diff = frozen_avg_logloss - replica_avg_logloss, two-sided sign-flip p-value.",
            "D5: replica AdamW weight_decay=0.0 (plan specifies only lr=1e-4).",
        ],
    }

    with open(os.path.join(STAGE_C_DIR, "final.json"), "w") as f:
        json.dump(final, f, indent=2)
    with open(os.path.join(STAGE_C_DIR, "records_frozen.json"), "w") as f:
        json.dump(frozen_records, f)
    with open(os.path.join(STAGE_C_DIR, "records_replica.json"), "w") as f:
        json.dump(replica_records, f)

    print(f"[stage_c] done: n_live={n_live} runtime={total_runtime_s:.1f}s")
    print(f"[stage_c] frozen  avg_logloss={frozen_summary['avg_logloss']:.4f} "
          f"top20={frozen_summary['top20_hit_rate']:.4f}")
    print(f"[stage_c] replica avg_logloss={replica_summary['avg_logloss']:.4f} "
          f"top20={replica_summary['top20_hit_rate']:.4f}")
    print(f"[stage_c] paired diff p={paired_diff['p_value_two_sided']:.4f} "
          f"observed={paired_diff['observed_mean_diff']:.6f}")
    return final


if __name__ == "__main__":
    main()
