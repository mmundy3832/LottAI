"""
train_lag.py - Phase 3 Stage B training loop (lag-credited loss, H2 test).
plans/PHASE3_PLAN.md section 4. Reuses train.py's data/model/eval plumbing;
train.py itself is not modified.

Loss (task instructions step 1), for a window at draw t:

    loss_t = - sum_p sum_k w_k * log P_p^k(digit at t+k, position p)
    w_k = exp(-lambda * k),  k in 0..K

DESIGN CHOICE (labeled, not in the plan text): the literal formula sums the
per-position NLL. The per-lag term here instead uses the MEAN over the 3
positions (not the sum) before applying w_k, so each lag's contribution
stays on the same ln(10)-ish scale Stage A already reports on, and only the
across-lag combination is weighted. This changes the training-loss/early-
stopping *scale* (irrelevant -- it is only used for gradient descent and
early stopping within a single run) but not the *ranking* of lags relative
to each other. All reported evaluation log-losses (final.json) are plain
per-position mean cross-entropy, unweighted, exactly Stage A's convention,
so they are directly comparable to Stage A numbers.

Architecture: same M1/M2 backbone as model.py, but build_lag_model() swaps
the 3-head output layer for 3*(K+1) heads (10-way each), ordered lag-major:
head index = k*NUM_DIGIT_POSITIONS + p. Heads 0..2 are therefore lag-0 and
have the exact same (B, 3, 10) layout Stage A's models produce, which lets
Lag0View wrap a Stage-B model so train.py's evaluate()/combo_eval()/
summarize_lag0_top20() run unmodified for the lag-0-only evaluation (task
instructions step 3a).

Config JSON (phase3/configs/*.json) fields, on top of train.py's Stage A
fields (model, W, seed, batch, lr, max_epochs, patience, weight_decay):
  K              max lag (task instructions step 2: 2)
  lambda         decay rate for w_k = exp(-lambda*k) (task instructions step 2: 1.0)
  comment        free text documenting K/lambda as placeholders (not measured)

Usage: python3 train_lag.py configs/m1_lag_W64_seed0.json
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import load_history, load_live, split_indices
from tokenizer import (
    load_vocab, tokenize_draws, FIRST_REAL_ID, MASK_ID,
    N_TOKENS_PER_DRAW, TARGET_KNOWN_COLS, EQUIPMENT_COLS, DIGIT_COLS,
)
from model import build_model, count_params, NUM_DIGIT_POSITIONS, DIGIT_OUT_SIZE

import train as stage_a
from train import (
    RUNS_DIR, set_seed, iterate_batches, evaluate, combo_eval,
    summarize_lag0_top20, BASELINE_LOGLOSS, BASELINE_TOP1, BASELINE_TOP20,
    IMPROVE_EPS,
)
from lag_null import run_lag_analysis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# GPU headroom required before attempting a Stage B run on cuda. Chosen (not
# measured): Stage A's W64 M1 run peaked at ~379MB (phase3/runs/m1_W64_seed0
# /metrics.jsonl); Stage B adds 3*(K+1) heads instead of 3 (small, output
# layer only) so a similar backbone footprint is expected, plus margin so we
# never contend with the three M3 grokking runs already resident on the GPU.
MIN_FREE_GPU_BYTES = 500_000_000

# Wall-clock budget per run (task instructions step 4: "if a run exceeds 15
# minutes, stop and report instead of waiting"). Checked after each epoch;
# on breach we stop training and evaluate/report on the best checkpoint
# found so far, same graceful path as early stopping.
TIME_BUDGET_S = 15 * 60


def pick_device():
    if not torch.cuda.is_available():
        return torch.device("cpu"), "cpu (no cuda)"
    free_bytes, _total = torch.cuda.mem_get_info()
    if free_bytes < MIN_FREE_GPU_BYTES:
        return torch.device("cpu"), f"cpu (gpu free={free_bytes/1e6:.0f}MB < {MIN_FREE_GPU_BYTES/1e6:.0f}MB headroom)"
    return torch.device("cuda"), f"cuda (gpu free={free_bytes/1e6:.0f}MB)"


# ---------------------------------------------------------------------------
# Windowing with K future targets (drops last K draws of each split/live so
# every window's targets exist and none crosses a split boundary -- task
# instructions step 1).
# ---------------------------------------------------------------------------
def make_lag_windows(tokens, W, K):
    """Like tokenizer.make_windows but target_digits_lag carries K+1 future
    draws' digits instead of one. Windows exist for t in [W, N-K)."""
    tokens = np.asarray(tokens, dtype=np.int64)
    n = tokens.shape[0]
    m = n - W - K
    if m <= 0:
        return {
            "context": np.empty((0, W, N_TOKENS_PER_DRAW), dtype=np.int64),
            "target_known": np.empty((0, len(TARGET_KNOWN_COLS)), dtype=np.int64),
            "target_digits_lag": np.empty((0, K + 1, len(DIGIT_COLS)), dtype=np.int64),
            "target_index": np.empty((0,), dtype=np.int64),
        }

    context = np.stack([tokens[t - W:t, :] for t in range(W, n - K)], axis=0)

    target_known = tokens[W:n - K, :][:, TARGET_KNOWN_COLS].copy()
    equip_rel_cols = [TARGET_KNOWN_COLS.index(c) for c in EQUIPMENT_COLS]
    target_known[:, equip_rel_cols] = MASK_ID

    target_digits_lag = np.stack(
        [tokens[W + k:n - K + k, :][:, DIGIT_COLS] for k in range(K + 1)], axis=1
    )  # (M, K+1, 3)
    target_index = np.arange(W, n - K, dtype=np.int64)

    return {
        "context": context,
        "target_known": target_known,
        "target_digits_lag": target_digits_lag,
        "target_index": target_index,
    }


def build_split_windows_lag(history_df, W, K, vocab):
    """Continuous history token stream, split by target_index, dropping any
    window whose t+K would cross into the next split (or, for test, past
    the end of history -- there is no next split to cross into, so those
    windows never exist in the first place under make_lag_windows' n-K
    bound). Returns (train, val, test, dropped) where dropped counts, per
    split, Stage-A single-target window count minus Stage-B count."""
    tokens = tokenize_draws(history_df, vocab)
    windows = make_lag_windows(tokens, W, K)
    train_end, val_end, test_end = split_indices(history_df)
    ti = windows["target_index"]

    train_mask = (ti < train_end) & (ti + K < train_end)
    val_mask = (ti >= train_end) & (ti < val_end) & (ti + K < val_end)
    test_mask = (ti >= val_end) & (ti < test_end) & (ti + K < test_end)

    def sel(mask):
        return {k: v[mask] for k, v in windows.items()}

    # Stage-A (K=0) single-target window count per split, for the drop log.
    stage_a_train_n = max(train_end - W, 0)
    stage_a_val_n = max(val_end - train_end, 0)
    stage_a_test_n = max(test_end - val_end, 0)

    dropped = {
        "train": stage_a_train_n - int(train_mask.sum()),
        "val": stage_a_val_n - int(val_mask.sum()),
        "test": stage_a_test_n - int(test_mask.sum()),
    }
    return sel(train_mask), sel(val_mask), sel(test_mask), dropped


def build_live_windows_lag(history_df, W, K, vocab):
    """Live token stream = tail(W) of full history + live draws (same
    construction as train.build_live_windows), so every live draw gets a
    full W-draw context. Drops the last K live draws (no data beyond the
    live file to supply their lag targets)."""
    live_df = load_live()
    tail_df = history_df.tail(W)
    combined = pd.concat([tail_df, live_df], ignore_index=True)
    tokens = tokenize_draws(combined, vocab)
    windows = make_lag_windows(tokens, W, K)
    dropped = len(live_df) - windows["context"].shape[0]
    return windows, dropped


def to_tensors_lag(w):
    return {
        "context": torch.from_numpy(w["context"]),
        "target_known": torch.from_numpy(w["target_known"]),
        "target_digits_lag": torch.from_numpy(w["target_digits_lag"] - FIRST_REAL_ID),
    }


def lag0_tensors(tl):
    """Slice out the lag-0 target as a Stage-A-shaped tensors dict (keys
    context/target_known/target_digits, target_digits (B,3)) so train.py's
    evaluate()/combo_eval() run unmodified."""
    return {
        "context": tl["context"],
        "target_known": tl["target_known"],
        "target_digits": tl["target_digits_lag"][:, 0, :],
    }


# ---------------------------------------------------------------------------
# Model: Stage A backbone, 3*(K+1)-head output layer.
# ---------------------------------------------------------------------------
def build_lag_model(model_name, vocab, W, K):
    base = build_model(model_name, vocab, W)
    in_features = base.heads[0].in_features
    n_heads_total = NUM_DIGIT_POSITIONS * (K + 1)
    base.heads = nn.ModuleList([nn.Linear(in_features, DIGIT_OUT_SIZE) for _ in range(n_heads_total)])
    return base


class Lag0View(nn.Module):
    """Adapter exposing only the model's lag-0 heads (indices 0..2 under the
    lag-major head ordering k*NUM_DIGIT_POSITIONS+p) as a (B, 3, 10) output,
    so train.py's evaluate()/combo_eval() -- written for a 3-head model --
    work unmodified on a Stage-B model."""
    def __init__(self, lag_model):
        super().__init__()
        self.lag_model = lag_model

    def forward(self, context, target_known):
        logits = self.lag_model(context, target_known)
        return logits[:, :NUM_DIGIT_POSITIONS, :]


# ---------------------------------------------------------------------------
# Loss / eval
# ---------------------------------------------------------------------------
def compute_lag_loss(model, batch, K, lag_weights):
    logits = model(batch["context"], batch["target_known"])  # (B, 3*(K+1), 10)
    targets = batch["target_digits_lag"]                       # (B, K+1, 3)
    lag_terms = []
    per_lag_pos_loss = {}
    for k in range(K + 1):
        pos_losses = []
        for p in range(NUM_DIGIT_POSITIONS):
            head_idx = k * NUM_DIGIT_POSITIONS + p
            ce = F.cross_entropy(logits[:, head_idx, :], targets[:, k, p], reduction="mean")
            pos_losses.append(ce)
            per_lag_pos_loss[(k, p)] = ce
        lag_mean = torch.stack(pos_losses).mean()
        lag_terms.append(lag_weights[k] * lag_mean)
    total_loss = torch.stack(lag_terms).sum()
    return logits, total_loss, per_lag_pos_loss


def evaluate_lag_total(model, tensors, K, lag_weights, batch_size):
    """Val TOTAL lag loss (early-stopping criterion, task instructions step 2)."""
    model.eval()
    n = tensors["context"].shape[0]
    total = 0.0
    with torch.no_grad():
        for batch in iterate_batches(tensors, batch_size, shuffle=False):
            _, loss, _ = compute_lag_loss(model, batch, K, lag_weights)
            bs = batch["context"].shape[0]
            total += loss.item() * bs
    return total / n


def evaluate_lag_heads(model, tensors, K, batch_size, device):
    """Plain (unweighted) per-(lag, position) mean cross-entropy over a full
    split, evaluated against the draw at t+k -- task instructions step 3b.
    Returns a (K+1, 3) nested list."""
    model.eval()
    n = tensors["context"].shape[0]
    sums = np.zeros((K + 1, NUM_DIGIT_POSITIONS))
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = {k: v[start:end].to(device) for k, v in tensors.items()}
            logits = model(batch["context"], batch["target_known"])
            targets = batch["target_digits_lag"]
            bs = end - start
            for k in range(K + 1):
                for p in range(NUM_DIGIT_POSITIONS):
                    head_idx = k * NUM_DIGIT_POSITIONS + p
                    ce = F.cross_entropy(logits[:, head_idx, :], targets[:, k, p], reduction="mean")
                    sums[k, p] += ce.item() * bs
    return (sums / n).tolist()


def extract_lag_table_top20(lag_result):
    """All lag 0..5 cells for set=top20, stat=exact_hit_rate -- task
    instructions step 3c."""
    out = {}
    for cell in lag_result["cells"]:
        if cell["set"] == "top20" and cell["stat"] == "exact_hit_rate":
            out[str(cell["lag"])] = {
                "observed": cell["observed"], "null_mean": cell["null_mean"], "p_value": cell["p_value"],
            }
    return out


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_run_lag(config, run_id):
    set_seed(config["seed"])
    K = config["K"]
    lam = config["lambda"]
    lag_weights = [float(np.exp(-lam * k)) for k in range(K + 1)]

    device, device_note = pick_device()
    stage_a.DEVICE = device  # runtime override so reused train.py functions target this device
    print(f"[{run_id}] device={device_note}")

    vocab = load_vocab()
    history_df = load_history()
    train_w, val_w, test_w, dropped = build_split_windows_lag(history_df, config["W"], K, vocab)
    live_w, live_dropped = build_live_windows_lag(history_df, config["W"], K, vocab)
    dropped["live"] = live_dropped
    print(f"[{run_id}] dropped windows (K={K}): {dropped}")

    train_t = to_tensors_lag(train_w)
    val_t = to_tensors_lag(val_w)
    test_t = to_tensors_lag(test_w)
    live_t = to_tensors_lag(live_w)

    model = build_lag_model(config["model"], vocab, config["W"], K).to(device)
    n_params = count_params(model)
    print(f"[{run_id}] model={config['model']} K={K} lambda={lam} params={n_params:,} "
          f"train={train_t['context'].shape[0]} val={val_t['context'].shape[0]} "
          f"test={test_t['context'].shape[0]} live={live_t['context'].shape[0]}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    best_val_loss = float("inf")
    best_epoch = -1
    best_state = None
    epochs_since_improve = 0
    time_budget_exceeded = False

    gen = torch.Generator().manual_seed(config["seed"])
    run_start = time.time()

    with open(metrics_path, "w") as mf:
        for epoch in range(1, config["max_epochs"] + 1):
            model.train()
            t0 = time.time()
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats()
            total_train_loss = 0.0
            n_train = train_t["context"].shape[0]
            for batch in iterate_batches(train_t, config["batch"], shuffle=True, generator=gen):
                optimizer.zero_grad()
                _, loss, _ = compute_lag_loss(model, batch, K, lag_weights)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * batch["context"].shape[0]
            train_loss = total_train_loss / n_train

            val_loss = evaluate_lag_total(model, val_t, K, lag_weights, batch_size=config["batch"])
            epoch_time = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated() if device.type == "cuda" else 0

            record = {
                "epoch": epoch, "train_lag_loss": train_loss, "val_lag_loss": val_loss,
                "epoch_time_s": epoch_time, "peak_gpu_mem_bytes": peak_mem,
            }
            mf.write(json.dumps(record) + "\n")
            mf.flush()
            print(f"[{run_id}] epoch {epoch} train_lag_loss={train_loss:.4f} val_lag_loss={val_loss:.4f} "
                  f"time={epoch_time:.2f}s peak_mem={peak_mem / 1e6:.1f}MB")

            if val_loss < best_val_loss - IMPROVE_EPS:
                best_val_loss = val_loss
                best_epoch = epoch
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= config["patience"]:
                    print(f"[{run_id}] early stop at epoch {epoch}, best_epoch={best_epoch}")
                    break

            if time.time() - run_start > TIME_BUDGET_S:
                time_budget_exceeded = True
                print(f"[{run_id}] TIME BUDGET ({TIME_BUDGET_S}s) exceeded at epoch {epoch}; stopping, "
                      f"reporting on best_epoch={best_epoch}")
                break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch

    model.load_state_dict(best_state)
    model.to(device)
    model.eval()
    lag0_view = Lag0View(model).to(device)
    lag0_view.eval()

    final = {
        "run_id": run_id, "config": config, "n_params": n_params,
        "device": device_note, "best_epoch": best_epoch, "best_val_lag_loss": best_val_loss,
        "time_budget_exceeded": time_budget_exceeded, "dropped_windows": dropped,
        "lag_weights": lag_weights,
        "baseline_logloss": BASELINE_LOGLOSS, "baseline_top1": BASELINE_TOP1, "baseline_top20": BASELINE_TOP20,
    }

    for split_name, tl in [("val", val_t), ("test", test_t), ("live", live_t)]:
        l0t = lag0_tensors(tl)
        avg_loss, pos_logloss, pos_acc = evaluate(lag0_view, l0t, batch_size=config["batch"])
        top20_hit, records = combo_eval(lag0_view, l0t, batch_size=config["batch"])
        lag_head_pos_logloss = evaluate_lag_heads(model, tl, K, config["batch"], device)
        final[split_name] = {
            "n": tl["context"].shape[0],
            "lag0_logloss": avg_loss, "lag0_pos_logloss": pos_logloss, "lag0_pos_acc": pos_acc,
            "lag0_top20_hit_rate": top20_hit,
            "lag_head_pos_logloss": lag_head_pos_logloss,  # (K+1, 3), index [k][p]
        }
        if split_name in ("test", "live"):
            with open(os.path.join(run_dir, f"records_{split_name}.json"), "w") as rf:
                json.dump(records, rf)
            lag_result = run_lag_analysis(records)
            with open(os.path.join(run_dir, f"lag_{split_name}.json"), "w") as lf:
                json.dump(lag_result, lf, indent=2)
            final[f"{split_name}_lag0_top20"] = summarize_lag0_top20(lag_result)
            final[f"{split_name}_lag_table_top20"] = extract_lag_table_top20(lag_result)

    with open(os.path.join(run_dir, "final.json"), "w") as ff:
        json.dump(final, ff, indent=2)
    print(f"[{run_id}] done: best_epoch={best_epoch} device={device_note} "
          f"test_lag0_logloss={final['test']['lag0_pos_logloss']} "
          f"test_lag0_top20={final['test']['lag0_top20_hit_rate']:.4f} "
          f"live_lag0_top20={final['live']['lag0_top20_hit_rate']:.4f}")
    return final


if __name__ == "__main__":
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)
    run_id = config.get("run_id") or os.path.splitext(os.path.basename(config_path))[0]
    train_run_lag(config, run_id)
