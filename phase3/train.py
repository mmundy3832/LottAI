"""
train.py - Phase 3 Stage A training loop (lag-0 control, H1 test).

Config JSON (phase3/configs/*.json) fields:
  run_id         string, optional (defaults to the config filename stem)
  model          "m1" | "m2"
  W              context window, number of prior draws
  seed           int
  batch          batch size
  lr             AdamW learning rate
  max_epochs     upper bound on epochs
  patience       epochs without val-loss improvement before stopping
  weight_decay   AdamW weight decay

Usage: python3 train.py configs/m1_W64_seed0.json
"""

import os
import sys
import json
import time
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data_loader import load_history, load_live, split_indices
from tokenizer import load_vocab, tokenize_draws, make_windows, DIGIT_BASE_SIZE, FIRST_REAL_ID
from model import build_model, count_params
from lag_null import run_lag_analysis

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_DIR = os.path.dirname(os.path.abspath(__file__))
RUNS_DIR = os.path.join(_DIR, "runs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_DIGIT_POSITIONS = 3
COMBO_SPACE = 1000
TOP20_SIZE = 20
CONSENSUS_SIZE = 30

BASELINE_LOGLOSS = 2.302585  # ln(10)
BASELINE_TOP1 = 0.10
BASELINE_TOP20 = 0.020

EVAL_BATCH_SIZE = 512      # only affects eval-loop chunking, not training dynamics
IMPROVE_EPS = 1e-4         # min val-loss decrease counted as "improvement" for early stopping

# Combo -> (d0, d1, d2) digit decomposition, index = combo int 0..999,
# f"{combo:03d}" matches ledger.py / lag_null.py's actual_combo string format.
_COMBO_DIGITS = np.array([[c // 100, (c // 10) % 10, c % 10] for c in range(COMBO_SPACE)])


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Data: windows per split, built from one continuous history token stream so
# every split's early windows get legitimate context from the prior split's
# tail (no lookahead -- context is always strictly before the target draw).
# ---------------------------------------------------------------------------
def build_split_windows(history_df, W, vocab):
    tokens = tokenize_draws(history_df, vocab)
    windows = make_windows(tokens, W)
    train_end, val_end, test_end = split_indices(history_df)
    ti = windows["target_index"]
    train_mask = ti < train_end
    val_mask = (ti >= train_end) & (ti < val_end)
    test_mask = (ti >= val_end) & (ti < test_end)

    def sel(mask):
        return {k: v[mask] for k, v in windows.items()}

    return sel(train_mask), sel(val_mask), sel(test_mask)


def build_live_windows(history_df, W, vocab):
    """Live token stream = tail(W) of full history + live draws, so every
    live draw gets a full W-draw context (task instructions step 2 note)."""
    live_df = load_live()
    tail_df = history_df.tail(W)
    combined = pd.concat([tail_df, live_df], ignore_index=True)
    tokens = tokenize_draws(combined, vocab)
    windows = make_windows(tokens, W)
    assert windows["context"].shape[0] == len(live_df), (
        f"live windows ({windows['context'].shape[0]}) != live draws ({len(live_df)})"
    )
    return windows


def to_tensors(w):
    # target_digits comes out of tokenizer.make_windows() as token ids
    # (raw digit + FIRST_REAL_ID, per tokenizer.py's _encode_offset). The
    # classification targets (cross-entropy class index, combo digits) need
    # the raw 0-9 digit, so subtract the offset here -- the only place
    # target_digits is used, since it is never fed back in as model input.
    return {
        "context": torch.from_numpy(w["context"]),
        "target_known": torch.from_numpy(w["target_known"]),
        "target_digits": torch.from_numpy(w["target_digits"] - FIRST_REAL_ID),
    }


def iterate_batches(tensors, batch_size, shuffle, generator=None):
    n = tensors["context"].shape[0]
    idx = torch.randperm(n, generator=generator) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        b = idx[i:i + batch_size]
        yield {k: v[b].to(DEVICE) for k, v in tensors.items()}


# ---------------------------------------------------------------------------
# Loss / eval
# ---------------------------------------------------------------------------
def compute_logits_loss(model, batch):
    logits = model(batch["context"], batch["target_known"])  # (B, 3, 10)
    targets = batch["target_digits"]                          # (B, 3)
    per_pos = [
        F.cross_entropy(logits[:, p, :], targets[:, p], reduction="mean")
        for p in range(NUM_DIGIT_POSITIONS)
    ]
    loss = torch.stack(per_pos).mean()
    return logits, loss, per_pos


def evaluate(model, tensors, batch_size=EVAL_BATCH_SIZE):
    model.eval()
    n = tensors["context"].shape[0]
    total_loss = 0.0
    pos_logloss_sum = np.zeros(NUM_DIGIT_POSITIONS)
    pos_correct = np.zeros(NUM_DIGIT_POSITIONS)
    with torch.no_grad():
        for batch in iterate_batches(tensors, batch_size, shuffle=False):
            logits, loss, per_pos = compute_logits_loss(model, batch)
            bs = batch["context"].shape[0]
            total_loss += loss.item() * bs
            for p in range(NUM_DIGIT_POSITIONS):
                pos_logloss_sum[p] += per_pos[p].item() * bs
                preds = logits[:, p, :].argmax(dim=-1)
                pos_correct[p] += (preds == batch["target_digits"][:, p]).sum().item()
    return total_loss / n, (pos_logloss_sum / n).tolist(), (pos_correct / n).tolist()


def combo_eval(model, tensors, batch_size=EVAL_BATCH_SIZE):
    """Per-record top20/top30 combo sets ranked by product of per-position
    probs, plus actual_combo, plus the top-20 exact-hit rate."""
    model.eval()
    n = tensors["context"].shape[0]
    records = []
    hits = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch = {k: v[start:end].to(DEVICE) for k, v in tensors.items()}
            logits = model(batch["context"], batch["target_known"])
            probs = torch.softmax(logits, dim=-1).cpu().numpy()  # (B, 3, 10)
            targets = batch["target_digits"].cpu().numpy()        # (B, 3)
            for i in range(probs.shape[0]):
                p = probs[i]
                combo_probs = (
                    p[0, _COMBO_DIGITS[:, 0]] * p[1, _COMBO_DIGITS[:, 1]] * p[2, _COMBO_DIGITS[:, 2]]
                )
                order = np.argsort(-combo_probs)
                top20_combos = [f"{c:03d}" for c in order[:TOP20_SIZE]]
                consensus_combos = [f"{c:03d}" for c in order[:CONSENSUS_SIZE]]
                actual_combo = f"{targets[i, 0] * 100 + targets[i, 1] * 10 + targets[i, 2]:03d}"
                if actual_combo in top20_combos:
                    hits += 1
                records.append({
                    "top20_combos": top20_combos,
                    "consensus_combos": consensus_combos,
                    "actual_combo": actual_combo,
                })
    return hits / n, records


def summarize_lag0_top20(lag_result):
    for cell in lag_result["cells"]:
        if cell["lag"] == 0 and cell["set"] == "top20" and cell["stat"] == "exact_hit_rate":
            return {"observed": cell["observed"], "null_mean": cell["null_mean"], "p_value": cell["p_value"]}
    return None


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------
def train_run(config, run_id, epoch_limit=None):
    """epoch_limit: if set, stop after this many epochs regardless of
    patience (used only for the one-epoch timing probe, never for a real run)."""
    set_seed(config["seed"])

    vocab = load_vocab()
    history_df = load_history()
    train_w, val_w, test_w = build_split_windows(history_df, config["W"], vocab)
    live_w = build_live_windows(history_df, config["W"], vocab)

    train_t = to_tensors(train_w)
    val_t = to_tensors(val_w)
    test_t = to_tensors(test_w)
    live_t = to_tensors(live_w)

    model = build_model(
        config["model"], vocab, config["W"],
        n_layers=config.get("n_layers"), d_model=config.get("d_model"),
        n_heads=config.get("n_heads"), ff_dim=config.get("ff_dim"),
        dropout=config.get("dropout"),
    ).to(DEVICE)
    n_params = count_params(model)
    print(f"[{run_id}] model={config['model']} params={n_params:,} "
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

    gen = torch.Generator().manual_seed(config["seed"])
    max_epochs = epoch_limit if epoch_limit is not None else config["max_epochs"]

    with open(metrics_path, "w") as mf:
        for epoch in range(1, max_epochs + 1):
            model.train()
            t0 = time.time()
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
            total_train_loss = 0.0
            n_train = train_t["context"].shape[0]
            for batch in iterate_batches(train_t, config["batch"], shuffle=True, generator=gen):
                optimizer.zero_grad()
                _, loss, _ = compute_logits_loss(model, batch)
                loss.backward()
                optimizer.step()
                total_train_loss += loss.item() * batch["context"].shape[0]
            train_loss = total_train_loss / n_train

            val_loss, val_pos_logloss, val_pos_acc = evaluate(model, val_t, batch_size=config["batch"])
            epoch_time = time.time() - t0
            peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

            record = {
                "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                "val_pos_logloss": val_pos_logloss, "val_pos_acc": val_pos_acc,
                "epoch_time_s": epoch_time, "peak_gpu_mem_bytes": peak_mem,
            }
            mf.write(json.dumps(record) + "\n")
            mf.flush()
            print(f"[{run_id}] epoch {epoch} train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
                  f"time={epoch_time:.2f}s peak_mem={peak_mem / 1e6:.1f}MB")

            if epoch_limit is not None:
                continue  # timing probe: skip early-stop bookkeeping

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

    if epoch_limit is not None:
        return None  # timing probe only

    if best_state is None:
        # never improved (shouldn't happen with IMPROVE_EPS this small, but guard anyway)
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = max_epochs

    model.load_state_dict(best_state)
    model.to(DEVICE)
    model.eval()

    final = {
        "run_id": run_id, "config": config, "n_params": n_params,
        "best_epoch": best_epoch, "best_val_loss": best_val_loss,
        "baseline_logloss": BASELINE_LOGLOSS, "baseline_top1": BASELINE_TOP1, "baseline_top20": BASELINE_TOP20,
    }

    for split_name, tset in [("val", val_t), ("test", test_t), ("live", live_t)]:
        avg_loss, pos_logloss, pos_acc = evaluate(model, tset, batch_size=config["batch"])
        top20_hit, records = combo_eval(model, tset, batch_size=config["batch"])
        final[split_name] = {
            "n": tset["context"].shape[0],
            "logloss": avg_loss, "pos_logloss": pos_logloss, "pos_acc": pos_acc,
            "top20_hit_rate": top20_hit,
        }
        if split_name in ("test", "live"):
            with open(os.path.join(run_dir, f"records_{split_name}.json"), "w") as rf:
                json.dump(records, rf)
            lag_result = run_lag_analysis(records)
            with open(os.path.join(run_dir, f"lag_{split_name}.json"), "w") as lf:
                json.dump(lag_result, lf, indent=2)
            final[f"{split_name}_lag0_top20"] = summarize_lag0_top20(lag_result)

    with open(os.path.join(run_dir, "final.json"), "w") as ff:
        json.dump(final, ff, indent=2)
    print(f"[{run_id}] done: best_epoch={best_epoch} "
          f"test_logloss={final['test']['pos_logloss']} test_top20={final['test']['top20_hit_rate']:.4f} "
          f"live_top20={final['live']['top20_hit_rate']:.4f}")
    return final


if __name__ == "__main__":
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)
    run_id = config.get("run_id") or os.path.splitext(os.path.basename(config_path))[0]
    epoch_limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    train_run(config, run_id, epoch_limit=epoch_limit)
