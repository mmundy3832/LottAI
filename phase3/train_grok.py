"""
train_grok.py - Phase 3 Stage M3 grokking run (plans/PHASE3_PLAN.md section 3,
M3; task instructions step 2).

Step-based training loop, reusing train.py's data/model/loss/eval plumbing.
Fixed train split, no early stopping, no epoch concept -- a fixed number of
gradient steps on the W=16 train split, weight decay swept via config.

Config JSON (phase3/configs/m3_*.json) fields:
  run_id         string, optional (defaults to config filename stem)
  model          "m1" (M3 uses the M1 transformer class with n_layers
                 overridden to 2, per plans/PHASE3_PLAN.md section 3)
  n_layers       transformer layers (2, per plan M3 spec)
  W              context window (16, per plan)
  seed           int (0)
  batch          batch size (512, per plan)
  lr             AdamW learning rate (1e-3, chosen)
  weight_decay   AdamW weight decay (swept: 0.1 / 0.3 / 1.0)
  max_steps      total gradient steps (1_000_000, per plan)

Logs to runs/<run_id>/metrics.jsonl every LOG_EVERY steps: step, train_loss
and train_pos_acc (on a fixed 2048-window train subset, cheap and constant
across the run so the curve is comparable step to step), val_loss,
val_pos_acc (full val split), elapsed_s. Every FULL_TRAIN_EVERY steps also
logs train_full_loss/train_full_pos_acc on the entire train split (the
pre-grokking memorization check). Checkpoints every CKPT_EVERY steps to
checkpoint_latest.pt (overwritten), plus a permanent checkpoint_step<N>.pt
every CKPT_KEEP_EVERY steps.

Usage: python3 train_grok.py configs/m3_wd1.0.json [max_steps_override]

Resume: python3 train_grok.py configs/m3_wd1.0.json --resume runs/m3_wd1.0/checkpoint_latest.pt --max-steps 10000000
  Loads model (+ optimizer, if the checkpoint has optimizer_state) and
  continues step numbering from checkpoint["step"]. Appends to the existing
  metrics.jsonl (never truncates) and keeps the same CKPT_EVERY /
  CKPT_KEEP_EVERY cadence, aligned to absolute step number. If the
  checkpoint has no optimizer_state, a fresh AdamW is built at config lr
  and a warning is printed to stdout and returned in the run summary.
"""

import os
import sys
import json
import time

import torch

from data_loader import load_history
from tokenizer import load_vocab
from model import build_model, count_params
from train import (
    DEVICE, RUNS_DIR, set_seed, build_split_windows, to_tensors,
    compute_logits_loss, evaluate,
)

# ---------------------------------------------------------------------------
# Constants (chosen, task instructions step 2)
# ---------------------------------------------------------------------------
LOG_EVERY = 1000
FULL_TRAIN_EVERY = 10000
CKPT_EVERY = 50000
CKPT_KEEP_EVERY = 250000
TRAIN_SUBSET_SIZE = 2048


def make_fixed_train_subset(train_t, seed):
    """A fixed (seeded, never reshuffled across the run) 2048-window sample
    of the train split, used for the cheap per-1k-step train-loss/acc log
    so successive points are directly comparable."""
    n = train_t["context"].shape[0]
    g = torch.Generator().manual_seed(seed)
    size = min(TRAIN_SUBSET_SIZE, n)
    idx = torch.randperm(n, generator=g)[:size]
    return {k: v[idx] for k, v in train_t.items()}


def step_batches(train_t, batch_size, seed):
    """Infinite generator of shuffled full-size minibatches (batch_size
    fixed every step; a trailing partial batch is dropped and picked up at
    the front of the next shuffled pass)."""
    gen = torch.Generator().manual_seed(seed)
    n = train_t["context"].shape[0]
    while True:
        idx = torch.randperm(n, generator=gen)
        for i in range(0, n, batch_size):
            b = idx[i:i + batch_size]
            if b.shape[0] < batch_size:
                continue
            yield {k: v[b].to(DEVICE) for k, v in train_t.items()}


def save_checkpoint(model, optimizer, config, step, path):
    tmp_path = path + ".tmp"
    torch.save({
        "step": step,
        "config": config,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
    }, tmp_path)
    os.replace(tmp_path, path)  # atomic swap: no truncated checkpoint on crash mid-write


def train_grok_run(config, run_id, max_steps_override=None, resume_path=None):
    set_seed(config["seed"])

    vocab = load_vocab()
    history_df = load_history()
    train_w, val_w, _test_w = build_split_windows(history_df, config["W"], vocab)
    train_t = to_tensors(train_w)
    val_t = to_tensors(val_w)
    fixed_train_t = make_fixed_train_subset(train_t, config["seed"])

    model = build_model(
        config["model"], vocab, config["W"], n_layers=config.get("n_layers"),
    ).to(DEVICE)
    n_params = count_params(model)

    start_step = 0
    optimizer_resumed = False
    if resume_path is not None:
        ckpt = torch.load(resume_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state"])
        start_step = ckpt["step"]
        if ckpt.get("optimizer_state") is not None:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],
            )
            optimizer.load_state_dict(ckpt["optimizer_state"])
            optimizer_resumed = True
        else:
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],
            )
            print(f"[{run_id}] RESUME WARNING: checkpoint {resume_path} has no "
                  f"optimizer_state -- resuming model weights only with a FRESH "
                  f"AdamW at config lr={config['lr']} (optimizer moments not "
                  f"preserved).", flush=True)
        print(f"[{run_id}] resumed from {resume_path} at step {start_step} "
              f"(optimizer_resumed={optimizer_resumed})", flush=True)
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"],
        )

    run_dir = os.path.join(RUNS_DIR, run_id)
    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    max_steps = max_steps_override if max_steps_override is not None else config["max_steps"]

    print(f"[{run_id}] model={config['model']} n_layers={config.get('n_layers')} "
          f"wd={config['weight_decay']} params={n_params:,} "
          f"train={train_t['context'].shape[0]} val={val_t['context'].shape[0]} "
          f"start_step={start_step} max_steps={max_steps}", flush=True)

    batches = step_batches(train_t, config["batch"], config["seed"])
    t_start = time.time()

    metrics_mode = "a" if resume_path is not None else "w"
    with open(metrics_path, metrics_mode) as mf:
        model.train()
        for step in range(start_step + 1, max_steps + 1):
            batch = next(batches)
            optimizer.zero_grad()
            _, loss, _ = compute_logits_loss(model, batch)
            loss.backward()
            optimizer.step()

            if step % LOG_EVERY == 0:
                model.eval()
                train_loss, _, train_pos_acc = evaluate(model, fixed_train_t, batch_size=config["batch"])
                val_loss, _, val_pos_acc = evaluate(model, val_t, batch_size=config["batch"])
                record = {
                    "step": step,
                    "elapsed_s": time.time() - t_start,
                    "train_loss": train_loss,
                    "train_pos_acc": train_pos_acc,
                    "val_loss": val_loss,
                    "val_pos_acc": val_pos_acc,
                }
                if step % FULL_TRAIN_EVERY == 0:
                    full_loss, _, full_pos_acc = evaluate(model, train_t, batch_size=config["batch"])
                    record["train_full_loss"] = full_loss
                    record["train_full_pos_acc"] = full_pos_acc
                mf.write(json.dumps(record) + "\n")
                mf.flush()
                print(f"[{run_id}] step {step} train_loss={train_loss:.4f} "
                      f"val_loss={val_loss:.4f} elapsed={record['elapsed_s']:.1f}s", flush=True)
                model.train()

            if step % CKPT_EVERY == 0:
                save_checkpoint(model, optimizer, config, step,
                                 os.path.join(run_dir, "checkpoint_latest.pt"))
                if step % CKPT_KEEP_EVERY == 0:
                    save_checkpoint(model, optimizer, config, step,
                                     os.path.join(run_dir, f"checkpoint_step{step}.pt"))

    print(f"[{run_id}] done at step {max_steps}", flush=True)


if __name__ == "__main__":
    config_path = sys.argv[1]
    with open(config_path) as f:
        config = json.load(f)
    run_id = config.get("run_id") or os.path.splitext(os.path.basename(config_path))[0]

    # Flag parsing: --resume <path> and --max-steps <int> in any order, plus
    # the legacy bare positional max_steps override (python3 train_grok.py
    # config.json 500) preserved for existing callers (run_m3_sequential.sh).
    resume_path = None
    max_steps_override = None
    rest = sys.argv[2:]
    i = 0
    while i < len(rest):
        arg = rest[i]
        if arg == "--resume":
            resume_path = rest[i + 1]
            i += 2
        elif arg == "--max-steps":
            max_steps_override = int(rest[i + 1])
            i += 2
        else:
            max_steps_override = int(arg)  # legacy positional form
            i += 1

    train_grok_run(config, run_id, max_steps_override=max_steps_override,
                    resume_path=resume_path)
