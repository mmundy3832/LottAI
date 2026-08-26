# LottAI Phase 2 — Council of Experts + Autonomous Prediction Loop

**Date:** 2026-04-25
**Status:** Approved for implementation

---

## Overview

Phase 1 built the autoresearch loop: a GA/LLM/random search over a 55-75D config space that
produces trained models scored by `optimal_ev` on a held-out val set. Phase 2 turns the best
products of that search into a live prediction service.

The system has two parallel tracks running on the OpenBrain Docker container:

**Track A — Prediction (runs first every cron fire)**
1. Download latest draws → append to live dataset
2. Council of 10 diverse experts retrain on all data through most recent draw
3. Council votes on the next draw → ranked probability matrix
4. Optimal-K ticket recommendation computed
5. Telegram notification sent with combos + analysis writeup
6. Prediction logged to ledger (outcome filled in after the draw)

**Track B — Discovery (launches in background while Track A runs)**
7. 10 GA/random experiments run
8. Top new experiments evaluated against live data
9. If any beat the current council floor, promote them in

The two tracks are independent. A new expert promoted in Track B will not appear in the
prediction run that triggered it — it enters the council for the *next* fire.

---

## Council of Experts

### What it is
A persisted set of 10 experiment configs stored in `autoresearch/council.json`.
Each member is a full config dict (feature_sets, pipeline, models) plus metadata:
exp_num, live_optimal_ev, val_optimal_ev, date_admitted.

### How it's selected (initial build)
1. Load `experiments_v4.jsonl` + `live_eval_results_v2.jsonl`
2. Restrict to experiments that have been live-evaluated
3. Score by `live_optimal_ev` (not val — val is compromised by 1600+ experiments of GA pressure)
4. Greedy diversity selection:
   - Compute `model_fingerprint` = sorted tuple of model types (e.g., `("et","lgb","xgb")`)
   - No two council members may share the same `model_fingerprint`
   - Secondary diversity: prefer different pipeline types and feature set combinations
   - Select top-N by live_ev subject to diversity constraints
5. Write `council.json`

### Why diversity matters
All the top GA experiments converge on `et+xgb+lgb / custom_interact / 5 features`.
A council of 10 clones would produce a single strong prior with no variance signal —
there would be no way to detect when they agree vs. disagree, and no way to identify
which digit positions have genuine signal vs. noise. A diverse council gives us:
- Disagreement as a signal of uncertainty
- Coverage of different parts of the 55-75D space
- A more honest ensemble that doesn't amplify a single basin's biases

### Promotion logic
After each GA discovery batch:
1. Find new experiments (exp_num > council's `last_checked_exp`)
2. Filter to val_ev > council's weakest member's live_ev (rough pre-filter)
3. Run `live_eval_v2.py` on those candidates
4. If any candidate's live_ev > council's weakest member's live_ev:
   - Replace weakest member
   - Log to `autoresearch/council_history.jsonl`

MiniMax is never told about the council, about live data, or about promotion logic.
It sees exactly what it saw in Phase 1: a config to try, a val score back.

---

## Prediction Engine

Each prediction run:
1. Load council.json → 10 experiment configs
2. For each config, render full experiment code (via template_engine.py)
3. Run the experiment, but instead of producing a val score, capture:
   - Three trained classifier objects (d1, d2, d3)
   - Feature-transformed test point for the next draw
4. Each classifier produces `predict_proba()` → 10-class probability vector
5. Weighted ensemble across council members:
   - Weight = `softmax(live_optimal_ev)` — higher live EV gets more vote
6. Final P(d1=x), P(d2=y), P(d3=z) vectors
7. Build 1000-combo probability matrix: `P(xyz) = P(d1=x) * P(d2=y) * P(d3=z)`
8. Sort descending; find optimal K using EV formula: `max_k(top_k_hit_rate * $500 - k)`
9. Output: top-K combos, per-combo probability, overall expected EV

### What the "next draw" context is
The prediction uses the most recent draw as the "previous draw" feature context.
The feature builder (prepare_v3.py) builds features for row N+1 using rows 0..N.
We set N = last row in the live dataset.

### Data boundary enforcement
- Live data flows INTO the prediction engine (as features)
- Live data does NOT flow out to any external LLM
- MiniMax never sees live draws, live metrics, or prediction outputs

---

## Operational Flow (per cron fire)

```
[cron fires]
    |
    v
1. live_update.py        Download + append new draws
    |
    v
2. predictor.py          Retrain council, vote on next draw
    |
    v
3. analysis.py           Generate writeup (Claude/local LLM)
    |
    v
4. telegram_bot.py       Send combos + writeup to Telegram
    |
    v
5. prediction_ledger.py  Log prediction record
    |
    +--- background ----> ga_discovery.py   Run 10 new experiments
                              |
                              v
                          promotion_checker.py  Live-eval candidates, update council
```

Track A (steps 1-5) runs synchronously and completes before the next draw.
Track B (steps 6-7) runs in a background subprocess and may complete after Track A.

---

## Cron Schedule

Texas Lottery draw times (Central Time), Mon-Sat:
- `27 10 * * 1-6`  — Morning (10:27am CT, fires after 10am draw results post)
- `57 12 * * 1-6`  — Day (12:57pm CT, fires after 12:27pm draw)
- `30 18 * * 1-6`  — Evening (6:30pm CT, fires after 6pm draw)
- `42 22 * * 1-6`  — Night (10:42pm CT, fires after 10:12pm draw)

No Sundays. Cron runs inside the OpenBrain container (UTC offset handled in compose env).

---

## Telegram Message Format

```
LottAI Council — [Draw Time] [Date]

Next draw prediction:

  Buy [K] combinations ($[K], EV $[ev]/draw):
  [d1][d2][d3]   p=[prob]
  [d1][d2][d3]   p=[prob]
  ...

Council composition: [N members, top live EV $X.XX]

[2-3 paragraph analysis from Claude]
```

---

## Prediction Ledger Schema

One JSONL record per prediction, written before the draw:

```json
{
  "prediction_id": "2026-04-25T22:42:00_night",
  "timestamp": "2026-04-25T22:42:00Z",
  "draw_slot": "night",
  "predicted_draw_date": "2026-04-26",
  "council_version": 3,
  "council_exp_nums": [1471, 880, 796, ...],
  "top_k": 16,
  "expected_ev": 11.78,
  "predictions": [
    {"combo": "123", "probability": 0.00412},
    ...
  ],
  "actual_combo": null,
  "hit": null,
  "hit_at_k": null,
  "updated_at": null
}
```

After the draw result is available, `actual_combo`, `hit`, and `hit_at_k` are filled in
by the next cron run (before Track A begins for that run).

---

## File Inventory

New files:
```
autoresearch/
  council.py              Council selection, persistence, promotion
  council.json            Current council state (gitignored? or tracked?)
  council_history.jsonl   Promotion log (append-only)
  predictor.py            Retrains council models, produces probability matrix
  prediction_ledger.py    Write/update prediction records
  prediction_ledger.jsonl All predictions + outcomes (append-only ledger)
  live_update.py          Downloads + appends live draws
  telegram_bot.py         Formats and sends Telegram notification
  analysis.py             Generates LLM writeup
  ga_discovery.py         Wrapper to launch 10-experiment GA batch
  promotion_checker.py    Identifies + evaluates promotion candidates
  orchestrator.py         Main cron entry point — runs both tracks

docker/
  Dockerfile
  docker-compose.yaml
  crontab

.env.example              TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, OLLAMA_HOST, etc.
```

`council.json` and `prediction_ledger.jsonl` should be git-tracked (unlike live CSVs).
They are derived data, not raw draws — no data boundary concern.

---

## Environment Variables

```
TELEGRAM_BOT_TOKEN=        Bot token from BotFather
TELEGRAM_CHAT_ID=          Your personal or group chat ID
OLLAMA_HOST=http://localhost:11434   Local Ollama for analysis writeup
LOTTAI_ROOT=/mnt/beastmode/lottai   Mount path in container
TZ=America/Chicago                  For correct draw-time scheduling
```

---

## What Phase 2 Does NOT Do

- Does not modify the GA/LLM search loop (runner_v4.py is untouched)
- Does not expose live data to MiniMax during GA experiments
- Does not buy tickets automatically (prediction output only)
- Does not retrain on live data (models train on historical only, predict into live)
- Does not use GPU models for prediction until they demonstrate live performance

---

## Whitepaper Data Collected

The prediction ledger will produce, over time:
- Hit rate at K=1, 5, 10, 16, 20 across council predictions
- Comparison vs random baseline (-$0.50/draw)
- Council consensus score (variance across member predictions) as a confidence proxy
- Council evolution over time (which experiments get promoted/demoted)

This is the live-forward validation that the whitepaper needs: not just retrospective
val/live EV scores, but prospective predictions made before draws and checked against reality.
