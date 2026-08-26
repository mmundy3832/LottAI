# Phase 3 Plan: Software 2.0 Predictor with Lag-Credited Online Learning

Status: DRAFT 2026-08-26, not approved
Depends on: Phase 2 backtest (autoresearch/backtest_lag.py) for the lag window and decay rate. Blocked on machine cores until Mark frees them.

## 1. Question

Phases 1 and 2 used hand-written features (244 of them) feeding tree ensembles, with a GA searching config space. Phase 3 removes the feature program entirely. The raw draw history is the input, a neural network is the program, and the training signal is a per-ball reward that credits a prediction if the ball lands within a short window of future draws.

Two hypotheses, both expected to return null:

H1. A sequence model trained end to end on raw Texas Pick 3 draws finds structure that the Phase 2 pipeline missed. Test: per-position log-loss below the uniform baseline ln(10) = 2.3026 on held-out draws, at a sample size where the gap is not noise.

H2. Phase 2 predictions were correct but early by one or two draws (Mark's observation from the live ledger). Test: a model trained with lag-credited targets beats the same model trained on lag-0 targets, on held-out draws, after subtracting the matched null for each lag.

A third, exploratory run tests whether long training with strong weight decay produces a delayed generalization transition (grokking, Power et al. 2022; Nanda et al. 2023). Expected result: none, because the target has no algebraic structure. The run is cheap and the negative result belongs in the paper.

## 2. Data

- History: pick3_combined.csv, 15,572 draws, 2013-09-09 to 2026-02-13, 4 slots/day (Morning, Day, Evening, Night), 3 balls each, digits 0-9.
- Live: pick3all_live.csv, 662 draws, 2026-02-14 to 2026-08-26. Eval only, never in offline training (see project_data_boundary memory).
- Splits (same as Phase 2 so results compare): train through 2022-05-23 (10,900), val to 2024-04-03 (2,336), test to 2026-02-13 (2,336), live 662.

Token encoding, one draw = 4 tokens: [slot] [d0] [d1] [d2]. Slot vocab 4, digit vocab 10, total vocab 14 plus a pad token. No date, no machine, no ball-set tokens in the base run (Phase 1 found none of these carry signal, and they are a leakage path). Ablation run adds day-of-week and machine ID tokens to check that claim under a 2.0 model.

Context window W: number of prior draws visible. Chosen values to sweep: 16, 64, 256 draws (64 to 1024 tokens). Default 64. These are my choices, not measured. The sweep is the measurement.

## 3. Model

Two architectures, both small enough to train on the GTX 1070 (8 GB) or CPU.

M1. Decoder-only transformer. 4 layers, d_model 128, 4 heads, feed-forward 512, learned positional embedding. About 1M parameters. Output: three 10-way softmax heads, one per ball position, from the final token's hidden state.

M2. GRU, 2 layers, hidden 256. Same heads. Baseline to check that any M1 result is not an artifact of attention.

M3 (grokking run). Transformer, 2 layers, d_model 128, AdamW, weight decay swept over {0.1, 0.3, 1.0}, batch 512, 1M steps on the fixed train split with W = 16. Log train and val per-position accuracy every 1k steps. Look for val accuracy rising long after train accuracy saturates.

Sizes are chosen, not measured. Parameter count is deliberately under the sample count (10,900 draws x 3 targets) so memorization is possible but not trivial, which is the regime where grokking has been observed.

## 4. Training objectives

Stage A, lag 0 (control). Standard cross-entropy on the next draw's three digits. This is the H1 test.

Stage B, lag-credited. For prediction at draw t, position p, and lag k in 0..K:

    loss_t = - sum_p sum_k w_k * log P_p(digit_{t+k, p})
    w_k = exp(-lambda * k)

K = 3 and lambda = 1.0 are placeholders. Both are set from the Phase 2 backtest lag table once it runs: K is the last lag whose 2-of-3 rate beats null, lambda is fit to the decay of that excess. If no lag beats null, K = 2 and lambda = 1.0 stay as the values for the H2 test, and the expected answer is null.

This is supervised learning with lagged targets. It is the same reward as the RL framing (hit = 1 scaled by exp(-lambda k), miss = 0, per ball) but with a dense gradient instead of a sampled one, so it converges with far fewer draws. A true policy-gradient variant (REINFORCE with the same reward) is Stage B2, run only if Stage B shows anything, to check the result survives the noisier estimator.

Lambda is fixed. A learnable lambda drifts toward whatever inflates the multi-lag reward and was parked in discussion on 2026-08-26.

Stage C, online. After each live draw lands: score pending predictions for lags 0..K, take one AdamW step on a replay batch of the newest 64 draws plus 64 sampled from history, predict the next draw, append to the ledger. Learning rate 1e-4 (chosen). Compare against the frozen Stage B model on the same live draws to measure whether online updates help, hurt, or do nothing.

## 5. Matched null and gaming pre-mortem

Every metric is reported as observed minus null, where null is the same metric on draw sequences with actual outcomes permuted (2000 permutations, seed 42), the same procedure as autoresearch/lag_analysis.py.

Known gaming vectors:

G1. Multi-lag inflation. Rewarding a hit at any of K+1 lags multiplies the chance rate by up to K+1. Per-lag null subtraction handles this. Never report a pooled hit rate.
G2. Near-miss inflation. A 20-combo set covers up to 540 of 1000 combos within one digit, a 30-combo set up to 810. The 2-of-3 metric has a null near 25-32 percent and is reported only against that null.
G3. Lookahead. Token stream must be built so position t sees only tokens from draws before t. Unit test: shuffle draw t+1 onward and assert prediction at t is unchanged.
G4. Slot leakage. Slot token is legitimate (the slot of the target draw is known before it happens) but must be the target's slot, not a feature derived from the target's digits.
G5. Run selection. Sweeps over W, architecture, weight decay, lambda, and seeds produce dozens of runs. Preregistered primary metrics are three: test per-position log-loss (H1), test lag-k excess over null at the backtest-chosen k (H2), and M3 val accuracy curve shape. Everything else is exploratory and labeled so. Bonferroni over the sweep size for any claim.
G6. Live contamination. Live draws enter training only in Stage C and only after they have been scored as predictions. Offline stages never see them.

## 6. Metrics

Primary: per-position log-loss vs 2.3026, per-position top-1 accuracy vs 0.10, top-20 combo hit rate vs 0.020, each on test and on live, each with null.
Secondary: optimal_ev on the same combo sets, for continuity with Phase 2 numbers.
Lag: full lag table 0..5 from the lag_analysis.py procedure on the model's top-20 and top-30 sets, on test and live.

## 7. Compute

1M-parameter transformer on 10,900 draws at W = 64: seconds per epoch on the GPU, minutes on CPU. Stage A and B sweeps (3 W values x 2 archs x 3 seeds = 18 runs) fit in an evening. M3 at 1M steps is the long one, roughly 8-12 hours on the 1070 at batch 512, which is the "leave it running over the weekend" run. All numbers are estimates and get measured on the first run.

## 8. Layout

phase3/ is self-contained. It imports nothing from autoresearch/. Anything reused is copied in and trimmed to what Phase 3 needs, so the experiment can be read and rerun without the Phase 2 tree.

    phase3/
      data_loader.py    copied and trimmed from autoresearch/live_update.py (download plus CSV parse) and prepare_v3.py (draw ordering only, no features)
      ledger.py         copied from autoresearch/prediction_ledger.py, schema extended with model_id, lag_hits, scored_through_lag
      lag_null.py       copied from autoresearch/lag_analysis.py, the permutation-null procedure only
      tokenize.py       raw CSV to token stream, lookahead unit test
      model.py          M1, M2, M3 definitions
      train.py          Stage A and B, config from JSON, writes runs/<id>/metrics.jsonl
      online.py         Stage C loop
      program_v4.md     Karpathy-loop spec: metric, editable surface, null procedure
      configs/          one JSON per run in the sweep
      runs/             outputs, gitignored except metrics.jsonl

Reads data/ for the CSVs. Writes only under phase3/. autoresearch/ is frozen as the Phase 2 record.

## 9. Order of work

1. tokenize.py plus lookahead test.
2. Stage A on M1 and M2, W = 64, 3 seeds. Report H1 numbers. Stop and review.
3. Phase 2 backtest runs when cores are free. Set K and lambda. Stop and review.
4. Stage B sweep. Report H2. Stop and review.
5. M3 grokking run in the background from step 2 onward.
6. Stage C online loop against the live window, only if Stage A or B shows anything above null, or as a final null check if Mark wants the complete story.

## 10. Decisions for Mark

D2. Token encoding: slot plus 3 digits only, or include day-of-week and machine ID from the start.
D3. Which target draws count for online Stage C: all four slots, or one slot as in Phase 2 live eval.
D4. Run Stage C even on a null Stage A/B result (completes the whitepaper) or gate it.
D5. Directory reorganization before Phase 3 starts (see reply of 2026-08-26).
