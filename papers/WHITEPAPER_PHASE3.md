# Phase 3: Sequence Models, a Preregistered Early-Hit Test, and Zero-Shot Forecasting on Texas Pick 3

Status: DRAFT, ready for review.

Author: Mark Mundy
AI collaborators: Claude (Anthropic). The specific model is not recorded in the Phase 3
commits or files.
Date: September 2026.
Repository: https://github.com/mmundy3832/LottAI

---

## The short version

Phase 2 searched a large space of hand-engineered features and tree-ensemble
configurations and found a validation score that climbed to $9.183 optimal EV
per draw, but that score did not survive contact with live, truly unseen
draws. Phase 3 asks a different question: does a neural network trained
directly on the raw sequence of Texas Pick 3 draws, with no hand-built
features at all, find anything the feature-engineering approach missed.

We tokenized every draw (day, time slot, machine, ball sets, three digits)
into a stream and trained small transformer and GRU models to predict the
next draw's digits. We swept context window size, ran a long weight-decay
sweep looking for the "grokking" phenomenon reported in other domains, built
an online-learning variant that updates on each new live draw, ran a
pretrained time-series foundation model zero-shot, and preregistered a
specific hypothesis (that Phase 2's production predictions were systematically
early by one or two draws) before testing it on 278 held-out draws.

Every one of these came back null. The sequence models never beat the
uniform-guessing baseline. Ten million steps of training under strong weight
decay showed no grokking. The preregistered early-hit hypothesis was rejected
(p=0.624). Online updating on live draws made the model measurably worse
(p=0.0005). The foundation model, run with no lottery-specific training at
all, was worse than a coin flip's worth of structure and worse than a naive
seasonal guess in log-loss terms. Phase 3 closes the sequence-model line of
inquiry on this data.

## Background you need

### Tokens and tokenizing a table of draws

A token is a small integer standing in for a category, the same trick a zip
code uses to stand in for a location. To feed a table of lottery draws into
a neural network, we turn every column of every row into one token. One Pick
3 draw has nine columns: which of the four daily slots it was (morning, day,
evening, night), the day of week, which machine drew it, which of three ball
sets were used, and the three winning digits, so one draw becomes nine
tokens in a row:

    [slot] [dow] [machine] [bs1] [bs2] [bs3] [d0] [d1] [d2]

String 15,572 draws together and you get a stream of 140,148 tokens, the
same shape of object a language model reads, except every "word" here is a
lottery fact instead of a piece of English. Turning every CSV column into a
token, rather than picking a subset, is decision D2 in the Phase 3 plan.
Source: `phase3/tokenizer.py`.

### Decoder-only transformers and next-token prediction

A decoder-only transformer is a network trained to guess the next item in a
sequence given everything before it. Phone autocomplete does a crude version
of this: type "I'm running a little" and it guesses "late" because that word
followed similar phrases many times in its training text. A transformer does
the same thing at larger scale, using attention, a mechanism that lets every
position weigh every earlier position when making its guess, instead of only
looking a fixed number of words back.

Here, "next token" is one of a draw's digits, not an English word. The model
reads a window of prior draws and outputs, for each of the three digit
positions, a probability over the ten possible digits 0-9. Reference:
Vaswani et al. (2017), "Attention Is All You Need" [1].

### Context window

The context window (W) is how many prior draws the model is allowed to look
at before making its guess, the same idea as how far back you scroll a text
thread before replying. We tried W=16 (144 tokens), W=64 (576 tokens, the
default), and W=256 (2,304 tokens, about two months of draws). A wider
window costs more compute and only helps if information from further back
actually matters to the next draw.

### Log-loss and the uniform baseline

Log-loss (cross-entropy loss) penalizes a probabilistic guess by how far off
it was, and penalizes a confident wrong guess much harder than a hedged
wrong guess. If a model has genuinely no information, the best it can do is
assign each of the 10 digits an equal 10% probability; the log-loss of that
flat guess is ln(10) = 2.302585, a fixed number independent of the data. A
model whose log-loss on held-out draws is measurably below 2.302585 found
something real. A model above 2.302585 is worse than flat guessing, usually
because it was confidently wrong some of the time. Every log-loss number
here is compared against that same 2.302585 baseline.

### Top-k accuracy

Top-k accuracy asks whether the true answer was among the model's k most
likely guesses. Top-1 accuracy for a single digit position is "did the
model's single best guess match", which chance gets right 10% of the time (1
in 10 digits). Top-20 combo accuracy asks whether the actual three-digit
combo (1 of 1,000 possible) was among the model's 20 highest-ranked combos,
which chance gets right about 2% of the time (20/1,000).

### Overfitting vs. generalization

A model overfits when it memorizes the specific examples it trained on
instead of learning a pattern that transfers to new examples, the textbook
picture being a student who memorizes last year's exact exam answers instead
of the material, and then fails this year's exam, which asks the same
questions in a different order. The standard signature is training accuracy
near 1.0 (it nailed every training example) alongside validation accuracy
far lower, on examples it never saw. That is exactly the shape our M3 runs
show: train accuracy 1.0, val accuracy about 0.10 (chance).

### Grokking

Grokking is a reported phenomenon (Power et al. 2022 [2]) where, on certain
small algorithmic tasks, a network first overfits (train accuracy 1.0, val
accuracy at chance) and then, if you keep training far past that point,
validation accuracy suddenly jumps up long after training accuracy already
saturated, as if the model "figured out" the underlying rule instead of just
memorizing. Follow-up work (Nanda et al. 2023 [3]) traced the mechanism: the
network quietly builds a general algorithm inside its weights the whole time,
even while validation accuracy looks flat, before that algorithm becomes
dominant enough to show up in the metric. We looked for a late rise in
validation accuracy after long training. Lottery draws have no algebraic rule
to discover, so the plan predicted none would appear, and ran the check
anyway because a negative result here is cheap and informative.

### Weight decay

Weight decay is a training-time penalty that pushes a model's internal
numbers (weights) toward zero unless the data gives them a reason to grow,
similar to a "use it or lose it" budget rule that forces you to justify every
expense. Grokking experiments use strong weight decay because it is thought
to be part of what pushes the network away from a memorizing solution and
toward a more compressed, general one. We used AdamW (Loshchilov & Hutter
2019 [4]), the standard weight-decay variant of the Adam optimizer, at three
decay strengths: 0.1, 0.3, and 1.0.

### Online learning vs. a frozen model

A frozen model's weights do not change after training ends; it makes every
prediction the same way regardless of what happens afterward. An online
model keeps learning, updating its weights after each new real-world result,
the way a weather forecaster might quietly adjust tomorrow's forecast after
seeing how wrong yesterday's was. This can help when the world is genuinely
changing in a learnable way. It can also hurt: a small number of new
examples is a noisy training signal, and repeatedly training on noise can
degrade a model that was already calibrated. Stage C tests this on live
draws.

### Paired test

A paired test compares two things measured on the exact same cases (the
frozen model and the online replica, scored on the same 662 live draws in
the same order) rather than on two different samples, removing noise that
would come from comparing draw-set A to a different draw-set B. We used a
sign-flip permutation test: take the observed per-draw differences, randomly
flip the sign of each many times, and see how often a "null world" with no
real difference produces a gap this large by chance alone.

### Preregistration

Preregistration means writing down the exact hypothesis, statistical test,
data, and pass/fail threshold, and committing that document (here, to git)
before running the test or looking at the outcome. The point is to remove
the researcher's ability to try several ways of slicing the data and report
only the one that looked good after the fact, a failure mode called
p-hacking. Reference: Nosek et al. (2018), "The Preregistration Revolution"
[9]. Our preregistration document was committed to git and the analysis ran
eleven minutes later, using the exact statistic and threshold written down
in that commit.

### Foundation models and zero-shot forecasting

A foundation model is a large model pretrained once on a huge, general
corpus of data, then reused on new problems without further training on
those problems, called zero-shot use ("zero" examples of the new task in its
training). TimesFM is Google's foundation model for numeric time series (Das,
Kong, Sen, Zhou, 2024 [5]), pretrained on a large corpus of real-world series
(weather, traffic, finance-like data) so it can forecast an unfamiliar series
by pattern-matching against structure it already learned. We ran the current
TimesFM-3 checkpoint on the Pick 3 digit sequence with no additional
training, to test whether real-world structure transfers to a sequence that
is, as far as Phases 1 and 2 could tell, cryptographically random.

### Lookahead leakage and why the tokenizer has a test for it

Lookahead leakage is when a model's input accidentally contains information
from the future relative to what it is predicting, making a prediction look
accurate for a reason that will not hold once you predict the future for
real. Phase 2's causal backtest found exactly this bug in its momentum
feature (it read a value computed from the target draw's own digits).
Because of that history, the Phase 3 tokenizer ships with an automated test
that shuffles every draw after the predicted position and checks that the
prediction does not change; if it changes, future information leaked in.

### The lag null

The "lag null" idea: to tell whether a match between a prediction and a
later draw means anything, you have to know how often that kind of match
would happen by pure chance. We build that baseline by shuffling the actual
draw sequence thousands of times and recomputing the same match statistic on
each shuffle. A real, unshuffled result far out in the tail of that shuffled
distribution is evidence of structure; one in the middle of the pack is
consistent with chance. Every result here is reported against a lag null
built this way (2,000 permutations, seed 42), never against a raw hit rate
alone.

## What we did

### Why Phase 3

Phase 2 searched 1,670 model configurations over 244 hand-engineered features
(basic, recency, gap, positional, temporal, momentum, and an equipment set
off by default) and tree-ensemble models, reaching a validation optimal EV
ceiling of $9.183 per draw. That search did not establish a live edge: the
strongest live result the Phase 2 paper reports is a 121-prediction ledger
with 2 of 72 lag-hit cells nominally significant and 0 surviving correction.
A causal backtest run after that paper (fixing a leaked feature) found 0 of
72 live cells significant, the strongest null in the project and not itself
reported in the Phase 2 paper.

Phase 3 removes the hand-built feature program entirely and asks whether a
model trained end to end on the raw token stream finds anything the
feature-engineering pipeline missed, and separately runs a single
preregistered test of a specific hypothesis Mark raised from the live
prediction stream: that Phase 2's predictions were correct, just early by a
draw or two.

Stage summary, detail follows in the sections below:

| Stage | Question | Method | Result |
|---|---|---|---|
| A, window sweep | Beat chance on held-out draws? | M1/M2, W in {16,64,256}, 21 runs | Log-loss 2.3085-2.3170, all above 2.302585 baseline |
| M3, grokking | Late generalization jump under weight decay? | 2-layer transformer, wd in {0.1,0.3,1.0}, 10M steps | Train acc 1.0, val acc ~0.10 throughout; no grokking |
| Prereg early-hit test | Were predictions early by 1-2 draws? | One statistic, 278 held-out draws | Observed 0.2423 vs null 0.2472, p=0.624; rejected |
| B, lag-credited training | Same idea, trained into the loss | M1/M2, K=2 lambda=1.0, 6 runs | Lag p-values at chance except one nominal cell |
| C, online replica | Does live updating help? | Frozen vs. replica, paired test, 662 draws | Replica worse: 2.65 vs. 2.31 frozen, p=0.0005 |
| D, TimesFM-3 zero-shot | Does a foundation model transfer? | Zero-shot, per-position series | Log-loss 4.7 vs. uniform 2.3; top-1/top-20 at chance |

### Data and tokenizer

History: `data/pick3_combined.csv`, 15,572 draws, 2013-09-09 to 2026-02-13,
four slots a day (Morning, Day, Evening, Night), Monday through Saturday.
Live: `data/pick3all_live.csv`, 662 draws, 2026-02-14 to 2026-08-26, eval only
and never used in offline training. Splits match Phase 2 so results compare:
train 10,900 draws through 2022-05-23, validation 2,336 through 2024-04-03,
test 2,336 through 2026-02-13, live 662.

Each draw becomes 9 tokens: slot, day-of-week, machine, three ball-set ids,
three digits. Vocabulary sizes: slot 4, day-of-week 7, machine 11, ball set
170 (one shared table across all three ball-set positions), digit 10, plus
three special ids (pad, mask, unknown) shared across every type. This is
decision D2: use every available CSV column as a token, rather than a
hand-picked subset. Live draws carry no machine or ball-set columns at all,
so those tokens encode as "unknown" throughout the live window.

The leakage rule: the target draw's slot and day of week are given to the
model as input (they are known before the draw happens), but its machine and
ball-set tokens are masked, because whether those are published before the
draw is not confirmed. This masking cannot be turned off by a config file; it
is unconditional in the tokenizer code.

The lookahead tests (`test_no_future_leakage`,
`test_lookahead_permutation_invariance` in `phase3/test_tokenize.py`) must
pass before any run is trusted.

### Stage A: window sweep

Stage A is the H1 test: does a sequence model beat the uniform baseline on
held-out draws. Two architectures: M1, a 4-layer decoder-only transformer
(d_model 128, 4 heads, feed-forward 512, roughly 0.83-0.86M parameters
depending on window size, described in the plan as "about 1M"), and M2, a
non-attention control, a 2-layer GRU (hidden size 256, about 0.73M
parameters). Training: cross-entropy loss on the next draw's three digits,
batch 128, learning rate 3e-4, up to 50 epochs with early stopping (patience
5) on validation loss, weight decay 0.01. Window sizes 16, 64 (default), and
256 draws, each at 3 seeds (0-2); W=64 additionally ran at seeds 3-5. 21
runs total.

| Config | Runs | Seeds | Notable individual result |
|---|---|---|---|
| M1, W=16 | 3 | 0-2 | none flagged |
| M1, W=64 | 6 | 0-5 | seed 0: test top-20 0.0278 vs null 0.0206, p=0.005 (nominal) |
| M1, W=256 | 3 | 0-2 | seed 2: 9 of 72 test lag cells p<.05 |
| M2, W=16 / W=64 / W=256 | 9 (3 each) | 0-2 | none flagged |

Across all 21 runs: test log-loss ranged 2.3085 to 2.3170, every run above
the 2.302585 baseline. Best validation loss was 2.3077 (M1, W=16, seed 2).
Test top-20 combo hit rate ranged 0.0133 to 0.0278 against a chance baseline
of 0.020; live log-loss ranged 2.3044 to 2.3213, live top-20 0.0136 to
0.0287. The one nominally significant cell (M1, W=64, seed 0, test top-20,
p=0.005) does not survive Bonferroni correction across 21 runs (adjusted
threshold 0.00238) and its live counterpart on the same run was not
significant (0.0287 vs null 0.0257, p=0.233). All other test-window
p-values fell between 0.13 and 0.98. Lag tables (72 cells per run, 3.6
expected false positives by chance) showed 0 to 13 cells below p=0.05 on
test and 0 to 8 on live, consistent with noise around 3.6 across 21
independent tables.

### M3: grokking runs

M3 is the exploratory grokking check: a smaller 2-layer transformer
(d_model 128), AdamW, W=16, batch 512, at three weight-decay values (0.1,
0.3, 1.0), logging train and validation per-position accuracy every 1,000
steps. The plan specified 1,000,000 steps (an estimated 8-12 hour run). All
three runs were in fact extended to 10,000,000 steps after the first
1,000,000 completed; the plan document still reads "1M steps" and was not
updated when this decision was made.

| Weight decay | Train loss | Train accuracy | Val loss | Val accuracy | Elapsed |
|---|---|---|---|---|---|
| 0.1 | 0.00039 | 1.0 / 1.0 / 1.0 | 8.149 | 0.106 / 0.101 / 0.113 | ~800,600 s (~222 h) |
| 0.3 | 0.0057 | 1.0 / 1.0 / 1.0 | 6.256 | 0.098 / 0.102 / 0.092 | ~800,600 s (~222 h) |
| 1.0 | 0.359 | 0.944 / 0.941 / 0.935 | 3.803 | 0.105 / 0.104 / 0.096 | ~787,300 s |

(Train/val accuracy triples are per digit position d0/d1/d2, at step
10,000,000.) At the original 1,000,000-step mark, the wd=0.1 run's
validation loss was already 7.574, showing the same shape it would still
show 9 million steps later.

We saw train accuracy saturate to 1.0 (0.94 under the strongest decay) at
all three weight decays, while validation accuracy stayed at approximately
0.10, chance for a 10-way digit guess, for the entire run. Validation loss
did not fall; it rose to 3.8-8.1, well above the 2.302585 baseline, meaning
the model became more confidently wrong on held-out data as training went
on, the opposite of a grokking transition. We think this means no grokking
transition occurred at these weight decays and this step count.

### The preregistered early-hit test

The hypothesis (H-early), Mark's observation from the live Telegram
prediction stream (May-June 2026): Phase 2's production predictions tend to
match the winning combo of a draw one or two draws LATER than the draw they
were made for, at an above-chance rate. The near-hit table
(`autoresearch/results/near_hits_table.md`, dated 2026-08-31) motivated this
observation but could not test it, since the table was built from the same
data the observation came from.

The preregistration (`plans/PREREG_EARLY_HIT.md`) was committed to git on
2026-08-31 and specified, before any analysis ran: the held-out data (all
draws in the live file strictly after 2026-06-05, expected approximately
280), the exact predictor (Config A, exp#1566, blended with Config B,
exp#1572, exactly as `predict_now.py` runs it in production, trained once on
all draws through 2026-06-05, no per-draw retraining), the single primary
statistic (the pooled rate, over prediction-to-actual-draw pairs at lag k in
{+1, +2}, of the best-in-top-20 combo matching at least 2 of 3 positions),
the null procedure (2,000 permutations of the actual draw sequence, seed
42), and the decision rule (one-sided p < 0.05 to keep H-early alive). The
analysis ran 2026-08-31T10:20:13, about ten minutes after the commit, on the
resulting 278 held-out draws (2026-06-06 to 2026-08-26), all scored.

| Statistic | Observed | Null mean | Null p95 | p-value | n |
|---|---|---|---|---|---|
| Primary: pooled >=2-of-3 match, k in {+1,+2} | 0.2423 | 0.2472 | 0.2785 | 0.624 | 553 pairs |
| Secondary: exact top-20 hit, k=+1 | 0.0108 | 0.0185 | - | 0.886 | 277 |
| Secondary: exact top-20 hit, k=+2 | 0.0181 | 0.0188 | - | 0.598 | 276 |

One deviation from the preregistered spec was logged: no per-draw retraining,
exactly as pre-declared (a confirmation that the pre-declared choice was
followed, not a departure from it). The full lag 0-5 table (24 cells) and a
negative control at k in {-1, -2} (8 cells, to check H-early's predicted
asymmetry: excess only at positive lags) were computed and stored but are
exploratory. Consensus set size across held-out draws ranged 13 to 46
(median 31, mean 30.4).

Observed (0.2423) fell below the null mean (0.2472), not above it, and
p=0.624 is far from the p<0.05 threshold. As preregistered, H-early is
rejected at this sample size, and the early-hit line of inquiry is closed
unless new live data reopens it.

### Stage B: lag-credited training and the causal backtest

Stage B is a second, independent test of the same early-hit idea (H2), built
into the training objective itself rather than tested on a frozen predictor.
Instead of cross-entropy on only the next draw, the loss credits correct
digit predictions at lags 0 through K with exponentially decaying weight
exp(-lambda * k). K=2 and lambda=1.0 were meant to be set from a Phase 2
causal backtest of the lag structure; that backtest found no lag beating its
own null, so K and lambda stayed at their placeholder values by the plan's
own fallback rule.

Six runs: M1 and M2, each at W=64, 3 seeds, trained with this lag-credited
objective (2,334 test records and 660 live records rather than 2,336/662,
since scoring a lag-2 credit needs two draws past the window edge). Test
lag-0 log-loss ranged 2.3089 to 2.3129, again above baseline. Lag-table
p-values at lags 0, 1, and 2 were all above 0.05 except one cell (M2, W=64,
seed 1, lag 2, p=0.032), a single nominal hit among many that does not
survive correction across 6 runs times 3 lags. Lag cells below p=0.05
ranged 0-3 per run on test and 1-10 on live (of 72 per table); the worst
live table (M2, W=64, seed 0) had 10 against roughly 3.6 expected by chance,
but this is one table of six and not isolated to one lag or direction.
Stage B2, a policy-gradient variant, was never run: the plan gated it on
Stage B showing something worth double-checking, and Stage B did not.

### Stage C: online replica vs. frozen control

The base model (M1, lag-credited, W=64, seed 0) was retrained deterministically
on GPU in 23.5 seconds (the original CPU-trained run left no checkpoint to
reuse). Two copies ran forward over all 662 live draws: a frozen copy that
never updates, and a replica copy that takes one AdamW optimizer step
(learning rate 1e-4, weight decay 0.0) after each live draw is scored, on a
replay batch of the 64 most recently scored live draws plus 64 sampled from
training history. Both copies predict every live draw before it is
revealed; only the replica's weights change, and those weights are never
reused offline. The two were compared with a paired sign-flip permutation
test on per-draw log-loss (2,000 resamples). Total wall time: 23.5 seconds
for the base model plus 30.3 seconds for the full online loop, 54.5 seconds
end to end.

| Model | Avg log-loss | Top-20 hit rate | Null top-20 | p (top-20) |
|---|---|---|---|---|
| Frozen | 2.3107 | 0.0151 | 0.0157 | 0.711 |
| Replica (online) | 2.6527 | 0.0242 | 0.0237 | 0.509 |

Paired difference (frozen log-loss minus replica log-loss): -0.342, against a
null of 0.000 +/- 0.025, two-sided p=0.0005. We saw the frozen model sit at
chance on live draws (top-20 not different from its null, log-loss close to
2.31) and the replica score reliably worse (log-loss 2.65) on the exact same
draws in the exact same order. We think this means the online updates hurt
rather than helped: a batch of 64 newest live draws is too small and too
noisy a signal to safely update a model on, and each update pushed the
replica away from, not toward, better calibration.

### Stage D: TimesFM-3 zero-shot

Stage D runs TimesFM-3 (checkpoint `google/timesfm-3.0-pytorch`, package
`timesfm[torch]==3.0.1`, CPU, 8 threads, 1.5 s load time, max context 15,360)
with no training on Pick 3 data at all. Each digit position (d0, d1, d2) is
its own univariate series in draw order, all four daily slots interleaved
chronologically (decision D6). For every test and live draw, the model
forecasts the next value from all prior values and returns 9 deciles, binned
to the nearest integer 0-9 (floor 1e-6, renormalized) into a 10-way digit
distribution; the three per-position distributions multiply into a
1,000-combo ranking scored the same way as the other stages. Comparison row:
seasonal-naive (same slot, one day earlier, period 4). 8,994 forecast calls
ran at a mean 0.589 s each, total wall time 5,342 seconds (89 minutes).

| Window | Model | Log-loss | Top-1 | Top-20 | Top-20 null | p |
|---|---|---|---|---|---|---|
| Test (2,336) | TimesFM-3 | 4.713 | 0.0986 | 0.0205 | 0.0189 | 0.281 |
| Test (2,336) | Naive (period 4) | 12.508 | 0.0946 | 0.0133 | - | 0.997 |
| Live (662) | TimesFM-3 | 4.756 | 0.104 | 0.0257 | 0.0214 | 0.238 |
| Live (662) | Naive (period 4) | 12.292 | 0.110 | 0.0211 | - | 0.609 |

(Uniform baseline for reference: log-loss 2.3026, top-1 0.10, top-20 0.020.)

We saw TimesFM-3's log-loss (4.7 on both windows) land well above the
uniform baseline (2.3): its binned forecast distributions were overconfident
for a target with no real structure. Top-1 and top-20 accuracy sat at
chance. The naive baseline scored worse still on log-loss (12.3-12.5),
because its point forecast concentrates probability mass more sharply than
TimesFM-3's decile spread, and a sharp wrong guess costs more under log-loss
than a broad one. We think TimesFM-3 treats the series as close to
unstructured (chance top-1/top-20 means it is not inventing the false
patterns the sharper naive baseline does), but its calibration is not tuned
for a target this flat, so its log-loss comes out worse than uniform
guessing. Claims about TimesFM-3 external benchmarks cited in the plan (330
million parameters, GIFT-Eval 0.640 against 1.0 seasonal-naive, non-commercial
license) could not be verified from anything in this repository.
[unverified: from plan]

## What we found

Six independent lines of inquiry (summarized in the stage table above), all
null, against the same uniform and permutation-null baselines. Stage A's 21
runs never beat log-loss 2.302585 on test data, and the one nominal top-20
significance (p=0.005) did not survive Bonferroni correction or replicate on
live data. M3's three 10-million-step runs saturated train accuracy to 1.0
while val accuracy stayed pinned at chance; no grokking transition appeared.
The preregistered early-hit test came in at 0.2423 against a null mean of
0.2472 (p=0.624), rejecting H-early. Stage B, the same early-hit idea tested
by training objective instead of a frozen predictor, showed lag-table
p-values at or above 0.05 except one unremarkable cell. Stage C's online
replica scored measurably worse than the frozen control (log-loss 2.65 vs.
2.31, paired p=0.0005), and the frozen model itself sat at chance on live
draws. Stage D's TimesFM-3, zero-shot, scored worse than uniform guessing on
log-loss (4.7 vs. 2.3) and at chance on top-1/top-20, though it beat the
naive seasonal baseline (12.5).

We think this means a from-scratch sequence model trained on the raw draw
token stream does not find structure the Phase 2 feature-engineering
pipeline missed, a pretrained real-world forecasting model does not either,
and Mark's early-hit observation from the live prediction stream does not
hold up on a held-out sample large enough to test it. This extends Phase 1's
null on direct statistical tests of the draw process and Phase 2's null on
live prediction: three independent approaches now agree.

## What this does not show

This does not show that no signal of any kind exists in Texas Pick 3 draws;
it shows that these specific models, sizes, window lengths, and training
budgets found none. A larger model, a longer M3 run, a different
tokenization, or a genuinely new source of information not in this CSV could
behave differently; nothing here rules that out, only that these attempts
did not find it.

The preregistered test rejects H-early at the effect size 278 draws can
detect (n_pairs 553 for the primary statistic); it cannot rule out a much
smaller early-hit effect that would need more data to see.

Stage B's K=2, lambda=1.0 lag-credit weights stayed at their placeholder
values because the Phase 2 causal backtest that was supposed to set them
found no lag beating null, by the plan's own documented fallback. The Stage
B result is therefore a test of the placeholder weights, not of weights fit
to a measured lag structure that does not appear to exist.

Notes on the record: the plan's decision log (`plans/PHASE3_PLAN.md` section
10) numbers its decisions D2 through D6; no D1 exists in any committed
version of the plan, and we report the decisions as numbered rather than
renumber them. Separately, decision D4 ("run Stage C even on a null Stage
A/B result, or gate it") was left "deferred until Stage A results are in" in
the plan text and was never updated, even though Stage C was in fact run on
a null Stage A/B result; this is an inconsistency between the plan document
and what the code and run history show, resolved in practice but not in the
prose. A third, unrelated "D1-D5" label set inside `phase3/online.py` and
its run output names Stage C implementation choices (loss scope, replay pool
source) and should not be conflated with the plan's decision log.

The plan estimated the M3 grokking run at 8-12 hours for 1,000,000 steps
(close to what the first million measured); the plan was never updated to
reflect that all three runs were in fact extended tenfold, to 10,000,000
steps.

## Where the code is

- Tokenizer and lookahead tests: `phase3/tokenizer.py`, `phase3/test_tokenize.py`
- Data loading and splits: `phase3/data_loader.py`
- Model definitions (M1, M2, M3): `phase3/model.py`
- Stage A/B training: `phase3/train.py`, `phase3/train_lag.py`
- Grokking run: `phase3/train_grok.py`, `phase3/runs/m3_wd0.1/`, `m3_wd0.3/`,
  `m3_wd1.0/` (`metrics.jsonl`, `final.json`, checkpoints)
- Lag null and permutation procedure: `phase3/lag_null.py`
- Online replica (Stage C): `phase3/online.py`, `phase3/runs/stage_c/final.json`
- Zero-shot forecasting (Stage D): `phase3/stage_d_timesfm.py`,
  `phase3/runs/stage_d/final.json`
- Preregistration document: `plans/PREREG_EARLY_HIT.md`
- Preregistered test runner and output: `autoresearch/prereg_early_hit.py`,
  `autoresearch/results/prereg_early_hit_output.json`,
  `autoresearch/results/prereg_predictions.jsonl`
- Plan: `plans/PHASE3_PLAN.md`
- Editable-surface spec (Karpathy-loop contract for Stage A):
  `phase3/program_v4.md`
- Phase 2 causal backtest referenced in "Why Phase 3":
  `autoresearch/backtest_lag.py`,
  `autoresearch/results/backtest_lag_output_test_causal.json`,
  `autoresearch/results/backtest_lag_output_live_causal.json`

## Open questions / next phase

- The Phase 3 plan's status line still reads "DRAFT, not approved" (last
  edited 2026-09-03); this paper reports work that ran under that unapproved
  plan.
- Whether the lookahead-clean Phase 3 null should be read alongside the
  Phase 2 causal-backtest correction as one combined null story, or kept
  separate, is not settled.
- TimesFM-3's own reported benchmarks (330M parameters, GIFT-Eval 0.640) need
  an external citation before they can be stated as fact here; we did not
  verify them.
- No further Phase 3 work is planned unless new live data reopens the
  early-hit question, per the preregistration's decision rule.
- Whether a larger model or longer training budget would change the Stage A
  result is open; nothing here tested that.

## References

[1] Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS 30.
https://arxiv.org/abs/1706.03762

[2] Power, A., Burda, Y., Edwards, H., Babuschkin, I., Misra, V. (2022).
"Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets."
https://arxiv.org/abs/2201.02177

[3] Nanda, N., Chan, L., Lieberum, T., Smith, J., Steinhardt, J. (2023).
"Progress Measures for Grokking via Mechanistic Interpretability."
https://arxiv.org/abs/2301.05217

[4] Loshchilov, I., Hutter, F. (2019). "Decoupled Weight Decay
Regularization." ICLR 2019. https://arxiv.org/abs/1711.05101

[5] Das, A., Kong, W., Sen, R., Zhou, Y. (2024). "A Decoder-Only Foundation
Model for Time-Series Forecasting." ICML 2024.
https://arxiv.org/abs/2310.10688

[6] Cho, K. et al. (2014). "Learning Phrase Representations using RNN
Encoder-Decoder for Statistical Machine Translation."
https://arxiv.org/abs/1406.1078

[7] Radford, A., Narasimhan, K., Salimans, T., Sutskever, I. (2018).
"Improving Language Understanding by Generative Pre-Training." URL needed.

[8] Aksu, T. et al. (2024). "GIFT-Eval: A Benchmark for General Time Series
Forecasting Model Evaluation." https://arxiv.org/abs/2410.10393

[9] Nosek, B. A., Ebersole, C. R., DeHaven, A. C., Mellor, D. T. (2018).
"The Preregistration Revolution." PNAS 115(11):2600-2606. URL needed.

[10] Good, P. (2005). Permutation, Parametric, and Bootstrap Tests of
Hypotheses. Springer. URL needed.

[11] Karpathy, A. (2017). "Software 2.0." Medium post. URL needed.

[12] Paszke, A. et al. (2019). "PyTorch: An Imperative Style,
High-Performance Deep Learning Library." NeurIPS 32. URL needed.

[13] TimesFM-3 model card and weights, Hugging Face,
`google/timesfm-3.0-pytorch`. URL needed.
