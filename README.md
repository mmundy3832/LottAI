# LottAI

Texas Lottery draw data as a test bed for a 25-year-old question about searching
very large spaces for a signal that probably is not there.

The lottery is a convenient subject. It publishes a clean historical record, it has
a dollar-denominated score (a $1 straight Pick 3 ticket pays $500), and the honest
prior is that there is nothing to find. That makes it a good place to test search
methods and evaluation discipline, because any "edge" that shows up is far more
likely to be a bug, a leak, or selection bias than a real pattern. Most of the work
in this repo is about telling those apart.

Expected result: null. So far, that is the result.

## The three phases

| Phase | Question | Method | Result | Paper |
|---|---|---|---|---|
| 1 | Is the Lotto Texas draw process statistically random? | 15 statistical tests on 2,308 draws, with multiple-testing correction | No deviation from random survived correction | [papers/WHITEPAPER_PHASE1.md](papers/WHITEPAPER_PHASE1.md) |
| 2 | Can a large automated search over model configurations find a Pick 3 model that beats chance on unseen draws? | Genetic algorithm plus LLM plus random search, 1,670 experiments, live prediction ledger | Validation score climbed to a ceiling; no edge on live draws after a leak was fixed | [papers/WHITEPAPER_PHASE2.md](papers/WHITEPAPER_PHASE2.md) |
| 3 | Does a sequence model trained directly on the draw stream find anything, and does a preregistered hypothesis survive held-out data? | Small transformer, 10M-step grokking runs, online-vs-frozen comparison, TimesFM-3 zero-shot, preregistered early-hit test | Everything at chance; preregistered hypothesis rejected | [papers/WHITEPAPER_PHASE3.md](papers/WHITEPAPER_PHASE3.md) |

All three papers are marked DRAFT, ready for review. Each one opens with a
plain-language summary and a "Background you need" section written for a reader
with high-school math, then goes into detail.

## Layout

- `phase1/` Phase 1. The Lotto Texas randomness scripts, the draw and Austin weather
  data, and their outputs under `phase1/results/`. Written on a different machine;
  four scripts still carry that machine's absolute paths.
- `autoresearch/` Phase 2. The experiment runner, genetic operators, feature
  pipelines, live prediction loop, and results ledgers. `program_v3.md` is the
  instruction file the LLM experimenter read.
- `phase3/` Phase 3. Tokenizer, transformer model, trainers, online replica,
  TimesFM-3 stage, and tests. Self-contained by design (copies, not imports, from
  Phase 2).
- `data/` Historical Pick 3 draws and Texas Lottery pretest (machine and ball set)
  files. Draws after the frozen cutoff used for live scoring are kept out of the
  repo by `.gitignore`.
- `papers/` The three whitepapers and working notes.
- `plans/` Phase plans and the preregistration document for the Phase 3 early-hit
  test.
- `BRIEFING.md` Internal onboarding notes written for the AI harness. Denser and
  older than the papers; read the papers first.


## Running it

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Phase 2's runner expects an Ollama server on localhost serving the experimenter
model (MiniMax M2.7 was used) and reads credentials from a `.env` file that is not
committed. Several scripts hardcode the original machine's path
(`/mnt/beastmode/lottai`) and will need that edited to run elsewhere. Phase 3
training runs on CUDA if available and falls back to CPU.

## Data boundary

Live draws (those after the frozen training cutoff) were never shown to the LLM
experimenter and were used only for scoring. The raw live CSVs are gitignored. The
prediction ledgers in `autoresearch/` do record the actual winning digits for the
draws they scored, because that is the evidence for the live results. Those digits
are public Texas Lottery records.

## License

MIT. See [LICENSE](LICENSE).

## Status

Research project, one author, run on personal hardware. Pull requests welcome,
especially ones that find a leak.
