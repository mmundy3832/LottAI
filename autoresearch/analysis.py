"""
analysis.py - Generate a prediction analysis writeup using Bonsai (local 8B model).

Spins up a second Bonsai instance on port 8081, generates a short analysis of
the prediction output, then tears the instance down.
"""

import subprocess, time, os, atexit, httpx
from pathlib import Path

BONSAI_DIR = Path("/mnt/beastmode/bonsai/server/llama-prism-b8846-d104cf1")
BONSAI_URL = "http://localhost:8081"

_BONSAI_PROC = None


def start_bonsai():
    global _BONSAI_PROC
    if _BONSAI_PROC is not None:
        return _BONSAI_PROC

    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = str(BONSAI_DIR) + ":" + env.get("LD_LIBRARY_PATH", "")

    _BONSAI_PROC = subprocess.Popen(
        [str(BONSAI_DIR / "llama-server"),
         "-m", str(BONSAI_DIR / "Ternary-Bonsai-8B-Q2_0.gguf"),
         "-c", "32768", "--port", "8081", "--host", "0.0.0.0"],
        env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    atexit.register(stop_bonsai)

    print("  Waiting for Bonsai to start...", flush=True)
    for _ in range(90):
        time.sleep(2)
        try:
            if httpx.get(f"{BONSAI_URL}/health", timeout=3).status_code == 200:
                print("  Bonsai ready.", flush=True)
                return _BONSAI_PROC
        except Exception:
            pass

    raise RuntimeError("Bonsai did not start within 3 minutes")


def stop_bonsai():
    global _BONSAI_PROC
    if _BONSAI_PROC is not None:
        _BONSAI_PROC.terminate()
        try:
            _BONSAI_PROC.wait(timeout=10)
        except subprocess.TimeoutExpired:
            _BONSAI_PROC.kill()
        _BONSAI_PROC = None


def generate_analysis(draw_label, draw_display, digit_summary, top_plays, best_k, best_ev):
    """
    Generate a 2-3 paragraph analysis of the prediction.

    Parameters
    ----------
    draw_label   : e.g. "Tuesday Night"
    draw_display : e.g. "10:12 PM CT"
    digit_summary: list of 3 strings, one per position, e.g.
                   ["9 (15.5%), 4 (13.3%), 7 (13.2%)", ...]
    top_plays    : list of strings, e.g. ["979", "999", "919", ...]
    best_k       : int, recommended ticket count
    best_ev      : float, projected EV
    """
    start_bonsai()

    top5 = ", ".join(top_plays[:5])
    pos_lines = "\n".join(
        f"  Position {i+1}: {s}" for i, s in enumerate(digit_summary)
    )

    prompt = f"""You are a concise analyst for a Pick 3 lottery prediction system.
Two machine learning ensemble models (trained on 15,000+ historical draws) have
produced the following prediction for the {draw_label} draw ({draw_display}):

Per-digit probabilities:
{pos_lines}

Top recommended combos: {top5}
Recommended buy: {best_k} tickets  |  Projected EV: ${best_ev:+.3f}

Write a 2-3 paragraph analysis (plain text, no markdown, no bullet points) that:
1. Describes what the models are signaling — which digits are dominant and whether
   the models agree or diverge across positions
2. Notes any interesting patterns in the probability distribution (concentration vs
   spread, any digit notably absent)
3. Closes with a one-sentence honest reminder about the nature of lottery predictions

Keep it tight — under 150 words total. Don't start with "The models" or "This prediction"."""

    response = httpx.post(
        f"{BONSAI_URL}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 300,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def evaluate_prediction(draw_label, digit_summary, consensus_combos,
                        consensus_ev, n_consensus, top20_combos):
    """
    Ask Bonsai to critically evaluate the prediction as a skeptical reviewer.

    Parameters
    ----------
    draw_label      : e.g. "Wednesday Night"
    digit_summary   : list of 3 strings (per-position top-5 with probabilities)
    consensus_combos: list of combo strings that both models agree on
    consensus_ev    : projected EV for consensus plays
    n_consensus     : how many consensus plays there are
    top20_combos    : list of top-20 combo strings from ensemble
    """
    start_bonsai()

    pos_lines = "\n".join(
        f"  Position {i+1}: {s}" for i, s in enumerate(digit_summary)
    )
    consensus_str = ", ".join(consensus_combos) if consensus_combos else "none"
    random_ev_20 = 20 * (1/1000) * 500 - 20   # = -10 (random baseline for 20 tickets)

    prompt = f"""You are a skeptical statistical reviewer for a Pick 3 lottery ML prediction system.
Two ensemble models produced predictions for the {draw_label} draw.

Per-digit probability distributions:
{pos_lines}

Consensus plays (both models independently ranked these in their top 50): {consensus_str}
Number of consensus plays: {n_consensus} out of a possible 50 (higher = stronger agreement)
Projected EV if buying all consensus plays: ${consensus_ev:+.3f}
  (EV formula: cumulative_hit_probability × $500 − number_of_tickets. Ticket cost is already deducted.)
Top-20 ensemble combos: {', '.join(top20_combos)}

Random baseline: buying 20 random tickets = EV ${random_ev_20:+.1f} per draw.

Evaluate this prediction critically. Consider:
1. How concentrated are the per-digit distributions? (flat = weak signal, peaked = stronger)
2. Is the consensus count high (strong agreement) or low (models diverge)?
3. Does the projected EV represent genuine lift over random?
4. Any patterns worth flagging (e.g., both models avoiding certain digits entirely)?

Be direct and honest. Under 120 words. No bullet points. No hedging phrases like "it's worth noting."
If the signal looks weak, say so plainly."""

    response = httpx.post(
        f"{BONSAI_URL}/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4,
            "max_tokens": 300,
        },
        timeout=60,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()
