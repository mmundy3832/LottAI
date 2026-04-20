"""
runner.py - Autonomous experiment loop controller for LottAI autoresearch.

Uses Ollama's local qwen2.5-coder:7b model to iteratively propose, execute,
and evaluate experiments for Pick 3 Evening prediction optimization.

Loop:
    1. Read program.md (instructions for the AI)
    2. Call Ollama to propose a new experiment (modification to experiment.py)
    3. Execute experiment.py in a subprocess
    4. Evaluate results via prepare.py metrics
    5. Log results to experiments.jsonl
    6. Feed results back to Ollama for the next iteration
    7. Repeat
"""

import os
import sys
import json
import re
import time
import subprocess
import argparse
from datetime import datetime, timezone

import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "qwen2.5-coder:7b"

# All paths relative to this file's directory
_DIR = os.path.dirname(os.path.abspath(__file__))
_PROGRAM_FILE = os.path.join(_DIR, "program.md")
_EXPERIMENT_FILE = os.path.join(_DIR, "experiment.py")
_LOG_FILE = os.path.join(_DIR, "experiments.jsonl")

# Fixed boilerplate prepended to every experiment - the model only writes model logic
BOILERPLATE = '''import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import prepare
from prepare import (load_data, get_train_val_test, build_features,
                     evaluate_predictions, NUM_COMBOS, combo_to_digits,
                     digits_to_combo, FEATURE_DIMS, load_cached_features)

# Common sklearn imports so experiments don't need to import them
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy as scipy_entropy

# Pre-loaded data - DO NOT redefine these
train_df, val_df, test_df = get_train_val_test()
full_df = load_data()
train_indices = list(range(0, len(train_df)))
val_indices = list(range(len(train_df), len(train_df) + len(val_df)))
y_train = train_df["combo_int"].values
y_val = val_df["combo_int"].values
n_val = len(val_df)
n_train = len(train_df)

# Per-digit labels (for per-digit modeling)
y_train_d1 = y_train // 100
y_train_d2 = (y_train // 10) % 10
y_train_d3 = y_train % 10
y_val_d1 = y_val // 100
y_val_d2 = (y_val // 10) % 10
y_val_d3 = y_val % 10

# FAST feature loading - use these instead of build_features():
#   X_train = load_cached_features("train", ["basic", "recency"])
#   X_val = load_cached_features("val", ["basic", "recency"])
# Available sets: basic(11), recency(121), gaps(40), positional(33), temporal(20), momentum(7)

'''


# ---------------------------------------------------------------------------
# Ollama Integration
# ---------------------------------------------------------------------------

def query_ollama(prompt, temperature=0.7, max_tokens=2000):
    """Send prompt to local Ollama, return response text.

    Parameters
    ----------
    prompt : str
        The full prompt to send.
    temperature : float
        Sampling temperature (higher = more creative).
    max_tokens : int
        Maximum tokens to generate.

    Returns
    -------
    str
        The model's response text.

    Raises
    ------
    ConnectionError
        If Ollama is unreachable.
    RuntimeError
        If the response is malformed.
    """
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=300,  # 5-minute timeout for generation
        )
        response.raise_for_status()
    except requests.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_URL}. "
            f"Is Ollama running? Start it with: ollama serve"
        )
    except requests.Timeout:
        raise RuntimeError("Ollama request timed out after 300 seconds.")
    except requests.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e}")

    data = response.json()
    if "response" not in data:
        raise RuntimeError(
            f"Unexpected Ollama response format. Keys: {list(data.keys())}"
        )
    return data["response"]


def check_ollama_connectivity():
    """Verify Ollama is running and the model is available.

    Returns
    -------
    bool
        True if connection succeeds and model responds.
    str
        Status message describing the result.
    """
    # Check if Ollama server is reachable
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.ConnectionError:
        return False, (
            "Cannot connect to Ollama at localhost:11434. "
            "Start it with: ollama serve"
        )
    except Exception as e:
        return False, f"Ollama connectivity check failed: {e}"

    # Check if model is available
    try:
        tags = resp.json()
        model_names = [m.get("name", "") for m in tags.get("models", [])]
        # Ollama often stores model names with tag suffix like ":latest"
        model_found = any(
            MODEL in name or name.startswith(MODEL.split(":")[0])
            for name in model_names
        )
        if not model_found:
            return False, (
                f"Model '{MODEL}' not found. Available models: {model_names}. "
                f"Pull it with: ollama pull {MODEL}"
            )
    except Exception as e:
        return False, f"Error checking model availability: {e}"

    # Quick generation test
    try:
        test_resp = query_ollama("Reply with only the word 'ready'.", temperature=0.0, max_tokens=10)
        if not test_resp.strip():
            return False, "Ollama returned empty response on test query."
    except Exception as e:
        return False, f"Ollama test generation failed: {e}"

    return True, f"Ollama OK. Model: {MODEL}. Test response: '{test_resp.strip()}'"


# ---------------------------------------------------------------------------
# File Helpers
# ---------------------------------------------------------------------------

def read_file(path):
    """Read a text file and return its contents, or None if missing."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    """Write content to a text file (overwrite)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def now_iso():
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Experiment Log (experiments.jsonl)
# ---------------------------------------------------------------------------

def load_experiment_log():
    """Load experiment log from JSONL file.

    Returns
    -------
    list of dict
        Each dict is one experiment's record.
    """
    if not os.path.exists(_LOG_FILE):
        return []

    entries = []
    with open(_LOG_FILE, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed line {line_num} in experiments.jsonl: {e}")
    return entries


def save_experiment_log(log):
    """Save entire experiment log to JSONL file (rewrite).

    Also appends the latest entry to avoid data loss if only
    one new entry was added.
    """
    with open(_LOG_FILE, "w", encoding="utf-8") as f:
        for entry in log:
            f.write(json.dumps(entry) + "\n")


def append_experiment_entry(entry):
    """Append a single experiment entry to the JSONL log (append-only)."""
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Code Extraction & Validation
# ---------------------------------------------------------------------------

def extract_code(ai_response):
    """Extract Python code from AI response.

    Looks for code between ```python and ``` markers.
    Falls back to detecting raw Python code if markers are absent.

    Parameters
    ----------
    ai_response : str
        The full text response from the AI model.

    Returns
    -------
    str or None
        Extracted Python code, or None if extraction fails validation.
    """
    # Try to find code between ```python and ``` markers
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, ai_response, re.DOTALL)

    if matches:
        # Use the longest match (in case of multiple code blocks)
        code = max(matches, key=len).strip()
    else:
        # Try generic code block markers
        pattern_generic = r"```\s*\n(.*?)```"
        matches_generic = re.findall(pattern_generic, ai_response, re.DOTALL)

        if matches_generic:
            code = max(matches_generic, key=len).strip()
        else:
            # Try to detect raw Python code (heuristic: lines starting with import/from/def/class)
            lines = ai_response.split("\n")
            code_lines = []
            in_code = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("import ", "from ", "def ", "class ", "#")):
                    in_code = True
                if in_code:
                    code_lines.append(line)
            if code_lines:
                code = "\n".join(code_lines).strip()
            else:
                return None

    # Basic validation: must have some actual code (at least a few lines)
    if len(code.strip().split("\n")) < 3:
        print("  VALIDATION FAIL: Extracted code too short (< 3 lines).")
        return None

    return code


def extract_description(ai_response):
    """Extract a brief description from the AI response.

    Looks for a # DESCRIPTION comment or takes the first non-empty line.

    Parameters
    ----------
    ai_response : str
        The full text response from the AI model.

    Returns
    -------
    str
        A short description of the experiment.
    """
    # Look for # DESCRIPTION comment in the response
    for line in ai_response.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# DESCRIPTION"):
            return stripped.replace("# DESCRIPTION", "").strip(" :-")
        if stripped.startswith("# ") and len(stripped) > 3:
            return stripped[2:].strip()

    # Fallback: first non-empty line (truncated)
    for line in ai_response.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith("```"):
            return stripped[:120]

    return "No description available"


# ---------------------------------------------------------------------------
# Experiment Execution
# ---------------------------------------------------------------------------

def execute_experiment(timeout=180):
    """Run experiment.py in a subprocess with timeout.

    Parameters
    ----------
    timeout : int
        Maximum seconds to allow the experiment to run.

    Returns
    -------
    tuple of (bool, str, str)
        (success, stdout, stderr)
    """
    try:
        result = subprocess.run(
            [sys.executable, "-u", "experiment.py"],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_DIR,
        )
        return (result.returncode == 0, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (False, "", f"Experiment timed out after {timeout} seconds")
    except Exception as e:
        return (False, "", f"Experiment execution error: {e}")


def parse_results(output):
    """Parse experiment results from stdout.

    Expects JSON on the last non-empty line of stdout.

    Parameters
    ----------
    output : str
        The full stdout from the experiment subprocess.

    Returns
    -------
    dict or None
        Parsed results dictionary, or None if parsing fails.
    """
    if not output or not output.strip():
        return None

    # Try each line from the end, looking for valid JSON
    lines = output.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        if not line:
            continue
        try:
            result = json.loads(line)
            if isinstance(result, dict):
                return result
        except json.JSONDecodeError:
            continue

    return None


# ---------------------------------------------------------------------------
# Prompt Templates
# ---------------------------------------------------------------------------

def build_iteration_prompt(program, recent_log, exp_num):
    """Build the standard iteration prompt for the AI.

    Parameters
    ----------
    program : str
        Contents of program.md.
    recent_log : list of dict
        Last 5 experiment log entries.
    exp_num : int
        Current experiment number (0-indexed).

    Returns
    -------
    str
        The formatted prompt.
    """
    # Format recent results
    if recent_log:
        recent_text = json.dumps(recent_log, indent=2, default=str)
    else:
        recent_text = "(No previous experiments. This is the first one.)"

    # Find best experiment so far
    best = _find_best_experiment(recent_log)
    if best:
        best_text = json.dumps(best, indent=2, default=str)
    else:
        best_text = "(No successful experiments yet.)"

    prompt = f"""You are an autonomous researcher optimizing a lottery prediction model.

INSTRUCTIONS:
{program}

RECENT RESULTS (last {len(recent_log)} experiments):
{recent_text}

CURRENT BEST:
{best_text}

EXPERIMENT NUMBER: {exp_num + 1}

YOUR TASK:
Write ONLY the model logic for experiment.py. A boilerplate header is automatically prepended that provides:
- All imports (numpy, json, warnings, prepare module)
- train_df, val_df, test_df = get_train_val_test()
- full_df = load_data()
- train_indices = list(range(0, len(train_df)))
- val_indices = list(range(len(train_df), len(train_df) + len(val_df)))
- y_train = train_df["combo_int"].values
- y_val = val_df["combo_int"].values
- n_val = len(val_df)
- NUM_COMBOS = 1000

DO NOT write imports, do not call get_train_val_test(), do not redefine train_indices or y_val.
These are ALREADY available. Just use them directly.

Your code must:
1. Load features FAST: X_train = load_cached_features("train", ["basic", "recency"])
                       X_val = load_cached_features("val", ["basic", "recency"])
   Available sets: basic(11), recency(121), gaps(40), positional(33), temporal(20), momentum(7)
2. Train a model on X_train, y_train
3. Generate prob_matrix with shape (n_val, 1000) - probability for each combo
4. Call: results = evaluate_predictions(prob_matrix, y_val)
5. LAST LINE must be: print(json.dumps(results))

Available feature sets: {list_feature_sets()}
Feature dimensions: basic=11, recency=121, gaps=40, positional=33, temporal=20, momentum=7

Start with # DESCRIPTION comment, then your model code.
Output ONLY code between ```python and ``` markers.
Only use: numpy, scipy, sklearn (already installed). Keep under 180 seconds.
"""
    return prompt


def build_strategy_prompt(program, full_log):
    """Build the strategy review prompt for the AI.

    Parameters
    ----------
    program : str
        Contents of program.md.
    full_log : list of dict
        The complete experiment log.

    Returns
    -------
    str
        The formatted prompt.
    """
    # Find top 5 and worst 5 by mean_rank
    successful = [e for e in full_log if e.get("status") == "success" and e.get("results")]
    failed = [e for e in full_log if e.get("status") == "error"]

    if successful:
        sorted_by_rank = sorted(
            successful,
            key=lambda x: x.get("results", {}).get("mean_rank", 999)
        )
        top_5 = sorted_by_rank[:5]
        worst_5 = sorted_by_rank[-5:]
        top_5_text = json.dumps(top_5, indent=2, default=str)
        worst_5_text = json.dumps(worst_5, indent=2, default=str)
    else:
        top_5_text = "(No successful experiments yet.)"
        worst_5_text = "(No successful experiments yet.)"

    # Summary statistics
    total = len(full_log)
    success_count = len(successful)
    fail_count = len(failed)

    prompt = f"""You are reviewing {total} experiments to decide the next research direction.

INSTRUCTIONS:
{program}

SUMMARY:
- Total experiments: {total}
- Successful: {success_count}
- Failed: {fail_count}

ALL RESULTS:
{json.dumps(full_log, indent=2, default=str)}

TOP 5 EXPERIMENTS (best mean_rank):
{top_5_text}

WORST 5 EXPERIMENTS (worst mean_rank):
{worst_5_text}

ANALYSIS QUESTIONS:
- Which feature sets appear in top experiments?
- Which model types perform best?
- What approaches have been exhausted?
- What hasn't been tried yet?
- Are there any promising patterns in the errors?

YOUR TASK:
1. Write a brief strategic analysis (2-3 paragraphs) as Python comments.
2. Then write ONLY the model logic (no imports, no data loading - boilerplate is prepended).

The boilerplate already provides: numpy as np, json, full_df, train_df, val_df,
train_indices, val_indices, y_train, y_val, n_val, NUM_COMBOS, build_features,
evaluate_predictions, combo_to_digits, digits_to_combo, FEATURE_DIMS.

Your code must:
1. Build features: X = build_features(full_df, indices, ["feature_set_names"])
2. Train model on X_train, y_train
3. Generate prob_matrix shape (n_val, 1000)
4. results = evaluate_predictions(prob_matrix, y_val)
5. LAST LINE: print(json.dumps(results))

Output ONLY code between ```python and ``` markers. Use numpy, scipy, sklearn only.
"""
    return prompt


def list_feature_sets():
    """Return a formatted string of available feature sets."""
    return "basic, recency, gaps, positional, temporal, momentum"


def _find_best_experiment(log):
    """Find the best experiment by mean_rank from a log subset.

    Parameters
    ----------
    log : list of dict
        Experiment log entries.

    Returns
    -------
    dict or None
        The best experiment entry, or None if no successful experiments.
    """
    successful = [
        e for e in log
        if e.get("status") == "success" and e.get("results")
    ]
    if not successful:
        return None

    return min(
        successful,
        key=lambda x: x.get("results", {}).get("mean_rank", 999)
    )


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------

def print_experiment_summary(entry):
    """Print a formatted summary of an experiment result.

    Parameters
    ----------
    entry : dict
        A single experiment log entry.
    """
    exp_id = entry.get("experiment_id", "?")
    status = entry.get("status", "unknown")
    desc = entry.get("description", "No description")

    print(f"\n  Experiment #{exp_id}: {status.upper()}")
    print(f"  Description: {desc[:100]}")

    if status == "error":
        error = entry.get("error", "Unknown error")
        print(f"  Error: {error[:200]}")
    elif status == "success" and entry.get("results"):
        r = entry["results"]
        print(f"  Mean Rank:      {r.get('mean_rank', 'N/A'):.1f}" if isinstance(r.get('mean_rank'), (int, float)) else f"  Mean Rank:      {r.get('mean_rank', 'N/A')}")
        print(f"  Brier Score:    {r.get('brier_score', 'N/A'):.6f}" if isinstance(r.get('brier_score'), (int, float)) else f"  Brier Score:    {r.get('brier_score', 'N/A')}")
        print(f"  Log-Likelihood: {r.get('log_likelihood', 'N/A'):.4f}" if isinstance(r.get('log_likelihood'), (int, float)) else f"  Log-Likelihood: {r.get('log_likelihood', 'N/A')}")
        print(f"  Top-10 Hit:     {r.get('top_10_hit', 'N/A'):.4f}" if isinstance(r.get('top_10_hit'), (int, float)) else f"  Top-10 Hit:     {r.get('top_10_hit', 'N/A')}")
        print(f"  Top-50 Hit:     {r.get('top_50_hit', 'N/A'):.4f}" if isinstance(r.get('top_50_hit'), (int, float)) else f"  Top-50 Hit:     {r.get('top_50_hit', 'N/A')}")
        print(f"  Top-100 Hit:    {r.get('top_100_hit', 'N/A'):.4f}" if isinstance(r.get('top_100_hit'), (int, float)) else f"  Top-100 Hit:    {r.get('top_100_hit', 'N/A')}")
        print(f"  Exact Hit:      {r.get('exact_hit', 'N/A'):.4f}" if isinstance(r.get('exact_hit'), (int, float)) else f"  Exact Hit:      {r.get('exact_hit', 'N/A')}")
        print(f"  Expected Value: ${r.get('expected_value', 'N/A'):.2f}" if isinstance(r.get('expected_value'), (int, float)) else f"  Expected Value: {r.get('expected_value', 'N/A')}")

        vs = r.get("vs_baseline", {})
        if vs:
            rank_imp = vs.get("rank_improvement", 0)
            direction = "BETTER" if rank_imp > 0 else "WORSE" if rank_imp < 0 else "SAME"
            print(f"  vs Baseline:    {abs(rank_imp):.1f} ranks {direction} than uniform" if isinstance(rank_imp, (int, float)) else "")
    else:
        print("  (No results parsed)")


def print_session_summary(log):
    """Print an overall session summary.

    Parameters
    ----------
    log : list of dict
        The complete experiment log.
    """
    total = len(log)
    successful = [e for e in log if e.get("status") == "success" and e.get("results")]
    failed = [e for e in log if e.get("status") == "error"]

    print(f"\n{'='*60}")
    print("SESSION SUMMARY")
    print(f"{'='*60}")
    print(f"  Total experiments:  {total}")
    print(f"  Successful:         {len(successful)}")
    print(f"  Failed:             {len(failed)}")

    if successful:
        best = min(successful, key=lambda x: x["results"].get("mean_rank", 999))
        worst = max(successful, key=lambda x: x["results"].get("mean_rank", 999))
        avg_rank = sum(e["results"]["mean_rank"] for e in successful) / len(successful)

        print(f"\n  Best mean_rank:     {best['results']['mean_rank']:.1f} (Experiment #{best['experiment_id']})")
        print(f"  Worst mean_rank:    {worst['results']['mean_rank']:.1f} (Experiment #{worst['experiment_id']})")
        print(f"  Average mean_rank:  {avg_rank:.1f}")
        print(f"  Baseline (uniform): 500.5")
        print(f"\n  Best experiment: {best.get('description', 'N/A')[:100]}")
    else:
        print("\n  No successful experiments.")

    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

def run_loop(max_experiments=100, strategy_review_every=25, temperature=0.7,
             resume=False):
    """Main autoresearch loop.

    Parameters
    ----------
    max_experiments : int
        Maximum number of experiments to run.
    strategy_review_every : int
        Every N experiments, perform a full strategy review instead of
        a normal iteration prompt.
    temperature : float
        Ollama sampling temperature.
    resume : bool
        If True, resume from existing experiments.jsonl log.
    """
    print(f"\n{'='*60}")
    print("  LottAI AutoResearch Runner")
    print(f"{'='*60}")
    print(f"  Model:             {MODEL}")
    print(f"  Max experiments:    {max_experiments}")
    print(f"  Strategy review:    every {strategy_review_every} experiments")
    print(f"  Temperature:        {temperature}")
    print(f"  Resume mode:        {resume}")
    print(f"{'='*60}")

    # Check Ollama connectivity
    print("\nChecking Ollama connectivity...")
    ok, msg = check_ollama_connectivity()
    print(f"  {msg}")
    if not ok:
        print("\nERROR: Cannot proceed without Ollama. Exiting.")
        sys.exit(1)

    # Load program.md
    program = read_file(_PROGRAM_FILE)
    if program is None:
        print(f"\nERROR: program.md not found at {_PROGRAM_FILE}")
        print("Create program.md with instructions for the AI researcher.")
        sys.exit(1)
    print(f"\nLoaded program.md ({len(program)} chars)")

    # Load or create experiment log
    if resume:
        log = load_experiment_log()
        print(f"Resumed from existing log: {len(log)} experiments")
    else:
        if os.path.exists(_LOG_FILE):
            # Back up existing log before overwriting
            backup_name = _LOG_FILE + f".backup.{int(time.time())}"
            os.rename(_LOG_FILE, backup_name)
            print(f"Backed up existing log to: {backup_name}")
        log = []
        print("Starting fresh experiment log.")

    start_exp = len(log)
    consecutive_failures = 0
    max_consecutive_failures = 5  # safety valve

    for exp_num in range(start_exp, max_experiments):
        print(f"\n{'='*60}")
        print(f"EXPERIMENT {exp_num + 1}/{max_experiments}")
        print(f"{'='*60}")

        # Safety valve: too many consecutive failures
        if consecutive_failures >= max_consecutive_failures:
            print(f"\nWARNING: {max_consecutive_failures} consecutive failures. "
                  f"Pausing for 10 seconds before retrying...")
            time.sleep(10)
            consecutive_failures = 0

        # Determine prompt type
        if exp_num > 0 and exp_num % strategy_review_every == 0:
            print("  [Strategy Review Mode]")
            prompt = build_strategy_prompt(program, log)
        else:
            print("  [Standard Iteration Mode]")
            recent = log[-5:] if len(log) >= 5 else log
            prompt = build_iteration_prompt(program, recent, exp_num)

        # Get AI response
        print("  Querying Ollama...")
        t0 = time.time()
        try:
            ai_response = query_ollama(prompt, temperature=temperature)
        except (ConnectionError, RuntimeError) as e:
            print(f"  ERROR querying Ollama: {e}")
            log_entry = {
                "experiment_id": exp_num + 1,
                "timestamp": now_iso(),
                "status": "error",
                "error": f"Ollama query failed: {e}",
                "description": "Ollama communication error",
            }
            log.append(log_entry)
            append_experiment_entry(log_entry)
            print_experiment_summary(log_entry)
            consecutive_failures += 1
            continue
        t_query = time.time() - t0
        print(f"  Ollama responded in {t_query:.1f}s ({len(ai_response)} chars)")

        # Extract code from response
        new_code = extract_code(ai_response)
        if new_code is None:
            print("  WARNING: Could not extract valid code from AI response.")
            # Log the failure with a snippet of the response for debugging
            log_entry = {
                "experiment_id": exp_num + 1,
                "timestamp": now_iso(),
                "status": "error",
                "error": "Code extraction failed",
                "ai_response_preview": ai_response[:300],
                "description": extract_description(ai_response),
            }
            log.append(log_entry)
            append_experiment_entry(log_entry)
            print_experiment_summary(log_entry)
            consecutive_failures += 1
            continue

        # Write experiment.py with boilerplate prepended
        full_code = BOILERPLATE + new_code
        write_file(_EXPERIMENT_FILE, full_code)
        print(f"  Wrote experiment.py ({len(new_code)} chars model code + boilerplate)")

        # Execute experiment.py with timeout
        print("  Running experiment...")
        t0 = time.time()
        success, output, error = execute_experiment(timeout=180)
        t_exec = time.time() - t0
        print(f"  Execution finished in {t_exec:.1f}s (success={success})")

        if not success:
            # Combine stdout and stderr for error context
            error_detail = error[:500] if error else "(no stderr)"
            if output:
                error_detail += f"\n--- stdout ---\n{output[-500:]}"

            log_entry = {
                "experiment_id": exp_num + 1,
                "timestamp": now_iso(),
                "status": "error",
                "error": error_detail[:1000],
                "description": extract_description(ai_response),
                "execution_time_s": round(t_exec, 2),
            }
            consecutive_failures += 1
        else:
            # Parse results from experiment output
            results = parse_results(output)
            if results is None:
                log_entry = {
                    "experiment_id": exp_num + 1,
                    "timestamp": now_iso(),
                    "status": "error",
                    "error": "Could not parse JSON results from stdout",
                    "stdout_tail": output[-500:] if output else "(empty)",
                    "description": extract_description(ai_response),
                    "execution_time_s": round(t_exec, 2),
                }
                consecutive_failures += 1
            else:
                log_entry = {
                    "experiment_id": exp_num + 1,
                    "timestamp": now_iso(),
                    "status": "success",
                    "description": extract_description(ai_response),
                    "results": results,
                    "execution_time_s": round(t_exec, 2),
                }
                consecutive_failures = 0  # reset on success

        # Append to log
        log.append(log_entry)
        append_experiment_entry(log_entry)

        # Print summary
        print_experiment_summary(log_entry)

    # Session complete
    print_session_summary(log)


# ---------------------------------------------------------------------------
# Self-test (initialization verification)
# ---------------------------------------------------------------------------

def self_test():
    """Run initialization checks without starting the full loop.

    Verifies:
    - Ollama connectivity and model availability
    - program.md existence
    - File read/write capability
    - Log file handling
    - Code extraction logic
    """
    print(f"\n{'='*60}")
    print("  LottAI AutoResearch Runner - Self Test")
    print(f"{'='*60}")

    all_ok = True

    # 1. Ollama connectivity
    print("\n[1/6] Ollama connectivity...")
    ok, msg = check_ollama_connectivity()
    print(f"  {'PASS' if ok else 'FAIL'}: {msg}")
    if not ok:
        all_ok = False

    # 2. program.md
    print("\n[2/6] program.md...")
    program = read_file(_PROGRAM_FILE)
    if program:
        print(f"  PASS: program.md found ({len(program)} chars)")
    else:
        print(f"  WARN: program.md not found at {_PROGRAM_FILE}")
        print("        (Will be needed before running experiments)")

    # 3. File read/write
    print("\n[3/6] File I/O...")
    test_path = os.path.join(_DIR, "_runner_test_tmp.txt")
    try:
        write_file(test_path, "test content")
        content = read_file(test_path)
        assert content == "test content", f"Read back: {content!r}"
        os.remove(test_path)
        print("  PASS: File read/write works")
    except Exception as e:
        print(f"  FAIL: File I/O error: {e}")
        all_ok = False

    # 4. Log handling
    print("\n[4/6] Experiment log handling...")
    try:
        log = load_experiment_log()
        print(f"  PASS: Log loaded ({len(log)} existing entries)")
    except Exception as e:
        print(f"  FAIL: Log load error: {e}")
        all_ok = False

    # 5. Code extraction
    print("\n[5/6] Code extraction logic...")
    test_response = '''Here's the experiment:

```python
# DESCRIPTION: Test experiment using basic features
import numpy as np
from prepare import get_train_val_test, build_features, evaluate_predictions

train, val, test = get_train_val_test()
print("done")
```

This should work well.'''

    code = extract_code(test_response)
    if code and "import numpy" in code and "prepare" in code:
        print(f"  PASS: Code extraction works ({len(code)} chars)")
    else:
        print(f"  FAIL: Code extraction returned: {code!r}")
        all_ok = False

    desc = extract_description(test_response)
    print(f"  Description extracted: '{desc}'")

    # 6. prepare.py import check
    print("\n[6/6] prepare.py availability...")
    prepare_path = os.path.join(_DIR, "prepare.py")
    if os.path.exists(prepare_path):
        print(f"  PASS: prepare.py found")
    else:
        print(f"  FAIL: prepare.py not found at {prepare_path}")
        all_ok = False

    # Summary
    print(f"\n{'='*60}")
    if all_ok:
        print("  ALL CHECKS PASSED - Ready to run experiments")
    else:
        print("  SOME CHECKS FAILED - Fix issues before running")
    print(f"{'='*60}\n")

    return all_ok


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LottAI AutoResearch Runner - Autonomous experiment loop"
    )
    parser.add_argument(
        "--max", type=int, default=100,
        help="Maximum number of experiments to run (default: 100)"
    )
    parser.add_argument(
        "--review-every", type=int, default=25,
        help="Strategy review interval in experiments (default: 25)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Ollama sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from existing experiments.jsonl log"
    )
    parser.add_argument(
        "--self-test", action="store_true",
        help="Run initialization checks only (don't start loop)"
    )
    args = parser.parse_args()

    if args.self_test:
        ok = self_test()
        sys.exit(0 if ok else 1)
    else:
        run_loop(
            max_experiments=args.max,
            strategy_review_every=args.review_every,
            temperature=args.temperature,
            resume=args.resume,
        )
