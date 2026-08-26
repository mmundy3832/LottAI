"""
runner_v3.py - LottAI AutoResearch Runner v3.

Key changes from v2:
- Config-driven experiments: GA mutates JSON configs, not Python code
- template_engine.config_to_code() renders configs to valid Python
- LLM exploration (20% of runs) asks for a JSON config, not Python
- Zero repair loop -- template always produces valid code
- Uses config_space.py, config_validator.py, template_engine.py, ga_ops.py

Usage:
    python -u runner_v3.py                    # Fresh start, 200 experiments
    python -u runner_v3.py --resume           # Resume from existing log
    python -u runner_v3.py --max 50           # Limit to 50 experiments
    python -u runner_v3.py --model minimax-m2.7:cloud
    python -u runner_v3.py --ga-only          # 100% GA mode, no LLM calls
    python -u runner_v3.py --self-test        # Smoke tests and exit
    python -u runner_v3.py --render cfg.json  # Render config to stdout and exit
"""

import os
import sys
import json
import re
import time
import random
import subprocess
from datetime import datetime, timezone

import socket
import ssl
import requests
from dotenv import load_dotenv

# All paths relative to this file's directory
_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env from project root (contains OLLAMA_API_KEY etc.)
load_dotenv(os.path.join(_DIR, '..', '.env'))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CLOUD_HOST = "ollama.com"
OLLAMA_CLOUD_PORT = 443

DEFAULT_MODEL = "minimax-m2.7:cloud"
DEFAULT_MAX = 200

_RESULTS_DIR = os.path.join(_DIR, "results")
_JSONL_V3 = os.path.join(_RESULTS_DIR, "experiments_v4.jsonl")
_EXPERIMENT_V3_FILE = os.path.join(_DIR, "experiment_v4.py")
_LOG_V3 = os.path.join(_RESULTS_DIR, "autoresearch_v4.log")
_ERROR_LOG = os.path.join(_RESULTS_DIR, "autoresearch_errors.log")

# ---------------------------------------------------------------------------
# Boilerplate (identical to v2 -- prepended to every experiment)
# ---------------------------------------------------------------------------

BOILERPLATE = '''import sys, os, json, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import prepare_v3 as prepare
from prepare_v3 import (load_data, get_train_val_test, build_features,
                     evaluate_predictions, NUM_COMBOS, combo_to_digits,
                     digits_to_combo, FEATURE_DIMS, load_cached_features)

# Common sklearn imports so experiments don't need to import them
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer, RobustScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
from scipy.stats import entropy as scipy_entropy
entropy = scipy_entropy  # alias so experiments can call entropy() directly
try:
    import xgboost as xgb
except ImportError:
    xgb = None
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    import catboost as cb
except ImportError:
    cb = None
try:
    import hmmlearn.hmm as hmm
except ImportError:
    hmm = None
try:
    import statsmodels.api as sm
except ImportError:
    sm = None
try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

# Additional sklearn imports
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.manifold import TSNE

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
# Logging
# ---------------------------------------------------------------------------

def log_msg(msg):
    """Print and append to _LOG_V3."""
    safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
    print(safe_msg)
    with open(_LOG_V3, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Timestamps / JSONL helpers
# ---------------------------------------------------------------------------

def now_iso():
    """Current UTC timestamp as ISO string."""
    return datetime.now(timezone.utc).isoformat()


def load_log(jsonl_path):
    """Read all entries from a JSONL file. Returns list of dicts."""
    if not os.path.exists(jsonl_path):
        return []
    entries = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                log_msg(f"WARNING: Skipping malformed line {line_num} in {jsonl_path}")
    return entries


def append_experiment_entry(entry, jsonl_path):
    """Append one dict as a JSON line to jsonl_path."""
    with open(jsonl_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Ollama Integration (local + cloud) -- copied verbatim from runner_v2.py
# ---------------------------------------------------------------------------

def _is_cloud_model(model):
    """Check if model is a cloud model (e.g. minimax-m2.7:cloud)."""
    return model.endswith(":cloud")


def _try_parse(text):
    """JSON parsing with fallbacks for control chars and trailing data."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Clean control characters (raw newlines/tabs inside JSON strings)
    cleaned = re.sub(r'[\x00-\x1f\x7f]', lambda m: {
        ord('\n'): '\\n', ord('\r'): '\\r', ord('\t'): '\\t'
    }.get(ord(m.group()), ''), text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    # Try raw_decode for extra trailing data
    try:
        decoder = json.JSONDecoder()
        result, _ = decoder.raw_decode(cleaned.lstrip())
        return result
    except json.JSONDecodeError:
        raise RuntimeError(f"Cannot parse cloud response: {text[:200]}")


def _cloud_request(body_dict, timeout=600):
    """Make a direct HTTPS request to ollama.com without SNI.

    Ollama.com rejects TLS connections with SNI on this system,
    so we connect without server_hostname to skip the SNI extension.
    """
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY not set. Add it to .env file in project root."
        )

    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    body = json.dumps(body_dict).encode()

    raw_sock = socket.create_connection((OLLAMA_CLOUD_HOST, 443), timeout=timeout)
    ssock = ctx.wrap_socket(raw_sock)  # no server_hostname = no SNI

    request_bytes = (
        f"POST /api/generate HTTP/1.1\r\n"
        f"Host: {OLLAMA_CLOUD_HOST}\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Authorization: Bearer {api_key}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
    ).encode() + body

    ssock.sendall(request_bytes)

    response = b""
    while True:
        chunk = ssock.recv(8192)
        if not chunk:
            break
        response += chunk
    ssock.close()

    resp_text = response.decode("utf-8", errors="replace")
    if "\r\n\r\n" not in resp_text:
        raise RuntimeError(f"Malformed HTTP response from cloud: {resp_text[:200]}")

    headers, body_text = resp_text.split("\r\n\r\n", 1)
    status_line = headers.split("\r\n")[0]

    if "200" not in status_line:
        raise RuntimeError(f"Cloud API error: {status_line} - {body_text[:200]}")

    # Handle chunked transfer encoding
    if "transfer-encoding: chunked" in headers.lower():
        decoded = []
        remaining = body_text
        while remaining:
            # Each chunk: size_hex\r\n...data...\r\n
            nl = remaining.find("\r\n")
            if nl == -1:
                break
            size_str = remaining[:nl].strip()
            if not size_str:
                remaining = remaining[nl + 2:]
                continue
            try:
                chunk_size = int(size_str, 16)
            except ValueError:
                break
            if chunk_size == 0:
                break
            chunk_data = remaining[nl + 2:nl + 2 + chunk_size]
            decoded.append(chunk_data)
            remaining = remaining[nl + 2 + chunk_size + 2:]  # skip trailing \r\n
        body_text = "".join(decoded)

    return _try_parse(body_text)


def query_ollama(prompt, model, temperature=0.7, max_tokens=6000):
    """Send prompt to Ollama (local or cloud), return response text."""
    t0 = time.time()

    if _is_cloud_model(model):
        data = _cloud_request({
            "model": model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }, timeout=600)

        elapsed = time.time() - t0
        eval_count = data.get("eval_count", 0)
        speed = eval_count / elapsed if elapsed > 0 else 0

        # Cloud models may put output in thinking and/or response
        resp_text = data.get("response", "")
        thinking = data.get("thinking", "")
        if not resp_text.strip() and thinking.strip():
            resp_text = thinking

        log_msg(f"  Cloud: {eval_count} tokens, {elapsed:.1f}s, {speed:.0f} tok/s")
        return resp_text

    # Local Ollama
    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "think": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "num_ctx": 32768,
                },
            },
            timeout=3600,
        )
        response.raise_for_status()
    except requests.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_URL}. "
            f"Is Ollama running? Start it with: ollama serve"
        )
    except requests.Timeout:
        raise RuntimeError("Ollama request timed out after 3600 seconds.")
    except requests.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e}")

    data = response.json()
    if "response" not in data:
        raise RuntimeError(
            f"Unexpected Ollama response format. Keys: {list(data.keys())}"
        )

    resp_text = _strip_thinking(data["response"])
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 0)
    speed = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0

    log_msg(f"  Ollama: {eval_count} tokens, {eval_duration/1e9:.1f}s, {speed:.1f} tok/s")

    return resp_text


# ---------------------------------------------------------------------------
# Experiment execution + result parsing -- copied from runner_v2.py
# ---------------------------------------------------------------------------

def execute_experiment(script_path, timeout=900):
    """Run a Python script in a subprocess with timeout.

    Returns (success: bool, stdout: str, stderr: str).
    """
    try:
        result = subprocess.run(
            [sys.executable, "-u", os.path.basename(script_path)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_DIR,
        )
        return (result.returncode == 0, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (False, "", f"Timed out after {timeout}s (limit: {timeout}s)")
    except Exception as e:
        return (False, "", f"Execution error: {e}")


def parse_results(output):
    """Find last valid JSON line in stdout. Returns dict or None."""
    if not output or not output.strip():
        return None
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
# Program (LLM prompt) loader
# ---------------------------------------------------------------------------

def load_program():
    """Load the LLM JSON prompt from program_v3.md."""
    path = os.path.join(_DIR, "program_v3.md")
    with open(path, encoding="utf-8") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Population management
# ---------------------------------------------------------------------------

def get_population_v3(log, top_n=15):
    """Return top-N successful experiments sorted by mean_rank (ascending)."""
    successful = [
        e for e in log
        if e.get("status") == "success"
        and isinstance(e.get("results"), dict)
        and e.get("config") is not None
    ]
    # Optimize for optimal_ev: best EV achievable across buying top 1-20 tickets
    successful.sort(key=lambda x: x["results"].get("optimal_ev", -999.0), reverse=True)
    return successful[:top_n]


# ---------------------------------------------------------------------------
# LLM response helpers
# ---------------------------------------------------------------------------

def _strip_thinking(text):
    """Remove <think>...</think> reasoning blocks that local models may emit."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# LLM JSON extraction
# ---------------------------------------------------------------------------

def extract_config_json(response):
    """Extract a JSON config dict from LLM response. Returns dict or None."""
    cleaned = _strip_thinking(response)

    # Try ```json...``` block first
    m = re.search(r"```json\s*\n?(.*?)```", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try any ``` code block
    m = re.search(r"```\s*\n?(.*?)```", cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try every { position in the response -- handles preamble text before JSON
    decoder = json.JSONDecoder()
    for i, ch in enumerate(cleaned):
        if ch == "{":
            try:
                obj, _ = decoder.raw_decode(cleaned, i)
                if isinstance(obj, dict) and "models" in obj:
                    return obj
            except json.JSONDecodeError:
                pass

    return None


# ---------------------------------------------------------------------------
# Run one experiment
# ---------------------------------------------------------------------------

def run_one_experiment(config, exp_num, mode, parent_ids, model_name):
    """Validate config, render code, execute, return JSONL entry dict."""
    from config_validator import validate_config
    from template_engine import config_to_code

    # 1. Validate
    ok, errors = validate_config(config)
    if not ok:
        return {
            "experiment_id": exp_num,
            "timestamp": now_iso(),
            "mode": mode,
            "parent_ids": parent_ids,
            "config": config,
            "status": "error",
            "results": None,
            "description": config.get("description", ""),
            "error": f"Invalid config: {'; '.join(errors[:2])}",
            "execution_time_s": 0,
        }

    # 2. Render code
    code = config_to_code(config)
    full_code = BOILERPLATE + "\n\n" + code

    # 3. Write to disk
    with open(_EXPERIMENT_V3_FILE, "w", encoding="utf-8") as f:
        f.write(full_code)

    # 4. Execute
    t0 = time.time()
    success, stdout, stderr = execute_experiment(_EXPERIMENT_V3_FILE, timeout=900)
    t_exec = round(time.time() - t0, 2)

    # 5. Parse results
    results = parse_results(stdout)

    # 6. Build entry
    status = "success" if (success and results and results.get("mean_rank") is not None) else "error"
    if status == "error" and stderr:
        with open(_ERROR_LOG, "a", encoding="utf-8") as _ef:
            _ef.write(f"=== exp#{exp_num} {now_iso()} ===\n{stderr}\n")
    entry = {
        "experiment_id": exp_num,
        "timestamp": now_iso(),
        "mode": mode,
        "parent_ids": parent_ids,
        "config": config,
        "code": code,  # rendered code for debugging
        "status": status,
        "description": config.get("description", ""),
        "results": results if status == "success" else None,
        "execution_time_s": t_exec,
        "error": stderr if status == "error" else None,
    }
    return entry


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def run_bootstrap(log, model_name):
    """Run any bootstrap configs not already in the log."""
    from config_space import BOOTSTRAP_CONFIGS
    existing_descs = {e.get("description", "") for e in log}
    new_entries = []
    for i, cfg in enumerate(BOOTSTRAP_CONFIGS, 1):
        if cfg.get("description", "") in existing_descs:
            log_msg(f"  [bootstrap {i}/{len(BOOTSTRAP_CONFIGS)}] already done: {cfg['description'][:50]}")
            continue
        exp_num = len(log) + len(new_entries) + 1
        log_msg(f"  [bootstrap {i}/{len(BOOTSTRAP_CONFIGS)}] Running: {cfg['description'][:60]}")
        entry = run_one_experiment(cfg, exp_num, "bootstrap", [], model_name)
        mr = entry.get("results") or {}
        mr_str = f"mean_rank={mr.get('mean_rank', 'error'):.2f}" if mr.get("mean_rank") else "error"
        log_msg(f"    -> {mr_str}  ({entry['execution_time_s']}s)")
        new_entries.append(entry)
        append_experiment_entry(entry, _JSONL_V3)
    return new_entries


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _repair_llm_config(config):
    """
    Repair common LLM schema mistakes in-place before validation.
    Returns the (possibly modified) config dict.
    Repairs:
      - feature_sets: remove unknown set names, keep valid ones
      - pipeline: coerce None / non-list to []
      - models: coerce None / non-list to [] (validator will catch empty list)
      - pipeline stage dicts: ignore non-dict entries
    """
    from config_space import VALID_FEATURE_SETS
    import copy
    config = copy.deepcopy(config)

    # feature_sets: filter to valid only
    fs = config.get("feature_sets")
    if isinstance(fs, list):
        original = fs
        config["feature_sets"] = [f for f in fs if f in VALID_FEATURE_SETS]
        removed = set(original) - set(config["feature_sets"])
        if removed:
            log_msg(f"  [repair] Removed unknown feature_sets: {sorted(removed)}")
    elif fs is None:
        config["feature_sets"] = VALID_FEATURE_SETS[:]
        log_msg("  [repair] feature_sets was None -- defaulting to all sets")

    # pipeline: coerce to list of dicts
    pl = config.get("pipeline")
    if pl is None:
        config["pipeline"] = []
        log_msg("  [repair] pipeline was None -- set to []")
    elif not isinstance(pl, list):
        config["pipeline"] = []
        log_msg(f"  [repair] pipeline was {type(pl).__name__} -- set to []")
    else:
        config["pipeline"] = [s for s in pl if isinstance(s, dict)]

    # models: coerce to list of dicts
    ml = config.get("models")
    if ml is None:
        config["models"] = []
        log_msg("  [repair] models was None -- set to [] (will fail validation)")
    elif not isinstance(ml, list):
        config["models"] = []
        log_msg(f"  [repair] models was {type(ml).__name__} -- set to []")
    else:
        config["models"] = [m for m in ml if isinstance(m, dict)]

    # Clamp all numeric params to valid ranges (handles -1, sklearn conventions, etc.)
    from config_space import PARAM_RANGES, VALID_MODEL_TYPES, VALID_SCALE_METHODS
    from config_space import VALID_SELECTOR_TARGETS, VALID_CALIB_METHODS, VALID_CALIB_CVS, VALID_POLY_DEGREES

    def _clamp(val, lo, hi, typ):
        try:
            v = float(val)
            v = max(lo, min(hi, v))
            return int(round(v)) if typ == "int" else v
        except (TypeError, ValueError):
            return int(round((lo + hi) / 2)) if typ == "int" else (lo + hi) / 2

    # Clamp model numeric params
    model_param_map = {
        "n_estimators":    "model.n_estimators",
        "max_depth":       "model.max_depth",
        "min_samples_leaf":"model.min_samples_leaf",
        "min_samples_split":"model.min_samples_split",
        "max_features":    "model.max_features",
        "weight":          "model.weight",
        "learning_rate":   "model.learning_rate",
        "subsample":       "model.subsample",
        "colsample_bytree":"model.colsample_bytree",
        "reg_alpha":       "model.reg_alpha",
        "reg_lambda":      "model.reg_lambda",
        "C":               "model.C",
        "max_iter":        "model.max_iter",
    }
    for m in config.get("models", []):
        mtype = m.get("type", "")
        if mtype not in VALID_MODEL_TYPES:
            m["type"] = "et"  # fallback to known-good type
            log_msg(f"  [repair] Unknown model type '{mtype}' -> 'et'")
        for key, range_key in model_param_map.items():
            if key in m and range_key in PARAM_RANGES:
                typ, lo, hi = PARAM_RANGES[range_key]
                orig = m[key]
                m[key] = _clamp(orig, lo, hi, typ)
                if m[key] != orig:
                    log_msg(f"  [repair] model.{key}: {orig} -> {m[key]} (clamped to [{lo},{hi}])")
        # Clamp hgb learning_rate (stored as learning_rate_hgb in PARAM_RANGES)
        if m.get("type") == "hgb" and "learning_rate" in m:
            typ, lo, hi = PARAM_RANGES["model.learning_rate_hgb"]
            orig = m["learning_rate"]
            m["learning_rate"] = _clamp(orig, lo, hi, typ)

    # Clamp pipeline stage numeric params
    stage_param_map = {
        "select": {
            "threshold_multiplier": "select.threshold_multiplier",
            "selector_n_estimators": "select.selector_n_estimators",
            "selector_max_depth":    "select.selector_max_depth",
        },
        "poly": {
            "top_k_variance": "poly.top_k_variance",
        },
        "custom_interact": {
            "n_head": "custom_interact.n_head",
            "n_tail": "custom_interact.n_tail",
        },
    }
    for stage in config.get("pipeline", []):
        stype = stage.get("stage", "")
        if stype in stage_param_map:
            for key, range_key in stage_param_map[stype].items():
                if key in stage and range_key in PARAM_RANGES:
                    typ, lo, hi = PARAM_RANGES[range_key]
                    orig = stage[key]
                    stage[key] = _clamp(orig, lo, hi, typ)
                    if stage[key] != orig:
                        log_msg(f"  [repair] stage {stype}.{key}: {orig} -> {stage[key]} (clamped)")

    # Fill in missing required model fields using defaults for each model type.
    # MiniMax sometimes omits required params (e.g. max_features for et/rf).
    from config_space import default_model
    for m in config.get("models", []):
        mtype = m.get("type", "et")
        defaults = default_model(mtype)
        for key, val in defaults.items():
            if key not in m:
                m[key] = val
                log_msg(f"  [repair] model.{key} missing for type '{mtype}' -- set to default {val}")

    return config


def run_loop(max_experiments, model_name, resume, ga_only, llm_frac=0.20, allow_equipment=False, random_frac=0.10):
    """Main experiment loop."""
    from ga_ops import pick_mutation, crossover, select_parents

    log = load_log(_JSONL_V3) if resume else []

    log_msg(f"Loaded {len(log)} existing experiments")

    # Bootstrap phase
    log_msg("Running bootstrap seeds...")
    new_entries = run_bootstrap(log, model_name)
    log.extend(new_entries)

    program = load_program()
    if not allow_equipment:
        # Strip equipment from the prompt so MiniMax doesn't propose it
        program = "\n".join(
            line for line in program.splitlines()
            if "equipment" not in line.lower()
        )

    start_num = len(log) + 1
    end_num   = start_num + max_experiments - 1
    log_msg(f"Starting main loop: experiments {start_num} to {end_num} ({max_experiments} to run)")

    for exp_num in range(start_num, end_num + 1):
        population = get_population_v3(log, top_n=15)

        # Random injection: fires first, independently of GA/LLM split.
        # Keeps the full space sampled throughout the run so the GA cannot
        # re-collapse into a single basin after corner seeding.
        roll = random.random()
        use_random = (not ga_only) and (roll < random_frac)
        use_ga = (not use_random) and (len(population) >= 2) and (ga_only or random.random() > llm_frac)

        if use_random:
            from config_space import generate_random_config
            config = generate_random_config(allow_equipment=allow_equipment)
            config["description"] = "RANDOM: " + config["description"]
            mode = "random"
            parent_ids = []

        elif use_ga:
            parent_a, parent_b = select_parents(population, top_n=15)
            parent_ids = [parent_a["experiment_id"]]

            if parent_b and random.random() < 0.30:
                config = crossover(parent_a["config"], parent_b["config"])
                parent_ids.append(parent_b["experiment_id"])
                # 50% chance: mutate the crossover child immediately
                if random.random() < 0.50:
                    config = pick_mutation(config)
                    config["description"] = (
                        f"CROSSOVER+MUTATE #{parent_a['experiment_id']}"
                        f" x #{parent_b['experiment_id']}"
                    )
                    mode = "ga_crossover_mutate"
                else:
                    config["description"] = (
                        f"CROSSOVER from #{parent_a['experiment_id']}"
                        f" x #{parent_b['experiment_id']}"
                    )
                    mode = "ga_crossover"
            else:
                config = pick_mutation(parent_a["config"])
                config["description"] = (
                    f"MUTATION from #{parent_a['experiment_id']}: "
                    f"{config.get('description', '')[:60]}"
                )
                mode = "ga_mutation"

        else:
            # LLM exploration: ask for JSON config
            mode = "llm"
            parent_ids = []
            try:
                llm_prompt = program + "\n\n---\nOUTPUT ONLY THE JSON CONFIG. Start your response with { and end with }. No explanation, no preamble, no markdown fences."
                response = query_ollama(llm_prompt, model_name, temperature=0.8, max_tokens=4000)
                config = extract_config_json(response)
                if config is None:
                    log_msg(f"  LLM response preview: {response[:200].encode('ascii','replace').decode('ascii')}")
                    log_msg("  LLM returned no valid JSON -- retrying with correction prompt...")
                    correction = (
                        "Your previous response did not contain valid JSON.\n"
                        f"Your response was:\n{response[:800]}\n\n"
                        "Please output ONLY the JSON config object. "
                        "Nothing before it, nothing after it. "
                        "Start with { and end with }."
                    )
                    response2 = query_ollama(correction, model_name, temperature=0.3, max_tokens=4000)
                    config = extract_config_json(response2)
                    if config is None:
                        log_msg(f"  Retry preview: {response2[:200].encode('ascii','replace').decode('ascii')}")
                        log_msg("  LLM retry also failed -- falling back to random config")
                        from config_space import generate_random_config
                        config = generate_random_config(allow_equipment=allow_equipment)
                        config["description"] = "random-fallback: " + config["description"]
                        mode = "random"
                    log_msg("  LLM retry succeeded.")
                # Repair common LLM schema mistakes before validating
                config = _repair_llm_config(config)
                from config_validator import validate_config
                ok, errors = validate_config(config)
                if not ok:
                    log_msg(f"  LLM config invalid after repair: {errors[:2]} -- skipping")
                    continue
                log_msg(f"  LLM config OK: {config.get('description', '')[:60]}")
            except Exception as e:
                log_msg(f"  LLM exploration error: {e} -- skipping")
                continue

        # Experiment header
        log_msg("")
        log_msg("=" * 60)
        log_msg(f"  EXPERIMENT {exp_num}/{end_num}  [{mode}]")
        log_msg("=" * 60)
        if mode == "llm":
            log_msg(f"  [LLM exploration] Querying {model_name}...")
        log_msg(f"  {config.get('description', '')[:80]}")

        entry = run_one_experiment(config, exp_num, mode, parent_ids, model_name)
        log.append(entry)
        append_experiment_entry(entry, _JSONL_V3)

        if entry["status"] == "success":
            mr   = entry["results"]["mean_rank"]
            oev  = entry["results"].get("optimal_ev", float("nan"))
            ok_  = entry["results"].get("optimal_k", "?")
            boev = entry["results"].get("box_optimal_ev", float("nan"))
            bok_ = entry["results"].get("box_optimal_k", "?")
            log_msg(f"  SUCCESS  rank={mr:.2f}  str_ev=${oev:.3f}(buy {ok_})  box_ev=${boev:.3f}(buy {bok_})  ({entry['execution_time_s']}s)")
        else:
            log_msg(f"  ERROR: {entry.get('error', '')[:100]}")

    log_msg("\nDone.")


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

def self_test():
    """Smoke-test: validate all bootstrap configs, render them, compile, run 50 mutations."""
    from config_space import DEFAULT_CONFIG, BOOTSTRAP_CONFIGS
    from config_validator import validate_config
    from template_engine import config_to_code
    from ga_ops import pick_mutation, crossover
    import copy

    passed = 0
    failed = 0

    def check(name, condition, msg=""):
        nonlocal passed, failed
        if condition:
            log_msg(f"  [PASS] {name}")
            passed += 1
        else:
            log_msg(f"  [FAIL] {name}: {msg}")
            failed += 1

    log_msg("Self-test starting...")

    # 1. Validate all bootstrap configs
    for i, cfg in enumerate(BOOTSTRAP_CONFIGS, 1):
        ok, errs = validate_config(cfg)
        check(f"Bootstrap {i} valid", ok, str(errs))

    # 2. Render and compile all bootstrap configs
    for i, cfg in enumerate(BOOTSTRAP_CONFIGS, 1):
        code = config_to_code(cfg)
        try:
            compile(code, f"<seed_{i}>", "exec")
            check(f"Bootstrap {i} compiles", True)
        except SyntaxError as e:
            check(f"Bootstrap {i} compiles", False, str(e))

    # 3. GA mutation smoke test (50 mutations)
    cfg_a = copy.deepcopy(DEFAULT_CONFIG)
    cfg_b = copy.deepcopy(BOOTSTRAP_CONFIGS[-1])
    mutation_ok = True
    for _ in range(50):
        mutated = pick_mutation(cfg_a)
        ok, _ = validate_config(mutated)
        if not ok:
            mutation_ok = False
            break
    check("50 mutations all valid", mutation_ok)

    # 4. Crossover
    crossed = crossover(cfg_a, cfg_b)
    ok, errs = validate_config(crossed)
    check("Crossover valid", ok, str(errs))

    # 5. Original not modified
    check("Mutations don't modify original", cfg_a == DEFAULT_CONFIG)

    log_msg(f"\nSelf-test complete: {passed} passed, {failed} failed")
    if failed:
        sys.exit(1)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="LottAI AutoResearch Runner v3")
    parser.add_argument("--max", type=int, default=DEFAULT_MAX)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ga-only", action="store_true", help="100%% GA mode, no LLM calls")
    parser.add_argument("--equipment", action="store_true",
                        help="Enable equipment features in LLM proposals and random configs (default off)")
    parser.add_argument("--llm-frac", type=float, default=0.20,
                        help="Fraction of experiments to fill with LLM/random exploration (default 0.20). "
                             "Set to 1.0 for pure exploration to seed a diverse initial population.")
    parser.add_argument("--random-frac", type=float, default=0.10,
                        help="Fraction of experiments that are fully random configs, drawn independently "
                             "of GA/LLM split (default 0.10). Keeps full space sampled to prevent "
                             "re-collapse into a single basin after corner seeding.")
    parser.add_argument("--self-test", action="store_true", help="Run smoke tests and exit")
    parser.add_argument("--render", metavar="CONFIG_JSON", help="Render a config file to stdout and exit")
    args = parser.parse_args()

    if args.render:
        from config_validator import validate_config
        from template_engine import config_to_code
        with open(args.render) as f:
            cfg = json.load(f)
        ok, errs = validate_config(cfg)
        if not ok:
            print(f"Invalid config: {errs}")
            sys.exit(1)
        print(config_to_code(cfg))
        return

    if args.self_test:
        self_test()
        return

    # Print header
    log_msg("")
    log_msg("=" * 60)
    log_msg("  LottAI AutoResearch Runner v3")
    log_msg("=" * 60)
    log_msg(f"  Model:    {args.model}")
    log_msg(f"  Max exp:  {args.max}")
    log_msg(f"  Resume:   {args.resume}")
    log_msg(f"  GA-only:  {args.ga_only}")
    log_msg(f"  LLM-frac: {args.llm_frac}")
    log_msg(f"  Rand-frac: {args.random_frac}")
    log_msg(f"  Equipment: {'ON' if args.equipment else 'OFF'}")
    log_msg("=" * 60)
    log_msg("")

    run_loop(
        max_experiments=args.max,
        model_name=args.model,
        resume=args.resume,
        ga_only=args.ga_only,
        llm_frac=args.llm_frac,
        random_frac=args.random_frac,
        allow_equipment=args.equipment,
    )


if __name__ == "__main__":
    main()
