"""
runner_v2.py - Upgraded autoresearch runner using qwen3-coder:30b.

Key improvements over v1:
- Uses qwen3-coder:30b (much better code quality than 7b)
- Seeded with 10 working experiment templates from strategic batch
- Results from prior experiments feed into prompts
- Longer timeouts for larger model inference
- Separate log file (experiments_v2.jsonl) from v1's data

Usage:
    python -u runner_v2.py                    # Fresh start, 100 experiments
    python -u runner_v2.py --resume           # Resume from existing log
    python -u runner_v2.py --max 50           # Limit to 50 experiments
    python -u runner_v2.py --model deepseek-r1:14b  # Use different model
    python -u runner_v2.py --self-test        # Verify setup without running
"""

import os
import sys
import json
import re
import time
import subprocess
import argparse
from datetime import datetime, timezone

import socket
import ssl
import http.client
import requests
from dotenv import load_dotenv

# All paths relative to this file's directory
_DIR = os.path.dirname(os.path.abspath(__file__))

# Load .env from project root (contains OLLAMA_API_KEY etc.)
load_dotenv(os.path.join(os.path.dirname(_DIR), ".env"))

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CLOUD_HOST = "ollama.com"
DEFAULT_MODEL = "minimax-m2.7:cloud"
_PROGRAM_FILE = os.path.join(_DIR, "program_v2.md")
_EXPERIMENT_FILE = os.path.join(_DIR, "experiment_v2.py")
_LOG_FILE = os.path.join(_DIR, "experiments_v2.jsonl")
_AUTORESEARCH_LOG = os.path.join(_DIR, "autoresearch_v2.log")

# Boilerplate prepended to every experiment (same as v1)
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
    """Print and append to log file."""
    # Sanitize for Windows cp1252 console
    safe_msg = msg.encode("ascii", errors="replace").decode("ascii")
    print(safe_msg)
    with open(_AUTORESEARCH_LOG, "a", encoding="utf-8") as f:
        f.write(msg + "\n")


# ---------------------------------------------------------------------------
# Ollama Integration (local + cloud)
# ---------------------------------------------------------------------------

def _is_cloud_model(model):
    """Check if model is a cloud model (e.g. minimax-m2.7:cloud)."""
    return model.endswith(":cloud")


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

    # Parse JSON — handle control chars, extra data, chunked leftovers
    def _try_parse(text):
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
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=900,
        )
        response.raise_for_status()
    except requests.ConnectionError:
        raise ConnectionError(
            f"Cannot connect to Ollama at {OLLAMA_URL}. "
            f"Is Ollama running? Start it with: ollama serve"
        )
    except requests.Timeout:
        raise RuntimeError("Ollama request timed out after 900 seconds.")
    except requests.HTTPError as e:
        raise RuntimeError(f"Ollama HTTP error: {e}")

    data = response.json()
    if "response" not in data:
        raise RuntimeError(
            f"Unexpected Ollama response format. Keys: {list(data.keys())}"
        )

    resp_text = data["response"]
    eval_count = data.get("eval_count", 0)
    eval_duration = data.get("eval_duration", 0)
    speed = eval_count / (eval_duration / 1e9) if eval_duration > 0 else 0

    log_msg(f"  Ollama: {eval_count} tokens, {eval_duration/1e9:.1f}s, {speed:.1f} tok/s")

    return resp_text


def check_ollama_connectivity(model):
    """Verify Ollama (local or cloud) is running and the model is available."""
    if _is_cloud_model(model):
        # Test cloud connectivity directly
        try:
            data = _cloud_request({
                "model": model,
                "prompt": "Reply with only the word 'ready'.",
                "stream": False,
                "think": False,
                "options": {"temperature": 0.0, "num_predict": 10},
            }, timeout=30)
            resp_text = data.get("response", "") or data.get("thinking", "")
            eval_count = data.get("eval_count", 0)
            log_msg(f"  Cloud: {eval_count} tokens")
            if not resp_text.strip():
                return False, "Cloud returned empty response."
            return True, f"OK. Model: {model} (cloud). Response: '{resp_text.strip()[:50]}'"
        except Exception as e:
            return False, f"Cloud connectivity failed: {e}"

    # Local Ollama check
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.ConnectionError:
        return False, "Cannot connect to Ollama at localhost:11434."
    except Exception as e:
        return False, f"Ollama connectivity check failed: {e}"

    try:
        tags = resp.json()
        model_names = [m.get("name", "") for m in tags.get("models", [])]
        model_found = any(
            model in name or name.startswith(model.split(":")[0])
            for name in model_names
        )
        if not model_found:
            return False, (
                f"Model '{model}' not found. Available: {model_names}. "
                f"Pull with: ollama pull {model}"
            )
    except Exception as e:
        return False, f"Error checking model availability: {e}"

    # Quick test
    try:
        test_resp = query_ollama(
            "Reply with only the word 'ready'.",
            model=model, temperature=0.0, max_tokens=10,
        )
        if not test_resp.strip():
            return False, "Ollama returned empty response."
    except Exception as e:
        return False, f"Ollama test failed: {e}"

    return True, f"OK. Model: {model}. Response: '{test_resp.strip()[:50]}'"


# ---------------------------------------------------------------------------
# File / Log Helpers
# ---------------------------------------------------------------------------

def read_file(path):
    """Read a text file, return contents or None."""
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def write_file(path, content):
    """Write content to file (overwrite)."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def now_iso():
    """UTC timestamp in ISO 8601."""
    return datetime.now(timezone.utc).isoformat()


def load_experiment_log():
    """Load experiment log from JSONL file."""
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
            except json.JSONDecodeError:
                log_msg(f"WARNING: Skipping malformed line {line_num}")
    return entries


def append_experiment_entry(entry):
    """Append one experiment entry to the JSONL log."""
    with open(_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


# ---------------------------------------------------------------------------
# Code Extraction
# ---------------------------------------------------------------------------

def extract_code(ai_response):
    """Extract Python code from AI response.

    Handles qwen3's <think>...</think> tags by stripping them first.
    Then looks for ```python...``` code blocks.
    """
    # Strip thinking tags if present (qwen3 thinking mode)
    cleaned = re.sub(r"<think>.*?</think>", "", ai_response, flags=re.DOTALL)

    # Look for ```python...``` blocks
    pattern = r"```python\s*\n(.*?)```"
    matches = re.findall(pattern, cleaned, re.DOTALL)

    if matches:
        code = max(matches, key=len).strip()
    else:
        # Try generic code blocks
        pattern_generic = r"```\s*\n(.*?)```"
        matches_generic = re.findall(pattern_generic, cleaned, re.DOTALL)
        if matches_generic:
            code = max(matches_generic, key=len).strip()
        else:
            # Try raw Python detection
            lines = cleaned.split("\n")
            code_lines = []
            in_code = False
            for line in lines:
                stripped = line.strip()
                if stripped.startswith(("# DESCRIPTION", "X_train", "X_val",
                                       "clf", "prob_matrix", "results",
                                       "from ", "import ", "def ", "for ")):
                    in_code = True
                if in_code:
                    code_lines.append(line)
            if code_lines:
                code = "\n".join(code_lines).strip()
            else:
                return None

    # Validation
    if len(code.strip().split("\n")) < 3:
        log_msg("  VALIDATION FAIL: Code too short (< 3 lines)")
        return None

    # Strip any imports that duplicate boilerplate
    lines = code.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        # Skip lines that re-import boilerplate items
        if stripped.startswith("import sys") or stripped.startswith("import os"):
            continue
        if stripped.startswith("import numpy") or stripped.startswith("import json"):
            continue
        if stripped.startswith("import warnings"):
            continue
        if stripped.startswith("from prepare import"):
            continue
        if stripped.startswith("from sklearn") and "import" in stripped:
            # Keep sklearn imports that might be NEW (not in boilerplate)
            boilerplate_imports = [
                "RandomForestClassifier", "GradientBoostingClassifier",
                "LogisticRegression", "GaussianNB", "KNeighborsClassifier",
                "MinMaxScaler", "StandardScaler", "DecisionTreeClassifier",
                "ExtraTreesClassifier", "CalibratedClassifierCV",
            ]
            # Check if ALL imported names are already in boilerplate
            import_names = stripped.split("import")[-1].strip().split(",")
            import_names = [n.strip() for n in import_names]
            all_in_boilerplate = all(
                any(bn in n for bn in boilerplate_imports)
                for n in import_names
            )
            if all_in_boilerplate:
                continue
        if "get_train_val_test()" in stripped and "=" in stripped:
            continue
        if "load_data()" in stripped and "full_df" in stripped:
            continue
        filtered.append(line)

    code = "\n".join(filtered).strip()

    if len(code.strip().split("\n")) < 3:
        log_msg("  VALIDATION FAIL: Code too short after filtering")
        return None

    return code


def extract_description(ai_response):
    """Extract description from AI response."""
    # Strip thinking tags
    cleaned = re.sub(r"<think>.*?</think>", "", ai_response, flags=re.DOTALL)

    for line in cleaned.split("\n"):
        stripped = line.strip()
        if stripped.startswith("# DESCRIPTION"):
            return stripped.replace("# DESCRIPTION", "").strip(" :-")
        if stripped.startswith("# ") and len(stripped) > 3 and "```" not in stripped:
            return stripped[2:].strip()

    for line in cleaned.split("\n"):
        stripped = line.strip()
        if stripped and not stripped.startswith(("```", "<", "import", "from")):
            return stripped[:120]

    return "No description"


# ---------------------------------------------------------------------------
# Static Code Verification
# ---------------------------------------------------------------------------

# Functions/variables that exist in the boilerplate
BOILERPLATE_NAMES = {
    "np", "json", "warnings", "sys", "os",
    "prepare", "load_data", "get_train_val_test", "build_features",
    "evaluate_predictions", "NUM_COMBOS", "combo_to_digits",
    "digits_to_combo", "FEATURE_DIMS", "load_cached_features",
    "RandomForestClassifier", "GradientBoostingClassifier",
    "ExtraTreesClassifier", "HistGradientBoostingClassifier",
    "LogisticRegression", "GaussianNB",
    "KNeighborsClassifier", "MinMaxScaler", "StandardScaler",
    "DecisionTreeClassifier", "CalibratedClassifierCV",
    "PolynomialFeatures",
    "scipy_entropy", "xgb", "lgb", "cb", "hmm", "sm",
    "torch", "nn", "MLPClassifier", "GaussianMixture",
    "BayesianGaussianMixture", "IsolationForest", "TSNE",
    "PolynomialFeatures", "QuantileTransformer", "RobustScaler",
    "SelectKBest", "mutual_info_classif", "f_classif", "entropy",
    "train_df", "val_df", "test_df", "full_df",
    "train_indices", "val_indices",
    "y_train", "y_val", "n_val", "n_train",
    "y_train_d1", "y_train_d2", "y_train_d3",
    "y_val_d1", "y_val_d2", "y_val_d3",
}

# Functions that DO NOT exist but the model hallucinates
HALLUCINATED_NAMES = {
    "load_X_train", "load_X_val", "load_train_features",
    "load_val_features", "get_features", "load_features",
    "prepare_features", "get_cached_features",
}

# Imports that are NOT available — catch before dry-run
BANNED_IMPORTS = {
    "tensorflow", "tf", "keras",
}

# Known deprecated API patterns -> fixes
API_FIXES = {
    "base_estimator=": "estimator=",
    "use_label_encoder=False": "",  # removed in newer xgboost
}


def verify_code(code):
    """Static verification of generated experiment code.

    Returns (is_valid, errors_list).
    Each error is a string describing the problem.
    """
    errors = []

    # 1. Syntax check via compile()
    full_code = BOILERPLATE + code
    try:
        compile(full_code, "<experiment>", "exec")
    except SyntaxError as e:
        errors.append(f"SyntaxError on line {e.lineno}: {e.msg}")
        return False, errors

    # 2. Check for hallucinated function calls
    for bad_name in HALLUCINATED_NAMES:
        if bad_name in code:
            errors.append(
                f"Undefined function '{bad_name}' used. "
                f"Use load_cached_features('train', [...]) and "
                f"load_cached_features('val', [...]) instead."
            )

    # 2b. Check for unavailable imports
    for banned in BANNED_IMPORTS:
        if f"import {banned}" in code or f"from {banned}" in code:
            errors.append(
                f"Module '{banned}' is NOT installed. Use only: "
                f"numpy, scipy, sklearn, xgboost (already imported in boilerplate)."
            )

    # 3. Check for deprecated API usage
    if "base_estimator=" in code:
        errors.append(
            "Deprecated API: use 'estimator=' instead of 'base_estimator=' "
            "in CalibratedClassifierCV (changed in sklearn 1.4+)."
        )

    # 4. Check that code produces required outputs
    if "prob_matrix" not in code:
        errors.append("Code must create a 'prob_matrix' variable.")
    if "evaluate_predictions" not in code:
        errors.append("Code must call evaluate_predictions(prob_matrix, y_val).")
    if "print(json.dumps(" not in code and "print(json.dumps (" not in code:
        errors.append("Last line must be: print(json.dumps(results))")

    # 5. Check for redefinition of boilerplate variables
    redefinition_patterns = [
        ("get_train_val_test()", "Do not call get_train_val_test() - already done in boilerplate"),
        ("= load_data()", "Do not call load_data() - full_df already available"),
        ("y_val = ", "Do not redefine y_val - already loaded in boilerplate"),
        ("n_val = ", "Do not redefine n_val - already loaded in boilerplate"),
    ]
    for pattern, msg in redefinition_patterns:
        # Only flag if it's a standalone assignment, not inside a function
        for line in code.split("\n"):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if pattern in stripped and not stripped.startswith("# "):
                errors.append(msg)
                break

    # 6. AST-based checks: undefined variables, invalid feature sets, variable assignments
    import ast as _ast
    valid_feature_sets = {"basic", "recency", "gaps", "positional", "temporal", "momentum"}

    try:
        tree = _ast.parse(full_code)

        # Collect all names assigned at top level in the experiment code
        # (boilerplate assignments are already in BOILERPLATE_NAMES)
        assigned_names = set(BOILERPLATE_NAMES)
        for node in _ast.walk(tree):
            # Assignments: X_train = ...
            if isinstance(node, _ast.Assign):
                for target in node.targets:
                    if isinstance(target, _ast.Name):
                        assigned_names.add(target.id)
                    elif isinstance(target, _ast.Tuple):
                        for elt in target.elts:
                            if isinstance(elt, _ast.Name):
                                assigned_names.add(elt.id)
            # Augmented assignments: x += ...
            elif isinstance(node, _ast.AugAssign):
                if isinstance(node.target, _ast.Name):
                    assigned_names.add(node.target.id)
            # For loop targets: for i in ...
            elif isinstance(node, _ast.For):
                if isinstance(node.target, _ast.Name):
                    assigned_names.add(node.target.id)
                elif isinstance(node.target, _ast.Tuple):
                    for elt in node.target.elts:
                        if isinstance(elt, _ast.Name):
                            assigned_names.add(elt.id)
            # Comprehension variables
            elif isinstance(node, _ast.comprehension):
                if isinstance(node.target, _ast.Name):
                    assigned_names.add(node.target.id)
            # Import names
            elif isinstance(node, _ast.Import):
                for alias in node.names:
                    assigned_names.add(alias.asname or alias.name)
            elif isinstance(node, _ast.ImportFrom):
                for alias in node.names:
                    assigned_names.add(alias.asname or alias.name)
            # Function/class defs
            elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                assigned_names.add(node.name)
            elif isinstance(node, _ast.ClassDef):
                assigned_names.add(node.name)
            # With ... as x
            elif isinstance(node, _ast.withitem):
                if node.optional_vars and isinstance(node.optional_vars, _ast.Name):
                    assigned_names.add(node.optional_vars.id)

        # Check for common undefined variable patterns
        # Focus on X_train/X_val which are the most frequent offenders
        for var_name in ["X_train", "X_val"]:
            if var_name not in assigned_names:
                # Check if it's actually used in the experiment code (not just boilerplate)
                used_in_code = False
                for line in code.split("\n"):
                    stripped = line.strip()
                    if stripped.startswith("#"):
                        continue
                    if var_name in stripped:
                        # Make sure it's not the assignment itself
                        if "=" in stripped and var_name in stripped.split("=")[0]:
                            break  # It IS assigned, just our AST check missed it
                        used_in_code = True
                        break
                if used_in_code:
                    split_name = "train" if "train" in var_name.lower() else "val"
                    errors.append(
                        f"'{var_name}' is used but never assigned. Add: "
                        f"{var_name} = load_cached_features('{split_name}', "
                        f"['basic', 'recency', 'gaps', 'positional', 'temporal', 'momentum'])"
                    )

        # Validate feature set names in load_cached_features calls
        for node in _ast.walk(tree):
            if (isinstance(node, _ast.Call) and
                isinstance(node.func, _ast.Name) and
                node.func.id == "load_cached_features"):
                if len(node.args) >= 2 and isinstance(node.args[1], _ast.List):
                    for elt in node.args[1].elts:
                        if isinstance(elt, _ast.Constant) and isinstance(elt.value, str):
                            if elt.value not in valid_feature_sets:
                                errors.append(
                                    f"Invalid feature set name '{elt.value}'. "
                                    f"Valid: {', '.join(sorted(valid_feature_sets))}"
                                )
    except Exception:
        pass  # AST parsing issues, skip these checks

    # 7. Check for common normalization mistake (multiplying prob_matrices)
    if code.count("prob_matrix *=") > 1 or "prob_matrix = prob_matrix *" in code:
        # Multiple multiplicative updates can cause underflow
        if "np.clip" not in code and "normalize" not in code.lower():
            errors.append(
                "Multiple multiplicative prob_matrix updates without normalization "
                "will cause underflow. Add normalization after combining."
            )

    return len(errors) == 0, errors


def auto_fix_code(code):
    """Apply automatic fixes for known issues. Returns fixed code."""
    # Fix deprecated sklearn API
    if "base_estimator=" in code:
        code = code.replace("base_estimator=", "estimator=")

    # Fix use_label_encoder (removed in newer xgboost)
    code = code.replace("use_label_encoder=False, ", "")
    code = code.replace("use_label_encoder=False,", "")
    code = code.replace(", use_label_encoder=False", "")
    code = code.replace("use_label_encoder=False", "")

    # Fix function-wrapped code: only unwrap if there is EXACTLY ONE top-level
    # function and no top-level executable statements outside it.
    # Helper functions (called from top-level code) must NOT be stripped.
    lines = code.split("\n")
    top_level_defs = [
        i for i, line in enumerate(lines)
        if line.startswith("def ") and "(" in line and "):" in line
    ]
    if len(top_level_defs) == 1:
        # Check that there are no meaningful executable statements outside the function
        func_start = top_level_defs[0]
        outside_lines = [
            line for line in lines[:func_start]
            if line.strip() and not line.strip().startswith("#")
        ]
        if not outside_lines:
            # Safe to unwrap — entire experiment is wrapped in one function
            unwrapped = []
            in_func = False
            func_indent = 0
            for line in lines:
                stripped = line.strip()
                if line.startswith("def ") and "):" in stripped:
                    in_func = True
                    func_indent = 4
                    continue
                if in_func:
                    if stripped.startswith("return "):
                        continue
                    if stripped == "return":
                        continue
                    if line.startswith(" " * func_indent):
                        unwrapped.append(line[func_indent:])
                    elif line.startswith("\t"):
                        unwrapped.append(line[1:])
                    else:
                        unwrapped.append(line)
                else:
                    unwrapped.append(line)
            if unwrapped:
                code = "\n".join(unwrapped)

    # Fix: remove any if __name__ == "__main__" blocks
    lines = code.split("\n")
    filtered = []
    skip_main = False
    for line in lines:
        if line.strip().startswith("if __name__"):
            skip_main = True
            continue
        if skip_main:
            if line.strip() and not line.startswith((" ", "\t")):
                skip_main = False
            else:
                continue
        if not skip_main:
            filtered.append(line)
    code = "\n".join(filtered)

    # Fix: replace hallucinated function names
    code = code.replace("load_X_train(", "load_cached_features(\"train\", ")
    code = code.replace("load_X_val(", "load_cached_features(\"val\", ")

    # Fix: common feature set name typos
    code = code.replace('"gap"', '"gaps"')
    code = code.replace("'gap'", "'gaps'")

    return code


def build_repair_prompt(code, errors, attempt):
    """Build a concise prompt asking the model to fix verification errors."""
    error_text = "\n".join(f"  - {e}" for e in errors[:3])

    # Only include the problematic portion of code (first 80 lines max)
    code_lines = code.split("\n")
    if len(code_lines) > 80:
        code_preview = "\n".join(code_lines[:80]) + "\n# ... (truncated)"
    else:
        code_preview = code

    return f"""Fix this code. Errors:
{error_text}

Code:
```python
{code_preview}
```

Rules: no imports (boilerplate provides everything), no function wrappers, top-level code only.
Available: load_cached_features("train"/"val", [...]), y_train, y_val, n_val, combo_to_digits(), evaluate_predictions(), json.dumps.
Use estimator= (not base_estimator=). Normalize prob_matrix rows. Last line: print(json.dumps(results))
Output fixed code in ```python``` block. Attempt {attempt}/3.
"""


# ---------------------------------------------------------------------------
# Experiment Execution
# ---------------------------------------------------------------------------

def execute_experiment(timeout=600):
    """Run experiment_v2.py in a subprocess with timeout."""
    try:
        result = subprocess.run(
            [sys.executable, "-u", os.path.basename(_EXPERIMENT_FILE)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_DIR,
        )
        return (result.returncode == 0, result.stdout, result.stderr)
    except subprocess.TimeoutExpired:
        return (False, "", f"Timed out after {timeout}s (limit: 600s)")
    except Exception as e:
        return (False, "", f"Execution error: {e}")


def parse_results(output):
    """Parse JSON results from experiment stdout (last valid JSON line)."""
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
# Prompt Construction
# ---------------------------------------------------------------------------

import random as _random

# GA mutation types
MUTATION_TYPES = [
    "hyperparameter",   # tweak n_estimators, max_depth, learning_rate, etc.
    "feature_set",      # add/remove/swap feature sets
    "model_swap",       # change classifier type (RF->ExtraTrees->GBM->LR)
    "interaction",      # add/remove polynomial or interaction features
    "calibration",      # add/remove/change calibration method
    "ensemble",         # combine with another parent's approach
]


def find_best_experiment(log):
    """Find best experiment by mean_rank."""
    successful = [
        e for e in log
        if e.get("status") == "success" and e.get("results")
    ]
    if not successful:
        return None
    return min(successful, key=lambda x: x["results"].get("mean_rank", 999))


def get_population(log, top_n=10):
    """Get the top-N experiments with archived code (the 'population')."""
    successful = [
        e for e in log
        if e.get("status") == "success" and e.get("results") and e.get("code")
    ]
    if not successful:
        return []
    # Sort by fitness (lower mean_rank = better)
    successful.sort(key=lambda x: x["results"].get("mean_rank", 999))
    return successful[:top_n]


def select_parents(population):
    """Fitness-proportional selection of 1-2 parents from population.

    Uses tournament selection: pick 3 random, take the best.
    Returns (parent1, parent2_or_None).
    """
    if not population:
        return None, None
    if len(population) == 1:
        return population[0], None

    # Tournament selection for parent 1
    candidates = _random.sample(population, min(3, len(population)))
    parent1 = min(candidates, key=lambda x: x["results"]["mean_rank"])

    # Tournament selection for parent 2 (different from parent 1)
    remaining = [p for p in population if p["experiment_id"] != parent1["experiment_id"]]
    if not remaining:
        return parent1, None
    candidates2 = _random.sample(remaining, min(3, len(remaining)))
    parent2 = min(candidates2, key=lambda x: x["results"]["mean_rank"])

    return parent1, parent2


def build_ga_prompt(program, log, exp_num, parent1, parent2, mutation_type):
    """Build a GA-style prompt: evolve from parent(s) with a specific mutation."""

    p1_id = parent1["experiment_id"]
    p1_rank = parent1["results"]["mean_rank"]
    p1_desc = parent1.get("description", "")
    p1_code = parent1.get("code", "# code not available")

    # Truncate code if too long
    if len(p1_code) > 2000:
        p1_code = p1_code[:2000] + "\n# ... (truncated)"

    if parent2 and mutation_type == "ensemble":
        p2_id = parent2["experiment_id"]
        p2_rank = parent2["results"]["mean_rank"]
        p2_desc = parent2.get("description", "")
        p2_code = parent2.get("code", "# code not available")
        if len(p2_code) > 2000:
            p2_code = p2_code[:2000] + "\n# ... (truncated)"

        crossover_section = f"""
## CROSSOVER: Combine these two parent experiments

PARENT A (Experiment #{p1_id}, mean_rank={p1_rank:.2f}):
Description: {p1_desc}
```python
{p1_code}
```

PARENT B (Experiment #{p2_id}, mean_rank={p2_rank:.2f}):
Description: {p2_desc}
```python
{p2_code}
```

Take the BEST elements from BOTH parents and combine them into a single
experiment. For example: use Parent A's model with Parent B's feature
engineering, or ensemble their probability outputs.
"""
    else:
        mutation_instructions = {
            "hyperparameter": (
                "MUTATE the hyperparameters. Change n_estimators, max_depth, "
                "min_samples_leaf, learning_rate, max_features, or other "
                "model parameters. Try values significantly different from the parent. "
                "Small tweaks AND large jumps are both valuable."
            ),
            "feature_set": (
                "MUTATE the feature sets. The parent uses certain feature sets from: "
                "basic, recency, gaps, positional, temporal, momentum. "
                "Try ADDING a set the parent doesn't use, REMOVING one it does, "
                "or using a completely different subset."
            ),
            "model_swap": (
                "SWAP the classifier for a different one. If the parent uses "
                "ExtraTreesClassifier, try RandomForestClassifier, "
                "GradientBoostingClassifier, LogisticRegression, or XGBoost. "
                "Keep the rest of the pipeline (features, calibration) the same."
            ),
            "interaction": (
                "MUTATE the feature engineering. Add polynomial interaction features "
                "(PolynomialFeatures degree=2 or 3), create custom interactions "
                "(feature products, ratios, differences), or REMOVE interactions "
                "if the parent uses them. Try np.column_stack with derived features."
            ),
            "calibration": (
                "MUTATE the probability calibration. Add CalibratedClassifierCV "
                "with isotonic or sigmoid method, or REMOVE calibration if parent "
                "uses it. Try different cv values (3, 5, 10)."
            ),
            "ensemble": (
                "Create an ENSEMBLE that includes this parent's approach plus "
                "a different model. Average their probability matrices. "
                "Try rank-based averaging or weighted averaging."
            ),
        }

        crossover_section = f"""
## MUTATION: Evolve from this parent experiment

PARENT (Experiment #{p1_id}, mean_rank={p1_rank:.2f}):
Description: {p1_desc}
```python
{p1_code}
```

MUTATION TYPE: {mutation_type.upper()}
{mutation_instructions.get(mutation_type, "Try something different.")}

Keep what works in the parent. Change ONLY the mutated aspect.
The goal is to BEAT the parent's mean_rank of {p1_rank:.2f}.
"""

    # Population summary for context
    population = get_population(log, top_n=5)
    if population:
        pop_text = "\n".join(
            f"  #{e['experiment_id']}: mean_rank={e['results']['mean_rank']:.2f} - "
            f"{e.get('description', '')[:60]}"
            for e in population
        )
    else:
        pop_text = "(no population yet)"

    prompt = f"""{program}

---

## Genetic Algorithm Mode - Experiment #{exp_num + 1}

GENERATION: {exp_num + 1}
POPULATION TOP 5:
{pop_text}

{crossover_section}

IMPORTANT: Your code will be STATICALLY VERIFIED before execution. The verifier checks:
- Syntax errors (compile check)
- Undefined functions (only use boilerplate functions)
- Deprecated APIs (use 'estimator=' not 'base_estimator=')
- Missing prob_matrix, evaluate_predictions, or print(json.dumps(results))
- Redefinition of boilerplate variables

Your primary objective is to write code that makes the verifier happy.

Requirements:
1. First line: # DESCRIPTION: [mutation_type] Evolved from #{p1_id}: brief description
2. Load features: X_train = load_cached_features("train", [...])
3. Load features: X_val = load_cached_features("val", [...])
4. Must produce prob_matrix of shape (n_val, 1000)
5. Must normalize rows to sum to 1.0 using:
   prob_matrix = np.clip(prob_matrix, 0, None)
   row_sums = prob_matrix.sum(axis=1, keepdims=True)
   row_sums[row_sums < 1e-15] = 1.0
   prob_matrix = prob_matrix / row_sums
6. results = evaluate_predictions(prob_matrix, y_val)
7. Last line: print(json.dumps(results))
8. Output code between ```python and ``` markers
"""
    return prompt


def build_prompt(program, log, exp_num):
    """Build the iteration prompt — alternates between exploration and GA evolution."""

    population = get_population(log, top_n=10)

    # GA mode: if we have archived code in the population, evolve 70% of the time
    if population and _random.random() < 0.7:
        parent1, parent2 = select_parents(population)
        if parent1:
            # Pick a mutation type
            if parent2 and _random.random() < 0.25:
                mutation_type = "ensemble"  # crossover
            else:
                mutation_type = _random.choice(MUTATION_TYPES[:-1])  # exclude ensemble

            log_msg(f"  GA MODE: {mutation_type} mutation from #{parent1['experiment_id']}"
                    + (f" x #{parent2['experiment_id']}" if mutation_type == "ensemble" and parent2 else ""))

            return build_ga_prompt(program, log, exp_num, parent1, parent2, mutation_type)

    # Exploration mode (original behavior)
    log_msg("  EXPLORATION MODE: free-form experiment")

    # Recent results (last 5)
    recent = log[-5:] if log else []
    if recent:
        recent_summaries = []
        for e in recent:
            s = {
                "id": e.get("experiment_id"),
                "status": e.get("status"),
                "description": e.get("description", ""),
            }
            if e.get("status") == "success" and e.get("results"):
                r = e["results"]
                s["mean_rank"] = r.get("mean_rank")
                s["rank_improvement"] = r.get("vs_baseline", {}).get("rank_improvement")
                s["top_10_hit"] = r.get("top_10_hit")
            elif e.get("status") == "error":
                s["error"] = e.get("error", "")[:200]
            recent_summaries.append(s)
        recent_text = json.dumps(recent_summaries, indent=2)
    else:
        recent_text = "(No previous experiments in this session. Start from a template.)"

    # Best so far
    best = find_best_experiment(log)
    if best:
        r = best["results"]
        vs = r.get("vs_baseline", {})
        rank_imp = vs.get("rank_improvement", 0)
        best_text = (
            f"Experiment #{best['experiment_id']}: {best.get('description', '')}\n"
            f"  mean_rank={r['mean_rank']:.2f}, "
            f"rank_improvement={rank_imp:+.2f}"
        )
    else:
        best_text = "No successful experiments yet. Start from Template A in program_v2.md."

    # All successful descriptions (to avoid repeating)
    tried = [
        e.get("description", "")
        for e in log
        if e.get("status") == "success"
    ]
    tried_text = "\n".join(f"  - {d}" for d in tried) if tried else "(none yet)"

    prompt = f"""{program}

---

## Current Session State

EXPERIMENT NUMBER: {exp_num + 1}
TOTAL EXPERIMENTS SO FAR: {len(log)}

RECENT RESULTS (last {len(recent)}):
{recent_text}

CURRENT BEST:
{best_text}

ALREADY TRIED:
{tried_text}

## Your Task

Write ONLY the experiment code (no imports, no data loading - boilerplate is prepended).
Base your code on one of the working templates in Section 6 above.
Try something DIFFERENT from what's already been tried.

IMPORTANT: Your code will be STATICALLY VERIFIED before execution. The verifier checks:
- Syntax errors (compile check)
- Undefined functions (only use boilerplate functions)
- Deprecated APIs (use 'estimator=' not 'base_estimator=')
- Missing prob_matrix, evaluate_predictions, or print(json.dumps(results))
- Redefinition of boilerplate variables

Your primary objective is to write code that makes the verifier happy.

Requirements:
1. First line: # DESCRIPTION: brief description of what this experiment does
2. Load features: X_train = load_cached_features("train", [...])
3. Load features: X_val = load_cached_features("val", [...])
4. Must produce prob_matrix of shape (n_val, 1000)
5. Must normalize rows to sum to 1.0 using:
   prob_matrix = np.clip(prob_matrix, 0, None)
   row_sums = prob_matrix.sum(axis=1, keepdims=True)
   row_sums[row_sums < 1e-15] = 1.0
   prob_matrix = prob_matrix / row_sums
6. results = evaluate_predictions(prob_matrix, y_val)
7. Last line: print(json.dumps(results))
8. Output code between ```python and ``` markers
"""
    return prompt


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_experiment_summary(entry):
    """Print formatted experiment result."""
    exp_id = entry.get("experiment_id", "?")
    status = entry.get("status", "unknown")
    desc = entry.get("description", "No description")

    log_msg(f"\n  Experiment #{exp_id}: {status.upper()}")
    log_msg(f"  Description: {desc[:100]}")

    if status == "error":
        error = entry.get("error", "Unknown error")
        log_msg(f"  Error: {error[:300]}")
    elif status == "success" and entry.get("results"):
        r = entry["results"]
        vs = r.get("vs_baseline", {})
        rank = r.get("mean_rank", 0)
        rank_imp = vs.get("rank_improvement", 0)
        direction = "BETTER" if rank_imp > 0 else "WORSE" if rank_imp < 0 else "SAME"

        log_msg(f"  Mean Rank:  {rank:.2f} ({abs(rank_imp):.2f} ranks {direction})")
        log_msg(f"  Brier:      {r.get('brier_score', 0):.6f}")
        log_msg(f"  Top-10 Hit: {r.get('top_10_hit', 0):.4f}")
        log_msg(f"  Top-50 Hit: {r.get('top_50_hit', 0):.4f}")
    else:
        log_msg("  (No results)")


def print_session_summary(log):
    """Print overall session summary."""
    successful = [e for e in log if e.get("status") == "success" and e.get("results")]
    failed = [e for e in log if e.get("status") == "error"]

    log_msg(f"\n{'='*60}")
    log_msg("  SESSION SUMMARY")
    log_msg(f"{'='*60}")
    log_msg(f"  Total:      {len(log)}")
    log_msg(f"  Successful: {len(successful)}")
    log_msg(f"  Failed:     {len(failed)}")
    log_msg(f"  Success rate: {len(successful)/max(len(log),1)*100:.0f}%")

    if successful:
        by_rank = sorted(successful, key=lambda x: x["results"]["mean_rank"])
        best = by_rank[0]
        log_msg(f"\n  BEST: #{best['experiment_id']} - {best.get('description', '')[:80]}")
        vs = best['results'].get('vs_baseline', {})
        log_msg(f"        mean_rank={best['results']['mean_rank']:.2f} "
                f"(rank_imp={vs.get('rank_improvement', 0):+.2f})")

        log_msg(f"\n  TOP 5:")
        for e in by_rank[:5]:
            r = e["results"]
            imp = r.get("vs_baseline", {}).get("rank_improvement", 0)
            log_msg(f"    #{e['experiment_id']:>3d}  rank={r['mean_rank']:.2f}  "
                    f"imp={imp:+.2f}  {e.get('description', '')[:50]}")

    log_msg(f"{'='*60}")


# ---------------------------------------------------------------------------
# Main Loop
# ---------------------------------------------------------------------------

def run_loop(model, max_experiments=100, temperature=0.7, resume=False):
    """Main autoresearch loop."""

    log_msg(f"\n{'='*60}")
    log_msg("  LottAI AutoResearch Runner v2")
    log_msg(f"{'='*60}")
    log_msg(f"  Model:          {model}")
    log_msg(f"  Max experiments: {max_experiments}")
    log_msg(f"  Temperature:     {temperature}")
    log_msg(f"  Resume:          {resume}")
    log_msg(f"  Log file:        {_LOG_FILE}")
    log_msg(f"{'='*60}")

    # Check Ollama
    log_msg("\nChecking Ollama...")
    ok, msg = check_ollama_connectivity(model)
    log_msg(f"  {msg}")
    if not ok:
        log_msg("ERROR: Cannot proceed. Exiting.")
        sys.exit(1)

    # Load program
    program = read_file(_PROGRAM_FILE)
    if program is None:
        log_msg(f"ERROR: {_PROGRAM_FILE} not found")
        sys.exit(1)
    log_msg(f"Loaded program ({len(program)} chars)")

    # Load or create log
    if resume:
        log = load_experiment_log()
        log_msg(f"Resumed: {len(log)} existing experiments")
    else:
        if os.path.exists(_LOG_FILE):
            backup = _LOG_FILE + f".backup.{int(time.time())}"
            os.rename(_LOG_FILE, backup)
            log_msg(f"Backed up existing log to: {backup}")
        log = []
        log_msg("Starting fresh.")

    start_exp = len(log)
    consecutive_failures = 0

    for exp_num in range(start_exp, max_experiments):
        log_msg(f"\n{'='*60}")
        log_msg(f"  EXPERIMENT {exp_num + 1}/{max_experiments}")
        log_msg(f"{'='*60}")

        # Safety valve
        if consecutive_failures >= 5:
            log_msg("WARNING: 5 consecutive failures. Pausing 10s...")
            time.sleep(10)
            consecutive_failures = 0

        # Build prompt
        prompt = build_prompt(program, log, exp_num)

        # Query model
        log_msg("  Querying model...")
        t0 = time.time()
        try:
            ai_response = query_ollama(prompt, model=model, temperature=temperature)
        except Exception as e:
            log_msg(f"  ERROR: {e}")
            entry = {
                "experiment_id": exp_num + 1,
                "timestamp": now_iso(),
                "status": "error",
                "error": f"Model query failed: {e}",
                "description": "Model communication error",
            }
            log.append(entry)
            append_experiment_entry(entry)
            consecutive_failures += 1
            continue
        t_query = time.time() - t0
        log_msg(f"  Response: {len(ai_response)} chars in {t_query:.1f}s")

        # Extract code
        code = extract_code(ai_response)
        description = extract_description(ai_response)
        if code is None:
            log_msg("  WARNING: Could not extract valid code")
            entry = {
                "experiment_id": exp_num + 1,
                "timestamp": now_iso(),
                "status": "error",
                "error": "Code extraction failed",
                "ai_response_preview": ai_response[:500],
                "description": description,
            }
            log.append(entry)
            append_experiment_entry(entry)
            consecutive_failures += 1
            continue

        # ── Verify + Repair Loop ────────────────────────────────────
        code = auto_fix_code(code)
        max_repairs = 3
        verified = False

        for attempt in range(max_repairs + 1):
            is_valid, errors = verify_code(code)

            if is_valid:
                if attempt > 0:
                    log_msg(f"  VERIFIED after {attempt} repair(s)")
                else:
                    log_msg("  VERIFIED (first pass)")
                verified = True
                break

            error_summary = "; ".join(errors[:3])
            log_msg(f"  VERIFY FAIL ({attempt+1}/{max_repairs+1}): {error_summary[:150]}")

            if attempt >= max_repairs:
                log_msg("  Max repairs reached, skipping experiment")
                break

            # Ask model to fix the code
            log_msg(f"  Requesting repair (attempt {attempt+1}/{max_repairs})...")
            repair_prompt = build_repair_prompt(code, errors, attempt + 1)
            try:
                repair_response = query_ollama(
                    repair_prompt, model=model,
                    temperature=max(0.3, temperature - 0.2),  # lower temp for repairs
                    max_tokens=6000,
                )
            except Exception as e:
                log_msg(f"  Repair query failed: {e}")
                break

            repaired_code = extract_code(repair_response)
            if repaired_code is None:
                log_msg("  Could not extract repaired code")
                break

            code = auto_fix_code(repaired_code)
            # Update description if the repair changed it
            new_desc = extract_description(repair_response)
            if new_desc and new_desc != "No description":
                description = new_desc

        if not verified:
            entry = {
                "experiment_id": exp_num + 1,
                "timestamp": now_iso(),
                "status": "error",
                "error": f"Verification failed after {max_repairs} repairs: {'; '.join(errors[:2])}",
                "description": description,
            }
            log.append(entry)
            append_experiment_entry(entry)
            consecutive_failures += 1
            continue

        # ── Dry-run execution check ─────────────────────────────────
        # Run the code with a quick timeout to catch runtime errors early.
        # If it crashes in < 5 seconds, it's a code bug, not a slow experiment.
        full_code = BOILERPLATE + code
        write_file(_EXPERIMENT_FILE, full_code)

        log_msg("  Dry-run check...")
        dry_ok, dry_out, dry_err = execute_experiment(timeout=10)

        if not dry_ok and "Timed out" not in dry_err:
            # Crashed fast = code bug. Send to repair loop.
            crash_msg = dry_err[:300] if dry_err else "(no stderr)"
            if dry_out:
                # Find the actual traceback line
                for line in dry_out.split("\n"):
                    if "Error" in line or "error" in line:
                        crash_msg = line.strip()
                        break

            log_msg(f"  DRY-RUN CRASH: {crash_msg[:120]}")

            # Try repair
            repaired = False
            for repair_attempt in range(max_repairs):
                repair_errors = [f"Runtime error: {crash_msg}"]
                log_msg(f"  Requesting runtime repair ({repair_attempt+1}/{max_repairs})...")
                repair_prompt = build_repair_prompt(code, repair_errors, repair_attempt + 1)
                try:
                    repair_response = query_ollama(
                        repair_prompt, model=model,
                        temperature=max(0.3, temperature - 0.2),
                        max_tokens=6000,
                    )
                except Exception as e:
                    log_msg(f"  Repair query failed: {e}")
                    break

                repaired_code = extract_code(repair_response)
                if repaired_code is None:
                    log_msg("  Could not extract repaired code")
                    break

                code = auto_fix_code(repaired_code)

                # Re-verify static
                is_valid, verify_errors = verify_code(code)
                if not is_valid:
                    log_msg(f"  Repair failed static verify: {verify_errors[0][:100]}")
                    continue

                # Re-run dry test
                full_code = BOILERPLATE + code
                write_file(_EXPERIMENT_FILE, full_code)
                dry_ok, dry_out, dry_err = execute_experiment(timeout=10)

                if dry_ok or "Timed out" in dry_err:
                    log_msg(f"  REPAIRED after {repair_attempt+1} attempt(s)")
                    repaired = True
                    break
                else:
                    crash_msg = dry_err[:300] if dry_err else "(no stderr)"
                    log_msg(f"  Still crashing: {crash_msg[:120]}")

            if not repaired and not dry_ok and "Timed out" not in dry_err:
                entry = {
                    "experiment_id": exp_num + 1,
                    "timestamp": now_iso(),
                    "status": "error",
                    "error": f"Runtime crash not fixable: {crash_msg[:500]}",
                    "description": description,
                }
                log.append(entry)
                append_experiment_entry(entry)
                print_experiment_summary(entry)
                consecutive_failures += 1
                continue

        # ── Execute verified + dry-run-tested code ──────────────────
        log_msg(f"  Wrote {os.path.basename(_EXPERIMENT_FILE)} ({len(code)} chars)")
        log_msg("  Running full experiment...")
        t0 = time.time()
        success, output, error = execute_experiment(timeout=600)
        t_exec = time.time() - t0
        log_msg(f"  Execution: {t_exec:.1f}s (success={success})")

        if not success:
            error_detail = error[:500] if error else "(no stderr)"
            if output:
                error_detail += f"\n--- stdout tail ---\n{output[-500:]}"
            entry = {
                "experiment_id": exp_num + 1,
                "timestamp": now_iso(),
                "status": "error",
                "error": error_detail[:1000],
                "description": description,
                "execution_time_s": round(t_exec, 2),
            }
            consecutive_failures += 1
        else:
            results = parse_results(output)
            if results is None:
                entry = {
                    "experiment_id": exp_num + 1,
                    "timestamp": now_iso(),
                    "status": "error",
                    "error": "Could not parse JSON from stdout",
                    "stdout_tail": output[-500:] if output else "(empty)",
                    "description": description,
                    "execution_time_s": round(t_exec, 2),
                }
                consecutive_failures += 1
            elif results.get("brier_score", 0) > 0.01:
                entry = {
                    "experiment_id": exp_num + 1,
                    "timestamp": now_iso(),
                    "status": "error",
                    "error": f"Invalid probabilities: brier_score={results.get('brier_score', 0):.4f} (baseline=0.000999)",
                    "description": description,
                    "execution_time_s": round(t_exec, 2),
                }
                log_msg(f"  REJECTED: Brier score {results.get('brier_score', 0):.4f} indicates broken prob_matrix")
                consecutive_failures += 1
            else:
                entry = {
                    "experiment_id": exp_num + 1,
                    "timestamp": now_iso(),
                    "status": "success",
                    "description": description,
                    "results": results,
                    "execution_time_s": round(t_exec, 2),
                }
                consecutive_failures = 0

        # Archive the experiment code with the entry
        if code:
            entry["code"] = code

        log.append(entry)
        append_experiment_entry(entry)
        print_experiment_summary(entry)

    print_session_summary(log)


# ---------------------------------------------------------------------------
# Self-Test
# ---------------------------------------------------------------------------

def self_test(model):
    """Verify setup without running experiments."""
    log_msg(f"\n{'='*60}")
    log_msg("  AutoResearch v2 - Self Test")
    log_msg(f"{'='*60}")
    all_ok = True

    # 1. Ollama
    log_msg("\n[1/4] Ollama connectivity...")
    ok, msg = check_ollama_connectivity(model)
    log_msg(f"  {'PASS' if ok else 'FAIL'}: {msg}")
    if not ok:
        all_ok = False

    # 2. Program file
    log_msg("\n[2/4] program_v2.md...")
    program = read_file(_PROGRAM_FILE)
    if program:
        log_msg(f"  PASS: Found ({len(program)} chars)")
    else:
        log_msg(f"  FAIL: Not found at {_PROGRAM_FILE}")
        all_ok = False

    # 3. prepare.py
    log_msg("\n[3/4] prepare.py...")
    if os.path.exists(os.path.join(_DIR, "prepare.py")):
        log_msg("  PASS: Found")
    else:
        log_msg("  FAIL: Not found")
        all_ok = False

    # 4. Feature cache
    log_msg("\n[4/4] Feature cache...")
    cache_dir = os.path.join(_DIR, ".feature_cache")
    if os.path.exists(cache_dir):
        cache_files = [f for f in os.listdir(cache_dir) if f.endswith(".npy")]
        log_msg(f"  PASS: {len(cache_files)} cached files")
    else:
        log_msg("  FAIL: .feature_cache/ not found")
        all_ok = False

    log_msg(f"\n{'='*60}")
    log_msg(f"  {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    log_msg(f"{'='*60}")
    return all_ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LottAI AutoResearch Runner v2")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL,
                        help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--max", type=int, default=100,
                        help="Max experiments (default: 100)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature (default: 0.7)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing experiments_v2.jsonl")
    parser.add_argument("--self-test", action="store_true",
                        help="Run setup checks only")
    args = parser.parse_args()

    if args.self_test:
        ok = self_test(args.model)
        sys.exit(0 if ok else 1)
    else:
        run_loop(
            model=args.model,
            max_experiments=args.max,
            temperature=args.temperature,
            resume=args.resume,
        )
