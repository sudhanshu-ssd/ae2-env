import subprocess
import sys
import json
import tempfile
import os
import time
import tracemalloc
from task_loader import TESTS

FORBIDDEN_IMPORTS = [
    'os.system',
    'subprocess.run', 'subprocess.call', 'subprocess.Popen',
    'shutil.rmtree',
    '__import__',
    'socket.connect',
    'urllib.request',
    'importlib.import_module',
]

ALLOWED_IMPORTS = [
    # Core
    'import pandas', 'import numpy', 'import json',
    'import re', 'import math', 'import collections',
    'import datetime', 'import logging', 'import traceback',
    
    # ML
    'import torch', 'from torch', 'import torch.nn',
    'from torch.utils.data',
    'import sklearn', 'from sklearn',
    
    # HuggingFace
    'import transformers', 'from transformers',
    'import datasets', 'from datasets',
    
    # Deployment
    'import fastapi', 'from fastapi',
    'import pydantic', 'from pydantic',
    'import sqlalchemy', 'from sqlalchemy',
]

def is_safe(code: str) -> tuple[bool, str]:
    for forbidden in FORBIDDEN_IMPORTS:
        if forbidden in code:
            return False, f"Forbidden pattern detected: {forbidden}"
    # Check that solution function exists
    if "def solution(" not in code:
        return False, "Code must contain a function named 'solution'"
    return True, ""


def run_single_test(
    code: str,
    test_input: dict,
    expected: any,
    timeout: int = 30
) -> tuple[bool, str, float, float]:
    """
    Run agent code against ONE test case in subprocess.
    Returns: (passed, output, time_ms, mem_mib)
    """
    # Before the runner_script f-string, compute these:
    test_input_json = json.dumps(test_input, ensure_ascii=True).replace("'", "\\'")
    expected_json = json.dumps(expected, ensure_ascii=True).replace("'", "\\'")
        
    # Build test runner script
    runner_script = f"""# -*- coding: utf-8 -*-
import json, time, tracemalloc, math, sys

def compare_results(actual, expected):
    import math

    # Handle "any_float" sentinel for model_ops_hard
    if expected == "any_float":
        return isinstance(actual, (int, float)) and not math.isnan(float(actual))

    # Unwrap torch tensors
    try:
        import torch
        if isinstance(actual, torch.Tensor):
            actual = actual.detach().cpu().tolist()
    except ImportError:
        pass

    # Unwrap numpy arrays
    try:
        import numpy as np
        if isinstance(actual, np.ndarray):
            actual = actual.tolist()
        if isinstance(expected, np.ndarray):
            expected = expected.tolist()
        if isinstance(actual, np.integer):
            actual = int(actual)
        if isinstance(actual, np.floating):
            actual = float(actual)
    except ImportError:
        pass

    # Single float
    if isinstance(expected, float) and isinstance(actual, (int, float)):
        return math.isclose(float(actual), expected, rel_tol=1e-3, abs_tol=1e-6)

    # Single int
    if isinstance(expected, int) and isinstance(actual, (int, float)):
        return int(actual) == expected

    # String
    if isinstance(expected, str) and isinstance(actual, str):
        return actual == expected

    # List — recursive
    if isinstance(expected, list) and isinstance(actual, list):
        if len(actual) != len(expected):
            return False
        return all(compare_results(a, e) for a, e in zip(actual, expected))

    # Dict
    if isinstance(expected, dict) and isinstance(actual, dict):
        if set(actual.keys()) != set(expected.keys()):
            return False
        return all(compare_results(actual[k], expected[k]) for k in expected)

    # Fallback
    return actual == expected 

# Agent's code
{code}

# Test input
test_input = json.loads('{test_input_json}')
expected = json.loads('{expected_json}')

# Run and measure
tracemalloc.start()
start = time.perf_counter()

try:
    # Call the function with unpacked input
    if isinstance(test_input, dict):
        result = solution(**test_input)
    else:
        result = solution(test_input)
        
    elapsed_ms = (time.perf_counter() - start) * 1000
    _, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    mem_mib = peak_mem / (1024 * 1024)
    
    passed = compare_results(result, expected)
    if passed:
        t = None
    else:
        t = f"Expected {{str(expected)}}, got {{str(result)}}"

    print(json.dumps({{
        "passed": passed,
        "result": str(result),
        "expected": str(expected),
        "time_ms": elapsed_ms,
        "mem_mib": mem_mib,
        "error": t
    }}))
    
except Exception as e:
    tracemalloc.stop()
    print(json.dumps({{
        "passed": False,        
        "result": None,
        "expected": str(expected),
        "time_ms": None,
        "mem_mib": None,
        "error": str(e)
    }}))
"""
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(
        mode='w',
        suffix='.py',
        delete=False,
        encoding='utf-8'
    ) as f:
        f.write(runner_script)
        tmp_path = f.name

    try:
        result = subprocess.run(
            [sys.executable, tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
            # cwd is removed so it uses the system default temp area
        )
        
        if result.returncode != 0:
            return False, result.stderr, None, None
            
        output = json.loads(result.stdout)
        return (
            output["passed"],
            output.get("error"),
            output["time_ms"],
            output["mem_mib"]
        )
        
        
    except subprocess.TimeoutExpired:
        return False, "Timeout: code took too long (>5s)", None, None
    except json.JSONDecodeError:
        return False, f"Output parse error: {result.stdout}", None, None
    finally:
        os.unlink(tmp_path)  # clean up temp file


def sandbox(
    code: str,
    task_id: str
) -> tuple[bool, str, float, float, int]:
    """
    Run agent code against ALL test cases.
    Returns: (exec_ok, output, avg_time_ms, avg_mem_mib, tests_passed)
    """
    
    # Static safety check first
    safe, reason = is_safe(code)
    if not safe:
        return False, reason, None, None, 0
    
    test_cases = TESTS[task_id]["cases"]
    
    tests_passed = 0
    times = []
    mems = []
    last_output = ""
    exec_ok = True
    
    for case in test_cases:
        passed, output, time_ms, mem_mib = run_single_test(
            code=code,
            test_input=case["input"],
            expected=case["expected"],
            timeout=30
        )
        
        if time_ms is not None:
            times.append(time_ms)
        if mem_mib is not None:
            mems.append(mem_mib)
            
        if passed:
            tests_passed += 1
        else:
            if output:
                last_output = output
            if output is not None and "Timeout" in output:
                exec_ok = False
                break  # only break on actual timeout
    
    avg_time = sum(times) / len(times) if times else None
    avg_mem = sum(mems) / len(mems) if mems else None
    
    return exec_ok, last_output, avg_time, avg_mem, tests_passed