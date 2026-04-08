from task_loader import TESTS
import math
import ast
from sandbox import sandbox


def grader(code: str, task_id: str) -> dict:
    grader_score = 0.02
    task_tests = TESTS[task_id]
    baseline_time = task_tests["baseline_time"]
    baseline_mem = task_tests["baseline_mem"]
    total_tests = len(task_tests["cases"])

    # syntax check
    syntax_ok, syntax_error = check_syntax(code)
    if not syntax_ok:
        return {
            "status": "syntax_error",
            "grader_score": 0.02,
            "tests_passed": 0,
            "total_tests": total_tests,
            "efficiency": {
                "runtime_ms": None,
                "baseline_ms": baseline_time,
                "memory_mib": None,
                "baseline_mib": baseline_mem,
                "speed_ratio": None,   # agent_time / baseline
                "memory_ratio": None,  # agent_mem / baseline
            },
            "error_message": syntax_error
        }

    # sandbox execution and  logic check
    exec_ok, output, time_ms, mem_mib, tests_passed = sandbox(code, task_id)

    if not exec_ok:
        status = "runtime_error"
        error_message = output
    elif tests_passed == total_tests:
        status = "success"
        error_message = None
    elif tests_passed > 0:
        status = "partial"        
        error_message = output
    else:
        status = "logic_error"
        error_message = output

    # Speed and memory ratios — key for reward tradeoff
    speed_ratio = (baseline_time / time_ms) if time_ms and time_ms > 0 else None
    memory_ratio = (baseline_mem / mem_mib) if mem_mib and mem_mib > 0 else None
    # ratio > 1.0 = better than baseline
    # ratio < 1.0 = worse than baseline

    if status == "syntax_error" or status == "runtime_error":
        grader_score = 0.02
    elif status == "logic_error":
        grader_score = 0.02
    elif status == "partial":
        raw = 0.6 * (tests_passed / total_tests)
        grader_score = round(max(0.01, min(raw, 0.99)), 3)
    elif status == "success":
        base = 0.6
        if speed_ratio and speed_ratio >= 1.0:
            base += min(0.25, 0.25 * (speed_ratio - 1.0))
        if memory_ratio and memory_ratio >= 1.0:
            base += min(0.15, 0.15 * (memory_ratio - 1.0))
        grader_score = round(min(base, 0.99), 3)

    # At the very end, clamp everything:
    grader_score = max(0.01, min(grader_score, 0.99))
    grader_score = float(grader_score)
    if grader_score >= 1.0:
        grader_score = 0.99
    if grader_score <= 0.0:
        grader_score = 0.01

    grader_score = _safe_score(grader_score)
    print(f"DEBUG - TASK: {task_id} | GRADE: {grader_score} | TYPE: {type(grader_score)}")

    return {
        "status": status,
        "grader_score":grader_score,
        "tests_passed": tests_passed,
        "total_tests": total_tests,
        "efficiency": {
            "runtime_ms": time_ms,
            "baseline_ms": baseline_time,
            "memory_mib": mem_mib,
            "baseline_mib": baseline_mem,
            "speed_ratio": speed_ratio,
            "memory_ratio": memory_ratio,
        },
        "error_message": error_message
    }

def compare_results(actual, expected):
    import math
    
    # Float comparison
    if isinstance(expected, float) and isinstance(actual, float):
        return math.isclose(actual, expected, rel_tol=1e-3)

    # Int
    if isinstance(expected, int) and isinstance(actual, int):
        return actual == expected

    # String
    if isinstance(expected, str) and isinstance(actual, str):
        return actual == expected

    # List
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return False
        for a, e in zip(actual, expected):
            if isinstance(e, float):
                if not math.isclose(a, e, rel_tol=1e-3):
                    return False
            else:
                if a != e:
                    return False
        return True

    # Dict
    if isinstance(expected, dict) and isinstance(actual, dict):
        return expected == actual

    # Numpy arrays
    try:
        import numpy as np
        if isinstance(actual, np.ndarray) or isinstance(expected, np.ndarray):
            return np.allclose(actual, expected, rtol=1e-3)
    except:
        pass

    # Pandas
    try:
        import pandas as pd
        if isinstance(actual, pd.Series) or isinstance(expected, pd.Series):
            return actual.equals(expected)
        if isinstance(actual, pd.DataFrame) or isinstance(expected, pd.DataFrame):
            return actual.equals(expected)
    except:
        pass

    # Torch tensors
    try:
        import torch
        if isinstance(actual, torch.Tensor) or isinstance(expected, torch.Tensor):
            return torch.allclose(actual, expected, rtol=1e-3)
    except:
        pass

    # Fallback
    return actual == expected


def check_syntax(code: str) -> tuple[bool, str]:
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, str(e)
    

def _safe_score(score) -> float:
    """Ensure score is strictly between 0 and 1, pure Python float."""
    s = float(score)
    if s <= 0.0 or s >= 1.0:
        s = max(0.01, min(s, 0.99))
    if s == 0.0:
        s = 0.01
    if s == 1.0:
        s = 0.99
    return s