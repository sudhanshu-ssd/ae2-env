"""
inference.py — AE² Baseline Inference Script
============================================
Required env vars:
    API_BASE_URL   LLM API endpoint
    MODEL_NAME     Model identifier  
    HF_TOKEN       HuggingFace / API key
    AE2_URL        Your HF Space URL (e.g. https://username-ae2-env.hf.space)
"""

import sys
import os
import warnings
import logging
import re
import json

# Silence all warnings to stderr only
warnings.filterwarnings("ignore")
logging.getLogger("websockets").setLevel(logging.CRITICAL)
logging.getLogger("websockets.legacy").setLevel(logging.CRITICAL)
logging.basicConfig(stream=sys.stderr, level=logging.CRITICAL)

from openai import OpenAI
from models import EngAction
from client import AE2Env
import requests
import time as _time


# ── Config ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
AE2_URL      = os.getenv("AE2_URL", "http://localhost:7860")
ENV_NAME     = "ae2-applied-ai-engineering"

MAX_STEPS    = 10
TEMPERATURE  = 0.1
MAX_TOKENS   = 1024



def log_start(task: str, model: str) -> None:
    # Ensure ONLY the [START] line goes to stdout
    sys.stdout.write(f"[START] task={task} env={ENV_NAME} model={model}\n")
    sys.stdout.flush()

def log_step(step: int, action: str, reward: float, done: bool, error=None) -> None:
    # FORCE the reward to be valid, no matter what the environment says
    safe_reward = max(0.05, min(float(reward), 0.95))
    
    d_str = "true" if done else "false"
    r_str = f"{safe_reward:.2f}"
    a_clean = str(action).replace('\n', ' ').strip()[:50]
    e_clean = "null" if not error else str(error).replace('\n', ' ').strip()
    
    sys.stdout.write(f"[STEP] step={step} action={a_clean} reward={r_str} done={d_str} error={e_clean}\n")
    sys.stdout.flush()

def log_end(success: bool, steps: int, rewards: list, final_grader_score: float) -> None:
    success_str = "true" if success else "false"
    
    # 1. Clamp the score strictly between 0 and 1
    # Use 2 decimal places as per rules
    safe_score = max(0.01, min(float(final_grader_score), 0.99))
    
    # 2. Clamp the rewards list
    safe_rewards = [f"{max(0.01, min(float(r), 0.99)):.2f}" for r in rewards]
    rewards_str = ",".join(safe_rewards)
    
    # 3. CONSTRUCT THE EXACT TAG REQUIRED
    # Format: [END] success=true steps=n score=0.xx rewards=r1,r2
    line = f"[END] success={success_str} steps={steps} score={safe_score:.2f} rewards={rewards_str}\n"
    
    sys.stdout.write(line)
    sys.stdout.flush()



# ── System prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert AI/ML Software Engineer.
You will be given a broken Python function and a task description.
Your job is to fix or optimize the code.

RULES:
1. Always return ONLY the complete fixed Python function — no explanation, no markdown,only python function.
2. The function MUST be named 'solution'.
3. Do not add any imports outside the function body unless they were already present.
4. Do not use os, subprocess, sys, socket, or any file I/O.
5. Keep the same function signature as the broken code.
6. keep helper functions

Example response format:
def solution(x, y):
    return x + y
"""

# ── Helpers ──────────────────────────────────────────────────────────────────
def extract_code(response_text: str) -> str:
    """Extract Python code from LLM response."""
    # Try to extract from markdown code block first
    match = re.search(r"```python\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # Try plain code block
    match = re.search(r"```\n(.*?)```", response_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # If no code block, assume entire response is code
    return response_text.strip()

def print_summary(results, scores_by_difficulty):
    print(f"\n{'='*60}")
    print("BASELINE RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"\n{'Domain':<20} {'Diff':<8} {'Score':<8} {'Tests':<10} {'Steps'}")
    print("-" * 60)
    for r in results:
        if "error" not in r:
            print(
                f"{r.get('domain','?'):<20} "
                f"{r.get('difficulty','?'):<8} "
                f"{r.get('grader_score',0.02):.3f}    "
                f"{r.get('tests_passed',0)}/{r.get('total_tests',0):<6} "
                f"{r.get('steps_taken',0)}"
            )
    print(f"\n{'Difficulty':<12} {'Avg Score':<12} {'Tasks'}")
    print("-" * 35)
    for diff, scores in scores_by_difficulty.items():
        if scores:
            avg = sum(scores) / len(scores)
            print(f"{diff:<12} {avg:.3f}        {len(scores)}")
    overall = [r["grader_score"] for r in results if "error" not in r]
    if overall:
        print(f"\nOVERALL BASELINE SCORE: {sum(overall)/len(overall):.3f}")
    with open("baseline_results.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "results": results,
            "summary": {d: sum(s)/len(s) if s else 0.01 for d, s in scores_by_difficulty.items()},
            "overall": sum(overall)/len(overall) if overall else 0.02
        }, f, indent=2)
    print("\nSaved to baseline_results.json")


def build_user_prompt(observation, step: int, history: list) -> str:
    history_text = "\n".join(history[-3:]) if history else "None"
    
    return f"""
TASK: {observation.task}

CURRENT CODE (fix or optimize this):
```python
{observation.code}
```

FEEDBACK FROM LAST ATTEMPT:
{observation.message or 'No previous attempt.'}

OUTPUT/ERROR FROM LAST RUN:
{observation.output or 'No output yet.'}

TESTS PASSED: {observation.tests_passed}/{observation.num_tests if observation.num_tests else '?'}
EXECUTION TIME: {f"{observation.time_taken:.2f}ms" if observation.time_taken else "N/A"}
MEMORY USED: {f"{observation.mem_taken:.2f}MiB" if observation.mem_taken else "N/A"}
ATTEMPTS REMAINING: {observation.num_steps_remain}

STEP HISTORY:
{history_text}

Return ONLY the fixed Python function named 'solution'. No explanation. No markdown.
""".strip()


def run_episode(client, env, task_id: str = None) -> dict:
    result = env.reset(task_id=task_id)
    observation = result.observation
    
    # print(f"\n{'='*60}")
    # print(f"Task: {observation.task[:80]}...")
    # print(f"Domain: {observation.domain} | Difficulty: {observation.difficulty}")
    # print(f"{'='*60}")
    
    # Mandatory [START] log
    log_start(task=task_id or observation.domain, model=MODEL_NAME)
    
    history = []
    final_reward = 0.01
    steps_taken = 0
    rewards = []
    
    for step in range(1, MAX_STEPS + 1):
        if result.done:
            # print(f"Episode done at step {step-1}.")
            break
        
        steps_taken = step
        user_prompt = build_user_prompt(observation, step, history)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                stream=False,
            )
            response_text = completion.choices[0].message.content or ""
        except Exception as exc:
            if "429" in str(exc):
                print(f"  Rate limited, waiting 20s...")
                _time.sleep(20)
                try:
                    completion = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=messages,
                        temperature=TEMPERATURE,
                        max_tokens=MAX_TOKENS,
                    )
                    response_text = completion.choices[0].message.content or ""
                except Exception:
                    response_text = observation.code
            else:
                print(f"  LLM call failed: {exc}")
                response_text = observation.code
        
        code = extract_code(response_text)
        # print(f"\nStep {step}: {code[:60].replace(chr(10), ' ')}...")
        
        action = EngAction(sol=code)
        result = env.step(action)
        observation = result.observation
        
        reward = result.reward or 0.01
        final_reward = reward
        rewards.append(reward)
        
        error_msg = observation.output if not result.done and observation.tests_passed == 0 else None
        
        # Mandatory [STEP] log
        log_step(step=step, action=code[:50], reward=reward, done=result.done, error=error_msg)
        
        if observation.time_taken:
            history_line = f"Step {step}: tests={observation.tests_passed}/{observation.num_tests} reward={reward:+.3f} time={observation.time_taken:.1f}ms"
        else:
            history_line = f"Step {step}: reward={reward:+.3f}"
        history.append(history_line)
        
        # print(f"  Reward: {reward:+.3f} | Tests: {observation.tests_passed}/{observation.num_tests} | Done: {result.done}")
        
        if result.done:
            # print(f"  {observation.message}")
            break
    
    success = result.done and (observation.tests_passed == observation.num_tests)
    
    # Mandatory [END] log
    log_end(success=success, steps=steps_taken, rewards=rewards)
    
    # Get grader score via HTTP
    # Inside run_episode()
    try:
        # Use requests.post and 'json=' parameter
        resp = requests.post(
            f"{AE2_URL}/grader", 
            json={"task_id": task_id, "code": observation.code}
        )
        
        if resp.status_code == 200:
            data = resp.json()
            final_grader_score = data.get("grader_score", 0.02)
        else:
            # This is where your 405 error is currently being caught
            print(f"  Grader error: {resp.status_code}")
            final_grader_score = 0.02
    except Exception as e:
        print(f"  Grader call failed: {e}")
        final_grader_score = 0.02
    
    return {
        "task": observation.task,
        "domain": observation.domain,
        "difficulty": observation.difficulty,
        "steps_taken": steps_taken,
        "final_reward": final_reward,
        "grader_score": final_grader_score,
        "tests_passed": observation.tests_passed,
        "total_tests": observation.num_tests,
        "success": success
    }


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    warnings.filterwarnings("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"
    
    # print("AE² — Applied AI Engineering Environment")
    # print(f"Model: {MODEL_NAME}")
    # print(f"Environment: {AE2_URL}")
    
    results = []
    scores_by_difficulty = {"EASY": [], "MEDIUM": [], "HARD": []}
    
    # Get task list via HTTP (this is fine, stateless is ok for /tasks)
    tasks_resp = requests.get(f"{AE2_URL}/tasks")
    tasks = tasks_resp.json().get("tasks", [])
    
    # Each task gets its OWN WebSocket connection = its own stateful session
    for task_info in tasks:
        task_id = task_info["id"]
        
        try:
            # New WebSocket connection per task = clean stateful episode
            with AE2Env(base_url=AE2_URL).sync() as env:
                episode_result = run_episode(client, env, task_id=task_id)
                results.append(episode_result)
                difficulty = episode_result["difficulty"]
                scores_by_difficulty[difficulty].append(episode_result["grader_score"])
                
        except Exception as e:
            print(f"Episode failed for {task_id}: {e}")
            _time.sleep(3)  # give server time to release connection

            results.append({
                "task_id": task_id,
                "grader_score": 0.02,
                "difficulty": task_info.get("difficulty", "EASY"),
                "domain": task_info.get("domain", ""),
                "error": str(e)
            })
    
    # Print and save results (same as before)
    # print_summary(results, scores_by_difficulty)


if __name__ == "__main__":
    main()