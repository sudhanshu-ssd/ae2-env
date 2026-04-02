---
title: AE2-Applied-AI-Engineering
emoji: 🏗️
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
---

# AE² — Applied AI Engineering Environment

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)
[![HuggingFace Space](https://img.shields.io/badge/🤗-Space-yellow)](https://huggingface.co/spaces/sudhanshu-ssd/ae2-env)
[![Python 3.11](https://img.shields.io/badge/python-3.11-green)](https://python.org)
[![Baseline Score](https://img.shields.io/badge/baseline-0.960-brightgreen)](baseline_results.json)

> A benchmark RL environment where AI agents fix and optimize broken production Python code across 5 real-world ML engineering domains.

---

## What is AE²?

AE² (Applied AI Engineering Environment) simulates the real-world task of an AI/ML engineer debugging and optimizing broken production code. Unlike toy environments, every task here mirrors genuine engineering work: fixing broken data pipelines, correcting model training bugs, optimizing slow inference code, and repairing API deployment issues.

Agents interact by submitting Python code fixes. The environment executes submissions in an isolated sandbox, runs deterministic test cases, measures runtime and memory efficiency, and returns a shaped reward signal with partial credit for incremental progress.

**Why this matters:** As LLM coding agents grow more capable, rigorous benchmarks must test real engineering judgment, not just syntax completion. AE² fills this gap by grounding evaluation in production ML engineering scenarios that engineers face daily.

---

## Environment Overview

| Property | Value |
|----------|-------|
| Domains | Data Engineering, Model Ops, NLP/LLM, Deployment, Eval Analysis |
| Total Tasks | 15 (5 domains × 3 difficulty levels) |
| Difficulty Levels | EASY → MEDIUM → HARD |
| Reward Range | [-0.3, 1.0] shaped with partial credit |
| Max Steps per Episode | 10 |
| Execution | Sandboxed subprocess with 30s timeout |

---

## Action & Observation Spaces

### Action
```python
class EngAction(Action):
    sol: str  # Complete Python function named 'solution'
```

### Observation
```python
class EngObservation(Observation):
    domain: str           # data_eng | model_ops | nlp_llm | deployment | eval_analysis
    difficulty: str       # EASY | MEDIUM | HARD
    task: str             # Full task description and objective
    code: str             # Current broken/partial code to fix
    done: bool            # Episode complete flag
    reward: float         # Reward for last action (-0.3 to 1.0)
    output: str           # Error message or assertion failure from last run
    tests_passed: int     # Number of test cases passed
    num_tests: int        # Total test cases
    time_taken: float     # Execution time in milliseconds
    mem_taken: float      # Memory usage in MiB
    message: str          # Human-readable feedback with error details
    num_steps_remain: int # Remaining attempts in episode
```

---

## Reward Function

Shaped to provide dense signal across the full episode trajectory:

| Condition | Reward |
|-----------|--------|
| Syntax error | -0.3 |
| Runtime error | -0.2 |
| All tests fail | 0.0 |
| Partial pass (k/n tests) | 0.6 × (k/n) |
| All tests pass | 0.6 base |
| + Speed faster than baseline | up to +0.25 |
| + Memory below baseline | up to +0.15 |
| MEDIUM difficulty bonus | +0.05 |
| HARD difficulty bonus | +0.10 |
| Maximum | 1.0 |

The efficiency bonus rewards agents that produce optimized code, not just correct code — directly reflecting production engineering values.

---

## Tasks

### EASY
| Domain | Task | Bug Type |
|--------|------|----------|
| data_eng | Fix Currency Parser | Crash on comma-formatted numbers |
| deployment | Fix Prediction Endpoint | NameError — missing class definition |
| eval_analysis | Fix Accuracy Calculation | Wrong division operand |
| model_ops | Fix sklearn Pipeline | Missing comma in step list |
| nlp_llm | Fix Tokenizer Argument | Invalid keyword argument |

### MEDIUM
| Domain | Task | Bug Type |
|--------|------|----------|
| data_eng | Fix Null Handling | Crash on None values in list |
| deployment | Fix Pydantic Validation | Wrong type annotation |
| eval_analysis | Fix F1 Averaging | Wrong averaging strategy |
| model_ops | Fix Feature Scaling | Missing StandardScaler |
| nlp_llm | Fix Batch Padding | Missing padding parameter |

### HARD
| Domain | Task | Challenge |
|--------|------|-----------|
| data_eng | Vectorize GroupBy | Replace O(n²) loop with pandas vectorization |
| deployment | Solve N+1 Latency | Replace 100 sequential calls with 1 bulk call |
| eval_analysis | Optimize Regex | Pre-compile pattern outside hot loop |
| model_ops | Mixed Precision Inference | Implement torch.amp.autocast |
| nlp_llm | Batched Tokenization | Replace loop with single batch call |

---

## Baseline Scores

Evaluated using `meta-llama/Llama-3.1-8B-Instruct` via HuggingFace Router (OpenAI-compatible):

| Domain | EASY | MEDIUM | HARD |
|--------|------|--------|------|
| data_eng | 1.000 | 1.000 | 0.600 |
| deployment | 1.000 | 0.793 | 1.000 |
| eval_analysis | 1.000 | 1.000 | 1.000 |
| model_ops | 1.000 | 1.000 | 1.000 |
| nlp_llm | 1.000 | 1.000 | 1.000 |
| **Average** | **1.000** | **0.957** | **0.920** |


**Overall Baseline Score: 0.960** — Full results in [baseline_results.json](baseline_results.json).
---

## Setup & Usage

### Local Development

```bash
# Install dependencies
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# Pre-download the model
python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('distilbert-base-uncased'); AutoModel.from_pretrained('distilbert-base-uncased')"

# Start server
uvicorn app:app --host 0.0.0.0 --port 7860

# Set env vars and run baseline
export API_KEY=your_api_key
export API_BASE_URL=https://router.huggingface.co/v1
export MODEL_NAME=meta-llama/Llama-3.1-8B-Instruct
export AE2_URL=http://localhost:7860
python inference.py
```

### Docker

```bash
docker build -t ae2-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key -e AE2_URL=http://localhost:7860 ae2-env
```

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | LLM API key | Yes |
| `API_BASE_URL` | LLM endpoint | No (defaults to HF router) |
| `MODEL_NAME` | Model identifier | No (defaults to Llama-3.1-8B) |
| `AE2_URL` | Environment server URL | No (defaults to localhost:7860) |

---

## Using the Environment

```python
from client import AE2Env
from models import EngAction

with AE2Env(base_url="https://sudhanshu-ssd-ae2-env.hf.space").sync() as env:
    result = env.reset(task_id="data_eng_easy_001")
    obs = result.observation
    print(f"Task: {obs.task}")
    
    result = env.step(EngAction(sol="""
def solution(value: str) -> float:
    cleaned = value.replace('$', '').replace(',', '')
    return float(cleaned)
"""))
    print(f"Reward: {result.reward} | Tests: {result.observation.tests_passed}/{result.observation.num_tests}")
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start episode, get initial observation |
| `/step` | POST | Submit code fix, get reward |
| `/ws` | WebSocket | Stateful connection (recommended) |
| `/tasks` | GET | List all 15 tasks with action schema |
| `/grader` | GET | Score a task+code pair directly |
| `/baseline` | POST | Return pre-computed baseline scores |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger UI |

---

## Project Structure

```
ae2-env/
├── environment.py        # EngEnv — OpenEnv Environment subclass
├── models.py             # EngAction, EngObservation, EngState
├── grader.py             # Deterministic grader + compare_results
├── reward.py             # Shaped reward function
├── sandbox.py            # Isolated subprocess execution
├── task_loader.py        # Task and test loader
├── client.py             # AE2Env WebSocket client
├── inference.py          # Baseline inference script
├── openenv.yaml          # OpenEnv spec metadata
├── requirements.txt
├── Dockerfile
├── app.py                # FastAPI server
├── baseline_results.json # Pre-computed baseline scores
└── tasks/                # 15 task directories
    └── {task_id}/
        ├── task.json     # Task definition + broken code
        └── tests.json    # Test cases
```

---

## Author

**Sudhanshu Saini** — B.E. AI & Data Science, CTAE Udaipur  
Top 3% Amazon ML Challenge 2025 · NFPC 2026 Top 12/400+  
[GitHub](https://github.com/sudhanshu-ssd) · [LinkedIn](https://linkedin.com/in/sudhanshu-saini-558575362/)