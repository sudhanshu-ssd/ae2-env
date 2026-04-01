import sys
import os

# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from models import EngAction, EngObservation

from environment import EngEnv
from task_loader import TASKS, TESTS
from grader import grader

# Base app from OpenEnv — gives /ws /reset /step /state /health /web /docs
app = create_fastapi_app(EngEnv,EngAction,EngObservation)# ── Additional required endpoints ────────────────────────────────────────────

@app.get("/tasks")
def list_tasks():
    """
    Returns all available tasks and their action schema.
    Required by PS: /tasks endpoint.
    """
    all_tasks = []
    for level in ['EASY', 'MEDIUM', 'HARD']:
        for task in TASKS[level]:
            all_tasks.append({
                "id": task["id"],
                "domain": task["domain"],
                "difficulty": level,
                "name": task["name"],
                "description": task["description"],
                "action_schema": {
                    "sol": "str  # Your complete Python solution code"
                }
            })
    return JSONResponse(content={"tasks": all_tasks, "total": len(all_tasks)})


    
@app.get("/grader")
def get_grader_score(task_id: str, code: str):
    try:
        result = grader(code, task_id)
        return JSONResponse(content={
            "task_id": task_id,
            "grader_score": result["grader_score"],
            "status": result["status"],
            "tests_passed": result["tests_passed"],
            "total_tests": result["total_tests"],
            "efficiency": result["efficiency"],
        })
    except KeyError:
        return JSONResponse(status_code=404, content={"error": f"Task '{task_id}' not found"})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.post("/baseline")
def trigger_baseline():
    """
    Triggers baseline inference and returns scores.
    Required by PS: /baseline endpoint.
    Note: Full inference.py should be run separately for complete results.
    Returns pre-computed baseline scores.
    """
    baseline_scores = {
        "EASY": {
            "avg_score": 0.0,
            "note": "Run inference.py to generate actual baseline scores"
        },
        "MEDIUM": {
            "avg_score": 0.0,
            "note": "Run inference.py to generate actual baseline scores"
        },
        "HARD": {
            "avg_score": 0.0,
            "note": "Run inference.py to generate actual baseline scores"
        }
    }
    
    # Try to load pre-computed results if available
    import json
    import os
    results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_results.json")
    if os.path.exists(results_path):
        with open(results_path) as f:
            data = json.load(f)
        return JSONResponse(content={
            "status": "pre_computed",
            "model": data.get("model"),
            "overall_score": data.get("overall"),
            "scores_by_difficulty": data.get("summary"),
            "results": data.get("results")
        })
    
    return JSONResponse(content={
        "status": "not_computed",
        "message": "Run inference.py first to generate baseline scores",
        "baseline_scores": baseline_scores
    })


@app.get("/health")
def health():
    """Health check — judges ping this first."""
    return JSONResponse(content={
        "status": "ok",
        "environment": "AE2 - Applied AI Engineering Environment",
        "version": "1.0.0",
        "tasks_available": sum(len(TASKS[l]) for l in TASKS),
        "domains": ["data_eng", "model_ops", "nlp_llm", "deployment", "eval_analysis"]
    })