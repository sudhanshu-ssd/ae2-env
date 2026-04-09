import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # when app is in server/


from openenv.core.env_server import create_fastapi_app
from fastapi import FastAPI
from fastapi.responses import JSONResponse

from models import EngAction, EngObservation

from environment import EngEnv
from task_loader import TASKS, TESTS
from grader import grader
from pydantic import BaseModel

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


    
# @app.get("/grader")
# def get_grader_score(task_id: str, code: str):
#     try:
#         result = grader(code, task_id)
#         return JSONResponse(content={
#             "task_id": task_id,
#             "grader_score": result["grader_score"],
#             "status": result["status"],
#             "tests_passed": result["tests_passed"],
#             "total_tests": result["total_tests"],
#             "efficiency": result["efficiency"],
#         })
#     except KeyError:
#         return JSONResponse(status_code=404, content={"error": f"Task '{task_id}' not found"})
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"error": str(e)})
    
from pydantic import BaseModel
from fastapi.responses import JSONResponse

# 1. Ensure the model is defined
class GraderRequest(BaseModel):
    task_id: str
    code: str

@app.post("/grader")
def get_grader_score(request: GraderRequest):
    try:
        # 2. CRITICAL: You must use request.code and request.task_id
        # If you used 'code' or 'task_id' directly, it caused the 500 error.
        result = grader(request.code, request.task_id)
        
        # 3. Apply the mandatory Phase 2 clamp (strictly between 0 and 1)
        raw_score = result.get("grader_score", 0.01)
        clamped_score = max(0.01, min(raw_score, 0.99))
        # res["grader_score"] = max(0.05, min(float(raw_score), 0.95))
        # return JSONResponse(content=res)
        
        return JSONResponse(content={
            "grader_score": float(clamped_score),
            "status": result.get("status", "success"),
            "tests_passed": result.get("tests_passed"),
            "total_tests": result.get("total_tests")
        })
    except Exception as e:
        # This will show up in your Hugging Face Space Logs
        print(f"ERROR IN /grader: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})


@app.get("/baseline")
def trigger_baseline():
    """
    Triggers baseline inference and returns scores.
    Required by PS: /baseline endpoint.
    Note: Full inference.py should be run separately for complete results.
    Returns pre-computed baseline scores.
    """
    baseline_scores = {
        "EASY": {
            "avg_score": 0.01,
            "note": "Run inference.py to generate actual baseline scores"
        },
        "MEDIUM": {
            "avg_score": 0.01,
            "note": "Run inference.py to generate actual baseline scores"
        },
        "HARD": {
            "avg_score": 0.01,
            "note": "Run inference.py to generate actual baseline scores"
        }
    }
    
    # Try to load pre-computed results if available
    import json
    import os
    # results_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_results.json") # when app is in root 
    results_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"baseline_results.json") # when app is in server/
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
    return JSONResponse(content={
        "status": "ok",
        "environment": "AE2 - Applied AI Engineering Environment",
        "version": "1.0.0",
        "tasks_available": sum(len(TASKS[l]) for l in TASKS),
        "domains": ["data_eng", "model_ops", "nlp_llm", "deployment", "eval_analysis"]
    })

@app.get("/")
def read_root():
    """
    Landing page for the AE² Environment.
    Prevents 'Detail Not Found' when visiting the base URL.
    """
    return JSONResponse(content={
        "name": "ae2-applied-ai-engineering",
        "version": "1.0.1",
        "description": "AE² is a benchmark environment for training and evaluating AI agents on real-world Applied AI Engineering tasks.",
        "status": "active",
        "repository": "https://github.com/sudhanshu-ssd/ae2-env"
    })


def main():
    """Main entry point for the server, required by OpenEnv validator."""
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()