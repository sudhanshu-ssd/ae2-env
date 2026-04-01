import json
import os

def load_all_tasks() -> tuple:
    tasks = {"EASY": [], "MEDIUM": [], "HARD": []}
    tests = {}
    
    # Find tasks dir relative to this file, not cwd
    base_dir = os.path.dirname(os.path.abspath(__file__))  
    tasks_dir = os.path.join(base_dir, "tasks")
    
    for task_folder in os.listdir(tasks_dir):
        task_path = os.path.join(tasks_dir, task_folder)
        if not os.path.isdir(task_path):
            continue
        with open(os.path.join(task_path, "task.json")) as f:
            task = json.load(f)
        with open(os.path.join(task_path, "tests.json")) as f:
            test_data = json.load(f)
        difficulty = task["difficulty"]
        tasks[difficulty].append(task)
        tests[task["id"]] = {
            "baseline_time": task["baseline_time_ms"],
            "baseline_mem": task["baseline_mem_mib"],
            "cases": test_data["cases"]
        }
    return tasks, tests

TASKS, TESTS = load_all_tasks()