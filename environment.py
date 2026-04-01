
from openenv.core.env_server import Environment
from models import EngAction, EngObservation, EngState
import random
from grader import grader
from reward import calculate_reward
import uuid
from task_loader import TASKS, TESTS

class EngEnv(Environment):
    def __init__(self):
        self._state = EngState()
        self.last_grader_result = None
        self.attempts_remaining = 10
        self.task_id = None
        self.task_name = None
        self.task_time_base = None
        self.task_memory_base = None
        self.difficulty = "EASY"
        self.task_description = ""
        self.task_domain = ""
        self.current_task = None

    def reset(self, task_id: str = None, seed: int = None, episode_id: str = None, **kwargs) -> EngObservation:
        levels = ['EASY', 'MEDIUM', 'HARD']

        if task_id:
            task = None
            level = None
            for lvl in levels:
                for t in TASKS[lvl]:
                    if t['id'] == task_id:
                        task = t
                        level = lvl
                        break
                if task:
                    break
            if task is None:
                raise ValueError(f"Task '{task_id}' not found")
        else:
            level = random.choice(levels)
            task = random.choice(TASKS[level])

        self.task_id = task['id']
        self.task_name = task['name']
        self.task_time_base = task['baseline_time_ms']
        self.task_memory_base = task['baseline_mem_mib']
        self.difficulty = level
        self.task_description = task["description"]
        self.task_domain = task["domain"]
        self.attempts_remaining = 10
        self.last_grader_result = None

        self._state = EngState(
            episode_id=str(uuid.uuid4()),
            step_count=0,
            total_steps=self.attempts_remaining,
            difficulty=level,
            task_domain=task["domain"],
        )

        return EngObservation(
            domain=task["domain"],
            difficulty=level,
            task=self.task_description,
            code=task["broken_code"],
            done=False,
            reward=None,
            output=None,
            tests_passed=None,
            num_tests=len(TESTS[task["id"]]["cases"]),
            time_taken=None,
            mem_taken=None,
            message=f"Fix the {level} {task['domain']} task. {self.attempts_remaining} attempts remaining.",
            num_steps_remain=self.attempts_remaining
        )
    
    

    def step(self, action: EngAction) -> EngObservation:
        self._state.step_count += 1
        self.attempts_remaining -= 1

        code = action.sol.strip()
        g_result = grader(code, self.task_id)
        self.last_grader_result = g_result
        reward = calculate_reward(g_result, self.difficulty)

        # Done logic
        out_of_attempts = self.attempts_remaining <= 0
        logic_passed = (
            g_result['status'] == 'success' or
            g_result['tests_passed'] == g_result['total_tests']
        )
        if self.difficulty == 'HARD':
            speed_ok = (g_result['efficiency'].get('speed_ratio') or 0) >= 1.0
            memory_ok = (g_result['efficiency'].get('memory_ratio') or 0) >= 1.0
            efficiency_met = speed_ok or memory_ok
        else:
            efficiency_met = True

        done = out_of_attempts or (logic_passed and efficiency_met)

        return EngObservation(
            domain=self.task_domain,
            difficulty=self.difficulty,
            task=self.task_description,
            code=code,
            done=done,
            reward=reward,
            output=g_result['error_message'],
            tests_passed=g_result['tests_passed'],
            num_tests=g_result['total_tests'],
            time_taken=g_result['efficiency'].get('runtime_ms'),
            mem_taken=g_result['efficiency'].get('memory_mib'),
            message=self._build_message(g_result, reward, done),
            num_steps_remain=self.attempts_remaining
        )

    @property
    def state(self) -> EngState:
        return self._state

    def list_tasks(self) -> list:
        all_tasks = []
        for level in ['EASY', 'MEDIUM', 'HARD']:
            for task in TASKS[level]:
                all_tasks.append({
                    "id": task["id"],
                    "domain": task["domain"],
                    "difficulty": level,
                    "name": task["name"],
                    "description": task["description"]
                })
        return all_tasks

    def get_grader_score(self) -> dict:
        if not self.last_grader_result:
            return {"grader_score": 0.0, "status": "no_attempt"}
        return {
            "task_id": self.task_id,
            "grader_score": self.last_grader_result.get("grader_score", 0.0),
            "status": self.last_grader_result.get("status"),
            "tests_passed": self.last_grader_result.get("tests_passed", 0),
            "total_tests": self.last_grader_result.get("total_tests", 0),
        }

    def _build_message(self, g_result: dict, reward: float, done: bool) -> str:
        status = g_result['status']
        passed = g_result['tests_passed']
        total = g_result['total_tests']
        eff = g_result['efficiency']

        if status == 'syntax_error':
            return f"Syntax error: {g_result['error_message']}"
        if status == 'runtime_error':
            return f"Runtime error: {g_result['error_message']}"
        if status == 'logic_error':
            return f"Code runs but fails all tests (0/{total}). Check your logic.Error: {g_result['error_message']}. {self.attempts_remaining} attempts left."
        if status == 'partial':
            error = g_result.get('error_message', '')
            return f"Partial: {passed}/{total} tests passed. Reward: {reward}. Error: {error}. {self.attempts_remaining} attempts left."
        if status == 'success' and not done:
            sr = eff.get('speed_ratio')
            mr = eff.get('memory_ratio')
            if self.difficulty == 'HARD':
                return f"All tests pass! Optimize further. Speed: {sr:.2f}x baseline. Memory: {mr:.2f}x baseline."
            return f"All tests passed! Reward: {reward}"
        if done and status == 'success':
            sr = eff.get('speed_ratio') or 0
            mr = eff.get('memory_ratio') or 0
            return f"Episode complete! Reward: {reward}. Speed: {sr:.2f}x. Memory: {mr:.2f}x."
        if done and status != 'success':
            return f"Out of attempts. Best: {passed}/{total} tests."
        return f"Reward: {reward}. {self.attempts_remaining} attempts remaining."