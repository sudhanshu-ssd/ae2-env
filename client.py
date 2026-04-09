from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from models import EngAction, EngObservation, EngState
import requests


class AE2Env(EnvClient[EngAction, EngObservation, EngState]):

    def _step_payload(self, action: EngAction) -> dict:
        return {"sol": action.sol}

    def _parse_result(self, payload: dict) -> StepResult:
        obs_data = payload.get("observation", {})
        obs = EngObservation(
            domain=obs_data.get("domain", ""),
            difficulty=obs_data.get("difficulty", ""),
            task=obs_data.get("task", ""),
            code=obs_data.get("code", ""),
            done=payload.get("done", False),
            reward=payload.get("reward",0.02),
            output=obs_data.get("output"),
            tests_passed=obs_data.get("tests_passed"),
            num_tests=obs_data.get("num_tests"),
            time_taken=obs_data.get("time_taken"),
            mem_taken=obs_data.get("mem_taken"),
            message=obs_data.get("message", ""),
            num_steps_remain=obs_data.get("num_steps_remain"),
        )
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> EngState:
        return EngState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            total_steps=payload.get("total_steps", 10),
            difficulty=payload.get("difficulty"),
            task_domain=payload.get("task_domain"),
        )

    def list_tasks(self) -> list:
        resp = requests.get(f"{self.base_url}/tasks")
        return resp.json().get("tasks", [])

    def get_grader_score(self, task_id: str = None, code: str = None) -> dict:
        params = {}
        if task_id:
            params["task_id"] = task_id
        if code:
            params["code"] = code
        resp = requests.get(f"{self.base_url}/grader", params=params)
        return resp.json()
    
    # In client.py, add:
    def reset(self, task_id: str = None, **kwargs):
        """Override reset to pass task_id."""
        payload = {}
        if task_id:
            payload["task_id"] = task_id
        return super().reset(**payload)