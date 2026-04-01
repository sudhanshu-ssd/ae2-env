from typing import Optional
from openenv.core.env_server import Action, State, Observation
from pydantic import BaseModel

class EngAction(Action):
    sol: str

class EngObservation(Observation):
    domain: str = ""
    difficulty: str = ""
    task: str = ""
    code: str = ""
    done: bool = False
    reward: Optional[float] = None
    output: Optional[str] = None
    tests_passed: Optional[int] = None
    num_tests: Optional[int] = None
    time_taken: Optional[float] = None
    mem_taken: Optional[float] = None
    message: Optional[str] = None
    num_steps_remain: Optional[int] = None

class EngState(State):
    episode_id: Optional[str] = None
    step_count: Optional[int] = 0
    total_steps: Optional[int] = 10
    difficulty: Optional[str] = None
    task_domain: Optional[str] = None
