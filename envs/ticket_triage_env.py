import json
import random
from pathlib import Path
from typing import Dict, List, Optional
from graders.grader import grade_agent_action
from models.schemas import AgentAction, TaskItem
class TicketTriageEnv:
    def __init__(self) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.tasks_dir = self.project_root / "tasks"
        self.current_task: Optional[TaskItem] = None
        self.done: bool = False
        self.tasks_by_difficulty: Dict[str, List[TaskItem]] = {
            "easy": self._load_tasks("easy.json"),
            "medium": self._load_tasks("medium.json"),
            "hard": self._load_tasks("hard.json")
        }
    def _load_tasks(self, filename: str) -> List[TaskItem]:
        file_path = self.tasks_dir / filename
        with open(file_path, "r", encoding="utf-8") as file:
            raw_tasks = json.load(file)
        return [TaskItem(**task) for task in raw_tasks]
    def reset(self, difficulty: str) -> Dict:
        if difficulty not in self.tasks_by_difficulty:
            raise ValueError(f"Invalid difficulty: {difficulty}")
        available_tasks = self.tasks_by_difficulty[difficulty]
        if not available_tasks:
            raise ValueError(f"No tasks found for difficulty: {difficulty}")
        self.current_task = random.choice(available_tasks)
        self.done = False
        return self._build_state()
    def state(self) -> Dict:
        if self.current_task is None:
            raise ValueError("Environment has not been reset yet")
        return {
            "task_id": self.current_task.task_id,
            "difficulty": self.current_task.difficulty,
            "state": {
                "customer_type": self.current_task.customer_type,
                "product": self.current_task.product,
                "message": self.current_task.message,
                "previous_status": self.current_task.previous_status
            },
            "done": self.done
        }
    def step(self, action: AgentAction) -> Dict:
        if self.current_task is None:
            raise ValueError("Environment has not been reset yet")
        if self.done:
            raise ValueError("Episode already finished. Please call reset() to start a new task")
        grading_result = grade_agent_action(
            expected=self.current_task.expected_output,
            actual=action
        )
        self.done = True
        return {
            "reward": grading_result["reward"],
            "score": grading_result["score"],
            "done": self.done,
            "feedback": grading_result["feedback"],
            "next_state": {
                "task_id": self.current_task.task_id,
                "difficulty": self.current_task.difficulty,
                "episode_status": "completed"
            }
        }
    def _build_state(self) -> Dict:
        if self.current_task is None:
            raise ValueError("No current task is loaded")
        return {
            "task_id": self.current_task.task_id,
            "difficulty": self.current_task.difficulty,
            "state": {
                "customer_type": self.current_task.customer_type,
                "product": self.current_task.product,
                "message": self.current_task.message,
                "previous_status": self.current_task.previous_status
            },
            "done": self.done
        }