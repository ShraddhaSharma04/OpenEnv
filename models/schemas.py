from typing import Any, Dict, List, Literal
from pydantic import BaseModel, Field
DifficultyLevel = Literal["easy", "medium", "hard"]
PriorityLevel = Literal["low", "medium", "high"]
class ResetRequest(BaseModel):
    difficulty: DifficultyLevel = Field(..., description="Difficulty level of the task")
class TicketState(BaseModel):
    task_id: str
    difficulty: DifficultyLevel
    customer_type: str
    product: str
    message: str
    previous_status: str
class AgentAction(BaseModel):
    category: str = Field(..., min_length=1)
    priority: PriorityLevel
    assigned_team: str = Field(..., min_length=1)
    next_action: str = Field(..., min_length=3)
class StepRequest(BaseModel):
    action: AgentAction
class GraderFeedback(BaseModel):
    category_correct: bool
    priority_correct: bool
    assigned_team_correct: bool
    matched_keywords: List[str]
    missing_keywords: List[str]
    keyword_score: float
class StateResponse(BaseModel):
    task_id: str
    difficulty: DifficultyLevel
    state: Dict[str, Any]
    done: bool
class StepResponse(BaseModel):
    reward: float
    score: float
    done: bool
    feedback: GraderFeedback
    next_state: Dict[str, Any]
class ExpectedOutput(BaseModel):
    category: str
    priority: PriorityLevel
    assigned_team: str
    next_action_keywords: List[str]
class TaskItem(BaseModel):
    task_id: str
    difficulty: DifficultyLevel
    customer_type: str
    product: str
    message: str
    previous_status: str
    expected_output: ExpectedOutput