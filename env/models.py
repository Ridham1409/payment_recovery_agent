from pydantic import BaseModel, Field
from typing import List

class Observation(BaseModel):
    client_name: str
    invoice_amount: float
    days_overdue: int
    client_status: str  # cooperative | ghosting | disputing
    client_personality: str
    background: str
    conversation_history: List[str]
    current_step: int
    amount_recovered: float
    is_done: bool

class Action(BaseModel):
    message: str

class Reward(BaseModel):
    score: float
    reason: str

class EpisodeResult(BaseModel):
    task_name: str
    final_score: float
    amount_recovered: float
    steps_taken: int
    conversation_history: List[str]
    explanation: str
