from typing import Optional

from fastapi import FastAPI, HTTPException

from envs.ticket_triage_env import TicketTriageEnv
from models.schemas import ResetRequest, StateResponse, StepRequest, StepResponse


app = FastAPI(
    title="Customer Support Ticket Triage OpenEnv",
    version="1.0.0",
    description="A real-world OpenEnv environment for customer support ticket triage.",
)

env = TicketTriageEnv()


@app.get("/")
def root():
    return {"message": "Customer Support Ticket Triage OpenEnv is running"}


@app.post("/reset", response_model=StateResponse)
def reset_environment(request: Optional[ResetRequest] = None):
    try:
        difficulty = request.difficulty if request else "easy"
        return env.reset(difficulty)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.get("/state", response_model=StateResponse)
def get_state():
    try:
        return env.state()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))


@app.post("/step", response_model=StepResponse)
def step_environment(request: StepRequest):
    try:
        return env.step(request.action)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
