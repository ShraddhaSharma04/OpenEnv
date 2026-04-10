import json
import os
from typing import Dict, List, Optional

import requests
from openai import OpenAI

API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY = os.environ["API_KEY"]
MODEL_NAME = os.environ["MODEL_NAME"]
ENV_BASE_URL = os.environ["ENV_BASE_URL"]

BENCHMARK = os.getenv("BENCHMARK", "customer-support-ticket-triage-openenv")
DIFFICULTIES = ["easy", "medium", "hard"]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    err = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(task_state: Dict) -> str:
    state = task_state["state"]
    return f"""
You are a customer support triage agent.
Return ONLY valid JSON with exactly these keys:
category
priority
assigned_team
next_action
Ticket:
customer_type: {state["customer_type"]}
product: {state["product"]}
message: {state["message"]}
previous_status: {state["previous_status"]}
""".strip()

def run_episode(client: OpenAI, difficulty: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    task_id = difficulty

    try:
        reset_response = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"difficulty": difficulty},
            timeout=30,
        )
        reset_response.raise_for_status()
        task_state = reset_response.json()
        task_id = task_state["task_id"]

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        action_dict = get_model_action(client, task_state)
        action_str = json.dumps(action_dict, separators=(",", ":"))

        step_response = requests.post(
            f"{ENV_BASE_URL}/step",
            json={"action": action_dict},
            timeout=30,
        )
        step_response.raise_for_status()
        result = step_response.json()

        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", False))
        score = float(result.get("score", 0.0))

        rewards.append(reward)
        steps_taken = 1
        success = score > 0.0

        log_step(step=1, action=action_str, reward=reward, done=done, error=None)

    except Exception as exc:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        log_step(step=1, action="null", reward=0.00, done=True, error=str(exc))
        success = False
        score = 0.0
    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def warmup_llm(client: OpenAI) -> None:
    client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": "Reply with exactly: ok"},
            {"role": "user", "content": "ok"},
        ],
        temperature=0,
        max_tokens=2,
        stream=False,
    )


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    warmup_llm(client)
    for difficulty in DIFFICULTIES:
        run_episode(client, difficulty)


if __name__ == "__main__":
    main()
