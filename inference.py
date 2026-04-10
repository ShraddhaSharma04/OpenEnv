import json
import os
from typing import Dict, List, Optional

import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


API_BASE_URL = os.getenv("API_BASE_URL", "")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")

BENCHMARK = "customer-support-ticket-triage-openenv"
DIFFICULTIES = ["easy", "medium", "hard"]


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_prompt(state: Dict) -> str:
    return f"""
You are a customer support ticket triage assistant.
Return only valid JSON with these keys:
category, priority, assigned_team, next_action

Ticket:
customer_type: {state.get("customer_type", "")}
product: {state.get("product", "")}
message: {state.get("message", "")}
previous_status: {state.get("previous_status", "")}
""".strip()


def heuristic_action(state: Dict) -> Dict:
    message = (state.get("message") or "").lower()

    category = "general"
    priority = "medium"
    assigned_team = "general_support"
    next_action = "review and respond to customer"

    if any(word in message for word in ["refund", "charged", "payment", "billing", "money deducted"]):
        category = "billing"
        assigned_team = "billing_team"
        next_action = "verify transaction and assist with billing issue"
        priority = "high" if "deducted" in message or "charged" in message else "medium"

    elif any(word in message for word in ["login", "password", "otp", "access", "account locked"]):
        category = "account_access"
        assigned_team = "account_support"
        next_action = "verify account and help restore access"
        priority = "high" if "locked" in message else "medium"

    elif any(word in message for word in ["delivery", "shipment", "late order", "delayed order", "order delayed"]):
        category = "delivery"
        assigned_team = "logistics_team"
        next_action = "check shipment status and update customer"
        priority = "medium"

    elif any(word in message for word in ["bug", "crash", "not working", "error", "issue in app"]):
        category = "technical_issue"
        assigned_team = "technical_support"
        next_action = "collect technical details and troubleshoot"
        priority = "high" if "crash" in message or "error" in message else "medium"

    return {
        "category": category,
        "priority": priority,
        "assigned_team": assigned_team,
        "next_action": next_action,
    }


def get_model_action(task_state: Dict) -> Dict:
    state = task_state.get("state", {})

    if not API_BASE_URL or not HF_TOKEN or OpenAI is None:
        return heuristic_action(state)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": build_prompt(state)},
            ],
            temperature=0,
            max_tokens=200,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        parsed = json.loads(text)
        return {
            "category": parsed.get("category", "general"),
            "priority": parsed.get("priority", "medium"),
            "assigned_team": parsed.get("assigned_team", "general_support"),
            "next_action": parsed.get("next_action", "review and respond to customer"),
        }
    except Exception:
        return heuristic_action(state)


def run_episode(difficulty: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    task_id = difficulty

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        reset_response = requests.post(
            f"{ENV_BASE_URL}/reset",
            json={"difficulty": difficulty},
            timeout=30,
        )
        reset_response.raise_for_status()
        task_state = reset_response.json()
        task_id = task_state.get("task_id", difficulty)

        action_dict = get_model_action(task_state)
        action_str = json.dumps(action_dict, separators=(",", ":"))

        step_response = requests.post(
            f"{ENV_BASE_URL}/step",
            json={"action": action_dict},
            timeout=30,
        )
        step_response.raise_for_status()
        result = step_response.json()

        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", True))
        score = float(result.get("score", reward))

        rewards.append(reward)
        steps_taken = 1
        success = score >= 0.0

        log_step(step=1, action=action_str, reward=reward, done=done, error=None)

    except Exception as exc:
        log_step(step=1, action="null", reward=0.00, done=True, error=str(exc))
        success = False
        score = 0.0

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    for difficulty in DIFFICULTIES:
        run_episode(difficulty)


if __name__ == "__main__":
    main()
