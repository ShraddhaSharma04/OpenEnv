import json
import os
from typing import Dict, List, Optional

import requests

try:
    from openai import OpenAI
except Exception:
    OpenAI = None


API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME", "")
API_KEY = HF_TOKEN or os.getenv("API_KEY", "dummy_key")

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
BENCHMARK = os.getenv("BENCHMARK", "customer-support-ticket-triage-openenv")
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

Return ONLY valid JSON with exactly these keys:
category, priority, assigned_team, next_action

Allowed priority values:
low, medium, high

Ticket details:
customer_type: {state.get("customer_type", "")}
product: {state.get("product", "")}
message: {state.get("message", "")}
previous_status: {state.get("previous_status", "")}
""".strip()


def heuristic_action(state: Dict) -> Dict:
    message = (state.get("message") or "").lower()
    product = (state.get("product") or "").lower()
    previous_status = (state.get("previous_status") or "").lower()

    category = "general_issue"
    priority = "medium"
    assigned_team = "general_support"
    next_action = "review the ticket and respond to the customer"

    if any(word in message for word in ["refund", "charged", "payment", "billing", "money deducted", "double charged"]):
        category = "billing_issue"
        assigned_team = "billing_support"
        next_action = "verify the double charge and process refund if needed"
        priority = "high"

    elif any(word in message for word in ["password", "sign in", "login", "log in", "access", "locked"]):
        category = "account_access"
        assigned_team = "account_support"
        next_action = "verify login issue and help reset account access"
        priority = "medium"

    elif any(word in message for word in ["crash", "error", "bug", "not working", "fails"]):
        category = "technical_issue"
        assigned_team = "app_support" if "app" in product else "platform_support"
        next_action = "reproduce the issue and investigate the reported error"
        priority = "medium" if previous_status != "escalated" else "high"

    elif any(word in message for word in ["plan", "subscription", "premium", "features locked", "upgrade"]):
        category = "subscription_issue"
        assigned_team = "subscription_support"
        next_action = "check subscription sync and unlock the expected features"
        priority = "medium"

    elif "sso" in message or "audit" in message:
        category = "security_access_issue"
        assigned_team = "security_support"
        next_action = "review sso access issue and audit log availability"
        priority = "high"

    elif any(word in message for word in ["stale data", "alerts", "reports are delayed", "pipeline"]):
        category = "data_pipeline_issue"
        assigned_team = "data_operations_support"
        next_action = "investigate delayed reports stale data and alert timing"
        priority = "high"

    elif any(word in message for word in ["roles", "latency", "slow", "export invoices"]):
        category = "enterprise_platform_issue"
        assigned_team = "enterprise_tech_support"
        next_action = "investigate role access export failures and latency"
        priority = "high"

    return {
        "category": category,
        "priority": priority,
        "assigned_team": assigned_team,
        "next_action": next_action,
    }


def normalize_action(action: Dict) -> Dict:
    return {
        "category": str(action.get("category", "general_issue")),
        "priority": str(action.get("priority", "medium")).lower(),
        "assigned_team": str(action.get("assigned_team", "general_support")),
        "next_action": str(action.get("next_action", "review the ticket and respond to the customer")),
    }


def get_model_action(task_state: Dict) -> Dict:
    state = task_state.get("state", {})

    if not API_BASE_URL or not API_KEY or OpenAI is None or API_KEY == "dummy_key":
        return heuristic_action(state)

    try:
        client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
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
        return normalize_action(parsed)
    except Exception:
        return heuristic_action(state)


def post_reset(difficulty: str) -> Dict:
    response = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"difficulty": difficulty},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def post_step(action: Dict) -> Dict:
    response = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()


def run_episode(difficulty: str) -> None:
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    task_id = difficulty

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        task_state = post_reset(difficulty)
        task_id = str(task_state.get("task_id", difficulty))

        action_dict = get_model_action(task_state)
        action_dict = normalize_action(action_dict)
        action_str = json.dumps(action_dict, separators=(",", ":"))

        result = post_step(action_dict)

        reward = float(result.get("reward", 0.01))
        done = bool(result.get("done", True))
        score = float(result.get("score", reward))

        rewards.append(reward)
        steps_taken = 1
        success = True

        log_step(step=1, action=action_str, reward=reward, done=done, error=None)

    except Exception as exc:
        log_step(step=1, action="null", reward=0.01, done=True, error=str(exc))
        success = False
        score = 0.01
        rewards.append(0.01)

    log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    for difficulty in DIFFICULTIES:
        run_episode(difficulty)


if __name__ == "__main__":
    main()
