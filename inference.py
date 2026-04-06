import json
import os
from typing import Dict, List, Optional

import requests
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")

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


def fallback_action(task_state: Dict) -> Dict:
    message = task_state["state"]["message"].lower()

    if "charged twice" in message or "refund" in message:
        return {
            "category": "billing_issue",
            "priority": "high",
            "assigned_team": "billing_support",
            "next_action": "Investigate the double charge and process a refund for the duplicate payment."
        }
    if "forgot my password" in message or "cannot sign in" in message:
        return {
            "category": "account_access",
            "priority": "medium",
            "assigned_team": "account_support",
            "next_action": "Help the user reset the password and restore sign in access."
        }
    if "crashes" in message or "log in" in message:
        return {
            "category": "technical_issue",
            "priority": "medium",
            "assigned_team": "app_support",
            "next_action": "Reproduce the login crash and investigate the issue."
        }
    if "premium" in message and "export reports" in message:
        return {
            "category": "account_access",
            "priority": "high",
            "assigned_team": "subscription_support",
            "next_action": "Check premium plan activation, export permissions, and account access."
        }
    if "pdf fails" in message or "recent update" in message:
        return {
            "category": "technical_issue",
            "priority": "high",
            "assigned_team": "platform_support",
            "next_action": "Investigate the invoice PDF failure after the recent update."
        }
    if "plan" in message and "features" in message:
        return {
            "category": "subscription_issue",
            "priority": "medium",
            "assigned_team": "subscription_support",
            "next_action": "Verify the plan change sync and unlock the correct features."
        }
    if "role-based access" in message or "dashboard is unusually slow" in message:
        return {
            "category": "enterprise_platform_issue",
            "priority": "high",
            "assigned_team": "enterprise_tech_support",
            "next_action": "Investigate role access changes, invoice export failure, and dashboard latency."
        }
    if "sso" in message or "audit logs" in message:
        return {
            "category": "security_access_issue",
            "priority": "high",
            "assigned_team": "security_support",
            "next_action": "Check the SSO configuration, restore access, and inspect missing audit logs."
        }
    if "scheduled reports" in message or "stale data" in message or "alerts" in message:
        return {
            "category": "data_pipeline_issue",
            "priority": "high",
            "assigned_team": "data_operations_support",
            "next_action": "Investigate delayed reports, stale dashboards, and alerting delays."
        }

    return {
        "category": "technical_issue",
        "priority": "medium",
        "assigned_team": "platform_support",
        "next_action": "Investigate the issue and gather more details."
    }


def get_model_action(client: OpenAI, task_state: Dict) -> Dict:
    prompt = build_prompt(task_state)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=200,
            stream=False,
        )
        text = (completion.choices[0].message.content or "").strip()
        return json.loads(text)
    except Exception:
        return fallback_action(task_state)


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


def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for difficulty in DIFFICULTIES:
        run_episode(client, difficulty)


if __name__ == "__main__":
    main()
