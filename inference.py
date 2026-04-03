import os
from typing import Dict, List

import requests
from dotenv import load_dotenv

load_dotenv()

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:8000")


def get_baseline_action(task_state: Dict) -> Dict:
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


def run_single_episode(difficulty: str) -> Dict:
    reset_response = requests.post(
        f"{ENV_BASE_URL}/reset",
        json={"difficulty": difficulty},
        timeout=30
    )
    reset_response.raise_for_status()
    task_state = reset_response.json()

    action = get_baseline_action(task_state)

    step_response = requests.post(
        f"{ENV_BASE_URL}/step",
        json={"action": action},
        timeout=30
    )
    step_response.raise_for_status()
    result = step_response.json()

    return {
        "difficulty": difficulty,
        "task_id": task_state["task_id"],
        "action": action,
        "result": result
    }


def main() -> None:
    difficulties: List[str] = ["easy", "medium", "hard"]
    results = []

    print("Running baseline inference...\n")

    for difficulty in difficulties:
        episode_result = run_single_episode(difficulty)
        results.append(episode_result)

        print(f"Difficulty: {episode_result['difficulty']}")
        print(f"Task ID: {episode_result['task_id']}")
        print(f"Score: {episode_result['result']['score']}")
        print(f"Reward: {episode_result['result']['reward']}")
        print(f"Done: {episode_result['result']['done']}")
        print("-" * 50)

    average_score = sum(item["result"]["score"] for item in results) / len(results)
    print(f"\nAverage Score: {round(average_score, 4)}")


if __name__ == "__main__":
    main()