from typing import Dict, List, Tuple

from models.schemas import AgentAction, ExpectedOutput


def normalize_text(value: str) -> str:
    return value.strip().lower()


def keyword_match_score(next_action: str, expected_keywords: List[str]) -> Tuple[float, List[str], List[str]]:
    action_text = normalize_text(next_action)
    matched_keywords: List[str] = []
    missing_keywords: List[str] = []

    for keyword in expected_keywords:
        if normalize_text(keyword) in action_text:
            matched_keywords.append(keyword)
        else:
            missing_keywords.append(keyword)

    if not expected_keywords:
        return 0.0, matched_keywords, missing_keywords

    score = len(matched_keywords) / len(expected_keywords)
    return score, matched_keywords, missing_keywords


def safe_open_score(score: float) -> float:
    if score <= 0.0:
        return 0.01
    if score >= 1.0:
        return 0.99
    return round(score, 4)


def grade_agent_action(expected: ExpectedOutput, actual: AgentAction) -> Dict:
    category_correct = normalize_text(actual.category) == normalize_text(expected.category)
    priority_correct = normalize_text(actual.priority) == normalize_text(expected.priority)
    assigned_team_correct = normalize_text(actual.assigned_team) == normalize_text(expected.assigned_team)

    keyword_score, matched_keywords, missing_keywords = keyword_match_score(
        actual.next_action,
        expected.next_action_keywords,
    )

    category_points = 0.4 if category_correct else 0.0
    priority_points = 0.2 if priority_correct else 0.0
    assigned_team_points = 0.2 if assigned_team_correct else 0.0
    next_action_points = 0.2 * keyword_score

    raw_score = category_points + priority_points + assigned_team_points + next_action_points
    final_score = safe_open_score(raw_score)

    feedback = {
        "category_correct": category_correct,
        "priority_correct": priority_correct,
        "assigned_team_correct": assigned_team_correct,
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "keyword_score": round(keyword_score, 4),
        "raw_score": round(raw_score, 4),
    }

    return {
        "score": final_score,
        "reward": final_score,
        "feedback": feedback,
    }
