from models.schemas import AgentAction, ExpectedOutput
from graders.grader import grade_agent_action

expected = ExpectedOutput(
    category="billing_issue",
    priority="high",
    assigned_team="billing_support",
    next_action_keywords=["double", "charge", "refund"]
)

actual = AgentAction(
    category="billing_issue",
    priority="high",
    assigned_team="account_support",
    next_action="Check double charge and process refund for the user"
)

result = grade_agent_action(expected, actual)
print(result)