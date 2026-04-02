#Customer Support Ticket Triage OpenEnv

This project is a real-world OpenEnv environment for AI agent to learn customer support ticket triage.

This agent reads a customer support ticket and predicts
-category
-priority
-assigned team
-next action

This environment will support
-reset()
-state()
-step()

It includes:
-easy, medium, hard tasks
-reward logic with partial scoring
-graders returning scores from 0.0 to 1.0
-inference script
-deployment - ready structure