
# Customer Support Ticket Triage OpenEnv

A real-world OpenEnv environment for customer support ticket triage.

## Overview

This project simulates a customer support triage workflow where an AI agent reads support tickets and predicts:

- issue category
- priority level
- assigned support team
- next action

The environment is designed for reinforcement learning style interaction through:

- `reset()`
- `state()`
- `step()`

## Why this is a real-world environment

Customer support triage is a real operational workflow used by companies to route incoming issues correctly and efficiently. This environment models realistic ticket classification and escalation behavior.

## Task Levels

The environment includes 3 difficulty levels:

- `easy`
- `medium`
- `hard`

Each level contains multiple support tickets.

## Observation / State Space

Each task provides the following state:

- `customer_type`
- `product`
- `message`
- `previous_status`

Example state:

```json
{
  "customer_type": "individual",
  "product": "mobile app",
  "message": "App crashes every time I try to log in.",
  "previous_status": "open"
}

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