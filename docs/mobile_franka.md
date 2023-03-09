# Mobile Franka task specification

## Observation space

- size 32, continuous (box) space, from -inf to inf

Each observation consists of

- base position in x and y coordinates (2)
- base yaw angle (1)
- scaled arm DoF positions (9)
- base velocity in x and y (2)
- base angular velocity (1)
- scaled arm DoF velocities (9)
- arm left finger position in xyz (3)
- target position in xyz (3)
- agent id (2)

Agent id is included as agents are heterogenous and parameter sharing is utilized. If agents use completely separate parameters (networks) agent id can be ommited.

Currently both agents observe the same full observations.

## Action space

Because the actions for the agents are heterogenous, action padding is used. Action padding means that we pad the actions to be the largest space and each agent only uses the actions that belong to its action space. As we include agent id in the observations the policy is conditioned to give different actions for different agents.

- size 9, continuous (box) space, from -1 to 1

### Arm agent
- 9 actions control roughly the efforts for each joint (7 angles and 2 distances for fingers)

### Base agent
- first action controls the target velocity in forward direction
- second action controls the target angular velocity
- rest are discarded

## Reward function

Weighted linear combination of several terms:
- action penalty: squared sum of each arm action to penalize big movements of the arm
- distance to target: calculated from arm left finger position to target position with Euclidean norm
- joint limit penalty: penalize arm DoF positions when they are far from neutral position

Weights:
- action penalty: -0.01
- distance to target: -0.2
- joint limit penalty: -0.02 

## Termination

Episode is terminated when the episode length reaches 500 steps. After termination the environment is reset and the robot is moved back to it's initial position with small random variation in the initial joint positions.