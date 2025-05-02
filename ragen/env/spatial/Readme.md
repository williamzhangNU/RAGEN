# Spatial Gym

The Spatial Gym is a gym environment for spatial reasoning.

## Main Components

1. `BaseEnv/room.py`: room (state / environment) for exploration and evaluation
    - SpatialGym interacts with room
    - Evaluation uses room to generate question and answer

2. `env.py`: SpatialGym environment
    - Main interface for agent to interact with the environment
    - Passive exploration: generate exploration history using DFS in `reset`
    - Semi-active exploration: agent ask about relationship between two objects in `step`
    - Active exploration: agent can only ask about one object relative to itself in `step`

3. `Evaluation.py`: Evaluation QA

## Exploration

### Passive Exploration
Exploration history is generated using DFS, and no exploration stage

### Semi-active Exploration
- Agent can ask about relationship between two objects
- Agent can not move in the room

### Active Exploration

Exploration:
- Agent can only ask about one object relative to itself
- Agent can only see the objects in front of it (NOTE -45~45 or -90~90 degree)
- Agent can move in the room

Actions:
- `move`: move to an object, format: "Move(A)"
- `rotate`: rotate to a specific direction, format: "Rotate(90)"
- `ask`: ask about relationship between one object and the agent, format: "Query(A)" for active, "Query(A, B)" for semi-active
- `return`: return to the original position, format: "Return()"

## Evaluation


