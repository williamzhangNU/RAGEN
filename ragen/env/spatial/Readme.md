# Spatial Gym

The Spatial Gym is a gym environment for text-based spatial reasoning.

## Main Components

1. `BaseEnv/room.py`: room (**state** / environment) for exploration and evaluation
    - SpatialGym interacts with room
    - Evaluation uses room to generate question and answer

2. `env.py`: SpatialGym environment
    - Main interface for agent to interact with the environment
    - Passive exploration: generate exploration history using DFS in `reset`
    - Semi-active exploration: agent ask about relationship between two objects in `step`
    - Active exploration: agent can only ask about one object relative to itself in `step`

3. `Evaluation.py`: Evaluation QA

## Exploration

### Active Exploration

1. Exploration:
- Agent can only see the objects in front of it (NOTE field of view: 90 degree)
- Ask:
    - Agent can only ask about one object relative to itself
    - **Agent can only ask visible object**
- Agent can move in the room

2. Actions:
- `move`: move to an object, format: "Move(A)"
- `rotate`: rotate to a specific direction, format: "Rotate(90)"
- `ask`: ask about relationship between one object and the agent, format: "Query(A)" for A object
- `return`: return to the original position, format: "Return()"
- `terminate`: terminate the exploration, format: "Terminate()"

### Passive Exploration

## Evaluation

1. Object preception: not included in text-based
2. Obejct relationship: 




## TODO
1. Change Evaluation task where spatial relationship involve two objects no agent, change it to allocentric (north, south, east, west)
