# Spatial Gym

A text-based spatial reasoning environment where agents explore rooms and answer spatial relationship questions.

## Overview

The Spatial Gym enables agents to:
- Explore spatial environments through movement and observation
- Learn spatial relationships between objects
- Answer spatial reasoning questions based on their exploration

## Quick Start

```python
from ragen.env.spatial import SpatialGym

# Create and use the environment
env = SpatialGym(config)
observation = env.reset()

# Agent explores
action = "Movement: [Move(table), Rotate(90)]; Final: Observe()"
observation, reward, done, info = env.step(action)

# End exploration
action = "Movement: []; Final: Term()"
observation, reward, done, info = env.step(action)
```

## Exploration Modes

### Active Exploration
- Agent has 180° field of view
- Can only query visible objects
- Must move strategically to discover all relationships

### Passive Exploration
- System provides complete exploration history
- No agent movement required

## Available Actions

- `Move(object)` - Move to a specific object
- `Rotate(degrees)` - Rotate by specified degrees
- `Query(object)` - Ask about object's relationship to agent
- `Observe()` - Observe current surroundings
- `Return()` - Return to starting position
- `Terminate()` - End exploration phase

## Evaluation Tasks

The environment includes various spatial reasoning tasks (`ragen/env/spatial/Base/tos_base/evaluation/task_types.py`):
- **all_pairs**: Understanding all pairs of objects
- **rot**: Mental Rotation
- **rot_dual**: Mental Rotation (dual task)
- **pov**: Spatial relationships from different perspectives
- **rot_pov**: Mental rotation from a different perspective
- **e2a**: Allocentric representation
- **loc**: Determine where the agent is
- **false_belief**: False Belief with object rotation only
- **false_belief (w/ movement)**: False Belief with object movement

## Configuration

Environment behavior is controlled through `config.py`. See the Base module documentation for detailed technical information and extension guidelines.

## Git Submodule Integration

### Overview
The `Base` directory is now implemented as a git submodule, containing the core spatial reasoning components. This allows for better modularity and independent development of the base spatial reasoning framework.

### Initial Setup
When cloning this repository, you need to initialize and update the submodule:

```bash
# Clone the repository with submodules
git clone --recursive https://github.com/your-org/RAGEN.git

# Or if you've already cloned without --recursive
git clone https://github.com/your-org/RAGEN.git
cd RAGEN
git submodule update --init --recursive
```

### Working with the Submodule

#### Updating the Submodule
To pull the latest changes from the submodule repository:

```bash
# Navigate to the project root
cd ragen/env/spatial/Base
git pull origin main

# Commit the submodule update in the main repository
cd ../../../..
git add ragen/env/spatial/Base
git commit -m "Update Base submodule to latest version"
```

#### Checking Submodule Status
To see the current state of submodules:

```bash
git submodule status
```

#### Making Changes to the Submodule
If you need to make changes to the Base module:

1. Navigate to the submodule directory: `cd ragen/env/spatial/Base`
2. Make your changes and commit them
3. Push to the submodule repository: `git push origin main`
4. Once pushed, update the main repository to point to the new commit


## NOTE
1. After exploration, the agent will return the its original position and orientation.


## Run Evaluation

### Prompts
1. Overall Prompts at `ragen/env/spatial/prompts.py`
2. Action Prompts at `ragen/env/spatial/Base/tos_base/actions/actions.py`
3. Evaluation Prompts at `ragen/env/spatial/Base/tos_base/evaluation/tasks.py`

### Settings
1. Add or modify environment settings at `config/envs.yaml`
2. Environment config file: `ragen/env/spatial/config.py`, when you edit `config/envs.yaml`, you need to refer to `config.py`
3. Add or modify models to be evaluated at `config/evaluate_api_llm.yaml`

### Evaluation
1. The main config file is `config/evaluate_spatial.yaml`
2. Run `python -m ragen.llm_agent.agent_proxy` to run evaluation


## TODO
- [x] Testing efficiency of exploration (90 and 180) ✓
- [x] Sometimes not all evaluation tasks are finished
- [x] Change move: agent can only move to objects it observed
- [x] Add an object as original position?
- [x] Add reason for invalid action / input
- [x] Before exploration, first tell the agent what question it needs to answer
- [x] Write a script to evaluate all tasks and all models