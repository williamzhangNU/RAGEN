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
action = "Move(table)"
observation, reward, done, info = env.step(action)

# Agent observes
action = "Observe()"
observation, reward, done, info = env.step(action)

# End exploration
action = "Terminate()"
observation, reward, done, info = env.step(action)
```

## Exploration Modes

### Active Exploration
- Agent has 90° field of view
- Can only query visible objects
- Must move strategically to discover all relationships

### Semi-Active Exploration  
- Agent can ask about any two objects
- More flexible than active mode

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

The environment includes various spatial reasoning tasks:
- **Direction**: Understanding cardinal directions (N, S, E, W)
- **Rotation**: Understanding rotational relationships  
- **Point of View**: Spatial relationships from different perspectives
- **Object Relations**: Relative positions between objects

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
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit them
4. Push to the submodule repository: `git push origin feature/your-feature`
5. Create a pull request in the submodule repository
6. Once merged, update the main repository to point to the new commit

**Important**: All imports from the Base module should use the path `ragen.env.spatial.Base.tos_base` instead of `ragen.env.spatial.Base`.

## TODO
- [x] Testing efficiency of exploration (90 and 180)
- [x] Change move: agent can only move to objects it observed
- [x] Add an object as original position?
- [x] Add reason for invalid action / input

## NOTE
1. After exploration, the agent will return the its original position and orientation.