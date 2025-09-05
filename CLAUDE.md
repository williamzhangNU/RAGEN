# RAGEN - Spatial Environment Development Guide

This guide focuses on the **spatial reasoning environment** (`ragen/env/spatial`) in the RAGEN project, which provides a text-based spatial reasoning environment for training and evaluating LLM agents.

## Project Overview

RAGEN (Reasoning AGENT) leverages reinforcement learning to train LLM reasoning agents in interactive environments. The spatial environment specifically focuses on spatial reasoning tasks where agents explore rooms and answer questions about spatial relationships between objects.

## Core Components

### Main Environment (`ragen/env/spatial/`)

- **`env.py`** - Main SpatialGym environment class implementing the Gymnasium interface
- **`config.py`** - Configuration dataclass for environment parameters 
- **`prompts/prompts.py`** - Prompt templates for different phases and tasks
- **`utils/`** - Utility functions for visualization, logging, and action processing

### Base Module (`ragen/env/spatial/Base/`)

The Base directory is a git submodule containing core spatial reasoning components:

- **Core** (`tos_base/core/`) - Room, Agent, Object, and Graph representations
- **Managers** (`tos_base/managers/`) - Exploration, Evaluation, and Cognitive Map managers
- **Actions** (`tos_base/actions/`) - Available agent actions (Move, Rotate, Query, etc.)
- **Evaluation** (`tos_base/evaluation/`) - Task types and evaluation logic

### Key Files

- `ragen/env/spatial/env.py:65` - SpatialGym class definition
- `ragen/env/spatial/config.py:8` - SpatialGymConfig with all parameters
- `ragen/env/spatial/Readme.md` - Detailed spatial environment documentation

## Environment Features

### Exploration Modes

1. **Active Exploration** - Agent actively explores with limited field of view
2. **Passive Exploration** - Agent receives complete exploration history

### Evaluation Tasks

Available in `ragen/env/spatial/Base/tos_base/evaluation/task_types.py`:
- **rot** - Mental Rotation
- **pov** - Perspective-taking 
- **e2a** - Allocentric representation
- **false_belief** - Theory of mind tasks

### Action Types

- `Move(object)` - Move to a specific object
- `Rotate(degrees)` - Rotate by specified degrees  
- `Query(object)` - Ask about object's spatial relationship
- `Observe()` - Observe current surroundings
- `Terminate()` - End exploration phase

## Development Commands

### Environment Setup

```bash
# Initialize git submodules (required for Base module)
git submodule update --init --recursive

# Install dependencies
pip install -r requirements.txt
```

### Running Evaluations

```bash
# Run spatial evaluation with default config
python -m ragen.llm_agent.agent_proxy

# Custom evaluation config
python -m ragen.llm_agent.agent_proxy --config config/evaluate_spatial.yaml
```

### Configuration Files

- **Main config**: `config/evaluate_spatial.yaml`
- **Environment settings**: `config/envs.yaml` 
- **Model settings**: `config/evaluate_api_llm.yaml`

### Testing

```bash
# Run spatial environment tests
python -m pytest tests/env/spatial/

# Test specific functionality
python -m pytest tests/env/spatial/test_env.py::test_spatial_gym_basic
```

## Development Workflow

### Adding New Tasks

1. Define task type in `Base/tos_base/evaluation/task_types.py`
2. Implement evaluation logic in `Base/tos_base/evaluation/tasks.py` 
3. Update configuration in `config/envs.yaml`

### Modifying Actions

1. Implement action class in `Base/tos_base/actions/actions.py`
2. Update action parsing in `Base/tos_base/actions/__init__.py`
3. Add corresponding prompt templates

### Submodule Updates

```bash
# Update Base submodule to latest version
cd ragen/env/spatial/Base
git pull origin main

# Commit submodule update in main repo
cd ../../../..
git add ragen/env/spatial/Base
git commit -m "Update Base submodule to latest version"
```

## Configuration Reference

### Key Config Parameters (`config.py`)

```python
@dataclass
class SpatialGymConfig:
    room_size: List[int] = [10, 10]        # Room dimensions
    n_objects: int = 3                      # Number of objects
    exp_type: str = 'passive'               # 'active' or 'passive'
    field_of_view: int = 90                 # 90 or 180 degrees
    max_exp_steps: int = 100                # Max exploration steps
    eval_tasks: List[Dict] = [...]          # Evaluation tasks list
```

### Environment Settings (`config/envs.yaml`)

```yaml
spatial_gym:
  name: "spatial-basic"
  room_size: [10, 10]
  n_objects: 5
  exp_type: "active"
  field_of_view: 180
  eval_tasks:
    - task_type: "rot"
      task_kwargs: {}
```

## Troubleshooting

### Common Issues

1. **Submodule not initialized**
   ```bash
   git submodule update --init --recursive
   ```

2. **Config validation errors**
   - Check `config/envs.yaml` syntax
   - Ensure task types exist in `task_types.py`
   - Verify object names in `CANDIDATE_OBJECTS`

3. **Action parsing failures**
   - Check action format: `"Movement: [Move(table)]; Final: Observe()"`
   - Ensure object names match room objects
   - Validate rotation degrees (multiples of 90)

### Debugging Tips

- Use `env.render()` to view current observation
- Check `env.turn_logs` for detailed execution history
- Enable verbose logging in config for detailed traces
- Visualize rooms using `utils/visualization/visualization.py`

### Performance Optimization

- Use `field_of_view: 180` for faster exploration
- Reduce `max_exp_steps` for quicker iterations  
- Enable `observation_mode: "dir"` for simpler observations
- Use passive exploration for reproducible results

## File Locations

- Main environment: `ragen/env/spatial/env.py:65`
- Configuration: `ragen/env/spatial/config.py:8`
- Action definitions: `ragen/env/spatial/Base/tos_base/actions/actions.py`
- Task types: `ragen/env/spatial/Base/tos_base/evaluation/task_types.py`
- Evaluation logic: `ragen/env/spatial/Base/tos_base/evaluation/tasks.py`

## Quick Reference

### Basic Usage

```python
from ragen.env.spatial import SpatialGym, SpatialGymConfig

# Create environment
config = SpatialGymConfig(exp_type='active', field_of_view=180)
env = SpatialGym(config)

# Reset and step
obs, info = env.reset()
obs, reward, done, info = env.step("Movement: [Move(table)]; Final: Observe()")
```

### Evaluation Flow

1. **Reset** - Generate room and initialize managers
2. **Exploration** - Agent explores room (active) or receives history (passive)  
3. **Evaluation** - Agent answers spatial reasoning questions
4. **Analysis** - Extract metrics from managers

This guide covers the essential aspects of developing with the RAGEN spatial environment. For more detailed information, refer to the individual README files in each module.