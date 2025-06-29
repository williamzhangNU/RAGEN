# Base Module Organization

This directory contains the core components for the spatial reasoning environment, organized into logical modules for better maintainability and clarity.

## Directory Structure

```
Base/
├── __init__.py                 # Main module interface with organized exports
├── core/                       # Core data structures and fundamental classes
│   ├── __init__.py
│   ├── object.py              # Object and Agent classes
│   ├── room.py                # Room state management
│   ├── relationship.py        # Spatial relationship system
│   ├── graph.py               # Directional graph for tracking relationships
│   └── constant.py            # Constants and predefined configurations
├── actions/                    # Action system for agent interactions
│   ├── __init__.py
│   ├── base.py                # Abstract base action class and common functionality
│   └── actions.py             # Concrete action implementations and ActionSequence
├── managers/                   # High-level management components
│   ├── __init__.py
│   ├── exploration_manager.py # Manages exploration phase logic
│   └── evaluation_manager.py  # Manages evaluation tasks and scoring
├── evaluation/                 # Evaluation tasks and utilities
│   ├── __init__.py
│   ├── tasks.py               # All evaluation task implementations
│   └── task_factory.py        # Factory function for creating tasks
└── utils/                      # Utility functions
    ├── __init__.py
    ├── room_utils.py          # Room generation utilities
    ├── eval_utilities.py      # Evaluation helper functions
    └── parse_exp_input.py     # Input parsing utilities
```

## Module Overview

### Core (`core/`)
Contains the fundamental data structures and logic:

- **`object.py`**: Defines `Object` and `Agent` classes representing entities in the spatial environment
- **`room.py`**: Manages room state, object relationships, and spatial queries
- **`relationship.py`**: Implements the spatial relationship system (directions, relative positions)
- **`graph.py`**: Maintains directional graphs for tracking known/unknown relationships
- **`constant.py`**: Defines constants, object names, and predefined room configurations

### Actions (`actions/`)
Implements the action system for agent interactions:

- **`base.py`**: Abstract base class `BaseAction` with common functionality and `ActionResult`
- **`actions.py`**: All concrete action implementations:
  - `MoveAction`: Move to a target object
  - `RotateAction`: Rotate by specified degrees
  - `ReturnAction`: Return to starting position
  - `ObserveAction`: Observe spatial relationships
  - `TermAction`: Terminate exploration
  - `QueryAction`: Query specific object relationships
  - `ActionSequence`: Parse and manage action sequences

### Managers (`managers/`)
High-level management components:

- **`exploration_manager.py`**: Manages the exploration phase, tracking agent movement and discoveries
- **`evaluation_manager.py`**: Manages evaluation tasks, scoring, and progress tracking

### Evaluation (`evaluation/`)
Evaluation system components:

- **`tasks.py`**: All evaluation task implementations (direction, rotation, POV, etc.)
- **`task_factory.py`**: Factory function to create evaluation tasks from configuration

### Utils (`utils/`)
Utility functions and helpers:

- **`room_utils.py`**: Functions for generating rooms with different configurations
- **`eval_utilities.py`**: Helper functions for evaluation tasks
- **`parse_exp_input.py`**: Functions for parsing action inputs

## Usage

### Import from the module

```python
from ragen.env.spatial.Base import (
    # Core components
    Room, Object, Agent, DirectionalGraph,
    
    # Actions
    ActionSequence, MoveAction, ObserveAction,
    
    # Managers
    ExplorationManager, EvaluationManager,
    
    # Utilities
    generate_room
)
```

### Create a room and explore

```python
# Generate a room
room = generate_room(n_objects=3, np_random=np_random)

# Set up exploration
exploration_manager = ExplorationManager(room)

# Parse and execute actions
action_seq = ActionSequence.parse("Move(table); Observe()")
result, info = exploration_manager.execute_action_sequence(action_seq)
```

## Design Principles

1. **Separation of Concerns**: Each module has a clear, focused responsibility
2. **Dependency Management**: Core components don't depend on higher-level managers
3. **Clean Imports**: All imports use relative paths within the module
4. **Consistent Naming**: Follow Python naming conventions throughout
5. **Extensibility**: Easy to add new actions, evaluation tasks, or utilities

## Migration Notes

When updating imports from the old structure:

**Old:**
```python
from ragen.env.spatial.Base.action import MoveAction
from ragen.env.spatial.Base.EvaluationManager import EvaluationManager
```

**New:**
```python
from ragen.env.spatial.Base import MoveAction, EvaluationManager
```

The main `__init__.py` provides a unified interface to all important components, making imports cleaner and the API more discoverable. 