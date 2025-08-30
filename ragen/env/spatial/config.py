from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from omegaconf import ListConfig, OmegaConf, DictConfig

from ragen.env.spatial.Base.tos_base import CANDIDATE_OBJECTS
from ragen.env.spatial.Base.tos_base.evaluation.task_types import EvalTaskType

@dataclass
class SpatialGymConfig:
    """
    Configuration for the SpatialGym environment.
    
    Parameters:
        name: Identifier for this configuration
        room_size: Size for room dimensions
        n_objects: Number of objects in the room
        candidate_objects: List of objects that can be placed in the room
        exp_type: Exploration type ('passive', 'active')
        field_of_view: Field of view in degrees (90 or 180)
        eval_tasks: List of evaluation tasks with their configurations
        max_exp_steps: Maximum exploration steps for active exploration
        render_mode: Rendering mode (currently only 'text' supported)
    """
    # Configuration name
    name: str = "default"
    
    # Room configuration
    room_size: List[int] = field(default_factory=lambda: [10, 10])
    n_objects: int = 3
    candidate_objects: List[str] = field(default_factory=lambda: CANDIDATE_OBJECTS)
    level: int = 0
    main: int = 6
    
    # Exploration configuration
    exp_type: str = 'passive'
    field_of_view: int = 90
    observation_mode: str = "full"
    max_exp_steps: int = 100
    proxy_agent_config: dict = field(default_factory=lambda: {"type": "analyst", "delegate": "oracle"})
    
    # Evaluation configuration
    eval_tasks: List[Dict[str, Any]] = field(default_factory=lambda: [{"task_type": "rot", "task_kwargs": {}}])
    
    # prompt configuration
    prompt_config: dict = field(default_factory=lambda: {"topdown": False, "cogmap": False, "type": "shorter"})

    # cognitive map configuration
    cogmap_config: dict = field(default_factory=lambda: {"cogmap_type": "standard", "pos_allow_scale": True, "scope": "all"})


    # Rendering configuration
    render_mode: str = "text"
    kwargs: Dict = None
    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.room_size[0] > 0 and self.room_size[1] > 0, "room_size must be positive"
        self._validate_exp_type()
        self._validate_field_of_view()
        self._validate_eval_tasks()
        self._validate_render_mode()


    def _validate_exp_type(self):
        """Validate exp_type parameter."""
        valid_exp_types = ["passive", "active"]
        if self.exp_type not in valid_exp_types:
            raise ValueError(f"exp_type must be one of {valid_exp_types}")

    def _validate_field_of_view(self):
        """Validate field_of_view parameter."""
        assert self.field_of_view == 90, "field_of_view must be 90 degrees"

    def _validate_eval_tasks(self):
        """Validate eval_tasks parameter."""
        valid_eval_tasks = EvalTaskType.get_short_names()

        if isinstance(self.eval_tasks, ListConfig):
            self.eval_tasks = OmegaConf.to_container(self.eval_tasks, resolve=True)
        if isinstance(self.room_size, ListConfig):
            self.room_size = OmegaConf.to_container(self.room_size, resolve=True)
        if isinstance(self.prompt_config, DictConfig):
            self.prompt_config = OmegaConf.to_container(self.prompt_config, resolve=True)
        if not self.eval_tasks:
            raise ValueError("eval_tasks must be non-empty")
        
        for i, task in enumerate(self.eval_tasks):
            if not isinstance(task, dict) or 'task_type' not in task:
                raise ValueError("Each eval_task must be a dict with 'task_type' key")
            
            task_type = task['task_type']
            if task_type not in valid_eval_tasks:
                raise ValueError(f"task_type '{task_type}' must be one of {valid_eval_tasks}")
            
            # Validate task-specific parameters
            task_kwargs = task.get('task_kwargs', {})
            self._validate_task_kwargs(task_type, task_kwargs)

    def _validate_task_kwargs(self, task_type: str, kwargs: Dict[str, Any]):
        """Validate task-specific parameters."""
        # if task_type == 'dir':
        #     movement = kwargs.get('movement', 'static')
        #     valid_movements = ['static', 'object_move', 'agent_move', 'agent_turn']
        #     if movement not in valid_movements:
        #         raise ValueError(f"dir task movement must be one of {valid_movements}")
        
        if task_type == 'rot':
            turn_direction = kwargs.get('turn_direction', 'clockwise')
            valid_directions = ['clockwise', 'counterclockwise']
            if turn_direction not in valid_directions:
                raise ValueError(f"rot task turn_direction must be one of {valid_directions}")

    def _validate_render_mode(self):
        """Validate render_mode parameter."""
        if self.render_mode != 'text':
            raise ValueError("Only 'text' rendering mode is currently supported")
    



    def get_room_config(self) -> Dict[str, Any]:
        """Get configuration for room generation."""
        return {
            'room_size': self.room_size,
            'n_objects': self.n_objects,
            'level': self.level,
            'main': self.main,
            # 'candidate_objects': self.candidate_objects,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'name': self.name,
            'room_size': self.room_size,
            'n_objects': self.n_objects,    
            'exp_type': self.exp_type,
            'field_of_view': self.field_of_view,
            'eval_tasks': self.eval_tasks,
            'max_exp_steps': self.max_exp_steps,
            'render_mode': self.render_mode,
            'prompt_config': self.prompt_config,
            'model': self.kwargs['model']
            # 'candidate_objects': self.candidate_objects,
        }
