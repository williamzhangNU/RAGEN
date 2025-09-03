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
    
    # New fine-grained control parameters (mutually exclusive with main/n_objects)
    fix_room_size: Optional[List[List[int]]] = None  # e.g., [[5,5], [6,6], [4,4]]
    fix_object_n: Optional[List[int]] = None         # e.g., [3, 4, 2]
    
    # Room size uniformity parameter (works with main, mutually exclusive with fix_room_size)
    same_room_size: bool = False                     # When True, all rooms use the same size as main room
    
    # Object density parameter
    objects_per_area: Optional[float] = None         # Objects per unit area (overrides n_objects if provided)
    
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
        self._validate_room_parameters()


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
    
    def _validate_room_parameters(self):
        """Validate room configuration parameters."""
        use_fixed = self.fix_room_size is not None or self.fix_object_n is not None
        
        if use_fixed and self.same_room_size:
            raise ValueError("fix_room_size and same_room_size are mutually exclusive")
        if self.objects_per_area and use_fixed:
            raise ValueError("objects_per_area cannot be used with fix_room_size/fix_object_n")
        
        if use_fixed:
            self._validate_fixed_params()
        if self.same_room_size:
            self._validate_same_room_size()
        if self.objects_per_area:
            self._validate_objects_per_area()

    def _validate_fixed_params(self):
        """Validate fix_room_size and fix_object_n parameters."""
        if not (self.fix_room_size and self.fix_object_n):
            raise ValueError("fix_room_size and fix_object_n must be provided together")
        
        expected = self.level + 1
        if len(self.fix_room_size) != expected or len(self.fix_object_n) != expected:
            raise ValueError(f"Lists must have {expected} elements (level + 1)")
        
        # Convert ListConfig
        if isinstance(self.fix_room_size, ListConfig):
            self.fix_room_size = OmegaConf.to_container(self.fix_room_size, resolve=True)
        if isinstance(self.fix_object_n, ListConfig):
            self.fix_object_n = OmegaConf.to_container(self.fix_object_n, resolve=True)
        
        # Validate room sizes and object counts
        for i, (size, count) in enumerate(zip(self.fix_room_size, self.fix_object_n)):
            if not isinstance(size, (list, tuple)) or len(size) != 2 or any(s <= 0 for s in size):
                raise ValueError(f"fix_room_size[{i}] must be [width, height] with positive values")
            if not isinstance(count, int) or count < 0:
                raise ValueError(f"fix_object_n[{i}] must be non-negative integer")
        
        total = sum(self.fix_object_n)
        if total < 3:
            raise ValueError(f"Total objects ({total}) must be at least 3")
        if total < 5 and any(t.get('task_type') in ['rot', 'rot_dual'] for t in self.eval_tasks):
            import warnings
            warnings.warn(f"Only {total} objects for rotation tasks. Consider ≥5 for better separation.")

    def _validate_same_room_size(self):
        """Validate same_room_size parameter."""
        if self.level == 0:
            import warnings
            warnings.warn("same_room_size=True has no effect when level=0")
        if not self.main:
            raise ValueError("main parameter required when using same_room_size=True")

    def _validate_objects_per_area(self):
        """Validate objects_per_area parameter."""
        if self.objects_per_area <= 0:
            raise ValueError("objects_per_area must be positive")
            

    



    def get_room_config(self) -> Dict[str, Any]:
        """Get configuration for room generation."""
        config = {
            'room_size': self.room_size,
            'level': self.level,
            # 'candidate_objects': self.candidate_objects,
        }
        
        # Use fixed parameters if provided, otherwise use traditional parameters
        if self.fix_room_size is not None and self.fix_object_n is not None:
            config.update({
                'fix_room_size': self.fix_room_size,
                'fix_object_n': self.fix_object_n,
                'n_objects': sum(self.fix_object_n),  # Total objects (for compatibility)
                'main': None,  # Not used when fix_room_size is provided
                'same_room_size': False,  # Not used when fix_room_size is provided
            })
        else:
            config.update({
                'n_objects': self.n_objects,
                'main': self.main,
                'same_room_size': self.same_room_size,
                'objects_per_area': self.objects_per_area,
            })
        
        # Always pass eval_tasks for validation
        config.update({
            'eval_tasks': self.eval_tasks,
            'min_angle_eps': 30.0,  # Default minimum angle separation
            'max_retries': 10,      # Maximum retry attempts
        })
        
        return config
    
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
