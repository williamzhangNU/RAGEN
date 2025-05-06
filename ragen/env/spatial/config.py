from dataclasses import dataclass, field, fields
from typing import Tuple, Optional, Dict, List
from ragen.env.spatial.Base.object import Object
from ragen.env.spatial.Base.constant import CANDIDATE_OBJECTS

@dataclass
class SpatialGymConfig:
    """
    Config for the SpatialGym

    Parameters:
        dim_room: Tuple containing room dimensions (width, height)
        dim_x: Optional parameter for room width (deprecated, use dim_room)
        dim_y: Optional parameter for room height (deprecated, use dim_room)
        candidate_objects: List of objects that can be placed in the room
        generation_type: Type of room generation ('rand', 'rot', 'a2e', 'pov')
        exp_type: Exploration type ('passive', 'semi', 'active')
        perspective: Perspective of exploration ('ego' or 'allo')
        eval_tasks: List of evaluation tasks, for reward calculation
        render_mode: Rendering mode
    """
    # Room config parameters
    room_range: List[int] = field(default_factory=lambda: [-10, 10])
    n_objects: int = 3
    candidate_objects: List[str] = field(default_factory=lambda: CANDIDATE_OBJECTS)
    generation_type: str = "rand"
    
    # Spatial gym parameters
    exp_type: str = 'semi'
    perspective: str = 'ego'
    eval_tasks: List[Dict] = field(default_factory=lambda: [{"task_type": "dir", "task_kwargs": {}}])
    max_exp_steps: int = 100
    render_mode: str = "text"

    def __post_init__(self):
        
        # Validate generation_type
        valid_generation_types = ["rand", "rot", "a2e", "pov"]
        assert self.generation_type in valid_generation_types, f"generation_type must be one of {valid_generation_types}"
        
        # Validate perspective
        assert self.perspective in ["ego", "allo"], f"perspective must be one of {['ego', 'allo']}"

        # Validate perspective and generation_type
        if self.generation_type in ['rot', 'pov']:
            assert self.perspective == 'ego', "pov and rot generation type only support ego perspective"
        
        # Validate exp_type
        valid_exp_types = ["passive", "semi", "active"]
        assert self.exp_type in valid_exp_types, f"exp_type must be one of {valid_exp_types}"


        # Validate eval_tasks
        valid_eval_tasks = ["dir", "rot", "pov", "a2e", "e2a", "rev"]
        assert len(self.eval_tasks) > 0, "eval_tasks must be non-empty"
        assert all(task['task_type'] in valid_eval_tasks for task in self.eval_tasks), f"eval_tasks must be a subset of {valid_eval_tasks}"

        # Validate render_mode
        assert self.render_mode == 'text', "Only support text rendering for now"

        # Validate task compatibility with perspective
        if self.perspective == 'ego':
            assert 'a2e' not in self.eval_tasks, "a2e is only supported for allocentric exploration"
        else:
            assert all(task not in ['rot', 'pov', 'e2a'] for task in self.eval_tasks), "invalid eval_tasks for allo perspective"

    def get_room_config(self):
        return {
            'room_range': self.room_range,
            'candidate_objects': self.candidate_objects,
            'generation_type': self.generation_type,
            'n_objects': self.n_objects,
            'perspective': self.perspective,  
        }
    
    # def to_dict(self):
    #     return {field.name: getattr(self, field.name) for field in fields(self)}
    def to_dict(self):
        from omegaconf import OmegaConf
        return {
            'room_range': self.room_range,
            'candidate_objects': self.candidate_objects,
            'generation_type': self.generation_type,
            'n_objects': self.n_objects,    
            'exp_type': self.exp_type,
            'perspective': self.perspective,
            'eval_tasks': OmegaConf.to_container(self.eval_tasks, resolve=True),
            'max_exp_steps': self.max_exp_steps,
            'render_mode': self.render_mode,
        }
    

@dataclass
class BaseEvaluationConfig:
    """
    Config for the base evaluation task
    """
    @classmethod
    def create_config(cls, config_type: str, **kwargs):
        """
        Factory method to create evaluation configs of different types
        
        Args:
            config_type: Type of evaluation config to create
            **kwargs: Additional arguments to pass to the config constructor
            
        Returns:
            An instance of the specified evaluation config
        """
        config_map = {
            'all_pairs': AllPairsEvaluationConfig,
            'dir': DirEvaluationConfig,
            'rot': RotEvaluationConfig,
            'pov': PovEvaluationConfig,
            'rev': ReverseDirEvaluationConfig,
            'a2e': A2EEvaluationConfig,
            'e2a': E2AEvaluationConfig,
        }
        
        if config_type not in config_map:
            raise ValueError(f"Unknown config type: {config_type}. Must be one of {list(config_map.keys())}")
        
        return config_map[config_type](**kwargs)
    
    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}
    
    @classmethod
    def from_dict(cls, config_dict: dict):
        """
        Factory method to create evaluation config from dictionary
        
        Args:
            config_dict: Dictionary containing configuration parameters
            
        Returns:
            An instance of the appropriate evaluation config
        """
        config_type = config_dict.pop('type', None)
        if config_type and config_type != cls.__name__:
            config_map = {
                'AllPairsEvaluationConfig': AllPairsEvaluationConfig,
                'DirEvaluationConfig': DirEvaluationConfig,
                'RotEvaluationConfig': RotEvaluationConfig,
                'PovEvaluationConfig': PovEvaluationConfig
            }
            return config_map[config_type].from_dict(config_dict)
        return cls(**config_dict)

@dataclass
class AllPairsEvaluationConfig(BaseEvaluationConfig):
    """
    Config for the all pairs evaluation task
    """
    pass

@dataclass
class DirEvaluationConfig(BaseEvaluationConfig):
    """
    Config for the direction evaluation task
    The movement can be 'static', 'object_move', 'agent_move', 'agent_turn'
        - static: nothing moves, all objects remain in place
        - object_move: the object moves <dir> of another object
        - agent_move: the agent moves <dir> of another object
        - agent_turn: the agent turns <degree>
    """
    movement: str = 'static'  # Options: 'static', 'object_move', 'agent_move', 'agent_turn'
    def __post_init__(self):
        assert self.movement in ['static', 'object_move', 'agent_move', 'agent_turn'],\
            "movement must be one of ['static', 'object_move', 'agent_move', 'agent_turn']"



@dataclass
class RotEvaluationConfig(BaseEvaluationConfig):
    """
    Config for the rotation evaluation task
    The rotation can be 'clockwise', 'counterclockwise'
        - clockwise: the agent turns clockwise
        - counterclockwise: the agent turns counterclockwise
    """
    turn_direction: str = 'clockwise'
    def __post_init__(self):
        assert self.turn_direction in ['clockwise', 'counterclockwise'],\
            "turn_direction must be one of ['clockwise', 'counterclockwise']"

@dataclass
class PovEvaluationConfig(BaseEvaluationConfig):
    """
    Config for the point of view evaluation task
    """
    pass

@dataclass
class ReverseDirEvaluationConfig(BaseEvaluationConfig):
    """
    Config for the reverse direction evaluation task
    """
    pass


@dataclass
class A2EEvaluationConfig(BaseEvaluationConfig):
    """
    Config for the A2E evaluation task
    """
    pass


@dataclass
class E2AEvaluationConfig(BaseEvaluationConfig):
    """
    Config for the E2A evaluation task
    """
    pass

if __name__ == "__main__":
    config = RotEvaluationConfig()
    print(config.to_dict())

    config = SpatialGymConfig()
    print(config.to_dict())