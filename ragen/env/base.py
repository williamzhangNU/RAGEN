from abc import ABC, abstractmethod
import re
from typing import Optional, List, Tuple, Any, Dict
from transformers import AutoTokenizer
import torch

class BaseEnv(ABC):
    """
    Abstract base class for all environments.
    The class needs to handle text-based input, input may be invalid
        - Environment will track the total reward for the trajectory

    """
    def __init__(self):
        pass

    @abstractmethod
    def reset(self, seed=None) -> Any:
        """
        Reset the environment.
        NOTE: the environment should be same for the same seed
        Returns:
            rendered environment
        """
        pass

    @abstractmethod
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        """
        Execute one step in the environment.
        NOTE should also handle predefined invalid action (0)
        Args:
            action: Action to take, must be in action space, or default invalid action
            
        Returns:
            observation (rendered environment), reward, done, info
        """
        pass

    @abstractmethod
    def render(self, mode: str = 'text') -> Any:
        """Render the environment."""
        pass


class BaseDiscreteActionEnv(BaseEnv, ABC):
    """
    Abstract base class for environments with discrete action spaces
    This class provides common functionality for environments like FrozenLakeEnv and SokobanEnv.
    """
    GRID_LOOKUP = {} # define the mapping from integer to string for rendering
    ACTION_LOOKUP = {} # define the mapping from integer to action string
    INVALID_ACTION = 0 # default invalid action
    PENALTY_FOR_INVALID = -1 # penalty for invalid action

    def get_all_actions(self) -> List[int]:
        """Get list of all valid actions."""
        pass


class BaseLanguageBasedEnv(BaseEnv, ABC):
    """
    Abstract base class for environments with language-based action space environment
    This class provides common functionality for environments like countdown from TinyZero
    """

    ACTION_LOOKUP = {} # TODO modify this as a method so can be called in a unified way
    INVALID_ACTION = "" # default invalid action

    def get_all_actions(self):
        raise NotImplementedError("Language-based environment does not have a finite action space")