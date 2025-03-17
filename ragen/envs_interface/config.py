from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass

@dataclass
class MultiEnvInterfaceConfig:
    envs_size: int = 5  # Total number of environments to create
    envs_type: Union[str, List[str]] = "sokoban"  # Type(s) of environment
    env_configs: Optional[Dict[str, Any]] = None  # Configurations for different environment types
    max_episode_steps: int = 5  # Maximum steps per episode
    
    # You can add more configuration parameters specific to your multi-env setup
    auto_reset: bool = True  # Whether to automatically reset completed environments
    multi_action_sep: str = "|"  # Separator for multiple actions in LLM response