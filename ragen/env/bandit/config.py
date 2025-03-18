from dataclasses import dataclass
from typing import Dict
@dataclass
class BanditEnvConfig:
    lo_arm_name: str = "phoenix"
    hi_arm_name: str = "dragon"
    action_space_start: int = 1
    lo_arm_score: float = 0.2
    hi_arm_loscore: float = 0.1
    hi_arm_hiscore: float = 1.0
    hi_arm_hiscore_prob: float = 0.25
    
    invalid_act: int = 0
    invalid_act_score: float = -0.1
    action_lookup: Dict[int, str] = None # defined in env.py