from typing import Optional, List, Dict

@dataclass
class FrozenLakeConfig:
    """Configuration for FrozenLake environment"""
    # Map config
    size: int = 8
    p: float = 0.8
    seed: Optional[int] = None
    
    # Environment config
    is_slippery: bool = True
    desc: Optional[List[str]] = None
    
    # Mappings
    action_map: Dict[int, int] = {1: 0, 2: 1, 3: 2, 4: 3}
    map_lookup: Dict[bytes, int] = {"P": 0, "F": 1, "H": 2, "G": 3}
    # P: Player; F: Frozen; H: Hole; G: Goal
    grid_lookup: Dict[int, str] = {0: " P ", 1: " _ ", 2: " O ", 3: " G ", 4: " X ", 5: " √ "}
    # P: Player; _: Frozen; O: Hole; G: Goal; X: Player in hole; √: Player on goal
    action_lookup: Dict[int, str] = {0: "None", 1: "Left", 2: "Down", 3: "Right", 4: "Up"}
    
    # Invalid action config
    invalid_act: int = 0
    invalid_act_score: float = -1.0
