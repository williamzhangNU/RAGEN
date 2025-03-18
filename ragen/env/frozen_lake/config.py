from typing import Optional, List, Dict
from dataclasses import dataclass, field

@dataclass
class FrozenLakeEnvConfig:
    """Configuration for FrozenLake environment"""
    # Map config
    size: int = 4
    p: float = 0.8
    is_slippery: bool = True
    map_seed: Optional[int] = None
        
    # Mappings
    action_map: Dict[int, int] = field(default_factory=lambda: {1: 0, 2: 1, 3: 2, 4: 3})
    map_lookup: Dict[bytes, int] = field(
        default_factory=lambda: {b"P": 0, b"F": 1, b"H": 2, b"G": 3}
    ) # b'' string is used for vectorization in numpy
    # P: Player; F: Frozen; H: Hole; G: Goal
    grid_lookup: Dict[int, str] = field(default_factory=lambda: {0:"P", 1:"_", 2:"O", 3:"G", 4:"X", 5:"√"})
    # P: Player; _: Frozen; O: Hole; G: Goal; X: Player in hole; √: Player on goal
    action_lookup: Dict[int, str] = field(default_factory=lambda: {0:"None", 1:"Left", 2:"Down", 3:"Right", 4:"Up"})
    
    # Invalid action config
    invalid_act: int = 0
    invalid_act_score: float = -0.1
