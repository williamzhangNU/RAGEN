
from ragen.env.base import BaseEnvConfig
from dataclasses import dataclass
@dataclass
class CountdownEnvConfig:
    max_instances: int = 20000

    invalid_act: str = ""
    invalid_act_score: float = 0
