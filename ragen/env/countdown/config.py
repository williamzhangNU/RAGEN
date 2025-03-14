
from ragen.env.base import BaseEnvConfig

@dataclass
class CountdownEnvConfig:
    max_instances: int = 20000

    invalid_act: str = ""
    invalid_act_score: float = 0
