
from ragen.env.base import BaseEnvConfig


class CountdownEnvConfig:
    def __init__(self, invalid_act: str = "", invalid_act_score: float = 0, max_instances: int = 20000):
        self.invalid_act = invalid_act
        self.invalid_act_score = invalid_act_score
        self.max_instances = max_instances

