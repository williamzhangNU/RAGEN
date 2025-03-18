from .frozen_lake.env import FrozenLakeEnv
from .frozen_lake.config import FrozenLakeEnvConfig
from .sokoban.env import SokobanEnv
from .sokoban.config import SokobanEnvConfig
from .bandit.env import BanditEnv
from .bandit.config import BanditEnvConfig
from .countdown.env import CountdownEnv
from .countdown.config import CountdownEnvConfig
from .base import BaseEnv
from .base import BaseEnvConfig

ENV_REGISTRY = {
    "frozen_lake": FrozenLakeEnv,
    "sokoban": SokobanEnv,
    "bandit": BanditEnv,
    "countdown": CountdownEnv,
}

ENV_CONFIG_REGISTRY = {
    "frozen_lake": FrozenLakeEnvConfig,
    "sokoban": SokobanEnvConfig,
    "bandit": BanditEnvConfig,
    "countdown": CountdownEnvConfig,
}
