import gymnasium as gym
import numpy as np
from typing import Optional
from ragen.env.base import BaseDiscreteActionEnv
from .config import BiArmBanditEnvConfig

class BiArmBanditEnv(BaseDiscreteActionEnv, gym.Env):
    def __init__(
            self,
            config: Optional[BiArmBanditEnvConfig] = None,
            seed: Optional[int] = None,
    ):
        BaseDiscreteActionEnv.__init__(self)
        
        # Use provided config or create default
        self.config = config if config is not None else BiArmBanditEnvConfig()
                
        # Set up action space
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(2, start=self.config.action_space_start)
        
        # Set up arm names and mappings
        self.low_risk_name = self.config.low_risk_name
        self.high_risk_name = self.config.high_risk_name
        
        # Action lookup mappings
        self.ACTION_LOOKUP = {
            self.INVALID_ACTION: "none",
            self.config.action_space_start: self.low_risk_name,
            self.config.action_space_start + 1: self.high_risk_name,
        }
        
        # Fixed mappings
        self.ARM_IDX_TO_NAME = self.ACTION_LOOKUP
        self.NAME_TO_ARM_IDX = {
            "none": self.INVALID_ACTION,
            self.low_risk_name: self.config.action_space_start,
            self.high_risk_name: self.config.action_space_start + 1,
        }
        
        # Store initialization parameters
        self.env_kwargs = {
            "n_arms": self.ACTION_SPACE.n,
            "config": self.config,
            "seed": seed,
        }
        
        # Initialize tracking variables
        self.last_action = None
        self._success = False
        self._finished = False
        
    def _low_risk_arm_reward_distribution(self):
        return self.config.low_risk_reward

    def _high_risk_arm_reward_distribution(self):
        if self.np_random.random() < self.config.high_risk_high_reward_prob:
            return self.config.high_risk_high_reward
        else:
            return self.config.high_risk_low_reward

    def reset(self, mode='text', seed=None):
        """Reset the environment and reward distributions"""
        # gym.Env.reset(self, seed=seed)
        gym.Env.reset(self, seed=seed)
        return self.render(mode)

    def step(self, action: int):
        """
        Take action (pull arm) and get reward
        - action = 1: pull low-risk arm
        - action = 2: pull high-risk arm
        """
        assert isinstance(action, int)
        self._finished = True
        
        if action == self.INVALID_ACTION:
            return self.render(), True, {"action_is_effective": False}
        
        self._success = True
        
        assert action in self.get_all_actions(), f"Invalid action {action}"
        
        if action == self.config.action_space_start:
            reward = self._low_risk_arm_reward_distribution()
        else:
            reward = self._high_risk_arm_reward_distribution()
            
        self.last_action = action
        
        info = {"action_is_effective": True}
        
        return self.render(), reward, True, info

    def render(self, mode='text'):
        """Render the current state"""
        if mode == 'text':
            if self.last_action is None:
                return f"You are facing {self.ACTION_SPACE.n} slot machines ({self.low_risk_name} and {self.high_risk_name}). Each machine has its own reward distribution.\nWhich machine will you pull?"
            else:
                return f"You pulled machine {self.ARM_IDX_TO_NAME[self.last_action]}."
        else:
            raise NotImplementedError(f"Render mode {mode} not implemented")

    def get_all_actions(self):
        return list(range(self.ACTION_SPACE.start, self.ACTION_SPACE.start + self.ACTION_SPACE.n))

if __name__ == "__main__":
    # 使用默认配置
    env = BiArmBanditEnv()
    print(env.reset())
    print(env.step(1))
    
    # 使用自定义配置
    custom_config = BiArmBanditEnvConfig(
        low_risk_name="safe",
        high_risk_name="risky",
        low_risk_reward=0.3,
        high_risk_low_reward=0.05,
        high_risk_high_reward=2.0,
        high_risk_high_reward_prob=0.2
    )
    env_custom = BiArmBanditEnv(config=custom_config, seed=0)
    print(env_custom.reset(seed=0))
    print(env_custom.step(1))