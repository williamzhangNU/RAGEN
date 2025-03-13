import gymnasium as gym
import numpy as np
from typing import Optional
from ragen.env.base import BaseDiscreteActionEnv
from .config import BiArmBanditEnvConfig

class BiArmBanditEnv(BaseDiscreteActionEnv, gym.Env):
    def __init__(
            self,
            config: Optional[BiArmBanditEnvConfig] = None
    ):
        BaseDiscreteActionEnv.__init__(self)
        
        # Use provided config or create default
        self.config = config if config is not None else BiArmBanditEnvConfig()
                
        # Set up action space
        self.ACTION_SPACE = gym.spaces.discrete.Discrete(2, start=self.config.action_space_start)
        self.invalid_act = self.config.invalid_act
        self.invalid_act_score = self.config.invalid_act_score
        # Set up arm names
        self.lo_arm_name = self.config.lo_arm_name
        self.hi_arm_name = self.config.hi_arm_name
        
    def _randomize_arms(self):
        """Randomize which position corresponds to which arm"""
        start = self.config.action_space_start
        if self.np_random.random() < 0.5:
            # Low risk is position 1, high risk is position 2
            self.ACTION_LOOKUP = {
                self.invalid_act: "none",
                start: self.lo_arm_name,
                start + 1: self.hi_arm_name,
            }
        else:
            # High risk is position 1, low risk is position 2
            self.ACTION_LOOKUP = {
                self.invalid_act: "none",
                start: self.hi_arm_name,
                start + 1: self.lo_arm_name,
            }
        
        # Update mappings
        self.ARM_IDX_TO_NAME = self.ACTION_LOOKUP
        self.NAME_TO_ARM_IDX = {name: idx for idx, name in self.ACTION_LOOKUP.items() if idx != self.invalid_act}

    def _lo_arm_reward(self):
        return self.config.lo_arm_score

    def _hi_arm_reward(self):
        if self.np_random.random() < self.config.hi_arm_hiscore_prob:
            return self.config.hi_arm_hiscore
        else:
            return self.config.hi_arm_loscore

    def reset(self, mode=None, seed=None):
        """Reset the environment and randomize arm positions"""
        gym.Env.reset(self, seed=seed)
        self._randomize_arms()
        pos1 = self.config.action_space_start
        pos2 = pos1 + 1
        machine1 = self.ARM_IDX_TO_NAME[pos1]
        machine2 = self.ARM_IDX_TO_NAME[pos2]
        
        init_text = f"You are facing {self.ACTION_SPACE.n} slot machines ({machine1} ({pos1}) and {machine2} ({pos2})). One has a higher risk and a higher reward, while the other has a lower risk and a lower reward. Which machine will you pull? Your answer should be in {self.get_all_actions()}"

        return init_text

    def step(self, action: int):
        assert action in self.get_all_actions(), f"Invalid action {action}"
        if action == self.invalid_act:
            reward = self.invalid_act_score
            next_obs = f"You give an invalid answer and receive {reward} points."
        else:
            arm_name = self.ARM_IDX_TO_NAME[action]
            if arm_name == self.lo_arm_name:
                reward = self._lo_arm_reward()
            else:
                reward = self._hi_arm_reward()                
            next_obs = f"You pull the {arm_name} slot machine and receive {reward} points."
        done, info = True, {}
        return next_obs, reward, done, info

    def get_all_actions(self):
        return [self.invalid_act, self.ACTION_SPACE.start, self.ACTION_SPACE.start + 1]



if __name__ == "__main__":
    
    def run_simulation(env, n_episodes=3000, action=1, start_seed=500):
        """Run simulation for n episodes and return statistics"""
        """will get 0.2 (low risk) * 0.5 (prob) + (0.25 * 1 + 0.75 * 0.1) (high risk expected reward) * 0.5 (prob) = 0.2625 reward for constant action"""
        rewards = []
        for i in range(start_seed, start_seed + n_episodes):
            env.reset(seed=i)
            reward = env.step(action)[1]
            rewards.append(reward)
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'n_episodes': n_episodes,
            'action': env.ARM_IDX_TO_NAME[action]
        }

    # Test default configuration
    env = BiArmBanditEnv()
    stats = run_simulation(env)
    print(f"\nDefault Configuration Results:")
    print(f"Arm: {stats['action']}")
    print(f"Mean reward: {stats['mean_reward']:.3f} ± {stats['std_reward']:.3f}")
    print(f"Number of episodes: {stats['n_episodes']}")