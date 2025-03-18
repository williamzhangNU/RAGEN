from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from ragen.env import ENV_REGISTRY, ENV_CONFIG_REGISTRY
from .config import MultiEnvInterfaceConfig


class MultiEnvInterface:
    """Multi-environment interface for LLM-Env interaction management"""

    def __init__(self, config: Optional[MultiEnvInterfaceConfig] = None):
        self.config = config or MultiEnvInterfaceConfig()
        self.envs = {}
        self.env_data = {k: {} for k in ['dones', 'steps', 'rewards', 'types']}
        self._init_envs()
    
    def _init_envs(self):
        env_types = [self.config.envs_type] if isinstance(self.config.envs_type, str) else self.config.envs_type
        count_per_type = [self.config.envs_size // len(env_types)] * len(env_types)
        for i in range(self.config.envs_size % len(env_types)):
            count_per_type[i] += 1
            
        env_id = 0
        for i, env_type in enumerate(env_types):
            env_class = ENV_REGISTRY[env_type]
            env_config = self.config.env_configs.get(env_type, ENV_CONFIG_REGISTRY[env_type]()) # If no config is provided, use the default one

            for _ in range(count_per_type[i]):
                self.envs[env_id] = env_class(config=env_config)
                self.env_data['dones'][env_id] = False
                self.env_data['steps'][env_id] = 0
                self.env_data['rewards'][env_id] = 0.0
                self.env_data['types'][env_id] = env_type
                env_id += 1

    def reset(self, env_ids=None, seeds=None):
        env_ids = env_ids or list(self.envs.keys())
        observations = {}
        
        for i, env_id in enumerate(env_ids):
            if env_id in self.envs:
                seed = seeds[i] if seeds and i < len(seeds) else None
                print(f"Resetting environment {env_id} with seed {seed}")
                observations[env_id] = self.envs[env_id].reset(seed=seed)
                self.env_data['dones'][env_id] = False
                self.env_data['steps'][env_id] = 0
                self.env_data['rewards'][env_id] = 0.0
            
        return observations
    
    def _set_default_result(self, results, env_id):
        """Set default results for skipped envs"""
        results['observations'][env_id] = "Environment is done"
        results['rewards'][env_id] = 0.0
        results['dones'][env_id] = True
        results['infos'][env_id] = {"info": "Environment completed previously"}
        
    def step(self, llm_responses):
        """Process LLM responses"""
        results = {k: {} for k in ['observations', 'rewards', 'dones', 'infos', 'actions_executed']}
        
        for env_id, response in llm_responses.items():
            if env_id not in self.envs or self.env_data['dones'].get(env_id, True):
                self._set_default_result(results, env_id)
                continue
            
            # Process multi-actions
            env_done, reward_sum, last_obs, infos = False, 0.0, None, []
            actions_count = 0
            
            for action_text in response.split(self.config.multi_action_sep):
                if not action_text.strip() or env_done:
                    continue
                
                for action in self._parse_action(action_text.strip(), env_id):
                    if action == 'None':
                        continue
                    
                    obs, reward, done, info = self.envs[env_id].step(action)
                    actions_count += 1
                    self.env_data['steps'][env_id] += 1
                    self.env_data['rewards'][env_id] += reward
                    reward_sum += reward
                    last_obs = obs
                    infos.append(info)
                    done = done or self.env_data['steps'][env_id] >= self.config.max_episode_steps
                    if done:
                        self.env_data['dones'][env_id] = True
                        env_done = True
                        break
            
            # Record results
            if last_obs:
                results['observations'][env_id] = last_obs
                results['rewards'][env_id] = reward_sum
                results['dones'][env_id] = env_done
                results['infos'][env_id] = infos
                results['actions_executed'][env_id] = actions_count
            else:
                results['observations'][env_id] = "Environment is done or action is invalid. No next observation."
                results['rewards'][env_id] = 0.0
                results['dones'][env_id] = self.env_data['dones'][env_id]
                results['infos'][env_id] = []
                results['actions_executed'][env_id] = 0
        
        return tuple(results.values())
    
    def _parse_action(self, text, env_id):
        """Parse text to extract actions"""
        if not text:
            return ['None']
            
        env = self.envs[env_id]
        text = text.lower().strip()
        
        # Try action lookup dictionary
        if hasattr(env, "ACTION_LOOKUP"):
            actions = {v.lower(): k for k, v in env.ACTION_LOOKUP.items()}
            for name, action_id in actions.items():
                if name == text:
                    return [action_id]
         
        return [getattr(env, "INVALID_ACTION", 'None')]



# Example usage
if __name__ == "__main__":
    from ragen.env.sokoban.config import SokobanEnvConfig
    from ragen.env.frozen_lake.config import FrozenLakeEnvConfig
    from ragen.env.countdown.config import CountdownEnvConfig
    from ragen.env.bandit.config import BanditEnvConfig

    bandit_config = BanditEnvConfig()
    countdown_config = CountdownEnvConfig()
    sokoban_config = SokobanEnvConfig(dim_room=(6, 6), num_boxes=1, max_steps=100)
    frozen_lake_config = FrozenLakeEnvConfig()

    multi_config = MultiEnvInterfaceConfig(
        envs_size=4,  
        envs_type=["sokoban", "frozen_lake", "bandit", "countdown"],
        env_configs={"sokoban": sokoban_config, "frozen_lake": frozen_lake_config, "bandit": bandit_config, "countdown": countdown_config},
        multi_action_sep="|"
    )
    
    multi_env = MultiEnvInterface(config=multi_config)
    
    observations = multi_env.reset(seeds=[42, 100, 200, 300])
    
    for env_id, obs in observations.items():
        env_type = multi_env.env_data['types'][env_id]
        print(f"Environment {env_id} (type: {env_type}) initial state:")
        print(obs)
        print()
    
    # Example of processing LLM responses with multiple actions
    llm_responses = {
        0: "up | right | down",
        1: "hello | right",
        2: "up | down",
        3: "hello | up"
    }
    
    obs, rewards, dones, infos, actions_executed = multi_env.step(llm_responses)
    
    # Print results
    for env_id in obs.keys():
        env_type = multi_env.env_data['types'][env_id]
        print(f"Environment {env_id} (type: {env_type}):")
        print(f"Final Observation: \n{obs[env_id]}")
        print(f"Cumulative Reward: {rewards[env_id]}")
        print(f"Done: {dones[env_id]}")
        print(f"All Infos: {infos[env_id]}")
        print(f"Actions executed: {actions_executed[env_id]}")
        print()