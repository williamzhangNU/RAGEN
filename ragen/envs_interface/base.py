from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import math
from ragen.env import ENV_REGISTRY, ENV_CONFIG_REGISTRY
from .config import MultiEnvInterfaceConfig
    

class MultiEnvInterface:
    """Interface for managing multiple environments.
    
    This class manages multiple instances of environments, tracks their states,
    and handles resetting and stepping through them.
    """
    
    def __init__(self, config: Optional[MultiEnvInterfaceConfig] = None):
        """Initialize the multi-environment interface.
        
        Args:
            config: Configuration for the multi-environment interface.
        """
        self.config = config or MultiEnvInterfaceConfig()
        self.envs = {}
        self.env_states = {}  # Tracks if environments are done
        self.env_steps = {}  # Tracks steps taken in each environment
        self.env_rewards = {}  # Tracks cumulative rewards
        self.env_types = {}   # Tracks environment type for each env_id
        
        # Initialize environments
        self._initialize_envs()
    
    def _initialize_envs(self):
        """Initialize environments distributed across types."""
        env_types = [self.config.envs_type] if isinstance(self.config.envs_type, str) else self.config.envs_type
        
        # Get environment classes and configs
        env_classes = [ENV_REGISTRY[t] for t in env_types]
        env_config_classes = [ENV_CONFIG_REGISTRY[t] for t in env_types]
        
        # Distribute environments across types
        num_types = len(env_types)
        envs_per_type = [self.config.envs_size // num_types] * num_types
        for i in range(self.config.envs_size % num_types):
            envs_per_type[i] += 1
            
        # Create environments
        env_id = 0
        for i, (env_type, env_class, config_class) in enumerate(zip(env_types, env_classes, env_config_classes)):
            # Get or create config
            env_config = self.config.env_configs.get(env_type, config_class()) if self.config.env_configs else config_class()
            
            # Create environments of this type
            for _ in range(envs_per_type[i]):
                self.envs[env_id] = env_class(config=env_config)
                self.env_states[env_id] = False
                self.env_steps[env_id] = 0
                self.env_rewards[env_id] = 0.0
                self.env_types[env_id] = env_type
                env_id += 1
    
    def reset(self, env_ids=None, seeds=None):
        """Reset environments with optional seeds."""
        env_ids = env_ids or list(self.envs.keys())
        observations = {}
        
        for i, env_id in enumerate(env_ids):
            if env_id not in self.envs:
                continue
                
            # Get seed if available
            seed = seeds[i] if seeds and i < len(seeds) else None
            
            # Reset environment and tracking
            observations[env_id] = self.envs[env_id].reset(seed=seed)
            self.env_states[env_id] = False
            self.env_steps[env_id] = 0
            self.env_rewards[env_id] = 0.0
            
        return observations
    
    def step(self, actions: Dict[int, int]):
        """Step through specified environments with given actions.
        
        Args:
            actions: Dictionary mapping environment IDs to actions.
            
        Returns:
            Tuple of (observations, rewards, dones, infos) dictionaries.
        """
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        for env_id, action in actions.items():
            if env_id not in self.envs:
                continue
                
            # Skip stepping if environment is already done
            if self.env_states.get(env_id, True):
                observations[env_id] = "Environment is done"
                rewards[env_id] = 0.0
                dones[env_id] = True
                infos[env_id] = {"info": "Environment completed previously"}
                continue
                
            # Step the environment
            obs, reward, done, info = self.envs[env_id].step(action)
            
            # Update tracking information
            self.env_steps[env_id] += 1
            self.env_rewards[env_id] += reward
            
            # Check if environment is done (either from environment or max steps)
            if done or self.env_steps[env_id] >= self.config.max_episode_steps:
                self.env_states[env_id] = True
                done = True
                
                # Auto-reset if configured
                if self.config.auto_reset:
                    new_obs = self.envs[env_id].reset()
                    info["next_obs"] = new_obs
                    self.env_steps[env_id] = 0
                    self.env_rewards[env_id] = 0.0
                    self.env_states[env_id] = False
            
            observations[env_id] = obs
            rewards[env_id] = reward
            dones[env_id] = done
            infos[env_id] = info
            
        return observations, rewards, dones, infos
    
    def get_env_states(self):
        """Get the current state of all environments.
        
        Returns:
            Dictionary of environment states.
        """
        return {
            "env_states": self.env_states.copy(),
            "env_steps": self.env_steps.copy(),
            "env_rewards": self.env_rewards.copy(),
            "env_types": self.env_types.copy()
        }
    
    def process_llm_response(self, llm_responses: Dict[int, str]):
        """Process LLM responses for each environment.
        
        This method takes LLM responses, converts them to actions,
        and steps through the environments accordingly. Supports multiple
        actions per environment separated by the configured separator.
        
        Args:
            llm_responses: Dictionary mapping environment IDs to LLM responses.
            
        Returns:
            Dict of results for each environment with cumulative rewards.
        """
        final_observations = {}
        final_rewards = {}
        final_dones = {}
        final_infos = {}
        
        for env_id, response in llm_responses.items():
            if env_id not in self.envs:
                continue
                
            # Skip processing if environment is done
            if self.env_states.get(env_id, True):
                final_observations[env_id] = "Environment is done"
                final_rewards[env_id] = 0.0
                final_dones[env_id] = True
                final_infos[env_id] = {"info": "Environment completed previously"}
                continue
            
            # Check if the response contains multiple actions
            action_responses = response.split(self.config.multi_action_sep)
            
            # Track cumulative rewards and last observation for this environment
            cumulative_reward = 0.0
            env_done = False
            last_obs = None
            last_info = {}
            actions_executed = 0
            
            # Process each action for this environment
            for action_response in action_responses:
                # Skip empty action strings
                if not action_response.strip():
                    continue
                    
                # Skip if environment became done during multi-action sequence
                if env_done:
                    continue
                
                # Parse the action(s)
                parsed_actions = self._parse_llm_response(action_response.strip(), env_id)
                
                # Skip if no valid action was parsed
                if not parsed_actions or parsed_actions == ['None']:
                    continue
                
                # Execute each parsed action
                for action in parsed_actions:
                    # Skip 'None' actions
                    if action == 'None':
                        continue
                        
                    # Execute the action
                    obs, reward, done, info = self.envs[env_id].step(action)
                    actions_executed += 1
                    
                    # Update tracking
                    self.env_steps[env_id] += 1
                    self.env_rewards[env_id] += reward
                    cumulative_reward += reward
                    last_obs = obs
                    last_info = info
                    
                    # Check if environment is done
                    if done or self.env_steps[env_id] >= self.config.max_episode_steps:
                        self.env_states[env_id] = True
                        env_done = True
                        
                        # Auto-reset if configured
                        if self.config.auto_reset:
                            new_obs = self.envs[env_id].reset()
                            info["next_obs"] = new_obs
                            self.env_steps[env_id] = 0
                            self.env_rewards[env_id] = 0.0
                            self.env_states[env_id] = False
                        
                        # No need to process more actions if done
                        break
            
            # Record final state after all actions are processed
            if last_obs is not None:
                final_observations[env_id] = last_obs
                final_rewards[env_id] = cumulative_reward
                final_dones[env_id] = env_done
                
                # Add information about number of actions executed
                last_info["actions_executed"] = actions_executed
                final_infos[env_id] = last_info
            else:
                # No valid actions were executed
                final_observations[env_id] = self.envs[env_id].render(mode="text")
                final_rewards[env_id] = 0.0
                final_dones[env_id] = self.env_states[env_id]
                final_infos[env_id] = {"actions_executed": 0, "info": "No valid actions were executed"}
        
        return final_observations, final_rewards, final_dones, final_infos
    
    def _parse_llm_response(self, response: str, env_id: int) -> List[Any]:
        """Parse LLM response to extract action(s).
        
        Args:
            response: LLM response text (single action).
            env_id: Environment ID for context.
            
        Returns:
            List of actions (could be integers or strings depending on environment).
            Returns ['None'] if no valid action could be parsed.
        """
        # Get environment type and environment instance
        env_type = self.env_types[env_id]
        env = self.envs[env_id]
        
        # Convert response to lowercase for case-insensitive matching
        response = response.lower().strip()
        
        # If response is empty, return None
        if not response:
            return ['None']
            
        # First, try to extract action using environment's ACTION_LOOKUP if available
        action_lookup = getattr(env, "ACTION_LOOKUP", None)
        
        # If environment has reverse action lookup (string -> int), use it
        if action_lookup is not None:
            # Create a reverse lookup (action name -> action id)
            reverse_lookup = {v.lower(): k for k, v in action_lookup.items()}
            
            # Try exact matches from the dictionary
            for action_name, action_id in reverse_lookup.items():
                if action_name.lower() in response:
                    return [action_id]
                    
            # Try partial matches for action names
            for action_name, action_id in reverse_lookup.items():
                if any(term in response for term in action_name.lower().split()):
                    return [action_id]
        
        # Environment-specific parsing for common terms
        if env_type == "sokoban":
            # Common directional terms for Sokoban
            if "up" in response or "north" in response:
                return [1]
            elif "down" in response or "south" in response:
                return [2]
            elif "left" in response or "west" in response:
                return [3]
            elif "right" in response or "east" in response:
                return [4]
        elif env_type == "some_other_env":  # Example for another environment
            if "forward" in response:
                return ["forward"]  # Some environments might use string actions
            elif "backward" in response:
                return ["backward"]
            # ... other action mappings
        
        # Generic number extraction for numeric actions
        try:
            import re
            numbers = re.findall(r'\d+', response)
            if numbers:
                # Get valid actions for this environment if available
                valid_actions = getattr(env, "get_all_actions", lambda: None)()
                
                for num in numbers:
                    action = int(num)
                    # If we have valid_actions, check if the action is valid
                    if valid_actions is not None:
                        if action in valid_actions:
                            return [action]
                    else:
                        # If we don't know the valid actions, just return the first number found
                        return [action]
        except:
            pass
            
        # If we've gotten here, we couldn't parse a valid action
        # Return a special 'None' value that can be handled by the step function
        invalid_action = getattr(env, "INVALID_ACTION", 'None')
        return [invalid_action]
    
    def render(self, env_ids: Optional[List[int]] = None, mode: str = 'text'):
        """Render specified environments.
        
        Args:
            env_ids: List of environment IDs to render. If None, render all environments.
            mode: Rendering mode.
            
        Returns:
            Dictionary mapping environment IDs to rendered outputs.
        """
        if env_ids is None:
            env_ids = list(self.envs.keys())
            
        rendered = {}
        
        for env_id in env_ids:
            if env_id not in self.envs:
                continue
                
            rendered[env_id] = self.envs[env_id].render(mode=mode)
            
        return rendered
    
    def close(self):
        """Close all environments."""
        for env in self.envs.values():
            env.close()


# Example usage
if __name__ == "__main__":
    # Example with multiple environment types
    from ragen.env.sokoban.config import SokobanEnvConfig
    from ragen.env.frozen_lake.config import FrozenLakeEnvConfig  # Assuming this exists
    
    # Create configurations for different environment types
    sokoban_config = SokobanEnvConfig(dim_room=(6, 6), num_boxes=1, max_steps=100)
    frozen_lake_config = FrozenLakeEnvConfig()  # Assuming this exists
    
    # Create multi-environment config with multiple types
    multi_config = MultiEnvInterfaceConfig(
        envs_size=5,  # Total of 5 environments
        envs_type=["sokoban", "frozen_lake"],  # Two types
        env_configs={"sokoban": sokoban_config, "frozen_lake": frozen_lake_config},
        multi_action_sep="|"
    )
    
    # Create multi-environment interface
    multi_env = MultiEnvInterface(config=multi_config)
    
    # Reset all environments
    observations = multi_env.reset(seeds=[42, 100, 200, 300, 400])
    
    # Print initial observations and environment types
    for env_id, obs in observations.items():
        env_type = multi_env.env_types[env_id]
        print(f"Environment {env_id} (type: {env_type}) initial state:")
        print(obs)
        print()
    
    # Example of processing LLM responses with multiple actions
    llm_responses = {
        0: "I think I should move up | then move right | finally move down",
        1: "Let me move the box to the right | go up",
        2: "I'll go with action 3 | action 4 | action 2",
        3: "Move forward | Turn left | Move forward",
        4: "Invalid action | Jump | Run"  # Testing with an invalid action
    }
    
    obs, rewards, dones, infos = multi_env.process_llm_response(llm_responses)
    
    # Print results
    for env_id in obs.keys():
        env_type = multi_env.env_types[env_id]
        print(f"Environment {env_id} (type: {env_type}) after multi-action sequence:")
        print(f"Final Observation: {obs[env_id]}")
        print(f"Cumulative Reward: {rewards[env_id]}")
        print(f"Done: {dones[env_id]}")
        print(f"Actions executed: {infos[env_id].get('actions_executed', 0)}")
        print()