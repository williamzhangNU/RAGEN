from abc import ABC, abstractmethod
import re
from typing import Optional, List, Tuple, Any, Dict
from copy import deepcopy
from transformers import AutoTokenizer
import torch

class BaseEnv(ABC):
    """
    Abstract base class for all environments.
    The class needs to handle text-based input, input may be invalid
        - Environment will track the total reward for the trajectory

    """
    INVALID_ACTION = 0
    PENALTY_FOR_INVALID = -1
    def __init__(self):
        self.reward = 0

        self._actions = [] # list of all actions (including all responses from LLM)
        self._actions_valid = [] # list of actions that are in the correct format
        self._actions_effective = [] # list of actions that are effective (actual moving in env)

    @staticmethod
    def _extract_answer(text: str) -> str:
        """Extract the answer from the text."""
        match = re.search(r"<answer>(.*?)</answer>", text)
        if match:
            return match.group(1).strip()
        else:
            return None

    def _reset_tracking_variables(self):
        self.reward = 0
        self._actions = []
        self._actions_valid = []
        self._actions_effective = []

    def get_tracking_variables(self) -> Dict:
        """Get statistics of valid actions."""
        return {
            "reward": self.reward,
            "actions": self._actions,
            "actions_valid": self._actions_valid,
            "actions_effective": self._actions_effective,
        }
    
    def _update_tracking_variables(
            self, 
            response: str,
            action: Any, 
            action_is_valid: bool,
            action_is_effective: bool,
            reward: float,
        ):
        """
        All of _actions, _actions_valid, _actions_effective are lists of the same length
            - None is used for _actions_valid and _actions_effective if the action is invalid or ineffective
        """
        self._actions.append(response)
        if action_is_valid:
            self._actions_valid.append(action)
        else:
            self._actions_valid.append(None)
        if action_is_effective:
            self._actions_effective.append(action)
        else:
            self._actions_effective.append(None)
        self.reward += reward if action_is_valid else (reward + self.PENALTY_FOR_INVALID)

    def _copy_tracking_variables(self, other: 'BaseEnv'):
        self.reward = other.reward
        self._actions = deepcopy(other._actions)
        self._actions_valid = deepcopy(other._actions_valid)
        self._actions_effective = deepcopy(other._actions_effective)



    @staticmethod
    def formulate_output(env_feedback: str, done: bool = False):
        """
        Formulate the environment feedback to as the input to the LLM
        - e.g., For Qwen, special tokens like <|im_start|>user and <|im_end|> should be added
        NOTE hard-coded now for Qwen
        """
        output = "\n <|im_start|>user\n" + env_feedback + "<|im_end|>\n"
        if not done:
            output += "<|im_start|>assistant\n<think>"
        return output

    @classmethod
    def execute_predictions(
        cls, 
        envs: List['BaseEnv'], 
        predictions: List[str], 
        prediction_ids: torch.Tensor,
        tokenizer: AutoTokenizer,
    ) -> List[str]:
        cur_actions, action_is_valid = cls.postprocess_predictions(envs, predictions)
        next_obs, dones = [], []
        
        for env, action, response, response_id, av in zip(envs, cur_actions, predictions, prediction_ids, action_is_valid):
            obs = ""
            if "<|im_end|>" not in response:
                obs += "<|im_end|>"

            if env.finished():
                obs += tokenizer.pad_token
                done = True
            else:
                thinking_reward = 0
                
                # step in environment
                observation, env_reward, done, extra_info = env.step(action)
                env_feedback = cls.parse_update_info_to_obs(
                    (observation, env_reward, done, extra_info), 
                    av
                )

                obs += cls.formulate_output(env_feedback, done)
                
                env._update_tracking_variables(
                    response=response, 
                    action=action, 
                    action_is_valid=av, 
                    action_is_effective=extra_info.get("action_is_effective", False), 
                    reward=thinking_reward + env_reward, 
                )
            next_obs.append(obs)
            dones.append(done)
        return next_obs, dones
    
    @classmethod
    @abstractmethod
    def postprocess_predictions(cls, envs: List['BaseEnv'], predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        pass


    @staticmethod
    @abstractmethod
    def parse_update_info_to_obs(update_info: Tuple[Any, float, bool, Dict], action_is_valid: bool) -> str:
        pass


    @abstractmethod
    def reset(self, seed: Optional[int] = None) -> Any:
        pass

    @abstractmethod
    def step(self, action) -> Tuple[Any, float, bool, Dict]:
        pass

    @abstractmethod
    def success(self) -> bool:
        pass

    @abstractmethod
    def finished(self) -> bool:
        pass

    @abstractmethod
    def render(self, mode: str = 'tiny_rgb_array') -> Any:
        pass

    @abstractmethod
    def copy(self) -> 'BaseEnv':
        pass




class BaseDiscreteActionEnv(BaseEnv, ABC):
    """
    Abstract base class for environments with discrete action spaces
    This class provides common functionality for environments like FrozenLakeEnv and SokobanEnv.
    """
    GRID_LOOKUP = {} # define the mapping from integer to string for rendering
    ACTION_LOOKUP = {} # define the mapping from integer to action string
    INVALID_ACTION = 0 # default invalid action
    PENALTY_FOR_INVALID = -1 # penalty for invalid action

    @staticmethod
    def parse_update_info_to_obs(update_info: Tuple[Any, float, bool, Dict], action_is_valid: bool) -> str:
        """
        Parse environment update information into observation string.
        
        Args:
            update_info: Tuple of (observation, reward, done, info)
            action_is_valid: Whether the action was valid
            
        Returns:
            Observation string
        """
        observation, reward, done, _ = update_info
        if not action_is_valid:
            return f"Action is invalid. You stay in the same position. The observation is: \n{observation}\nreward: {reward}\ndone: {done}\n"
        return f"After you take this action, the observation is: \n{observation}\nreward: {reward}\ndone: {done}\n"


    @classmethod
    def postprocess_predictions(cls, envs: List['BaseDiscreteActionEnv'], predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        actions = []
        action_is_valid = []
        
        for env, prediction in zip(envs, predictions):
            if isinstance(prediction, str): # for llm output
                action = cls._extract_answer(prediction)
                if action is None:
                    action = env.INVALID_ACTION
                else:
                    action = env.extract_action(action)
                action_is_valid.append(action != env.INVALID_ACTION)
            elif isinstance(prediction, int):
                action = prediction if prediction in env.get_all_actions() else env.INVALID_ACTION
                action_is_valid.append(action != env.INVALID_ACTION)
            elif isinstance(prediction, list):
                action = prediction
                action_is_valid.append(True)
            elif prediction is None:
                action = env.INVALID_ACTION
                action_is_valid.append(False)
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            
        return actions, action_is_valid

    def get_all_actions(self) -> List[int]:
        return list(range(self.ACTION_SPACE.start, self.ACTION_SPACE.start + self.ACTION_SPACE.n))
    
    @abstractmethod
    def extract_action(self, text: str) -> int:
        pass

    @abstractmethod
    def reset(self, mode: str = 'tiny_rgb_array', seed: Optional[int] = None) -> Any:
        pass

    @abstractmethod
    def step(self, action: int) -> Tuple[Any, float, bool, Dict]:
        pass

    @abstractmethod
    def success(self) -> bool:
        pass

    @abstractmethod
    def finished(self) -> bool:
        pass

    @abstractmethod
    def render(self, mode: str = 'tiny_rgb_array') -> Any:
        pass

    @abstractmethod
    def copy(self) -> 'BaseDiscreteActionEnv':
        pass







class BaseLanguageBasedEnv(BaseEnv, ABC):
    """
    Abstract base class for environments with language-based action space environment
    This class provides common functionality for environments like countdown from TinyZero
    """

    ACTION_LOOKUP = {} # TODO modify this as a method so can be called in a unified way
    INVALID_ACTION = "" # default invalid action
    PENALTY_FOR_INVALID = -1 # penalty for invalid action

    @staticmethod
    def parse_update_info_to_obs(update_info: Tuple[Any, float, bool, Dict], action_is_valid: bool) -> str:
        if not action_is_valid:
            return f"Action is invalid. You stay in the same position. The observation is: \n{observation}\nreward: {reward}\ndone: {done}\n"
        return f"After you take this action, the observation is: \n{observation}\nreward: {reward}\ndone: {done}\n"

    @classmethod
    def postprocess_predictions(cls, envs: List['BaseLanguageBasedEnv'], predictions: List[Any]) -> Tuple[List[int], List[bool]]:
        actions = []
        action_is_valid = []
        
        for env, prediction in zip(envs, predictions):
            if isinstance(prediction, str):
                action = cls._extract_answer(prediction)
                if action is None:
                    action = env.INVALID_ACTION
                else:
                    action = env.extract_action(action)
                action_is_valid.append(action != env.INVALID_ACTION)
            else:
                # raise ValueError(f"Invalid prediction type: {type(prediction)}")
                action = env.INVALID_ACTION
                action_is_valid.append(False)
            
            actions.append(action)
            
        return actions, action_is_valid
    

    @abstractmethod
    def extract_action(self, text: str) -> int:
        pass
    
    def get_all_actions(self):
        raise NotImplementedError("Language-based environment does not have a finite action space")

    @abstractmethod
    def reset(self, mode: str = 'tiny_rgb_array', seed: Optional[int] = None) -> Any:
        pass

    @abstractmethod
    def step(self, action: str) -> Tuple[Any, float, bool, Dict]:
        pass

    @abstractmethod
    def success(self) -> bool:
        pass

    @abstractmethod
    def finished(self) -> bool:
        pass

    @abstractmethod
    def render(self, mode: str = 'tiny_rgb_array') -> Any:
        pass

    @abstractmethod
    def copy(self) -> 'BaseDiscreteActionEnv':
        pass
