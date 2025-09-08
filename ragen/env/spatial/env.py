import gymnasium as gym
from typing import List, Dict, Any

from ragen.env.spatial.config import SpatialGymConfig
from ragen.env.spatial.Base.tos_base import (
    EvaluationManager,
    ActionSequence,
    ExplorationManager,
    CognitiveMapManager,
    HistoryManager,
    RoomGenerator,
    BaseAction,
    ObserveAction,
)
from ragen.env.spatial.Base.tos_base.managers.agent_proxy import get_agent_proxy
from ragen.env.spatial.prompts import Prompter
from ragen.env.spatial.Base.tos_base.utils.action_utils import action_results_to_text
from ragen.env.spatial.Base.tos_base.utils.env_logger import EnvTurnLog
from ragen.env.spatial.Base.tos_base.utils.utils import extract_think_and_answer
from ragen.env.spatial.Base.tos_base.actions.actions import ForcedTermAction, ActionSequence


class SpatialGym(gym.Env):
    """
    Spatial Gym Environment with exploration and evaluation phases.
    
    This environment uses an EvaluationManager to handle all evaluation tasks,
    separating evaluation logic from the main environment logic.
    """
    def __init__(self, config: SpatialGymConfig):
        super().__init__()
        self.config = config
        self.prompter = None

        self.is_exploration_phase = None
        self.remaining_exp_steps = None
        self.render_cache = None

        # Room state management
        self.initial_room = None
        self.initial_agent = None
        
        # Managers
        self.exploration_manager = None
        self.evaluation_manager = None
        self.cognitive_map_manager = None

        # Turn logging
        self.turn_logs: List[EnvTurnLog] = None
        self.current_turn_number = None

    def _set_args(self):
        # Set field of view for all actions
        BaseAction.set_field_of_view(self.config.field_of_view)
        
        # Set observation mode: default to 'full' (dir+degree+distance), allow override via config
        mode = getattr(self.config, 'observation_mode', 'full')
        ObserveAction.MODE = 'full' if mode == 'full' else 'dir'


    def _generate_initial_observation(self) -> str:
        """Generate initial observation based on exploration type."""
        exp_history = ""
        if self.config.exp_type == 'passive' and not self.config.prompt_config["topdown"]:
            proxy = get_agent_proxy(
                self.config.proxy_agent_config["type"],
                self.initial_room,
                self.agent,
                delegate=self.config.proxy_agent_config.get("delegate"),
                observer_delegate=self.config.proxy_agent_config.get("observer_delegate"), # TODO change name
            )
            proxy.run()
            exp_history = proxy.to_text()
            # expose proxy manager so metrics are available via env.get_exp_summary()
            self.exploration_manager = proxy.mgr
        return self.prompter.get_initial_observation_prompt(
            room=self.initial_room,
            agent=self.agent,
            eval_manager=self.evaluation_manager,
            exp_history=exp_history,
        )
    
    def get_history(self):
        return self.history_manager.get_responses() if self.history_manager else None

    def reset(self, seed: int = None):
        """Reset environment for a new episode."""
        super().reset(seed=seed)

        self.prompter = Prompter(self.config, self.np_random)
        
        # Generate initial room
        self.initial_room, self.agent = RoomGenerator.generate_room(
            **self.config.get_room_config(),
            np_random=self.np_random,
        )
        self.initial_agent = self.agent.copy()

        # Initialize episode state
        self.remaining_exp_steps = self.config.max_exp_steps
        
        # Reset turn tracking
        self.turn_logs = []
        self.current_turn_number = 0
        
        # Set exploration phase
        self.is_exploration_phase = self.config.exp_type == 'active'
        
        self._set_args()
        
        # Initialize managers
        # create decoupled agent with initial pose (0,0,N) and store init pose
        # always create exploration manager (also used to generate passive history)
        self.exploration_manager = ExplorationManager(self.initial_room, self.agent)
        self.evaluation_manager = EvaluationManager(self.config.eval_tasks, self.np_random, self.initial_room, self.agent) if len(self.config.eval_tasks) > 0 else None
        self.cognitive_map_manager = CognitiveMapManager(**self.config.cogmap_config) if self.config.prompt_config["cogmap"] else None
        self.history_manager = HistoryManager(seed, self.config) if self.config.exp_type == 'active' else None
        
        obs = self._generate_initial_observation()
        self.render_cache = obs
        return obs, {}
    
    def _step_exploration(self, action: str):
        """
        Handle exploration phase step.
        """
        obs = ""
        reward = -0.1 # per step penalty
        self.remaining_exp_steps -= 1
        # proceed and consume a step

        exp_log = None
        info = {'is_valid_action': True}
        action_sequence = ActionSequence.parse(action)
        # Pre-check remaining steps: if below zero, force-terminate via proxy sequence
        if self.remaining_exp_steps < 0:
            action_sequence = ActionSequence(motion_actions=[], final_action=ForcedTermAction())
        if not action_sequence:
            obs += "Invalid action\n"
            reward += -0.5 # format penalty
            info['is_valid_action'] = False
        else:
        # Execute action
            _ , action_results = self.exploration_manager.execute_action_sequence(action_sequence)
            obs += action_results_to_text(action_results)
            exp_log = self.exploration_manager.turn_logs[-1]
            if action_sequence.final_action and action_sequence.final_action.is_term():
                self.is_exploration_phase = False
                obs += self.prompter.get_evaluation_prompt(self.evaluation_manager)
            else:
                obs += f"\nYou have a maximum of {self.remaining_exp_steps} exploration steps left."
        
        return obs, reward, False, info, exp_log
    
    def _step_evaluation(self, action: str):
        """Handle evaluation phase step."""
        # TODO: different reward for different tasks

        # Evaluate answer
        correct, _ = self.evaluation_manager.evaluate_answer(action)
        eval_log = self.evaluation_manager.turn_logs[-1]
        reward = 1 if correct else 0
        
        # Check for next task
        if self.evaluation_manager.next_task():
            next_question = self.evaluation_manager.get_current_question()
            assert next_question, "No question found after evaluation phase"
            return next_question, reward, False, {}, eval_log
        
        # All tasks completed
        return "Task finished", reward, True, {}, eval_log

    def step(self, llm_response: str):
        """Process agent actions in the spatial gym environment."""
        self.current_turn_number += 1
        exp_log, eval_log = None, None
        think_content, action = extract_think_and_answer(llm_response)
        room_state = next((turn_log.room_state for turn_log in self.turn_logs[::-1] if turn_log.room_state), self.initial_room)
        agent_state = next((turn_log.agent_state for turn_log in self.turn_logs[::-1] if turn_log.agent_state), self.agent)
        
        # Log turn at start with current state
        current_obs = self.render_cache
        img_path = None
        # step the environment
        if action and think_content:
            if self.is_exploration_phase:
                obs, reward, done, step_info, exp_log = self._step_exploration(action)
                if exp_log:
                    room_state, agent_state = exp_log.room_state, exp_log.agent_state
                if self.history_manager:
                    if self.history_manager.is_history_exist():
                        img_path = self.history_manager.get_image_path(self.current_turn_number)
                    else:
                        img_path = self.history_manager.update_response(llm_response, room_state, agent_state)
                    # has terminated
                    if not self.is_exploration_phase:
                        self.history_manager.save()
            else:
                obs, reward, done, step_info, eval_log = self._step_evaluation(action)
                room_state, agent_state = self.evaluation_manager.get_last_room_state()
        else:
            reward, obs, done, step_info = -0.5, "Invalid input format.\n", False, {}

        self.render_cache = obs

        turn_log = EnvTurnLog(
            turn_number=self.current_turn_number,
            user_message=current_obs,
            assistant_raw_message=llm_response,
            assistant_think_message=think_content,
            assistant_parsed_message=action,
            is_exploration_phase=self.is_exploration_phase,
            room_state=room_state,
            agent_state=agent_state,
            observed_items=list(self.exploration_manager.observed_items),
            room_image=img_path,
            exploration_log=exp_log,
            evaluation_log=eval_log,
            info={"reward": reward, "is_done": done, **step_info}
        )
        self.turn_logs.append(turn_log)
        return obs, reward, done, step_info

    def render(self):
        return self.render_cache







    # =============== Analysis Methods ===============
    
    def get_exp_summary(self):
        """Get exploration efficiency metrics."""
        return self.exploration_manager.get_exp_summary() if self.exploration_manager else ExplorationManager.DEFAULT_EXP_SUMMARY
    
    def get_eval_summary(self):
        """Get evaluation performance metrics."""
        return self.evaluation_manager.get_eval_summary() if self.evaluation_manager else EvaluationManager.DEFAULT_EVAL_SUMMARY.copy()

    def get_cogmap_summary(self):
        """Get cognitive map summary."""
        return self.cognitive_map_manager.get_cogmap_summary() if self.cognitive_map_manager else CognitiveMapManager.DEFAULT_COGMAP_SUMMARY.copy()

    def get_env_summary(self) -> Dict[str, Any]:
        """Aggregate environment metrics from all turns."""

        return {
            'env_info': self._get_env_info(),
            'env_turn_logs': [turn_log.to_dict() for turn_log in self.turn_logs],
            'summary': {
                'total_turns': len(self.turn_logs),
                'exp_summary': self.get_exp_summary(),
                'eval_summary': self.get_eval_summary(),
                'cogmap_summary': self.get_cogmap_summary()
            }
        }

    def _get_env_info(self):
        """Get environment state information."""
        return {
            "config": self.config.to_dict(),
            "initial_room": self.initial_room.to_dict(),
            "initial_agent": self.initial_agent.to_dict(),
        }

if __name__ == "__main__":
    # Simple test cases for SpatialGym environment
    
    # TODO: add test cases
    pass