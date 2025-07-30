import gymnasium as gym
import re
from typing import Optional, List, Dict, Union, Any
from dataclasses import dataclass, field
import copy

from ragen.env.spatial.config import SpatialGymConfig
from ragen.env.spatial.Base.tos_base import (
    EvaluationManager,
    EvaluationTurnLog,
    Room,
    ActionSequence,
    ExplorationManager,
    ExplorationTurnLog,
    generate_room,
    BaseAction
)
from ragen.env.spatial.utils.generate_history import AutoExplore
from ragen.env.spatial.prompts import (
    ACTIVE_INSTRUCTION, 
    PASSIVE_INSTRUCTION, 
    EVALUATION_INSTRUCTION,
    SHORT_EXPLORATION_PROMPT, 
    SHORT_EVALUATION_PROMPT
)
from ragen.env.spatial.utils.action_utils import action_results_to_text
from ragen.env.spatial.utils.text_utils import extract_think_and_answer

@dataclass
class EnvTurnLog:
    """Log data for a single environment turn."""
    turn_number: int
    user_message: str = ""  # Environment observation
    assistant_raw_message: str = ""  # Raw assistant input
    assistant_think_message: str = ""  # Think part of assistant message
    assistant_parsed_message: str = ""  # Parsed assistant action
    is_exploration_phase: bool = False
    exploration_log: Optional["ExplorationTurnLog"] = None
    evaluation_log: Optional["EvaluationTurnLog"] = None
    room_state: Optional["Room"] = None
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "turn_number": self.turn_number,
            "user_message": self.user_message,
            "assistant_raw_message": self.assistant_raw_message,
            "assistant_think_message": self.assistant_think_message,
            "assistant_parsed_message": self.assistant_parsed_message,
            "is_exploration_phase": self.is_exploration_phase,
            "exploration_log": self.exploration_log.to_dict() if self.exploration_log else {},
            "evaluation_log": self.evaluation_log.to_dict() if self.evaluation_log else {},
            "room_state": self.room_state.to_dict() if self.room_state else {},
            "info": self.info
        }




class SpatialGym(gym.Env):
    """
    Spatial Gym Environment with exploration and evaluation phases.
    
    This environment uses an EvaluationManager to handle all evaluation tasks,
    separating evaluation logic from the main environment logic.
    """
    def __init__(self, config: SpatialGymConfig):
        super().__init__()
        self.config = config
        self.is_exploration_phase = None
        self.remaining_exp_steps = None
        self.render_cache = None

        # Room state management
        self.initial_room = None
        self.final_room = None
        
        # Managers
        self.exploration_manager = None
        self.evaluation_manager = None

        # Exploration metrics
        self.n_valid_queries = None
        self.n_redundant_queries = None

        # Turn logging
        self.turn_logs: List[EnvTurnLog] = []
        self.current_turn_number = 0

        self.original_render_read = None 
        # this is in case ctx_manager parses no valid action, so if the original render is read, we will use one short prompt to replace it

    def _generate_initial_observation(self) -> str:
        """Generate initial observation based on exploration type."""
        room_desc = self.initial_room.get_room_description(with_topdown=self.config.with_topdown)
        
        if self.config.exp_type == 'passive':
            exp_history = f"## Exploration History\n{AutoExplore(self.initial_room, self.np_random).gen_exp_history()}" if not self.config.with_topdown else ""
            eval_question = self.evaluation_manager.get_current_question(self.initial_room.copy())
            assert eval_question, "No question found after exploration phase"
            obs = PASSIVE_INSTRUCTION.format(
                room_info=room_desc,
                exp_history=exp_history,
            )
            obs += EVALUATION_INSTRUCTION.format(eval_question=f"## Evaluation Question\n{eval_question}")

        else:
            exp_instructions = f"## Action Instructions\n{ActionSequence.get_usage_instructions()}\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."
            obs = ACTIVE_INSTRUCTION.format(
                room_info=room_desc,
                exp_instructions=exp_instructions
            )

        return obs
    
    def _update_render_cache(self, obs: str):
        assert self.original_render_read, "Observation is not read yet"
        self.render_cache = obs
        self.original_render_read = False


    def reset(self, seed: int = None):
        """Reset environment for a new episode."""
        super().reset(seed=seed)
        
        # Generate initial room
        self.initial_room = generate_room(
            **self.config.get_room_config(),
            np_random=self.np_random,
        )

        # Initialize episode state
        self.remaining_exp_steps = self.config.max_exp_steps
        self.n_valid_queries = 0
        self.n_redundant_queries = 0
        
        # Reset turn tracking
        self.turn_logs = []
        self.current_turn_number = 0
        
        # Set exploration phase
        self.is_exploration_phase = self.config.exp_type not in ['passive', 'overview']
        
        # Set field of view for all actions
        BaseAction.set_field_of_view(self.config.field_of_view)
        
        # Initialize managers
        if self.config.exp_type in ['active', 'active_overview']:
            self.exploration_manager = ExplorationManager(self.initial_room)
        self.evaluation_manager = EvaluationManager(self.config.eval_tasks, self.np_random, self.initial_room)

        # Generate initial observation
        obs = self._generate_initial_observation()
        self.original_render_read = True
        self._update_render_cache(obs)
        return obs, {}
    
    def _step_exploration(self, action: str):
        """
        Handle exploration phase step.
        TODO:
        1. Add reward for invalid action
        2. Add reward for Terminate: based on exploration summary
        """
        obs = ""
        reward = -0.1 # per step penalty

        # Parse and validate action
        action_sequence = ActionSequence.parse(action)
        if not action_sequence:
            obs += "Invalid action\n"
            reward += -0.5 # format penalty
        else:
            self.n_valid_queries += 1 if not action_sequence.final_action.is_term() else 0

        self.remaining_exp_steps -= 1
        if self.remaining_exp_steps < 0 or (action_sequence and action_sequence.final_action.is_term()):
            # End exploration phase
            self.is_exploration_phase = False
            obs += "Exploration phase ended\n"
            self.final_room = self.exploration_manager.finish_exploration()
            
            # Transition to evaluation, NOTE question is generated based on the initial room
            eval_question = self.evaluation_manager.get_current_question(self.initial_room.copy())
            assert eval_question, "No question found after exploration phase"
            obs += EVALUATION_INSTRUCTION.format(eval_question=f"## Evaluation Question\n{eval_question}")
        else:
            # Execute exploration action, TODO give reward to efficient exploration
            if action_sequence:
                exp_info, action_results = self.exploration_manager.execute_action_sequence(action_sequence)
                # Track redundant queries
                if exp_info.get('redundant', False):
                    self.n_redundant_queries += 1
                    reward += -1 # redundant observe penalty
                obs += action_results_to_text(action_results)
            obs += f"\nYou have a maximum of {self.remaining_exp_steps} exploration steps left."
        
        self._update_render_cache(obs)
        return obs, reward, False, {}
    
    def _step_evaluation(self, action: str):
        """Handle evaluation phase step."""
        # TODO: different reward for different tasks

        # Evaluate answer
        correct, info = self.evaluation_manager.evaluate_answer(action)
        reward = 1 if correct else 0
        
        # Check for next task
        if self.evaluation_manager.next_task():
            next_question = self.evaluation_manager.get_current_question(self.initial_room.copy())
            assert next_question, "No question found after evaluation phase"
            self._update_render_cache(next_question)
            return next_question, reward, not bool(next_question), {}
        
        # All tasks completed
        self._update_render_cache("Task finished")
        return "Task finished", reward, True, {}

    def step(self, action: str):
        """Process agent actions in the spatial gym environment."""
        self.current_turn_number += 1
        exp_log, eval_log = None, None
        
        # Log turn at start with current state
        current_obs = self.render_cache if hasattr(self, 'render_cache') else ""
        
        
        # Execute action
        if self.is_exploration_phase:
            obs, reward, done, step_info = self._step_exploration(action)
            exp_log = self.exploration_manager.turn_logs[-1] if self.exploration_manager and self.exploration_manager.turn_logs else None
        else:
            obs, reward, done, step_info = self._step_evaluation(action)
            eval_log = self.evaluation_manager.turn_logs[-1] if self.evaluation_manager.turn_logs else None
        
        # Get room state from turn logs
        room_state = None
        if exp_log and exp_log.room_state:
            room_state = exp_log.room_state
        elif eval_log and eval_log.room_state:
            room_state = eval_log.room_state
        
        turn_log = EnvTurnLog(
            turn_number=self.current_turn_number,
            user_message=current_obs,
            assistant_parsed_message=action,
            is_exploration_phase=self.is_exploration_phase,
            room_state=room_state,
            exploration_log=exp_log,
            evaluation_log=eval_log,
            info={"reward": reward, "is_done": done, **step_info}
        )
        self.turn_logs.append(turn_log)
        
        return obs, reward, done, step_info

    def render(self):
        if not self.original_render_read:
            self.original_render_read = True
            return self.render_cache
        if self.is_exploration_phase:
            return SHORT_EXPLORATION_PROMPT + f"You have a maximum of {self.remaining_exp_steps} exploration steps left."
        else:
            return SHORT_EVALUATION_PROMPT







    # =============== Analysis Methods ===============
    
    def get_exp_summary(self):
        """Get exploration efficiency metrics."""
        return self.exploration_manager.get_exp_summary() if self.exploration_manager else ExplorationManager.DEFAULT_EXP_SUMMARY
    
    def get_eval_summary(self):
        """Get evaluation performance metrics."""
        return self.evaluation_manager.get_eval_summary() if self.evaluation_manager else EvaluationManager.DEFAULT_EVAL_SUMMARY.copy()
    
    def get_env_summary(self) -> Dict[str, Any]:
        """Aggregate environment metrics from all turns."""

        return {
            'env_info': self._get_env_info(),
            'env_turn_logs': [turn_log.to_dict() for turn_log in self.turn_logs],
            'summary': {
                'total_turns': len(self.turn_logs),
                'exp_summary': self.get_exp_summary(),
                'eval_summary': self.get_eval_summary()
            }
        }
    
    @staticmethod
    def aggregate_env_data(
        envs: Dict[int, "SpatialGym"], 
        messages: List[str], 
        env_ids: List[int]
    ) -> Dict:
        """
        Group environments by config name and calculate aggregate metrics.
        
        Returns:
            Dict with structure:
            {
                'config_groups': {
                    'config_name': {
                        'env_data': [list of env data for this config]
                    }
                },
                'exp_summary': {
                    'overall_performance': {...},
                    'group_performance': {'config_name': {...}}
                },
                'eval_summary': {
                    'overall_performance': {...},
                    'group_performance': {'config_name': {...}}
                }
            }
        """
        from collections import defaultdict
        
        # Group environments by config name
        config_groups = defaultdict(list)
        for message, env_id in zip(messages, env_ids):
            env = envs[env_id]
            config_name = env.config.name
            
            # Split messages and assign to turn logs
            SpatialGym._assign_raw_messages(message, env.turn_logs)
            
            env_data = {**env.get_env_summary(), "message": message}
            config_groups[config_name].append(env_data)
        
        # Initialize result structure
        result = {
            "config_groups": {},
            "exp_summary": {"overall_performance": {}, "group_performance": {}},
            "eval_summary": {"overall_performance": {}, "group_performance": {}}
        }
        
        # Collect all metrics for overall calculation
        all_exp_data, all_eval_data = [], []
        
        for config_name, env_data_list in config_groups.items():
            # Store environment data
            result["config_groups"][config_name] = {"env_data": env_data_list}
            
            # Extract metrics for this group
            exp_summaries = [d['summary']['exp_summary'] for d in env_data_list]
            eval_summaries = [d['summary']['eval_summary'] for d in env_data_list]
            
            # Calculate group performance using manager methods
            result["exp_summary"]["group_performance"][config_name] = ExplorationManager.aggregate_group_performance(exp_summaries)
            result["eval_summary"]["group_performance"][config_name] = EvaluationManager.aggregate_group_performance(eval_summaries)
            
            # Collect for overall calculation
            all_exp_data.extend(exp_summaries)
            all_eval_data.extend(eval_summaries)
        
        # Calculate overall performance using manager methods
        if all_exp_data:
            result["exp_summary"]["overall_performance"] = ExplorationManager.aggregate_group_performance(all_exp_data)
        
        if all_eval_data:
            result["eval_summary"]["overall_performance"] = EvaluationManager.aggregate_group_performance(all_eval_data)
        
        return result
    



    def _get_env_info(self):
        """Get environment state information."""
        return {
            "config": self.config.to_dict(),
            "initial_room": self.initial_room.to_dict(),
            "final_room": self.final_room.to_dict() if self.final_room else None,
        }

    @staticmethod
    def _assign_raw_messages(message: List[Dict], turn_logs: List[EnvTurnLog]) -> List[EnvTurnLog]:
        """Assign raw assistant messages to turn logs."""
        # Extract assistant messages from conversation
        assistant_messages = []
        for msg in message:
            if msg.get("role") == "assistant":
                assistant_messages.append(msg.get("content", ""))
        
        # Check if number of assistant messages matches turn logs
        if len(assistant_messages) != len(turn_logs):
            raise ValueError(f"Mismatch: {len(assistant_messages)} assistant messages vs {len(turn_logs)} turns")
        
        # Assign raw messages to turn logs
        for turn_log, raw_msg in zip(turn_logs, assistant_messages):
            think_content, answer_content = extract_think_and_answer(raw_msg)
            turn_log.assistant_raw_message = raw_msg
            turn_log.assistant_think_message = think_content
        



    
    







if __name__ == "__main__":
    # Simple test cases for SpatialGym environment
    
    def test_render_cache():
        """Test render cache functionality."""
        print("=== Testing Render Cache ===")
        
        # Create simple config
        config = SpatialGymConfig(
            name="render_test",
            exp_type='active',
            max_exp_steps=2,
            eval_tasks=[{"task_type": "rot", "task_kwargs": {}}],
            n_objects=2,
            generation_type="rand",
            room_range=[-3, 3]
        )
        
        # Test environment creation and initial render cache
        env = SpatialGym(config)
        _, info = env.reset(seed=42)
        
        # Test render returns cached observation
        rendered = env.render()
        print(f"Initial render matches observation: {rendered}")

        rendered = env.render()
        print(f"Initial render matches observation: {rendered}")
        
        # Test render cache updates after exploration step
        _, reward, done, info = env.step("Movement: []\nFinal: Observe()")
        rendered2 = env.render()
        print(f"Render cache updated after step: {rendered2}")
        rendered2 = env.render()
        print(f"Render cache updated after step again: {rendered2}")
        
        # Test render cache updates after evaluation transition
        _, reward, done, info = env.step("Movement: []\nFinal: Term()")  # End exploration
        rendered3 = env.render()
        print(f"Render cache updated after evaluation transition: {rendered3}")

        rendered3 = env.render()
        print(f"Render cache updated after evaluation transition again: {rendered3}")
        
        print(f"Render cache test completed successfully!")
        print()


    def test_active_exploration():
        """Test active exploration mode."""
        print("=== Testing Active Exploration ===")
        
        # Create config for active exploration
        config = SpatialGymConfig(
            name="active_test",
            exp_type='active',
            max_exp_steps=3,
            eval_tasks=[{"task_type": "rot", "task_kwargs": {}}],
            n_objects=3,
            generation_type="rand",
            room_range=[-5, 5]
        )
        
        
        # Create and reset environment
        env = SpatialGym(config)
        env.reset(seed=42)
        obs = env.render()
        print(f"Room: {env.initial_room}")
        print(f"Initial observation: {obs[:100]}...")
        
        # Take exploration steps
        _, reward, done, info = env.step("Movement: []\nFinal: Observe()")
        obs = env.render()
        print(f"Step 1 - Reward: {reward}, Done: {done}, Obs: {obs}")

        _, reward, done, info = env.step("Movement: [Move(microphone), Rotate(180)]\nFinal: Observe()")
        obs = env.render()
        print(f"Step 2 - Reward: {reward}, Done: {done}, Obs: {obs}")

        print(f"Exploration Room: {env.exploration_manager.exploration_room}")
        
        # End exploration and get evaluation question
        _, reward, done, info = env.step("Movement: []\nFinal: Term()")
        obs = env.render()
        print(f"Exploration ended - Reward: {reward}, Done: {done}")
        print(f"Evaluation question: {obs[:100]}...")
        
        # Answer evaluation question
        _, reward, done, info = env.step("['keyboard', 'sofa', 'microphone']")
        obs = env.render()
        print(f"Final - Reward: {reward}, Done: {done}")

        print(f"Exploration Room: {env.exploration_manager.exploration_room}")
        print(f"Final Room: {env.final_room}")

        
        # Get metrics
        exp_metrics = env.get_exp_efficiency()
        eval_metrics = env.get_eval_performance()
        print(f"Exploration efficiency: {exp_metrics}")
        print(f"Evaluation performance: {eval_metrics}")
        print()
    
    def test_passive_exploration():
        """Test passive exploration mode."""
        print("=== Testing Passive Exploration ===")
        
        # Create config for passive exploration
        config = SpatialGymConfig(
            name="passive_test",
            exp_type='passive',
            max_exp_steps=0,  # Not used in passive mode
            eval_tasks=[{"task_type": "rot", "task_kwargs": {}}],
            n_objects=3,
            generation_type="rand",
            room_range=[-5, 5],
            with_topdown=True
        )
        
        # Create and reset environment
        env = SpatialGym(config)
        env.reset(seed=42)
        print(f"Initial room: {env.initial_room}")
        obs = env.render()
        print(f"Initial observation with auto-exploration: {obs}")
        
        # Directly answer evaluation question (no exploration phase)
        _, reward, done, info = env.step("['keyboard', 'sofa', 'microphone']")
        obs = env.render()
        print(f"Answer - Reward: {reward}, Done: {done}")
        
        # Get metrics
        eval_metrics = env.get_eval_performance()
        print(f"Evaluation performance: {eval_metrics}")
        print()
    
    def test_basic_functionality():
        """Test basic environment functionality."""
        print("=== Testing Basic Functionality ===")
        
        # Create simple config
        config = SpatialGymConfig(
            name="basic_test",
            exp_type='active',
            max_exp_steps=2,
            eval_tasks=[{"task_type": "rot", "task_kwargs": {}}],
            n_objects=2,
            generation_type="rand",
            room_range=[-3, 3]
        )
        
        # Test environment creation and reset
        env = SpatialGym(config)
        obs, info = env.reset(seed=123)
        print(f"Environment created and reset successfully")
        print(f"Initial observation length: {len(obs)}")

        obs, info = env.reset(seed=1234)
        print(f"Environment created and reset successfully")
        print(f"Initial observation length: {len(obs)}")
        
        # Test render
        rendered = env.render()
        print(f"Render works: {rendered == obs}")
        
        # Test env info
        env_info = env.get_env_info()
        print(f"Environment info available: {'initial_room' in env_info}")
        print()

    
    def test_field_of_view():
        """Test field of view configuration."""
        print("=== Testing Field of View ===")
        
        # With seed=42, 'keyboard' is at a position that is outside 90 FOV but inside 180 FOV.
        # Let's use Observe() to check visibility.
        def run_test(fov, seed=0):
            print(f"\n--- Testing {fov}-degree FOV ---")
            config = SpatialGymConfig(
                name=f"fov_test_{fov}",
                exp_type='active',
                max_exp_steps=1,
                eval_tasks=[{"task_type": "rot", "task_kwargs": {}}],
                n_objects=3,
                generation_type="rand",
                room_range=[-5, 5],
                field_of_view=fov
            )
            env = SpatialGym(config)
            env.reset(seed=seed)
            print(f"Room for {fov} FOV test: {env.initial_room}")
            obs = env.render()
            print(f"Initial observation: {obs}")
            print(f"Room for {fov} FOV test: {env.initial_room}")
            obs, _, _, _ = env.step("Movement: []\nFinal: Observe()")
            print(f"Observation with {fov} FOV: {obs}")
            return obs

        # Test with 90-degree field of view
        obs_90 = run_test(90)
        assert "whiteboard" not in obs_90, "whiteboard should not be visible with 90 FOV"
        print("Keyboard not in observation, as expected.")

        # Test with 180-degree field of view
        obs_180 = run_test(180)
        assert "whiteboard" in obs_180, "whiteboard should be visible with 180 FOV"
        print("Keyboard in observation, as expected.")
        print()
    
    def test_config_grouping():
        """Test config grouping in aggregate_env_data."""
        print("=== Testing Config Grouping ===")
        
        # Create environments with different config names
        config1 = SpatialGymConfig(name="config_A", exp_type='active', max_exp_steps=1, n_objects=2)
        config2 = SpatialGymConfig(name="config_B", exp_type='passive', n_objects=3)
        config3 = SpatialGymConfig(name="config_A", exp_type='active', max_exp_steps=1, n_objects=2)  # Same as config1
        
        envs = {}
        messages = ["msg1", "msg2", "msg3"]
        env_ids = [1, 2, 3]
        
        # Create and reset environments
        for i, config in enumerate([config1, config2, config3], 1):
            env = SpatialGym(config)
            env.reset(seed=42)
            envs[i] = env
        
        # Test aggregation
        result = SpatialGym.aggregate_env_data(envs, messages, env_ids)
        
        print(f"Config groups found: {list(result['config_groups'].keys())}")
        print(f"Config A has {len(result['config_groups']['config_A']['env_data'])} environments")
        print(f"Config B has {len(result['config_groups']['config_B']['env_data'])} environments")
        print(f"Exploration efficiency sections: {list(result['exp_summary'].keys())}")
        print(f"Evaluation performance sections: {list(result['eval_summary'].keys())}")
        print(f"Result: {result}")
    
    # Run all tests
    try:
        # test_render_cache()
        # test_active_exploration()
        test_passive_exploration()
        # test_basic_functionality()
        # test_field_of_view()
        # test_config_grouping()
        print("All tests completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


