import gymnasium as gym
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, field

from ragen.env.spatial.config import SpatialGymConfig
from ragen.env.spatial.Base.tos_base import (
    EvaluationManager,
    EvaluationTurnLog,
    Room,
    ActionSequence,
    ExplorationManager,
    ExplorationTurnLog,
    CognitiveMapManager,
    CognitiveMapTurnLog,
    generate_room,
    BaseAction
)
from ragen.env.spatial.prompts import Prompter
from ragen.env.spatial.utils.action_utils import action_results_to_text
from ragen.env.spatial.utils.utils import extract_think_and_answer
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
    cogmap_log: Optional["CognitiveMapTurnLog"] = None
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
            "cogmap_log": self.cogmap_log.to_dict() if self.cogmap_log else {},
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
        self.prompter = None

        self.is_exploration_phase = None
        self.remaining_exp_steps = None
        self.render_cache = None

        # Room state management
        self.initial_room = None
        self.final_room = None
        
        # Managers
        self.exploration_manager = None
        self.evaluation_manager = None
        self.cognitive_map_manager = None

        # Turn logging
        self.turn_logs: List[EnvTurnLog] = None
        self.current_turn_number = None


    def _generate_initial_observation(self) -> str:
        """Generate initial observation based on exploration type."""
        return self.prompter.get_initial_observation_prompt(
            room=self.initial_room, 
            eval_manager=self.evaluation_manager,
            cogmap_manager=self.cognitive_map_manager
        )
    

    def reset(self, seed: int = None):
        """Reset environment for a new episode."""
        super().reset(seed=seed)

        self.prompter = Prompter(self.config, self.np_random)
        
        # Generate initial room
        self.initial_room = generate_room(
            **self.config.get_room_config(),
            np_random=self.np_random,
        )

        # Initialize episode state
        self.remaining_exp_steps = self.config.max_exp_steps
        
        # Reset turn tracking
        self.turn_logs = []
        self.current_turn_number = 0
        
        # Set exploration phase
        self.is_exploration_phase = self.config.exp_type == 'active'
        
        # Set field of view for all actions
        BaseAction.set_field_of_view(self.config.field_of_view)
        
        # Initialize managers
        self.exploration_manager = ExplorationManager(self.initial_room) if self.config.exp_type == 'active' else None
        self.evaluation_manager = EvaluationManager(self.config.eval_tasks, self.np_random, self.initial_room) if len(self.config.eval_tasks) > 0 else None
        self.cognitive_map_manager = CognitiveMapManager() if self.config.prompt_with_cogmap else None

        # Generate initial observation
        obs = self._generate_initial_observation()
        self.render_cache = obs
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
        self.remaining_exp_steps -= 1
        exp_log = None

        # Parse and validate action
        action_sequence = ActionSequence.parse(action)
        if not action_sequence:
            obs += "Invalid action\n"
            reward += -0.5 # format penalty
        else:
        # Execute action
            exp_info, action_results = self.exploration_manager.execute_action_sequence(action_sequence)
            reward += -1 if exp_info.get('redundant', False) else 0 # redundant observe penalty
            obs += action_results_to_text(action_results)
            exp_log = self.exploration_manager.turn_logs[-1]

        # End exploration phase
        if self.remaining_exp_steps < 0 or (action_sequence and action_sequence.final_action.is_term()):
            self.is_exploration_phase = False
            obs += "Exploration phase ended\n"
            self.final_room = self.exploration_manager.finish_exploration()
            # Transition to evaluation, NOTE question is generated based on the initial room
            obs += self.prompter.get_evaluation_prompt(self.evaluation_manager)
        else:
            obs += f"\nYou have a maximum of {self.remaining_exp_steps} exploration steps left."
        
        return obs, reward, False, {}, exp_log
    
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
        exp_log, eval_log, cogmap_log = None, None, None
        think_content, action = extract_think_and_answer(llm_response)
        room_state_last_turn = next((turn_log.room_state for turn_log in self.turn_logs[::-1] if turn_log.room_state), self.initial_room)
        
        # Log turn at start with current state
        current_obs = self.render_cache if hasattr(self, 'render_cache') else ""
        
        if action and think_content:
            # Evaluate cognitive map if enabled and we have assistant response
            if self.cognitive_map_manager:
                self.cognitive_map_manager.evaluate_cognitive_map(think_content, room_state_last_turn)
                cogmap_log = self.cognitive_map_manager.turn_logs[-1]

            if self.is_exploration_phase:
                obs, reward, done, step_info, exp_log = self._step_exploration(action)
            else:
                obs, reward, done, step_info, eval_log = self._step_evaluation(action)
        else:
            reward = -0.5 # format penalty
            obs = "Invalid input format.\n"
            done, step_info = False, {}

        obs += self.prompter.COGMAP_REQUIRED_INSTRUCTION if self.config.prompt_with_cogmap else ""
        self.render_cache = obs

        # Get room state from turn logs
        room_state = room_state_last_turn
        if exp_log and exp_log.room_state:
            room_state = exp_log.room_state
        elif eval_log and eval_log.room_state:
            room_state = eval_log.room_state
        
        turn_log = EnvTurnLog(
            turn_number=self.current_turn_number,
            user_message=current_obs,
            assistant_raw_message=llm_response,
            assistant_think_message=think_content,
            assistant_parsed_message=action,
            is_exploration_phase=self.is_exploration_phase,
            room_state=room_state,
            exploration_log=exp_log,
            evaluation_log=eval_log,
            cogmap_log=cogmap_log,
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
            "final_room": self.final_room.to_dict() if self.final_room else None,
        }





    
    







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
        exp_metrics = env.get_exp_summary()
        eval_metrics = env.get_eval_summary()
        print(f"Exploration summary: {exp_metrics}")
        print(f"Evaluation summary: {eval_metrics}")
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
        eval_metrics = env.get_eval_summary()
        print(f"Evaluation summary: {eval_metrics}")
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
        env_info = env._get_env_info()
        print(f"Environment info available: {'initial_room' in env_info}")
        print()

    
    def test_field_of_view():
        """Test field of view configuration."""
        print("=== Testing Field of View ===")
        config = SpatialGymConfig(
            name="fov_test",
            exp_type='active',
            max_exp_steps=1,
            eval_tasks=[{"task_type": "rot", "task_kwargs": {}}],
            n_objects=3,
            generation_type="rand",
            room_range=[-5, 5],
            field_of_view=180
        )
        env = SpatialGym(config)
        env.reset(seed=42)
        assert env.config.field_of_view == 180
        print("Field of view test passed.")
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
        test_render_cache()
        test_active_exploration()
        test_passive_exploration()
        test_basic_functionality()
        test_field_of_view()
        print("All tests completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


