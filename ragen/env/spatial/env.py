import gymnasium as gym
import re
from typing import Optional

from ragen.env.spatial.config import SpatialGymConfig
from ragen.env.spatial.Base.tos_base import (
    EvaluationManager,
    Room,
    ActionSequence,
    ExplorationManager,
    generate_room
)
from ragen.env.spatial.utils.generate_history import AutoExplore
from ragen.env.spatial.prompts import (
    ACTIVE_INSTRUCTION, 
    PASSIVE_INSTRUCTION, 
    SHORT_EXPLORATION_PROMPT, 
    SHORT_EVALUATION_PROMPT
)




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

        self.original_render_read = None 
        # this is in case ctx_manager parses no valid action, so if the original render is read, we will use one short prompt to replace it

    def _generate_initial_observation(self) -> str:
        """Generate initial observation based on exploration type."""
        room_desc = self.initial_room.get_room_description()
        
        if self.config.exp_type == 'passive':
            auto_explore = AutoExplore(self.initial_room, self.np_random)
            exp_history = f"## Exploration History\n{auto_explore.gen_exp_history()}"
            eval_question = self.evaluation_manager.get_current_question(self.initial_room.copy())
            assert eval_question, "No question found after exploration phase"
            obs = PASSIVE_INSTRUCTION.format(
                room_info=room_desc,
                exp_history=exp_history,
                eval_question=f"## Evaluation Question\n{eval_question}"
            )

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
        
        # Set exploration phase
        self.is_exploration_phase = self.config.exp_type != 'passive'
        
        # Initialize managers
        if self.config.exp_type == 'active':
            self.exploration_manager = ExplorationManager(self.initial_room)
        self.evaluation_manager = EvaluationManager(self.config.eval_tasks, self.np_random)

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
            
            # Transition to evaluation
            question = self.evaluation_manager.get_current_question(self.initial_room.copy())
            assert question, "No question found after exploration phase"
            obs += question
        else:
            # Execute exploration action, TODO give reward to efficient exploration
            if action_sequence:
                result, exp_info = self.exploration_manager.execute_action_sequence(action_sequence)
                # Track redundant queries
                if exp_info.get('redundant', False):
                    self.n_redundant_queries += 1
                    reward += -1 # redundant observe penalty
                obs += result
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
        if self.is_exploration_phase:
            return self._step_exploration(action)
        else:
            return self._step_evaluation(action)

    def render(self):
        if not self.original_render_read:
            self.original_render_read = True
            return self.render_cache
        if self.is_exploration_phase:
            return SHORT_EXPLORATION_PROMPT + f"You have a maximum of {self.remaining_exp_steps} exploration steps left."
        else:
            return SHORT_EVALUATION_PROMPT

    # =============== Analysis Methods ===============
    def get_env_info(self):
        """Get environment state information."""
        return {
            "config": self.config.to_dict(),
            "initial_room": self.initial_room.to_dict(),
            "final_room": self.final_room.to_dict() if self.final_room else None,
        }

    def get_exp_efficiency(self):
        """Get exploration efficiency metrics."""
        if self.config.exp_type == 'passive':
            return {
                "coverage": 0,
                "redundancy": self.n_redundant_queries / self.n_valid_queries if self.n_valid_queries > 0 else 0,
                "n_valid_queries": self.n_valid_queries,
                "n_redundant_queries": self.n_redundant_queries,
            }
        
        assert self.exploration_manager, "Exploration manager not initialized"
        return self.exploration_manager.get_exploration_efficiency()
    
    def get_eval_performance(self):
        """Get evaluation performance metrics."""
        if not self.evaluation_manager:
            return {
                "accuracy": 0.0,
                "accuracy_completed": 0.0,
                "task_results": [],
                "completed_tasks": 0,
                "unanswered_tasks": 0
            }
        
        return self.evaluation_manager.get_evaluation_summary()


if __name__ == "__main__":
    # Simple test cases for SpatialGym environment
    
    def test_render_cache():
        """Test render cache functionality."""
        print("=== Testing Render Cache ===")
        
        # Create simple config
        config = SpatialGymConfig(
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
            exp_type='active',
            max_exp_steps=3,
            eval_tasks=[{"task_type": "rot", "task_kwargs": {}}],
            n_objects=3,
            generation_type="rand",
            room_range=[-5, 5]
        )
        
        
        # Create and reset environment
        env = SpatialGym(config)
        obs, info = env.reset(seed=42)
        print(f"Room: {env.initial_room}")
        print(f"Initial observation: {obs[:100]}...")
        
        # Take exploration steps
        obs, reward, done, info = env.step("Movement: []\nFinal: Observe()")
        print(f"Step 1 - Reward: {reward}, Done: {done}, Obs: {obs}")

        obs, reward, done, info = env.step("Movement: [Move(microphone), Rotate(180)]\nFinal: Observe()")
        print(f"Step 2 - Reward: {reward}, Done: {done}, Obs: {obs}")

        print(f"Exploration Room: {env.exploration_manager.exploration_room}")
        
        # End exploration and get evaluation question
        obs, reward, done, info = env.step("Movement: []\nFinal: Term()")
        print(f"Exploration ended - Reward: {reward}, Done: {done}")
        print(f"Evaluation question: {obs[:100]}...")
        
        # Answer evaluation question
        obs, reward, done, info = env.step("['keyboard', 'sofa', 'microphone']")
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
            exp_type='passive',
            max_exp_steps=0,  # Not used in passive mode
            eval_tasks=[{"task_type": "rot", "task_kwargs": {}}],
            n_objects=3,
            generation_type="rand",
            room_range=[-5, 5]
        )
        
        # Create and reset environment
        env = SpatialGym(config)
        obs, info = env.reset(seed=42)
        print(f"Initial observation with auto-exploration: {obs[:100]}...")
        
        # Directly answer evaluation question (no exploration phase)
        obs, reward, done, info = env.step("['keyboard', 'sofa', 'microphone']")
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
    
    # Run all tests
    try:
        test_render_cache()
        # test_active_exploration()
        # test_passive_exploration()
        # test_basic_functionality()
        print("All tests completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

