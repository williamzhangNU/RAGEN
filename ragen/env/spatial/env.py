import gymnasium as gym
import re
from typing import Optional, List, Dict

from ragen.env.spatial.config import SpatialGymConfig
from ragen.env.spatial.Base.tos_base import (
    EvaluationManager,
    Room,
    ActionSequence,
    ExplorationManager,
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
        
        # Set exploration phase
        self.is_exploration_phase = self.config.exp_type not in ['passive', 'overview']
        
        # Set field of view for all actions
        BaseAction.set_field_of_view(self.config.field_of_view)
        
        # Initialize managers
        if self.config.exp_type in ['active', 'active_overview']:
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
    def get_exploration_per_turn_metrics(self):
        """Get exploration per turn metrics"""
        if self.config.exp_type == 'passive':
            return []
        assert self.exploration_manager, "Exploration manager not initialized"
        return self.exploration_manager.get_metrics_log()
    def get_evaluation_per_turn_metrics(self):
        """Get evaluation per turn metrics"""
        return self.evaluation_manager.get_eval_metrics_log()

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
                'exploration_efficiency': {
                    'overall_performance': {...},
                    'group_performance': {'config_name': {...}}
                },
                'evaluation_performance': {
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
            env_data = {
                "message": message,
                "env_info": env.get_env_info(),
                "exploration_efficiency": env.get_exp_efficiency(),
                "evaluation_performance": env.get_eval_performance(),
                "exploration_metrics_log": env.get_exploration_per_turn_metrics(),
                "evaluation_metrics_log": env.get_evaluation_per_turn_metrics()
            }
            config_groups[config_name].append(env_data)
        
        # Initialize result structure
        result = {
            "config_groups": {},
            "exploration_efficiency": {"overall_performance": {}, "group_performance": {}},
            "evaluation_performance": {"overall_performance": {}, "group_performance": {}}
        }
        
        # Collect all metrics for overall calculation
        all_exp_data, all_eval_data = [], []
        
        for config_name, env_data_list in config_groups.items():
            # Store environment data
            result["config_groups"][config_name] = {"env_data": env_data_list}
            
            # Extract metrics for this group
            exp_metrics = [d['exploration_efficiency'] for d in env_data_list]
            eval_metrics = [d['evaluation_performance'] for d in env_data_list]
            
            # Calculate group performance
            result["exploration_efficiency"]["group_performance"][config_name] = {
                'avg_coverage': sum(m.get('coverage', 0) for m in exp_metrics) / len(exp_metrics),
                'avg_redundancy': sum(m.get('redundancy', 0) for m in exp_metrics) / len(exp_metrics),
            }
            
            result["evaluation_performance"]["group_performance"][config_name] = {
                'avg_accuracy': sum(m.get('accuracy', 0) for m in eval_metrics) / len(eval_metrics),
            }
            
            # Collect for overall calculation
            all_exp_data.extend(exp_metrics)
            all_eval_data.extend(eval_metrics)
        
        # Calculate overall performance
        if all_exp_data:
            result["exploration_efficiency"]["overall_performance"] = {
                'avg_coverage': sum(m.get('coverage', 0) for m in all_exp_data) / len(all_exp_data),
                'avg_redundancy': sum(m.get('redundancy', 0) for m in all_exp_data) / len(all_exp_data),
            }
        
        if all_eval_data:
            result["evaluation_performance"]["overall_performance"] = {
                'avg_accuracy': sum(m.get('accuracy', 0) for m in all_eval_data) / len(all_eval_data),
            }
        
        return result


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
        print(f"Exploration efficiency sections: {list(result['exploration_efficiency'].keys())}")
        print(f"Evaluation performance sections: {list(result['evaluation_performance'].keys())}")
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


