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
from ragen.env.spatial.prompts import ACTIVE_INSTRUCTION, PASSIVE_INSTRUCTION




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
        self.render_cache = obs
        return obs, {}
    
    def _step_exploration(self, action: str):
        """Handle exploration phase step."""
        obs = ""
        reward = 0

        # Parse and validate action
        action_sequence = ActionSequence.parse(action)
        if not action_sequence:
            obs += "Invalid action\n"
            reward += -0.1
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
                obs += result
            obs += f"You have a maximum of {self.remaining_exp_steps} exploration steps left."
        
        self.render_cache = obs
        return obs, reward, False, {}
    
    def _step_evaluation(self, action: str):
        """Handle evaluation phase step."""
        # Evaluate answer
        correct, reward, info = self.evaluation_manager.evaluate_answer(action)
        
        # Check for next task
        if self.evaluation_manager.next_task():
            next_question = self.evaluation_manager.get_current_question(self.initial_room.copy())
            assert next_question, "No question found after evaluation phase"
            self.render_cache = next_question
            return next_question, reward, not bool(next_question), {}
        
        # All tasks completed
        self.render_cache = "Task finished"
        return "Task finished", reward, True, {}

    def step(self, action: str):
        """Process agent actions in the spatial gym environment."""
        if self.is_exploration_phase:
            return self._step_exploration(action)
        else:
            return self._step_evaluation(action)

    def render(self):
        return self.render_cache

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

    # TODO
    def test_passive_exploration():
        """Test passive exploration mode."""
        print("Testing Passive Exploration...")
        
        config = SpatialGymConfig(
            exp_type='passive',
            n_objects=3,
            room_range=[-5, 5],
            eval_tasks=[
                {"task_type": "dir", "task_kwargs": {}},
                {"task_type": "all_pairs", "task_kwargs": {}}
            ],
            max_exp_steps=50
        )
        
        env = SpatialGym(config)
        obs, info = env.reset(seed=42)
        print(f"room: {env.initial_room}")
        print(f"Initial observation <<{obs}>>")
        print(f"Contains exploration history: {'Exploration History' in obs}")
        
        # Simulate evaluation answers
        done = False
        step_count = 0
        while not done and step_count < 10:
            # Simple answer format for testing
            answer = "(unknown, unknown)"
            print(f"ground truth answer: {env.evaluation_manager._get_current_eval_task().answer}")
            obs, reward, done, info = env.step(answer)
            step_count += 1
            print(f"observation <<{obs}>>, Step {step_count}: Reward={reward}, Done={done}")
        
        # Check evaluation performance
        eval_perf = env.get_eval_performance()
        print(f"Evaluation accuracy: {eval_perf['accuracy']:.2f}")
        print("Passive exploration test completed.\n")

    def test_active_exploration():
        """Test active exploration mode."""
        print("Testing Active Exploration...")
        
        config = SpatialGymConfig(
            exp_type='active',
            n_objects=4,
            room_range=[-8, 8],
            eval_tasks=[{"task_type": "dir", "task_kwargs": {}}],
            max_exp_steps=20
        )
        
        env = SpatialGym(config)
        obs, info = env.reset(seed=123)
        print(f"room: {env.initial_room}")
        env.initial_room.plot()
        print(f"Initial observation contains action format: {'Available Actions' in obs}")
        
        # Test exploration phase
        exploration_actions = [
            "Observe()",
            "Rotate(90); Observe()",
            "Rotate(180); Observe()",
            "Rotate(90), Move(chair); Observe()",
            # "Move(keyboard), Rotate(90); Observe()",
            # "Rotate(90); Observe()",
        ]
        
        step_count = 0
        for action in exploration_actions:
            if env.is_exploration_phase:
                obs, reward, done, info = env.step(action)
                step_count += 1
                print(f"Observation <<{obs}>>, Exploration step {step_count}: Action='{action}', Valid response received")
                if not env.is_exploration_phase:
                    print("Transitioned to evaluation phase")
                    break
            else:
                break

        print(f"all objects in exploration manager: {env.exploration_manager.exploration_room.all_objects}")
        print(f"Exploration graph: {env.exploration_manager.exp_graph.to_dict()}")
        
        # Test evaluation phase
        if not env.is_exploration_phase:
            answer = "right"
            print(f"ground truth answer: {env.evaluation_manager._get_current_eval_task().answer}")
            obs, reward, done, info = env.step(answer)
            print(f"Evaluation answer: Reward={reward}, Done={done}")
        
        # Check exploration efficiency
        exp_eff = env.get_exp_efficiency()
        print(f"Exploration coverage: {exp_eff['coverage']:.2f}")
        print(f"Redundancy: {exp_eff['redundancy']:.2f}")
        print(f"Valid queries: {exp_eff['n_valid_queries']}")
        print(f"Redundant queries: {exp_eff['n_redundant_queries']}")
        print("Active exploration test completed.\n")

    def test_different_generation_types():
        """Test different room generation types."""
        print("Testing Different Generation Types...")
        
        generation_types = ["rand", "rot", "a2e", "pov"]
        
        for gen_type in generation_types:
            try:
                print(f"Testing generation type: {gen_type}")
                
                # Adjust perspective based on generation type
                perspective = "ego" if gen_type in ["rot", "pov"] else "ego"
                
                config = SpatialGymConfig(
                    generation_type=gen_type,
                    perspective=perspective,
                    exp_type='passive',
                    n_objects=3,
                    eval_tasks=[{"task_type": "dir", "task_kwargs": {}}]
                )
                
                env = SpatialGym(config)
                obs, info = env.reset(seed=42)
                print(f"room: {env.initial_room}")
                
                # Get environment info
                env_info = env.get_env_info()
                print(f"  Room generated with {len(env_info['initial_room']['all_objects'])} objects")
                print(f"  Generation type: {env_info['config']['generation_type']}")
                
            except Exception as e:
                print(f"  Error with {gen_type}: {e}")

        print("Generation types test completed.\n")

    def test_evaluation_tasks():
        """Test different evaluation task types."""
        print("Testing Different Evaluation Tasks...")
        
        task_configs = [
            {"task_type": "dir", "task_kwargs": {}},
            {"task_type": "rot", "task_kwargs": {"turn_direction": "clockwise"}},
            {"task_type": "pov", "task_kwargs": {}},
            {"task_type": "all_pairs", "task_kwargs": {}}
        ]
        
        for task_config in task_configs:
            try:
                print(f"Testing task: {task_config['task_type']}")
                
                config = SpatialGymConfig(
                    exp_type='passive',
                    n_objects=3,
                    eval_tasks=[task_config],
                    perspective='ego',
                    generation_type='pov'
                )
                
                env = SpatialGym(config)
                obs, info = env.reset(seed=42)
                print(f"room: {env.initial_room}")
                print(f"observation: {obs}")
                
                # Try one evaluation step
                answer = env.evaluation_manager._get_current_eval_task().answer
                print(f"ground truth answer: {answer}")
                obs, reward, done, info = env.step(answer)
                print(f"  Task executed successfully, reward: {reward}")
                
            except Exception as e:
                print(f"  Error with {task_config['task_type']}: {e}")
        
        print("Evaluation tasks test completed.\n")

    def test_action_parsing():
        """Test action sequence parsing."""
        # TODO more test cases
        print("Testing Action Parsing...")
        
        from ragen.env.spatial.Base.tos_base import ActionSequence
        
        test_actions = [
            "Observe()",
            "Move(chair), Rotate(90); Observe()",
            "Rotate(90); Observe()",
            "Return(); Observe()",
            "Term()",
            "Invalid action",
            "Observe() Move(chair)",  # Multiple actions
            ""
        ]
        
        for action_str in test_actions:
            action_seq = ActionSequence.parse(action_str)
            if action_seq:
                print(f"  '{action_str}' -> Valid: {action_seq}")
            else:
                print(f"  '{action_str}' -> Invalid")
        
        print("Action parsing test completed.\n")

    def test_environment_states():
        """Test environment state transitions."""
        print("Testing Environment States...")
        
        config = SpatialGymConfig(
            exp_type='active',
            n_objects=3,
            max_exp_steps=5
        )
        
        env = SpatialGym(config)
        obs, info = env.reset(seed=42)
        
        print(f"Initial state - Is exploration: {env.is_exploration_phase}")
        
        # Force transition to evaluation by terminating
        obs, reward, done, info = env.step("Term()")
        print(f"After termination - Is exploration: {env.is_exploration_phase}")
        
        # Test evaluation phase
        if not env.is_exploration_phase:
            print(f"ground truth answer: {env.evaluation_manager._get_current_eval_task().answer}")
            obs, reward, done, info = env.step("left")
            print(f"Evaluation step completed - Done: {done}")
        
        # Check final states
        env_info = env.get_env_info()
        print(f"Room states available - initial_room: {bool(env_info['initial_room'])}, "
              f"final_room: {bool(env_info['final_room'])}")
        
        print("Environment states test completed.\n")

    def test_configuration_validation():
        """Test configuration validation."""
        print("Testing Configuration Validation...")
        
        # Test valid configurations
        valid_configs = [
            {"exp_type": "passive", "perspective": "ego"},
            {"exp_type": "active", "perspective": "ego"},
            {"generation_type": "rand", "perspective": "ego"},
            {"generation_type": "rot", "perspective": "ego"}
        ]
        
        for config_dict in valid_configs:
            try:
                config = SpatialGymConfig(**config_dict)
                print(f"  Valid config: {config_dict}")
            except Exception as e:
                print(f"  Unexpected error with {config_dict}: {e}")
        
        # Test invalid configurations
        invalid_configs = [
            {"generation_type": "invalid_type"},
            {"exp_type": "invalid_exp"},
            {"perspective": "invalid_perspective"},
            {"generation_type": "rot", "perspective": "allo"}  # Incompatible combination
        ]
        
        for config_dict in invalid_configs:
            try:
                config = SpatialGymConfig(**config_dict)
                print(f"  Unexpected success with invalid config: {config_dict}")
            except Exception as e:
                print(f"  Expected error with {config_dict}: {type(e).__name__}")
        
        print("Configuration validation test completed.\n")

    # Run all tests
    print("="*50)
    print("SPATIAL GYM ENVIRONMENT TESTS")
    print("="*50)
    
    try:
        # test_passive_exploration()
        # test_active_exploration()
        # test_different_generation_types()
        # test_evaluation_tasks()
        # test_action_parsing()
        # test_environment_states()
        test_configuration_validation()
        
        print("="*50)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*50)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()