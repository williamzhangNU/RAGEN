import gymnasium as gym
import re
from typing import Optional

from ragen.env.spatial.config import SpatialGymConfig
from ragen.env.spatial.ToS_Base.tos_base import (
    EvaluationManager,
    Room,
    ActionSequence,
    ExplorationManager,
    generate_room
)
from ragen.env.spatial.utils.generate_history import AutoExplore


instruction = (
    "# Spatial Mapping Task\n"
    "\n"
    "You are exploring a room to discover spatial relationships between objects.\n"
    "Build a complete mental map by finding where each object is relative to others.\n"
    "\n"
    "## Spatial Relationships\n"
    "When you query an object, you get its position relative to you: (horizontal, vertical)\n"
    "\n"
    "- Horizontal: left, right, same\n"
    "- Vertical: front, back, same\n"
    "- Example: (left, front) means object is to your left and in front of you\n"
    "\n"
    "## Key Points\n"
    "- Relationships are relative: if A is left of B, then B is right of A\n"
    "- Terminate when you have enough information to map all object pairs\n"
    "\n"
    "## Room Layout\n"
    "{room_info}\n"
    "\n"
    "{exp_history}\n"
    "\n"
    "{exp_answer_format}\n"
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
        self.is_exp_stage = None  # indicates exploration or evaluation stage
        self.max_exp_steps = None
        self.render_cache = None

        # Room state management
        self.room_s_t = None  # latest/current state of the room
        self.room_s_0 = None  # initial state of the room
        self.room_s_end = None  # final state of the room, agent may return to its original state
        
        # Managers
        self.exploration_manager = None  # handles exploration logic
        self.evaluation_manager = None  # handles evaluation logic

        # Action space configuration
        self.move_action = ["Move", "Rotate", "Return"] if self.config.exp_type == 'active' else []
        self.query_action = ["Query"] if self.config.exp_type != 'passive' else []
        self.term_action = ["Term"] if self.config.exp_type != 'passive' else []
        self.action_space = self.move_action + self.query_action

        # Exploration metrics
        self.n_novel_queries = None
        self.n_valid_queries = None

    def _gen_initial_obs(self):
        """
        Generate initial observation as a user message (instruction).
        """
        exp_history = ""
        room_desc = self.room_s_0.get_room_description()
        exp_answer_format = ""
        
        if self.config.exp_type == 'passive':
            # Generate exploration history using DFS
            auto_explore = AutoExplore(self.room_s_0, self.np_random)
            exp_history = auto_explore.gen_exp_history()
            exp_history = f"## Exploration History\n{exp_history}"
        else:
            # Generate action format instructions for active exploration
            exp_answer_format = (
                "## Available Actions\n"
                f"Movement: {', '.join(self.move_action)}\n"
                f"Query: {', '.join(self.query_action)}\n"
                f"Term: {', '.join(self.term_action)}\n"
                "\n" +
                ActionSequence.get_usage_instructions()
            )
        
        # Format the instruction
        obs = instruction.format(
            room_info=room_desc,
            exp_history=exp_history,
            exp_answer_format=exp_answer_format
        )

        # For passive exploration, add the first evaluation question
        if self.config.exp_type == 'passive':
            first_question = self.evaluation_manager.get_current_question(self.room_s_0.copy())
            if first_question:
                obs = obs + "\n\n" + first_question

        return obs

        


    
    
    def reset(self, seed: int = None):
        """
        Reset the environment for a new episode.
        
        1. Generate initial room
        2. Initialize evaluation manager
        3. Set up exploration manager if needed
        4. Generate initial observation

        Returns:
            - obs (str): Initial observation/instruction
            - info (dict): Additional information
        """
        super().reset(seed=seed)
        self.max_exp_steps = self.config.max_exp_steps
        self.n_valid_queries = 0
        self.n_novel_queries = 0

        # Generate initial room
        self.room_s_0: Room = generate_room(
            **self.config.get_room_config(),
            np_random=self.np_random,
        )
        self.room_s_t = self.room_s_0.copy()
        
        self.is_exp_stage = True if self.config.exp_type != 'passive' else False
        # Initialize exploration manager for active exploration
        if self.config.exp_type == 'active':
            self.exploration_manager = ExplorationManager(self.room_s_0)
        
        # Initialize evaluation manager
        self.evaluation_manager = EvaluationManager(self.config.eval_tasks, self.np_random)

        # Generate initial observation
        obs = self._gen_initial_obs()
        self.render_cache = obs
        return obs, {}
        
    
    def step(self, action: str):
        """
        Process agent actions in the spatial gym environment.
        
        Args:
            action (str): Either an exploration command or evaluation answer
        
        Returns:
            tuple: (observation, reward, done, info)
                - observation: Current environment state description
                - reward: Numerical reward signal
                - done: Whether episode is complete
                - info: Additional information dictionary
        """
        
        # Exploration stage
        if self.is_exp_stage:
            self.max_exp_steps -= 1
            
            # Parse action using exploration manager
            action_sequence = ActionSequence.parse(action)
            if not action_sequence:
                self.render_cache = "Invalid action"
                return "Invalid action", -0.1, False, {}
            
            # Check if exploration phase should end
            if (action_sequence.final_action.is_term() or self.max_exp_steps < 0):
                self.is_exp_stage = False
                self.room_s_end = self.exploration_manager.finish_exploration()
                
                # Transition to first evaluation task
                question = self.evaluation_manager.get_current_question(self.room_s_0.copy())
                self.render_cache = question or "Task finished"
                return question or "Task finished", 0, not bool(question), {}
            else:
                # Continue exploration
                self.n_valid_queries += 1
                result, exp_info = self.exploration_manager.execute_action_sequence(action_sequence)
                if exp_info['novel_query']:
                    self.n_novel_queries += 1
                
                self.render_cache = result
                return result, 0, False, {}
        
        # Evaluation stage
        else:
            # Evaluate current task answer using evaluation manager
            correct, reward, info = self.evaluation_manager.evaluate_answer(action)
            
            if self.evaluation_manager.next_task():
                # Get next question
                next_question = self.evaluation_manager.get_current_question(self.room_s_0.copy())
                self.render_cache = next_question or "Task finished"
                return next_question or "Task finished", reward, not bool(next_question), {}
            else:
                # All tasks completed
                self.render_cache = "Task finished"
                return "Task finished", reward, True, {}
        

    def render(self):
        return self.render_cache




    #=============== for analysis ===============
    def get_env_info(self):
        return {
            "config": self.config.to_dict(),
            "room_s_0": self.room_s_0.to_dict(),
            "room_s_t": self.room_s_t.to_dict(),
            "room_s_end": self.room_s_end.to_dict() if self.room_s_end else None,
        }

    def get_exp_efficiency(self):
        """
        Get the exploration efficiency
        - Coverage: percentage of pairs covered (known / all relations)
        - Novelty: percentage of novel pairs (inferable / all queries)
        TODO use exploration manager to get efficiency
        """
        assert self.config.exp_type in ["active", "passive"]
        if self.config.exp_type == 'passive':
            return {
                "coverage": 0,
                "novelty": 0,
                "n_valid_queries": 0,
                "n_novel_queries": 0,
            }
        # if self.exploration_manager:
        #     unknown_pairs = self.exploration_manager.get_unknown_pairs()
        #     n_object = len(self.room_s_0.all_objects)
        #     max_rels = int(n_object * (n_object - 1) / 2)
        #     coverage = (max_rels - len(unknown_pairs)) / max_rels
        # else:
        #     coverage = 0
            
        # return {
        #     "coverage": coverage,
        #     "novelty": self.n_novel_queries / self.n_valid_queries if self.n_valid_queries > 0 else 0,
        #     "n_valid_queries": self.n_valid_queries,
        #     "n_novel_queries": self.n_novel_queries,
        # }
        if self.exploration_manager:
            return self.exploration_manager.get_exploration_efficiency()
        else:
            raise ValueError("Exploration manager not initialized")
    
    def get_eval_performance(self):
        """
        Get the evaluation performance using the EvaluationManager.
        
        Returns:
            Dictionary containing evaluation metrics and detailed results
        """
        if self.evaluation_manager is None:
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
        print(f"room: {env.room_s_0}")
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
        print(f"room: {env.room_s_0}")
        print(f"Initial observation contains action format: {'Available Actions' in obs}")
        
        # Test exploration phase
        exploration_actions = [
            "Observe()",
            "Rotate(90); Observe()",
            "Rotate(180); Observe()",
            "Move(keyboard); Rotate(90); Observe()",
        ]
        
        step_count = 0
        for action in exploration_actions:
            if env.is_exp_stage:
                obs, reward, done, info = env.step(action)
                step_count += 1
                print(f"Observation <<{obs}>>, Exploration step {step_count}: Action='{action}', Valid response received")
                if not env.is_exp_stage:
                    print("Transitioned to evaluation phase")
                    break
            else:
                break

        print(f"all objects in exploration manager: {env.exploration_manager.exploration_room.all_objects}")
        print(f"Exploration graph: {env.exploration_manager.exp_graph.to_dict()}")
        
        # Test evaluation phase
        if not env.is_exp_stage:
            answer = "right"
            print(f"ground truth answer: {env.evaluation_manager._get_current_eval_task().answer}")
            obs, reward, done, info = env.step(answer)
            print(f"Evaluation answer: Reward={reward}, Done={done}")
        
        # Check exploration efficiency
        exp_eff = env.get_exp_efficiency()
        print(f"Exploration coverage: {exp_eff['coverage']:.2f}")
        print(f"Novel queries: {exp_eff['n_novel_queries']}/{exp_eff['n_valid_queries']}")
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
                print(f"room: {env.room_s_0}")
                
                # Get environment info
                env_info = env.get_env_info()
                print(f"  Room generated with {len(env_info['room_s_0']['all_objects'])} objects")
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
                print(f"room: {env.room_s_0}")
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
        print("Testing Action Parsing...")
        
        from ragen.env.spatial.ToS_Base.tos_base import ActionSequence
        
        test_actions = [
            "Query(table)",
            "Move(chair), Rotate(90); Query(table)",
            "Rotate(90); Query(table)",
            "Return(); Query(table)",
            "Term()",
            "Invalid action",
            "Query(table) Move(chair)",  # Multiple actions
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
        
        print(f"Initial state - Is exploration: {env.is_exp_stage}")
        
        # Force transition to evaluation by terminating
        obs, reward, done, info = env.step("Term()")
        print(f"After termination - Is exploration: {env.is_exp_stage}")
        
        # Test evaluation phase
        if not env.is_exp_stage:
            print(f"ground truth answer: {env.evaluation_manager._get_current_eval_task().answer}")
            obs, reward, done, info = env.step("left")
            print(f"Evaluation step completed - Done: {done}")
        
        # Check final states
        env_info = env.get_env_info()
        print(f"Room states available - s_0: {bool(env_info['room_s_0'])}, "
              f"s_t: {bool(env_info['room_s_t'])}, s_end: {bool(env_info['room_s_end'])}")
        
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
        test_evaluation_tasks()
        # test_action_parsing()
        # test_environment_states()
        # test_configuration_validation()
        
        print("="*50)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*50)
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()