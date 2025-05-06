import gymnasium as gym
import re
from typing import Optional

from ragen.env.spatial.config import (
    SpatialGymConfig,
    AllPairsEvaluationConfig,
    DirEvaluationConfig,
    RotEvaluationConfig,
    PovEvaluationConfig
)
from ragen.env.spatial.Base.room import Room, Action, ActionType
from ragen.env.spatial.Base.utils.room_utils import generate_room
from ragen.env.spatial.utils.generate_history import AutoExplore
from ragen.env.spatial.utils.parse_exp_input import parse_action
from ragen.env.spatial.utils.get_eval_task import get_eval_task


instruction = """\
# Spatial Mapping Task
You are exploring the room with several objects. 
Your goal is to uncover spatial relationships between object pairs to build a complete mental map.
You should terminate your exploration when you have explored the room and found all the spatial relationships.

## Spatial Relationships
(A, B): (<Horizontal>, <Vertical>) means A is to the <Horizontal> and <Vertical> of B, where:
- Horizontal: left, right, same
- Vertical: front, back, same
- "same" means objects are aligned on that axis, (e.g., (same, front) means directly front, not leaning left or right)
- Relationships are relative (if A is left of B, then B is right of A)
- Relationships can be transitive (if A is left of B and B is left of C, then A is left of C)
- No distance information is included

## Room Description
{room_info}

{exp_history}

{exp_answer_format}
"""



class SpatialGym(gym.Env):
    """
    Spatial Gym Environment, explore then evaluate
    NOTE for evaluation, always use s_0 room to generate question
    TODO working on eval_performance, check
    TODO finish obs, reward, done, info for step
    TODO check room's all_objects_exp, all_objects
    """
    def __init__(self, config: SpatialGymConfig):
        super().__init__()
        self.config = config
        self.is_exp_stage = None  # indicates exploration or evaluation stage
        self.max_exp_steps = None
        self.render_cache = None

        # for state
        self.room_s_t = None  # latest/current state of the room
        self.room_s_0 = None  # initial state of the room
        self.room_s_end = None  # final state of the room, agent may return to its original state

        # for action space
        # Set available actions based on exploration type
        self.move_action = [ActionType.MOVE, ActionType.ROTATE, ActionType.RETURN] if self.config.exp_type == 'active' else []
        self.query_action = [ActionType.QUERY, ActionType.TERM] if self.config.exp_type != 'passive' else []
        self.action_space = self.move_action + self.query_action

        # for analysis
        self.n_novel_queries = None
        self.n_valid_queries = None
        self.eval_results = None

    def _gen_initial_obs(self):
        """
        Generate initial observation as a user message (instruction)
        """
        exp_history = ""
        room_desc = self.room_s_0.get_room_desc()
        exp_answer_format = ""
        if self.config.exp_type == 'passive':
            # TODO for passive, generate exploration history using DFS here
            auto_explore = AutoExplore(self.room_s_0, self.np_random)
            exp_history = auto_explore.generate_history()
            exp_history = f"## Exploration History\n{exp_history}"

        else:
            action_formats = []
            for action in self.action_space:
                is_active = self.config.exp_type == 'active'
                action_formats.append(f"- {ActionType.get_format(action, is_active=is_active)}")
            exp_answer_format = "## Response format\nAvailable actions:\n" + "\n".join(action_formats)
            if self.config.exp_type == 'active':
                exp_answer_format += (
                    "\n\nYou can perform multiple movement actions (" + 
                    ", ".join([action.value for action in self.move_action]) + 
                    ") separated by commas, and use semicolons to separate groups of movement actions from the final query action (" + 
                    ", ".join([action.value for action in self.query_action]) + 
                    ").\n\nExample: Move(chair), Rotate(90); Query(table)" +
                    "\n\nThe last action must always be a query action and you can only perform one query action at a time."
                    "\n\nIf you choose to terminate with Term(), it should be the only action without any movement actions before it. Example: Term()"
                )
            else:
                exp_answer_format += "\n\nYou can only perform one query action at a time. Example: Query(table, plant) or Term()"

        obs = instruction.format(
            room_info=room_desc,
            exp_history=exp_history,
            exp_answer_format=exp_answer_format
        )

        if self.config.exp_type == 'passive':
            obs = obs + "\n\n" + self.eval_tasks[0].generate_question(self.room_s_0.copy())

        return obs

        


    
    
    def reset(self, seed: int = None):
        """
        1. Generate initial room
        2. Initialize evaluation question(s)
        3. For eval only (passive): generate exp history

        For exp + eval:
            - obs: exploration instruction
        For eval only:
            - obs: exp instruction + exp history + eval question

        Returns:
            - obs (str): user message (instruction)
            - info (dict)
        """
        super().reset(seed=seed)
        self.is_exp_stage = True if self.config.exp_type != 'passive' else False  # starts with exploration stage, 
        self.max_exp_steps = self.config.max_exp_steps
        self.n_valid_queries = 0
        self.n_novel_queries = 0

        self.room_s_0: Room = generate_room(
            **self.config.get_room_config(),
            np_random=self.np_random,
        )
        self.room_s_t = self.room_s_0.copy()
        self.eval_tasks = [get_eval_task(task['task_type'], self.np_random, task['task_kwargs']) for i, task in enumerate(self.config.eval_tasks)]
        if self.config.exp_type == 'passive':
            for task in self.eval_tasks:
                task.generate_question(self.room_s_0.copy())
        self.eval_results = []
            

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
        # Check if transitioning from exploration to evaluation
        if self.is_exp_stage:
            self.max_exp_steps -= 1
            motion_list, query_action = parse_action(action, self.room_s_t, 
                                                is_active=self.config.exp_type == 'active')
            
            # Handle invalid actions
            if query_action is None:
                self.render_cache = "Invalid action"
                return "Invalid action", -0.1, False, {}
            
            # Check if exploration phase should end
            if query_action.action_type == ActionType.TERM or self.max_exp_steps < 0:
                self.is_exp_stage = False
                self.room_s_end = self.room_s_t.copy()
                self.room_s_end.finish_exploration()
                
                # Transition to first evaluation task
                question = self.eval_tasks[0].generate_question(self.room_s_0.copy())
                self.render_cache = question
                return question, 0, False, {}
            else:
                # Continue exploration
                self.n_valid_queries += 1
                dir_pair_str, exp_info = self.room_s_t.explore(motion_list, query_action, 
                                                is_active=self.config.exp_type == 'active')
                if exp_info['novel_query']:
                    self.n_novel_queries += 1
            
                self.render_cache = dir_pair_str
                return dir_pair_str, 0, False, {}
        
        # Evaluation stage
        else:
            # Evaluate current task answer
            correct, _info = self.eval_tasks[0].evaluate(action)
            self.eval_results.append({
                "task_type": self.eval_tasks[0].to_string(),
                "correct": correct,
                "info": _info,
            })
            reward = 1 if correct else 0
            self.eval_tasks.pop(0)
            
            # Check if all tasks are completed
            if len(self.eval_tasks) == 0:
                self.render_cache = "Task finished"
                return "Task finished", reward, True, {}
            else:
                question = self.eval_tasks[0].generate_question(self.room_s_0.copy())
                self.render_cache = question
                return question, reward, False, {}
        

    def render(self):
        return self.render_cache




    #=============== for analysis ===============
    def get_env_info(self):
        print(self.config.to_dict())
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
        """
        assert self.config.exp_type in ["active", "semi", "passive"]
        if self.config.exp_type == 'passive':
            return {
                "coverage": 0,
                "novelty": 0,
                "n_valid_queries": 0,
                "n_novel_queries": 0,
            }
        self.room_s_t.finish_exploration()
        unknown_pairs = self.room_s_t.get_unknown_pairs()

        n_object = len(self.room_s_t.all_objects)
        max_rels = int(n_object * (n_object - 1) / 2)
        return {
            "coverage": (max_rels - len(unknown_pairs)) / max_rels,
            "novelty": self.n_novel_queries / self.n_valid_queries if self.n_valid_queries > 0 else 0,
            "n_valid_queries": self.n_valid_queries,
            "n_novel_queries": self.n_novel_queries,
        }
    
    def get_eval_performance(self):
        """
        Get the evaluation performance
        - Accuracy: average accuracy across all evaluation tasks
        - Task-specific results: detailed performance for each task type
        """
        # Include unanswered evaluation tasks in the results
        all_results = self.eval_results.copy()
        
        # Add unanswered tasks as incorrect answers
        for task in self.eval_tasks[len(self.eval_results):]:
            all_results.append({
                "task_type": task.to_string(),
                "correct": False,
                "info": {}
            })
        
        return {
            "accuracy": sum(result["correct"] for result in all_results) / len(all_results) if len(all_results) > 0 else 0,
            'accuracy_completed': sum(result["correct"] for result in self.eval_results) / len(self.eval_results) if len(self.eval_results) > 0 else 0,
            "task_results": all_results,
            "completed_tasks": len(self.eval_results),
            "unanswered_tasks": len(self.eval_tasks)
        }


    



if __name__ == "__main__":
    # config = SpatialGymConfig(eval_tasks=["rot"])
    # env = SpatialGym(config)
    # env.reset(seed=42)
    # print(env.room_s_0)

    # # result = env.step("Move(monitor), Rotate(90); Query(monitor)")
    # result = env.step("Rotate(90); Query(plant)")
    # print(result)
    # result = env.step("Term()")
    # print(result)
    # result = env.step("[plant, sofa, headphones, monitor]")
    # print(result)


    config = SpatialGymConfig(eval_tasks=[{"task_type": "dir", "task_kwargs": {}}], exp_type="passive")
    env = SpatialGym(config)
    obs, _ = env.reset(seed=25)
    print(env.room_s_0)
    print(obs)


    config = SpatialGymConfig(eval_tasks=[{"task_type": "dir", "task_kwargs": {}}], exp_type="semi")
    env = SpatialGym(config)
    obs, _ = env.reset(seed=25)
    print(env.room_s_0)
    print(obs)

    result = env.step("Query(agent, flower)")
    print(result)
    print(env.room_s_t.exp_graph._v_matrix)
    print(env.room_s_t.exp_graph._h_matrix)
    # result = env.step("Move(table), Rotate(90); Query(flower)")
    # print(result)
    # print(env.room_s_t.exp_graph._v_matrix)
    # print(env.room_s_t.exp_graph._h_matrix)
    result = env.step("Term()")
    print(result)
    result = env.step("(right, unknown)")
    print(result)
    print(env.get_exp_efficiency())
    print(env.get_eval_performance())