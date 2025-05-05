"""
The script defines the different evaluation metrics for the SpatialGym.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List
import re
import numpy as np
from typing_extensions import override
import copy

from ragen.env.spatial.Base.room import Room
from ragen.env.spatial.utils.parse_eval import dir_eval_fn, obj_seq_eval_fn, deg_seq_eval_fn, list_dir_eval_fn
from ragen.env.spatial.Base.constant import CANDIDATE_OBJECTS
from ragen.env.spatial.Base.graph import DirectionalGraph
from ragen.env.spatial.Base.object import Object, Agent
from ragen.env.spatial.Base.relationship import DirPair, Dir, DirectionSystem
from ragen.env.spatial.config import (BaseEvaluationConfig,
                                      AllPairsEvaluationConfig,
                                      DirEvaluationConfig,
                                      RotEvaluationConfig,
                                      PovEvaluationConfig)

class BaseEvaluationTask(ABC):
    """Base class for all spatial evaluation tasks"""
    
    def __init__(self, np_random: np.random.Generator, config: BaseEvaluationConfig = BaseEvaluationConfig()):
        """
        Initialize the evaluation task
        
        Args:
            config: Configuration for the evaluation task
        """
        self.config = config
        self.np_random = np_random
        self.question = None
        self.reasoning = None
        self.answer = None

    @abstractmethod
    def _generate_reasoning(self, env_state: Dict[str, Any]) -> str:
        """
        Generate reasoning for the evaluation task

        Args:
            env_state: Current state of the environment
            
        Returns:
            Reasoning string
        """
        pass
    
    @abstractmethod
    def generate_question(self, env_state: Dict[str, Any]) -> str:
        """
        Generate evaluation questions based on the environment state
        
        Args:
            env_state: Current state of the environment
            
        Returns:
            Question string
        """
        pass
    
    @abstractmethod
    def evaluate(self, answer: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate an answer to the given question
        
        Args:
            answer: Agent's answer
            
        Returns:
            Tuple of (correct, info)
        """
        pass

    def to_string(self) -> str:
        """
        Convert the evaluation task to a string
        Default implementation: return the class name, e.g. "BaseEvaluationTask()"
        """
        return f"{self.__class__.__name__}()"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEvaluationTask':
        """
        Initialize the evaluation task from a dictionary
        """
        return cls(
            question=data['question'],
            answer=data['answer'],
            reasoning=data['reasoning'],
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the evaluation task to a dictionary
        """
        return {
            'question': self.question,
            'answer': self.answer,
            'reasoning': self.reasoning,
        }

    @classmethod
    def create_task_from_dict(cls, data: Dict[str, Any]) -> 'BaseEvaluationTask':
        """
        Initialize a single evaluation task from a dictionary with type information
        """
        task_types = {
            'AllPairsEvaluationTask': AllPairsEvaluationTask,
            'DirEvaluationTask': DirEvaluationTask,
            'RotEvaluationTask': RotEvaluationTask,
            'PovEvaluationTask': PovEvaluationTask,
            'ReverseDirEvaluationTask': ReverseDirEvaluationTask,
            'E2AEvaluationTask': E2AEvaluationTask,
            'A2EEvaluationTask': A2EEvaluationTask,
        }
        
        task_type = data.get('type', cls.__name__)
        return task_types.get(task_type, cls).from_dict(data)
    

class AllPairsEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for checking all spatial relationships between object pairs
    Q: spatial relationship between all pairs of objects
    A: [<dir>, <dir>, ...]
    
    This task evaluates spatial relationships between all pairs of objects in the room.
    For N objects, there are N*(N-1)/2 distinct relationships.
    The question contains all distinct relationships with randomly shuffled object orders.
    The answer is the list of all distinct relationships.

    Question format:
    Consider the following spatial relationships:
    1. (<obj1>, <obj2>)
    2. (<obj1>, <obj3>)
    ...
    
    Response format:
    1. (<horiz>, <vert>)
    2. (<horiz>, <vert>)
    ...
    """
    
    def __init__(self, np_random: np.random.Generator, config: AllPairsEvaluationConfig = AllPairsEvaluationConfig()):
        super().__init__(np_random, config)
        self.object_pairs = []
        self.relationship_answers = []

    def _generate_reasoning(self, env_state: Dict[str, Any]) -> str:
        """Generate reasoning for the evaluation task"""
        return "Testing spatial reasoning between all object pairs"
    
    def generate_question(self, room: Room) -> str:
        """
        Generate a question that asks about all spatial relationships between pairs of objects
        
        Args:
            room: Room object containing objects
            
        Returns:
            Question string
        """
        room = room.copy()

        n = len(room.all_objects)
        pairs = [(i, j) if self.np_random.random() >= 0.5 else (j, i) 
                for i in range(n) for j in range(i+1, n)]
        self.np_random.shuffle(pairs)
        
        rel_questions = []
        for i, j in pairs:
            obj1 = room.all_objects[i]
            obj2 = room.all_objects[j]
            _, dir_pair_str = room.get_direction(obj1.name, obj2.name)
            self.answer.append(dir_pair_str)
            rel_questions.append(f"({obj1.name}, {obj2.name})")
        
        # Generate the question
        self.question = "Consider the following spatial relationships:\n"
        for i, question in enumerate(rel_questions, 1):
            self.question += f"{i}. {question}\n"
        
        self.question += "\nList all of these relationships in the format:\n1. (<horiz>, <vert>)\n2. (<horiz>, <vert>)\n..."
        return self.question
    
    def evaluate(self, answer: Any) -> Tuple[bool, Dict[str, Any]]:
        """
        Evaluate an answer to the given question
        
        Args:
            answer: Agent's answer formatted as a list of relationships
            
        Returns:
            Tuple of (correct, info)
        """
        # Default response
        info = {
            "score": 0.0,
            "correct_count": 0,
            "total_count": 0,
        }
        
        # Validate input
        if not isinstance(answer, str):
            info["message"] = "Answer must be a string"
            return False, info
        
        # Use list_dir_eval_fn to check all pairs
        correct_count = list_dir_eval_fn(answer, self.answer)
        
        # Calculate score as percentage of correct answers
        total_answers = len(self.answer)
        score = correct_count / total_answers if total_answers > 0 else 0.0
        
        # Set results
        is_perfect = correct_count == total_answers
        info["score"] = score
        info["correct_count"] = correct_count
        info["total_count"] = total_answers

        return is_perfect, info

    

class DirEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for direction questions
    Always start from the original position and orientation of the agent
    Q: Ask spatial relationship between two objects (a, b)
    A: <dir>

    1. Add a new object <dir> to anchor_obj (static)
        - a = new_obj, b = another obj
    2. Move target_obj <dir> to anchor_obj (object_move)
        - a = target_obj, b = another obj
    3. Move agent <dir> to anchor_obj (agent_move)
        - a = agent, b = another obj
    4. Rotate agent <degree> (agent_turn)
        - a = obj1, b = obj2

    TODO:
    2. Move object to a new position
    3. Move or rotate the agent to a new position

    TODO: deal with:
    1. Too many unknown relationships
    """
    
    def __init__(self, np_random: np.random.Generator, config: DirEvaluationConfig = DirEvaluationConfig()):
        super().__init__(np_random, config)
        
    def _generate_reasoning(self, env_state: Dict[str, Any]) -> str:
        """Generate reasoning for the evaluation task"""
        return "Testing spatial reasoning between a new object and a random object"

    def generate_question(self, room: Room) -> str:
        room = room.copy()
        if self.config.movement == 'agent_move':
            if room.agent is None:
                raise ValueError("Agent must be in the room for agent movement")
        elif self.config.movement == 'agent_turn':
            if room.agent is None:
                raise ValueError("Agent must be in the room for agent turn")

        graph = DirectionalGraph(room.all_objects, is_explore=False)
        graph.is_explore = True

        # 1. Get target object
        obs = ""
        if self.config.movement == 'static':
            target_name = room.objects[0].name
            while room.is_object_in_room(target_name):
                target_name = self.np_random.choice(CANDIDATE_OBJECTS)
            target_obj_idx = len(room.all_objects)
            obs = f"A new object {target_name} is placed in the room.\n"
        elif self.config.movement == 'object_move':
            # Choose a target object that is not the agent
            non_agent_objects_indices = [i for i, obj in enumerate(room.all_objects) if room.agent is not None and obj.name != room.agent.name]
            target_obj_idx = self.np_random.choice(non_agent_objects_indices)
            target_name = room.all_objects[target_obj_idx].name
            obs = f"{target_name} is moved to a new position.\n"
        elif self.config.movement == 'agent_move':
            target_name = room.agent.name
            target_obj_idx = next(i for i, obj in enumerate(room.all_objects) if obj == room.agent)
            obs = f"{target_name} moves to a new position.\n"
        elif self.config.movement == 'agent_turn':
            target_obj_idx = self.np_random.integers(0, len(room.all_objects))
            target_name = room.all_objects[target_obj_idx].name
        else:
            raise ValueError(f"Invalid movement type: {self.config.movement}")
        

        # 2. update the graph by moving or turning around
        if self.config.movement != 'agent_turn':
            # Randomly choose another object as anchor in the room
            anchor_obj_idx = self.np_random.integers(0, len(room.all_objects))
            anchor_obj = room.all_objects[anchor_obj_idx]
            anchor_name = anchor_obj.name
            # Get a new position
            min_x_bound, max_x_bound, min_y_bound, max_y_bound = room.get_boundary()
            new_pos = np.array([self.np_random.uniform(min_x_bound, max_x_bound), self.np_random.uniform(min_y_bound, max_y_bound)])
            # Get the direction between the new object and the chosen object
            dir_pair = DirectionSystem.get_direction(new_pos, anchor_obj.pos, anchor_obj.ori)
            dir_pair_str = DirectionSystem.to_string(dir_pair, perspective='ego' if room.agent is not None else 'allo')
            obs += f"{target_name} moves such that it is {dir_pair_str} of {anchor_name}."

            if self.config.movement == 'static':
                graph.add_node(anchor_obj_idx, dir_pair)
            else:
                graph.move_node(target_obj_idx, anchor_obj_idx, dir_pair)

        else:
            anchor_name = ""
            degree = self.np_random.choice([90, 180, 270])
            graph.rotate_axis(degree)
            obs += f"You turn {degree} degrees clockwise."

        # 3. Randomly choose another object in the room as query object
        query_obj_idx = self.np_random.integers(0, len(room.all_objects))
        while room.all_objects[query_obj_idx].name in [target_name, anchor_name]:
            query_obj_idx = self.np_random.integers(0, len(room.all_objects))
        query_obj = room.all_objects[query_obj_idx]
        

        # 4. Generate the QA
        self.question = f"{obs} {target_name} is what direction to {query_obj.name}?\n" \
                "Answer format: <answer>(X, Y)</answer>. X can be left, right, or same; Y can be front, back, or same. " \
                "If the relationship cannot be deduced, answer <answer>(unknown, unknown)</answer>."
        dir_pair_query = graph.get_direction(target_obj_idx, query_obj_idx)
        self.answer = DirectionSystem.to_string(dir_pair_query, perspective='ego' if room.agent is not None else 'allo')
        return self.question
        
    def evaluate(self, answer: Any) -> Tuple[bool, Dict[str, Any]]:
        print(f"[DEBUG] GT Answer: {self.answer}")
        print(f"[DEBUG] Model Answer: {answer}")
        return dir_eval_fn(answer, self.answer), {}


class ReverseDirEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for reverse direction questions:
    Q: which object is also <dir> to <new_obj>?
    A: <obj2>
    
    1. Add a new object and guarentee inferable direction
        - Given new_obj --> anchor_obj, ask **which (target) object** is also new_obj --> target_obj
        - Provide any one of correct answer is acceptable
    """
    
    def __init__(self, np_random: np.random.Generator, config: Dict[str, Any] = None):
        super().__init__(np_random, config)

    def _generate_reasoning(self, env_state: Dict[str, Any]) -> str:
        """Generate reasoning for the evaluation task"""
        return "Testing spatial reasoning between a new object and a random object"

    def generate_question(self, room: Room) -> str:
        room = room.copy()
        # For inferable question asking
        # 1. Randomly choose a new object name
        new_obj_name = room.objects[0].name
        while room.is_object_in_room(new_obj_name):
            new_obj_name = self.np_random.choice(CANDIDATE_OBJECTS)

        # 2. Randomly choose a pair of objects in the room
        anchor_obj_idx = self.np_random.integers(0, len(room.all_objects))
        target_obj_idx = self.np_random.integers(0, len(room.all_objects))
        while target_obj_idx == anchor_obj_idx:
            anchor_obj_idx = self.np_random.integers(0, len(room.all_objects))
        target_obj = room.all_objects[target_obj_idx]
        anchor_obj = room.all_objects[anchor_obj_idx]

        # 3. new_obj --> anchor_obj --> target_obj
        dir_pair, dir_pair_str = room.get_direction(anchor_obj.name, target_obj.name)

        # 4. find all target_objs that satisfies new_obj --> anchor_obj --> target_obj
        graph = DirectionalGraph(room.all_objects, is_explore=False)
        new_obj_idx = len(room.all_objects)
        graph.is_explore = True
        graph.add_node(anchor_obj_idx, dir_pair)
        inferable_pairs = graph.get_inferable_pairs()
        self.answer = [room.all_objects[j].name for i, j in inferable_pairs if i == new_obj_idx]
        self.answer.extend([room.all_objects[i].name for i, j in inferable_pairs if j == new_obj_idx])

        self.question = f"A new object {new_obj_name} is {dir_pair_str} to {anchor_obj.name}. {new_obj_name} is also {dir_pair_str} to which object, give ONLY one of the possible answers."
        
        return self.question
    
    def evaluate(self, answer: Any) -> Tuple[bool, Dict[str, Any]]:
        return answer.strip().lower() in [ans.strip().lower() for ans in self.answer], {}



class RotEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for rotation questions
    Q: What is the sequence of objects when agent turns around at its original position?
    A: [<obj1>, <obj2>, ...]
    """
    
    def __init__(self, np_random: np.random.Generator, config: Dict[str, Any] = None):
        super().__init__(np_random, config)

    def _generate_reasoning(self, env_state: Dict[str, Any]) -> str:
        """Generate reasoning for the evaluation task"""
        return "Testing spatial reasoning between a new object and a random object"

    def generate_question(self, room: Room) -> str:
        room = room.copy()
        
        # Get the answer (object sequence when agent turns around)
        def _get_angle(pos: np.ndarray):
            # get angle from 0 (0, 1) to 90 (1, 0) to 180 (0, -1) to 270 (-1, 0)
            angle = np.arctan2(pos[0], pos[1])
            if angle < 0:
                angle += 2 * np.pi  
            return angle
        
        turn_direction = self.config.turn_direction
        objects = room.objects
        objects.sort(key=lambda x: _get_angle(x.pos), reverse=(turn_direction == 'counterclockwise'))
        self.answer = [obj.name for obj in objects]
        self.question = f"What is the sequence of objects when agent turns around {turn_direction} at its original position?"
        
        return self.question
    
    def evaluate(self, answer: Any) -> Tuple[bool, Dict[str, Any]]:
        correct = obj_seq_eval_fn(answer, self.answer)
        return correct, {}
    
    @override
    def to_string(self) -> str:
        return f"{self.__class__.__name__}({self.config.turn_direction})"


class PovEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for perspective taking questions
    Q: Ask spatial relationship between two objects (a, b) from the perspective of c
    A: <dir>

    """
    
    def __init__(self, np_random: np.random.Generator, config: Dict[str, Any] = None):
        super().__init__(np_random, config)

    def _generate_reasoning(self, env_state: Dict[str, Any]) -> str:
        """Generate reasoning for the evaluation task"""
        return "Testing spatial reasoning between a new object and a random object"
    
    def generate_question(self, room: Room) -> str:
        room = room.copy()
        obj1_idx = self.np_random.integers(0, len(room.all_objects))
        obj2_idx = self.np_random.integers(0, len(room.all_objects))
        anchor_obj_idx = self.np_random.integers(0, len(room.all_objects))
        while obj2_idx == obj1_idx:
            obj2_idx = self.np_random.integers(0, len(room.all_objects))

        _, dir_pair_str = room.get_direction(obj1_idx, obj2_idx, anchor_obj_idx)
        self.question = room.get_objects_orientation()    
        self.question += f"From {room.all_objects[anchor_obj_idx].name}'s perspective, {room.all_objects[obj1_idx].name} is what direction to {room.all_objects[obj2_idx].name}?"
        self.answer = dir_pair_str
        return self.question
    
    def evaluate(self, answer: Any) -> Tuple[bool, Dict[str, Any]]:
        print(f"[DEBUG] GT Answer: {self.answer}")
        print(f"[DEBUG] Model Answer: {answer}")
        return dir_eval_fn(answer, self.answer), {}
        
        

class E2AEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for ego2allo questions
    Exploration from ego-centric view, evaluate how it performs in allocentric view
    Q: Given a list of coordinates, what is the sequence of objects that corresponds to each coordinate?
    A: [<obj1>, <obj2>, ...]

    TODO check uniqueness of the answer
    """
    
    def __init__(self, np_random: np.random.Generator, config: Dict[str, Any] = None):
        super().__init__(np_random, config)

    def _generate_reasoning(self, env_state: Dict[str, Any]) -> str:
        """Generate reasoning for the evaluation task"""
        return "Testing spatial reasoning between a new object and a random object"
    
    def generate_question(self, room: Room) -> str:
        """
        Get the question for the ego2allo evaluation task
        """
        # Shuffle objects and get their positions
        objects = copy.deepcopy(room.all_objects)
        self.np_random.shuffle(objects)
        coordinates = np.array([obj.pos for obj in objects])
        
        min_x, max_x, min_y, max_y = room.get_boundary()
        random_offset = np.column_stack([
            self.np_random.integers(min_x, max_x, len(objects)),
            self.np_random.integers(min_y, max_y, len(objects))
        ])
        coordinates = coordinates + random_offset

        coordinates_str = ", ".join([f"({coord[0]},{coord[1]})" for coord in coordinates])
        instruction_str = (
            "Map objects to coordinates in the given order.\n"
            "Example:\n"
            "- Ground truth: A at (0, 0), B at (1, 1), C at (2, 2)"
            "- Given coordinates: (0, 0), (1, 1), (2, 2)\n"
            "- answer: ['A', 'B', 'C']\n"
        )
        format_str = "Answer format: `[<object1>, <object2>, ...]`"
        question = f"Given a list of coordinates: {coordinates_str}, what is the sequence of objects that corresponds to each coordinate?"
        self.question = instruction_str + "\n" + question + "\n" + format_str
        self.answer = [obj.name for obj in objects]
        return self.question
    
    def evaluate(self, answer: Any) -> Tuple[bool, Dict[str, Any]]:
        print(f"[DEBUG] GT Answer: {self.answer}")
        print(f"[DEBUG] Model Answer: {answer}")
        return obj_seq_eval_fn(answer, self.answer), {}
        
        



class A2EEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for allo2ego questions
    Exploration from allocentric view, evaluate how it performs in ego-centric view
    Q: What is the sequence of degrees you need to turn to traverse all objects in order?
    A: [<degree1>, <degree2>, ...]
    """
    
    def __init__(self, np_random: np.random.Generator, config: Dict[str, Any] = None):
        super().__init__(np_random, config)

    def _generate_reasoning(self, env_state: Dict[str, Any]) -> str:
        """Generate reasoning for the evaluation task"""
        return "Testing spatial reasoning between a new object and a random object"
    
    def generate_question(self, room: Room) -> str:
        """
        Get the object traverse sequence and gt action sequence
        1. Traverse all objects in the room follow the order they in objects list
        NOTE:
            - Agent starts from the first object and facing north
            - When moving to the next object, agent should identify:
                - degree: Agent should rotate to keep the next object at its front (0, 90, 180, 270)
            - Result should be a list of degrees
        """

        room = room.copy()

        def _get_angle(vec: np.ndarray, ref_vec: np.ndarray = np.array([0, 1])):
            # get angle between vec and ref_vec
            # angle is clockwise from ref_vec to vec
            angle = np.arctan2(vec[0], vec[1]) - np.arctan2(ref_vec[0], ref_vec[1]) # radian
            angle = angle * 180 / np.pi # degree
            if angle < 0:
                angle += 360
            return angle

        def _get_orientation(ori: np.ndarray, angle: float):
            # what is orientation after rotating clockwise by angle degree
            angle = angle * np.pi / 180
            rotation_matrix = np.array([
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ])
            new_ori = ori @ rotation_matrix
            new_ori = np.round(new_ori, 0).astype(int)
            return new_ori            

        gt_turning_degrees = []
        traverse_agent = Object(name="agent", pos=room.all_objects[0].pos, ori=(0, 1))
        for next_obj in (room.all_objects[1:]):
            angle = _get_angle(next_obj.pos - traverse_agent.pos, traverse_agent.ori)
            # Round to nearest 0, 90, 180, or 270 degrees
            rotation_degree = int(round(angle / 90) * 90) % 360
            gt_turning_degrees.append(rotation_degree)
            traverse_agent.ori = _get_orientation(traverse_agent.ori, rotation_degree)
            traverse_agent.pos = next_obj.pos

        object_sequence_str = ", ".join([obj.name for obj in room.all_objects])
        format_str = "Format your answer as a comma-separated list of degrees, e.g., '[0, 90, 180, 270]'."
        instruction_str = (
            "After exploring the room from bird-eye view, you'll now traverse it from an egocentric view. \n"
            "Starting at the first object facing north, report the clockwise turning degrees (0, 90, 180, or 270) "
            "needed to face each subsequent object before moving to it. Your orientation remains fixed while moving."
        )
        question = f"Given a sequence of objects:\n{object_sequence_str}\nWhat is the sequence of degrees you need to turn to traverse all objects in order?"
        self.question = instruction_str + "\n" + question + "\n" + format_str
        self.answer = gt_turning_degrees
        return self.question
    
    def evaluate(self, answer: Any) -> Tuple[bool, Dict[str, Any]]:
        print(f"[DEBUG] GT Answer: {self.answer}")
        print(f"[DEBUG] Model Answer: {answer}")
        return deg_seq_eval_fn(answer, self.answer), {}





if __name__ == "__main__":
    from ragen.env.spatial.config import SpatialGymConfig
    from ragen.env.spatial.Base.utils.room_utils import generate_room
    from gymnasium.utils import seeding
    import numpy as np

    config = SpatialGymConfig(n_objects=3, generation_type='rot', perspective='ego')
    np_random = seeding.np_random(10)[0]
    room = generate_room(**config.get_room_config(), np_random=np_random)
    print(room)
    

    # # Direction evaluation task
    # task = DirEvaluationTask(np_random=np_random, config=DirEvaluationConfig(movement='agent_turn'))
    # question = task.generate_question(room)
    # print(question)
    # print(task.answer)
    # correct, info = task.evaluate("(unknown, back)")
    # print(correct)

    # # Reverse direction evaluation task
    # task = ReverseDirEvaluationTask(np_random=np_random)
    # question = task.generate_question(room)
    # print(question)
    # print(task.answer)
    # correct, info = task.evaluate("folder")
    # print(correct)

    # # Pov evaluation task
    # task = PovEvaluationTask(np_random=np_random)
    # question = task.generate_question(room)
    # print(question)
    # print(task.answer)
    # correct, info = task.evaluate("(right, back)")
    # print(correct)
    
    # # A2E evaluation task
    # task = A2EEvaluationTask(np_random=np_random)
    # question = task.generate_question(room)
    # print(question)
    # print(task.answer)
    # correct, info = task.evaluate("[0, 90, 180, 270]")
    # print(correct)

    # # E2A evaluation task
    # task = E2AEvaluationTask(np_random=np_random)
    # question = task.generate_question(room)
    # print(question)
    # print(task.answer)
    # correct, info = task.evaluate("['television', 'microphone', 'agent', 'eraser']")
    # print(correct)

    # Rotation evaluation task
    task = RotEvaluationTask(np_random=np_random, config=RotEvaluationConfig(turn_direction='counterclockwise'))
    question = task.generate_question(room)
    print(question)
    print(task.answer)
    correct, info = task.evaluate("['television', 'microphone', 'agent', 'eraser']")
    print(correct)

    
    