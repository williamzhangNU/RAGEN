"""
The script defines the different evaluation metrics for the SpatialGym.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List
import re
import numpy as np

from ragen.env.spatial.Base.room import Room
from ragen.env.spatial.utils.parse_eval import dir_eval_fn, obj_seq_eval_fn, deg_seq_eval_fn, list_dir_eval_fn
from ragen.env.spatial.Base.constant import CANDIDATE_OBJECTS
from ragen.env.spatial.Base.graph import DirectionalGraph
from ragen.env.spatial.Base.relationship import DirPair, Dir, DirectionSystem

class BaseEvaluationTask(ABC):
    """Base class for all spatial evaluation tasks"""
    
    def __init__(self, np_random: np.random.Generator, config: Dict[str, Any] = None):
        """
        Initialize the evaluation task
        
        Args:
            config: Configuration for the evaluation task
        """
        self.config = config or {}
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
            Tuple of (reward, info)
        """
        pass

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
    
    def __init__(self, np_random: np.random.Generator, config: Dict[str, Any] = None):
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
            Tuple of (reward, info)
        """
        # Default response
        info = {
            "correct": False,
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
        info["correct"] = is_perfect
        info["score"] = score
        info["correct_count"] = correct_count
        info["total_count"] = total_answers

        return is_perfect, info

    

class DirEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for direction questions
    Always start from the original position and orientation of the agent
    1. Add a new object at random position, ask the direction between the new object and a random object
    TODO:
    2. Move object to a new position
    3. Move or rotate the agent to a new position
    """
    
    def __init__(self, np_random: np.random.Generator, config: Dict[str, Any] = None):
        super().__init__(np_random, config)

    def _generate_reasoning(self, env_state: Dict[str, Any]) -> str:
        """Generate reasoning for the evaluation task"""
        return "Testing spatial reasoning between a new object and a random object"

    def generate_question(self, room: Room) -> str:
        room = room.copy()
        # For inferable question asking

        # 1. Randomly generate a new object
        new_obj_name = room.objects[0].name
        while room.is_object_in_room(new_obj_name):
            new_obj_name = self.np_random.choice(CANDIDATE_OBJECTS)
        min_x_bound, max_x_bound, min_y_bound, max_y_bound = room.get_boundary()
        new_pos = np.array([self.np_random.uniform(min_x_bound, max_x_bound), self.np_random.uniform(min_y_bound, max_y_bound)])


        # 3. Randomly choose another object as anchor in the room
        anchor_obj_idx = self.np_random.integers(0, len(room.all_objects))
        anchor_obj = room.all_objects[anchor_obj_idx]


        # 4. Get the direction between the new object and the chosen object
        dir_pair = DirectionSystem.get_direction(new_pos, anchor_obj.pos, anchor_obj.ori)
        dir_pair_str = DirectionSystem.to_string(dir_pair, perspective='ego' if room.agent is not None else 'allo')

        # 5. Randomly choose another object in the room as query object
        query_obj_idx = self.np_random.integers(0, len(room.all_objects))
        while query_obj_idx == anchor_obj_idx:
            query_obj_idx = self.np_random.integers(0, len(room.all_objects))
        query_obj = room.all_objects[query_obj_idx]

        # 6. Add the new object to the graph to check the inferable direction (ground-truth, not oracle relationship, may be unknown)
        graph = DirectionalGraph(room.all_objects, is_explore=False)
        graph.is_explore = True
        graph.add_node(anchor_obj_idx, dir_pair)
        new_obj_idx = len(room.all_objects)

        # 7. Generate the QA
        self.question = f"A new object {new_obj_name} is {dir_pair_str} to {anchor_obj.name}. {new_obj_name} is what direction to {query_obj.name}?"
        dir_pair_query = graph.get_direction(new_obj_idx, query_obj_idx)
        self.answer = DirectionSystem.to_string(dir_pair_query, perspective='ego' if room.agent is not None else 'allo')
        return self.question
        
    def evaluate(self, answer: Any) -> Tuple[bool, Dict[str, Any]]:
        correct = dir_eval_fn(answer, self.answer)
        return correct, {}


class ReverseDirEvaluationTask(BaseEvaluationTask):
    """
    Evaluation task for reverse direction questions:
    1. Add a new object and guarentee inferable direction
        - Given new_obj --> obj1, ask **which object** is also new_obj --> obj2
        - Provide any one of correct answer is acceptable
    TODO
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
        print(f'[DEBUG] new_obj_name: {new_obj_name}')

        # 2. Randomly choose a pair of objects in the room
        obj1_idx = self.np_random.integers(0, len(room.all_objects))
        obj2_idx = self.np_random.integers(0, len(room.all_objects))
        while obj1_idx == obj2_idx:
            obj2_idx = self.np_random.integers(0, len(room.all_objects))
        obj1 = room.all_objects[obj1_idx]
        obj2 = room.all_objects[obj2_idx]

        # 3. Put the new object in the room
        graph = DirectionalGraph(room.all_objects, is_explore=False)
        min_x_bound, max_x_bound, min_y_bound, max_y_bound = room.get_boundary()

        result = DirPair(
            Dir.RIGHT if graph._h_matrix[obj1_idx, obj2_idx] > 0 else Dir.LEFT,
            Dir.FORWARD if graph._v_matrix[obj1_idx, obj2_idx] > 0 else Dir.BACKWARD
        )

        self.question = f"A new object {new_obj_name} is {DirectionSystem.to_string(result, 'ego')} to {obj1.name}."
        
        return question
    def evaluate(self, answer: Any) -> Tuple[bool, Dict[str, Any]]:
        correct = dir_eval_fn(answer, self.answer)
        return correct, {}



class RotEvaluationTask(BaseEvaluationTask):
    """Evaluation task for rotation questions"""
    
    def __init__(self, np_random: np.random.Generator, config: Dict[str, Any] = None):
        super().__init__(np_random, config)

    def _get_answer(self, room: Room) -> List[str]:
        """
        Get the object sequence when agent turns around at its original position
        """
        def _get_angle(pos: np.ndarray):
            # get angle from 0 (0, 1) to 90 (1, 0) to 180 (0, -1) to 270 (-1, 0)
            angle = np.arctan2(pos[0], pos[1])
            if angle < 0:
                angle += 2 * np.pi  
            return angle
        turn_direction = self.config.turn_direction
        objects = room.objects
        objects.sort(key=lambda x: _get_angle(x.pos), reverse=(turn_direction == 'counterclockwise'))
        return [obj.name for obj in objects]
    
    def _generate_reasoning(self, env_state: Dict[str, Any]) -> str:
        """Generate reasoning for the evaluation task"""
        return "Testing spatial reasoning between a new object and a random object"
    
    def _get_question(self, room: Room) -> str:
        """
        Get the question for the rotation evaluation task
        """
        object_names = [obj.name for obj in room.objects]
        return f"Objects in the room: {object_names}. What is the sequence of objects when agent turns around at its original position?"

    def generate_question(self, room: Room) -> str:
        room = room.copy()
        self.answer = self._get_answer(room)
        self.question = self._get_question(room)
        return self.question
    
    def evaluate(self, answer: Any) -> Tuple[bool, Dict[str, Any]]:
        correct = obj_seq_eval_fn(answer, self.answer)
        return correct, {}


class E2AEvaluationTask(BaseEvaluationTask):
    """Evaluation task for ego2allo questions"""
    
    def __init__(self, np_random: np.random.Generator, config: Dict[str, Any] = None):
        super().__init__(np_random, config)


class A2EEvaluationTask(BaseEvaluationTask):
    """Evaluation task for allo2ego questions"""
    
    def __init__(self, np_random: np.random.Generator, config: Dict[str, Any] = None):
        super().__init__(np_random, config)


class PovEvaluationTask(BaseEvaluationTask):
    """Evaluation task for perspective taking questions"""
    
    def __init__(self, np_random: np.random.Generator, config: Dict[str, Any] = None):
        super().__init__(np_random, config)




if __name__ == "__main__":
    from ragen.env.spatial.config import SpatialGymConfig
    from ragen.env.spatial.Base.room import generate_room
    from gymnasium.utils import seeding
    import numpy as np

    config = SpatialGymConfig()
    np_random = seeding.np_random(0)[0]
    room = generate_room(**config.get_room_config(), np_random=np_random)
    print(room)
    

    # Direction evaluation task
    task = DirEvaluationTask(np_random=np_random)
    question = task.generate_question(room)
    correct, info = task.evaluate("(unknown, back)")
    print(question)
    print(task.answer)
    print(correct)
    
    # # Rotation evaluation task
    # task = RotEvaluationTask()
    # question = task.generate_question(room)
    # print(question)
    # print(task.answer)
    # print(task.evaluate("scanner, vase, projector, television"))

    # # Ego2Allo evaluation task
    # task = E2AEvaluationTask()
    # question = task.generate_question(room)
    # print(question)
    # print(task.answer)

    # # Allo2Ego evaluation task
    # task = A2EEvaluationTask()
    # question = task.generate_question(room)
    # print(question)
    # print(task.answer)

    # # Perspective taking evaluation task
    # task = PovEvaluationTask()
    # question = task.generate_question(room)
    # print(question)
    # print(task.answer)

    
    