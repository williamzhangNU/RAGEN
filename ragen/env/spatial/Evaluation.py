"""
The script defines the different evaluation metrics for the SpatialGym.
"""

from abc import ABC, abstractmethod
from typing import Dict, Tuple, Any, List
from textwrap import indent, dedent
from collections import defaultdict
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
    def _generate_reasoning(self, room) -> str:
        """
        Generate reasoning for the evaluation task

        Args:
            room: Current state of the environment
            
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
    def _wrap(self, reason: str, ans) -> str:
        """Wrap reasoning + answer in the required XML‑like tags."""
        return f"<think>\n{indent(reason.strip(), '    ')}\n</think> <answer> {ans} </answer>"
    def _pair_to_xy(self, pair: DirPair) -> Tuple[int, int]:
        """Convert DirPair → (dx, dy) in {-1,0,1}×{-1,0,1}."""
        h = {Dir.LEFT: -1, Dir.SAME: 0, Dir.RIGHT: 1}.get(pair.horiz, 0)
        v = {Dir.BACKWARD: -1, Dir.SAME: 0, Dir.FORWARD: 1}.get(pair.vert, 0)
        return h, v

    def _xy_to_pair(self, dx: int, dy: int) -> DirPair:
        """Inverse of _pair_to_xy(…) with saturation to ±1."""
        h_dir = Dir.LEFT if dx < 0 else Dir.RIGHT if dx > 0 else Dir.SAME
        v_dir = Dir.BACKWARD if dy < 0 else Dir.FORWARD if dy > 0 else Dir.SAME
        return DirPair(h_dir, v_dir)

    def _compose(self, p1: DirPair, p2: DirPair) -> DirPair:
        """Return p1 ⊕ p2 (vector addition in {-1,0,1} with clipping)."""
        dx1, dy1 = self._pair_to_xy(p1)
        dx2, dy2 = self._pair_to_xy(p2)
        dx = max(-1, min(1, dx1 + dx2))
        dy = max(-1, min(1, dy1 + dy2))
        return self._xy_to_pair(dx, dy)


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
    def _parse_dir_pair(self, s: str) -> Tuple[Dir, Dir]:
        m = re.match(r"\(\s*([^,]+?)\s*,\s*([^)]+?)\s*\)", s)
        to_dir = lambda w: (
            Dir.LEFT if w.lower()=='left' else
            Dir.RIGHT if w.lower()=='right' else
            Dir.FORWARD if w.lower() in ('front','north') else
            Dir.BACKWARD if w.lower() in ('back','south') else
            Dir.SAME
        )
        return to_dir(m.group(1)), to_dir(m.group(2)) if m else (Dir.UNKNOWN, Dir.UNKNOWN)

    def _generate_reasoning(self, room) -> str:
        """Generate reasoning for the evaluation task"""
        # 1) Grab the full movement block (may be one or two sentences),
        #    the target name, and the query name.
        #
        #    e.g. movement_desc = 
        #      "A new object pen is placed.\npen moves (left,front) to desk."
        #    or "You turn 90 degrees."
        pattern = re.compile(
            r'^(?P<movement>[\s\S]+?)\s*'      # up to just before "<tgt>"
            r'(?P<tgt>\S+)\s+is what direction to\s+'  
            r'(?P<qry>\S+)\?',  
            flags=re.DOTALL
        )
        m = pattern.match(self.question)
        if not m:
            raise ValueError(f"Couldn’t parse question: {self.question!r}")

        movement_desc = m.group("movement").strip()   # full movement text
        tgt_name      = m.group("tgt")               # e.g. "cabinet"
        qry_name      = m.group("qry")               # e.g. "agent"
        post_str      = self.answer                  # e.g. "(right, front)"

        # 2) Compute the "before" relationship on a fresh, allocentric graph
        g0 = DirectionalGraph(room.all_objects, is_explore=False)
        idx = {obj.name: i for i, obj in enumerate(room.all_objects)}
        pre_pair = g0.get_direction(idx[tgt_name], idx[qry_name])
        persp = "ego" if room.agent is not None else "allo"
        pre_str = DirectionSystem.to_string(pre_pair, perspective=persp)

        # 3) Branch on movement type
        mv = self.config.movement
        if mv in ("static", "object_move", "agent_move"):
            # extract the (h,v) tuple and the anchor from movement_desc
            # works for all three since they end with "... moves (h,v) to X."
            m2 = re.search(rf"{re.escape(tgt_name)} moves \(([^)]+)\) to (\S+)\.", movement_desc)
            if not m2:
                raise ValueError(f"Couldn’t parse move in: {movement_desc!r}")
            dir_str, anchor = m2.group(1), m2.group(2)

            reasoning = dedent(f"""\
                Because {tgt_name} was {pre_str} of {qry_name} before movement,
                and we moved {tgt_name} {dir_str} relative to {anchor},
                it is now {post_str}.
            """).strip()

        else:  # agent_turn
            # 1) extract the angle
            deg_match = re.search(r"(\d+)", movement_desc)
            deg = int(deg_match.group(1)) if deg_match else None

            # 2) parse the original pair into “h” and “v”
            orig_h, orig_v = pre_str.strip("()").split(", ")
            orig_h, orig_v = orig_h.strip(), orig_v.strip()

            # 3) helper to flip a component
            def opposite(comp: str) -> str:
                return {
                    "left":    "right",
                    "right":   "left",
                    "front":   "back",
                    "back":    "front",
                    "same":    "same"
                }[comp]
            def rotate_90(h: str) -> str:
                mapping = {
                    "right": "front",
                    "left":  "back",
                    "back":  "right",
                    "front": "left",
                    "same":  "same"
                }
                return mapping[h]

            # 4) compute new h/v under each rotation
            if deg == 90:
                new_h = rotate_90(orig_v)
                new_v = rotate_90(orig_h)
                rule = (
                    f"horizontally, what was {orig_v} is now {new_h}; "
                    f"vertically, what was {orig_h} is now {new_v}"
                )
            elif deg == 180:
                new_h = opposite(orig_h)
                new_v = opposite(orig_v)
                rule = (
                    f"horizontally, left/right are swapped {orig_h} → {new_h}; "
                    f"vertically, front/back are swapped {orig_v} → {new_v}"
                )
            elif deg == 270:
                new_h = opposite(rotate_90(orig_v))
                new_v = opposite(rotate_90(orig_h))
                rule = (
                    f"horizontally, what was {orig_v} is now {new_h}; "
                    f"vertically, what was {orig_h} is now {new_v}"
                )

            # 5) format the rule description
            reasoning = dedent(f"""\
                {tgt_name} was {pre_str} of {qry_name} before turning {deg}°:
                 and {rule},
                it is now {post_str}.
            """).strip()

        # 4) Wrap in <think> and return
        return f"<think>\n{reasoning}\n</think> <answer> {post_str} </answer>"

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
            obs += f"{target_name} moves {dir_pair_str} to {anchor_name}."

            if self.config.movement == 'static':
                graph.add_node(anchor_obj_idx, dir_pair)
            else:
                graph.move_node(target_obj_idx, anchor_obj_idx, dir_pair)

        else:
            anchor_name = ""
            degree = self.np_random.choice([90, 180, 270])
            graph.rotate_axis(degree)
            obs += f"You turn {degree} degrees."

        # 3. Randomly choose another object in the room as query object
        query_obj_idx = self.np_random.integers(0, len(room.all_objects))
        while room.all_objects[query_obj_idx].name in [target_name, anchor_name]:
            query_obj_idx = self.np_random.integers(0, len(room.all_objects))
        query_obj = room.all_objects[query_obj_idx]
        

        # 4. Generate the QA
        self.question = f"{obs} {target_name} is what direction to {query_obj.name}?"
        dir_pair_query = graph.get_direction(target_obj_idx, query_obj_idx)
        self.answer = DirectionSystem.to_string(dir_pair_query, perspective='ego' if room.agent is not None else 'allo')
        return self.question
        
    def evaluate(self, answer: Any) -> Tuple[bool, Dict[str, Any]]:
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

    def _generate_reasoning(self, room) -> str:
        """Generate reasoning for the evaluation task"""
        q = self.question

        # 1) pull out the new object, the direction string, and the anchor
        m = re.match(r"A new object (\S+) is (.+?) to (\S+)\.", q)
        if not m:
            raise ValueError(f"Couldn't parse question for reasoning: {q!r}")
        new_obj, dir_str, anchor = m.group(1), m.group(2), m.group(3)

        # 2) pick one of the acceptable answers
        #    (self.answer is a list of all objects that satisfy the relation)
        if isinstance(self.answer, (list, tuple)):
            ans_obj = self.answer[0]
        else:
            ans_obj = self.answer

        # 3) compute the original anchor→answer relation
        _, pre_str = room.get_direction(anchor, ans_obj)

        # 4) build a simple two‐step reasoning
        lines = [
            f"1. Before adding anything, {anchor} is {pre_str} of {ans_obj}.",
            f"2. Then we place {new_obj} so that it is {dir_str} of {anchor}.",
            f"3. Horizontally, what is {pre_str[1:pre_str.index('t')+1]} of {pre_str[1:pre_str.index('t')+1]} is also {pre_str[1:pre_str.index('t')+1]};",
            f" Vertically, what is {pre_str[pre_str.index('t')+3:len(pre_str)-1]} of {pre_str[pre_str.index('t')+3:len(pre_str)-1]} is also {pre_str[pre_str.index('t')+3:len(pre_str)-1]}. Therefore {new_obj} is also {pre_str} of {ans_obj}."]
        think = "\n".join(lines)

        return f"<think>\n{think}\n</think> <answer> {ans_obj} </answer>"

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

    def _generate_reasoning(self, room) -> str:
        turn_dir = self.config.turn_direction  # 'clockwise' or 'counterclockwise'
        agent = room.agent.name

        # 1) group into quadrants
        def quadrant(obj):
            _, rel = room.get_direction(obj.name, agent)
            h, v = (x.strip().lower() for x in rel.strip("()").split(","))
            if v == "front":
                return "front-right" if h == "right" else "front-left"
            if v == "back":
                return "back-right"  if h == "right" else "back-left"
            if h == "right":
                return "front-right" if v == "same"  else "back-right"
            return "front-left"      if v == "same"  else "back-left"

        quads = defaultdict(list)
        for obj in room.objects:
            quads[quadrant(obj)].append(obj)

        detailed = []
        ordered_quads = {}

        # 2) sort within each quadrant
        for qname, objs in quads.items():
            if len(objs) < 2:
                ordered_quads[qname] = [o.name for o in objs]
                continue

            is_up    = qname.startswith("front")
            is_right = qname.endswith("right")

            # horizontal policy
            if turn_dir == "clockwise":
                left_first = is_up
            else:  # counterclockwise
                left_first = not is_up

            ordered = [objs[0]]
            for nxt in objs[1:]:
                placed = False
                for i, cur in enumerate(ordered):
                    _, rel = room.get_direction(nxt.name, cur.name)
                    h, v   = (x.strip().lower() for x in rel.strip("()").split(","))

                    if h != "same":
                        # horizontal ordering
                        take = (h == "left"  and left_first) or \
                               (h == "right" and not left_first)
                        policy = "horizontal"
                    else:
                        # vertical tie-break
                        if turn_dir == "counterclockwise":
                            take = (v == "back") if is_right else (v == "front")
                        else:  # clockwise
                            take = (v == "front") if is_right else (v == "back")
                        policy = "vertical"

                    if take:
                        ordered.insert(i, nxt)
                        if policy == "horizontal":
                            detailed.append(
                                f"  • In {qname}: comparing {nxt.name} vs {cur.name}, "
                                f"{nxt.name} is {rel} of {cur.name}; "
                                f"{turn_dir} sweep → {nxt.name} comes before {cur.name}."
                            )
                        else:
                            # describe tie-break policy
                            tie_policy = (
                                "back-first" if (turn_dir == "counterclockwise" and is_right)
                                           or (turn_dir == "clockwise"      and not is_right)
                                else "front-first"
                            )
                            detailed.append(
                                f"  • In {qname}: {nxt.name} vs {cur.name} share the same horizontal; "
                                f"{nxt.name} is {rel} of {cur.name}; "
                                f"{turn_dir} sweep → "
                                f"{nxt.name} comes before {cur.name}."
                            )
                        placed = True
                        break

                if not placed:
                    ordered.append(nxt)
                    if h != "same":
                        detailed.append(
                            f"  • In {qname}: {nxt.name} is {rel} of {cur.name}; "
                            f"{turn_dir} sweep → placed after."
                        )
                    else:
                        tie_policy = (
                            "back-first" if (turn_dir == "counterclockwise" and is_right)
                                       or (turn_dir == "clockwise"      and not is_right)
                            else "front-first"
                        )
                        detailed.append(
                            f"  • In {qname}: {nxt.name} is {rel} of {cur.name}; horizontal tie; {turn_dir} sweep → placed after."
                        )

                ordered_quads[qname] = [o.name for o in ordered]

        # 3) sweep quadrants
        quad_ring = (
            ["front-right","back-right","back-left","front-left"]
            if turn_dir == "clockwise"
            else ["front-left","back-left","back-right","front-right"]
        )

        final = []
        sweep = []
        for q in quad_ring:
            if q in ordered_quads:
                names = ordered_quads[q]
                sweep.append(f"• Sweep hits {q}: {', '.join(names)}")
                final.extend(names)

        reasoning = dedent(f"""\
            1. Divide objects into quadrants relative to the agent.
            2. Sweep {turn_dir} through quadrants:
        {chr(10).join('    '+s for s in sweep)}
            3. Within each quadrant, decide orders using pair-wise relationships
        {chr(10).join(detailed)}
        """)

        return (
            f"<think>\n{indent(reasoning,'    ')}\n</think> "
            f"<answer> {self.answer} </answer>"
        )
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

    def _generate_reasoning(self, room) -> str:
        # 1. Extract the POV line
        pov_line = None
        for line in self.question.splitlines():
            if line.strip().lower().startswith("from ") and " is what direction to " in line.lower():
                pov_line = line.strip()
                break
        if not pov_line:
            raise ValueError(f"Could not find POV question in:\n{self.question!r}")

        # 2. Parse perspective, A, and B
        m = re.match(
            r"From\s+(.+?)'s perspective,\s+(.+?)\s+is what direction to\s+(.+?)\?",
            pov_line,
            flags=re.IGNORECASE
        )
        if not m:
            raise ValueError(f"Couldn't parse POV line: {pov_line!r}")
        pov_name, a_name, b_name = m.group(1), m.group(2), m.group(3)

        # 3. Build index map
        idx = {obj.name: i for i, obj in enumerate(room.all_objects)}
        agent_idx, a_idx, b_idx, pov_idx = idx[room.agent.name], idx[a_name], idx[b_name], idx[pov_name]
        
        # 4. Re‐compute relative directions in the pov frame
        _, dir_agent = room.get_direction(a_idx, b_idx, agent_idx)
        _, dir_ab = room.get_direction(a_idx, b_idx, pov_idx)       # final answer
        def orientation_to_string(ori: np.ndarray) -> int:
            direction_map = {
        (1, 0): 0,
        (0, 1): 90,
        (-1, 0): 180,
        (0, -1): 270
        }
    
            ori_tuple = tuple(ori.tolist())
            if ori_tuple not in direction_map:
                raise ValueError(f"Invalid orientation vector: {ori}. Must be one of the four cardinal directions.")
    
            return direction_map[ori_tuple]
        pov_ori = orientation_to_string(room.objects[pov_idx-1].ori)
        agent_ori = orientation_to_string(room.agent.ori)
        turn_angle = (pov_ori-agent_ori)%360
        orig_h, orig_v = dir_agent.strip("()").split(", ")
        # 3) helper to flip a component
        def opposite(comp: str) -> str:
            return {
                    "left":    "right",
                    "right":   "left",
                    "front":   "back",
                    "back":    "front",
                    "same":    "same"
                }[comp]
        def rotate_90(h: str) -> str:
            mapping = {
                    "right": "back",
                    "left":  "front",
                    "back":  "left",
                    "front": "right",
                    "same":  "same"
                }
            return mapping[h]

            # 4) compute new h/v under each rotation
        if turn_angle == 90:
            new_h = rotate_90(orig_v)
            new_v = rotate_90(orig_h)
            rule = (
                    f"horizontally, what was {orig_v} is now {new_h}; "
                    f"vertically, what was {orig_h} is now {new_v}"
                )
        elif turn_angle == 180:
            new_h = opposite(orig_h)
            new_v = opposite(orig_v)
            rule = (
                    f"horizontally, left/right are swapped {orig_h} → {new_h}; "
                    f"vertically, front/back are swapped {orig_v} → {new_v}"
                )
        elif turn_angle == 270:
            new_h = opposite(rotate_90(orig_v))
            new_v = opposite(rotate_90(orig_h))
            rule = (
                    f"horizontally, what was {orig_v} is now {new_h}; "
                    f"vertically, what was {orig_h} is now {new_v}"
                )
        elif turn_angle == 0:
            new_h = orig_h
            new_v = orig_v
            rule = (
                    f"horizontally, what was {orig_h} is now {new_h}; "
                    f"vertically, what was {orig_v} is now {new_v}"
                )
        # 5. Compose detailed reasoning
        lines = [
            f"1. From agent's(my) perspective, {a_name} is {dir_agent} to {b_name}.",
            f"2. Because {pov_name} is at {pov_ori} degrees, I need to turn {turn_angle} clockwise to think at its perspective.",
            f"3. {rule}.",
            f"4. Therefore, the direction from {b_name} to {a_name} is {dir_ab!r}."
        ]
        if agent_idx == pov_idx:
            think = f"From agent's(my) perspective, {a_name} is {dir_agent} to {b_name}."
        else:
            think = "\n".join(lines)

        return f"<think>\n{think}\n</think> <answer> {dir_ab} </answer>"
    
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

    def _generate_reasoning(self, room) -> str:
        """Generate reasoning for the evaluation task"""
        # 1. Extract the coordinate list from the question
        coords_line = next(
            (line for line in self.question.splitlines() if "Given a list of coordinates:" in line),
            None
        )
        if not coords_line:
            raise ValueError(f"Couldn't find coordinates in question:\n{self.question!r}")
        coords = re.findall(r"\((-?\d+),\s*(-?\d+)\)", coords_line)
        coords = [(int(x), int(y)) for x, y in coords]

        # 2. Build a map from each allocentric direction to its object
        graph   = DirectionalGraph(room.all_objects, is_explore=False)
        idx_map = {obj.name: i for i, obj in enumerate(room.all_objects)}
        agent_idx = idx_map[room.agent.name]

        dir_to_obj = {}
        for name, i in idx_map.items():
            # get_direction(a, b) with a=agent, b=object
            dp, _ = graph.get_direction(agent_idx, i)
            dp_str = DirectionSystem.to_string(dp, perspective="ego")
            dir_to_obj[dp_str] = name

        # 3. For each coordinate, derive its direction tuple and look up the object
        seq = []
        for x, y in coords:
            h = "right" if x > 0 else "left" if x < 0 else "same"
            v = "front" if y > 0 else "back" if y < 0 else "same"
            dp_str = f"({h},{v})"
            obj = dir_to_obj.get(dp_str, None)
            seq.append(obj)

        # 4. Compose the reasoning
        lines = [
            "1. We don’t know the true (x,y) of each object, but we can infer allocentric directions relative to the agent.",
            "2. Query each object’s direction from the agent and record:",
        ]
        for dp_str, name in dir_to_obj.items():
            lines.append(f"   - {name} is {dp_str} of agent.")
        lines.append("3. For each given coordinate, we translate it to a direction tuple:")
        for (x, y), dp_str, obj in zip(coords, [f"({('right' if x>0 else 'left' if x<0 else 'same')},{('front' if y>0 else 'back' if y<0 else 'same')})" for x,y in coords], seq):
            lines.append(f"   - Coordinate ({x},{y}) → direction {dp_str} → {obj}")
        lines.append(f"4. Thus the sequence of objects is {seq!r}.")

        think = "\n".join(lines)
        return f"<think>\n{think}\n</think> <answer> {seq} </answer>"
    
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

    def _generate_reasoning(self, room) -> str:
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
        return deg_seq_eval_fn(answer, self.answer), {}





if __name__ == "__main__":
    from ragen.env.spatial.config import SpatialGymConfig
    from ragen.env.spatial.Base.utils.room_utils import generate_room
    from gymnasium.utils import seeding
    import numpy as np

    config = SpatialGymConfig(n_objects=8, generation_type='pov', perspective='ego')
    np_random = seeding.np_random(10)[0]
    room = generate_room(**config.get_room_config(), np_random=np_random)
    print(room)
    

    # # Direction evaluation task
    task = DirEvaluationTask(np_random=np_random, config=DirEvaluationConfig(movement='agent_turn'))
    question = task.generate_question(room)
    print(question)
    print(task.answer)
    correct, info = task.evaluate("(unknown, back)")
    print(correct)
    reasoning = task._generate_reasoning(room=room)
    print(reasoning)

    # # Reverse direction evaluation task
    """task = ReverseDirEvaluationTask(np_random=np_random)
    question = task.generate_question(room)
    print(question)
    print(task.answer)
    correct, info = task.evaluate("folder")
    print(correct)
    reasoning = task._generate_reasoning(room=room)
    print(reasoning)"""

    # # Pov evaluation task
    """task = PovEvaluationTask(np_random=np_random)
    question = task.generate_question(room)
    print(question)
    print(task.answer)
    correct, info = task.evaluate("(right, back)")
    print(correct)
    reasoning = task._generate_reasoning(room=room)
    print(reasoning)"""
    
    # # A2E evaluation task
    """task = A2EEvaluationTask(np_random=np_random)
    question = task.generate_question(room)
    print(question)
    print(task.answer)
    correct, info = task.evaluate("[0, 90, 180, 270]")
    print(correct)
    reasoning = task._generate_reasoning(room=room)
    print(reasoning)"""
    # # E2A evaluation task
    """task = E2AEvaluationTask(np_random=np_random)
    question = task.generate_question(room)
    print(question)
    print(task.answer)
    correct, info = task.evaluate("['television', 'microphone', 'agent', 'eraser']")
    print(correct)
    reasoning = task._generate_reasoning(room=room)
    print(reasoning)"""
    # Rotation evaluation task
    """
    task = RotEvaluationTask(np_random=np_random, config=RotEvaluationConfig(turn_direction='clockwise'))
    question = task.generate_question(room)
    print(question)
    print(task.answer)
    correct, info = task.evaluate("['television', 'microphone', 'agent', 'eraser']")
    print(correct)
    reasoning = task._generate_reasoning(room=room)
    print(reasoning)
"""
    
    