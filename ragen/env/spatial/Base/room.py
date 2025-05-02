import numpy as np
import json
from typing import List, Union, Dict, Any, Tuple
import copy

from ragen.env.spatial.Base.object import Object, Agent
from ragen.env.spatial.Base.relationship import DirPair, DirectionSystem, Dir
from ragen.env.spatial.Base.constant import CANDIDATE_OBJECTS, AGENT_NAME
from ragen.env.spatial.Base.graph import DirectionalGraph


from enum import Enum
from typing import Optional, Union, Tuple


class ActionType(Enum):
    """Enum for different types of actions in the spatial gym environment"""
    MOVE = "Move"
    ROTATE = "Rotate"
    RETURN = "Return"
    QUERY = "Query"
    TERM = "Term"
    
    @classmethod
    def from_string(cls, action_str: str) -> Optional['ActionType']:
        """Convert a string to an ActionType enum value"""
        for action_type in cls:
            if action_str == action_type.value:
                return action_type
        return None
    
    @staticmethod
    def get_format(action_type: 'ActionType', **kwargs):
        """
        Get the string format description for each action type.
        
        Parameters:
            action_type: The type of action to get format for
            **kwargs: Additional parameters
                - is_active: Boolean indicating if this is an active exploration mode
                  (affects Query format)
        
        Returns:
            String describing the proper format for the specified action type
        """
        formats = {
            ActionType.MOVE: "Move to an object by specifying: Move(<object_name>). Example: Move(chair)",
            ActionType.ROTATE: "Rotate by a specific degree (90, 180, or 270) clockwise: Rotate(<degree>). Example: Rotate(90)",
            ActionType.RETURN: "Return to your starting position and orientation: Return()",
            ActionType.TERM: "End the current exploration: Term()"
        }
        
        if action_type == ActionType.QUERY:
            if kwargs.get('is_active', False):
                return "Query the spatial relationship between yourself and an object: Query(<object_name>). Example: Query(table)"
            else:
                return "Query the spatial relationship between two objects: Query(<object1_name>, <object2_name>). Example: Query(chair, table)"
        
        return formats[action_type]
            


class Action:
    """Class representing an action in the spatial gym environment"""
    
    def __init__(self, action_type: ActionType, parameters: Optional[Union[str, int, Tuple[str, str]]] = None):
        """
        Initialize an Action object
        
        Parameters:
            action_type: The type of action
            parameters: Parameters for the action
                - For MOVE: object name (str)
                - For ROTATE: degrees (int)
                - For RETURN: None
                - For QUERY: object name (str) or tuple of two object names (str, str)
                - For TERM: None
        """
        self.action_type = action_type
        self.parameters = parameters
    
    def __repr__(self):
        """String representation of the action"""
        if self.parameters is None:
            return f"[Action] {self.action_type.value}()"
        elif isinstance(self.parameters, tuple):
            return f"[Action] {self.action_type.value}({self.parameters[0]}, {self.parameters[1]})"
        else:
            return f"[Action] {self.action_type.value}({self.parameters})"
            
    def is_final_action(self) -> bool:
        """Check if this is a final action (QUERY or TERM)"""
        return self.action_type in [ActionType.QUERY, ActionType.TERM]
    
    def get_format(self, **kwargs):
        """Get the format of the action"""
        return ActionType.get_format(self.action_type, **kwargs)
    


class Room:
    """
    Room as the environment for spatial tasks, also serve as the state for the gym
    Attributes:
        - name: name of the room
        - objects: list of objects in the room
        - agent (optional): agent

    NOTE
        - For egocentric exploration, objects are tracked as:
            - agent: the agent itself
            - objects
            - original position of the agent
        - For allocentric exploration, objects are tracked as:
            - objects

    Room defines the following attributes:
        - room layout (room_desc)
        - action space (action_space)
        - state / observation (exp_graph)
    """

    def _validate(self):
        assert len(self._all_objects_exp) == len(set([obj.name for obj in self._all_objects_exp])),\
            "Objects' names must be unique"

    def _name_to_id(self, name: str) -> int:
        object_names = [obj.name for obj in self._all_objects_exp]
        assert name in object_names, f"name {name} not in room {object_names}"
        return object_names.index(name)
    
    def _get_object_by_name(self, name: str) -> Object:
        return self._all_objects_exp[self._name_to_id(name)]

    def _check_visibility(self, anchor_obj: Object, obj: Object) -> bool:
        """
        Check if obj is visible from anchor_obj's perspective
        NOTE currently 180 degree
        TODO 90 degree
        """
        dir_pair = DirectionSystem.get_direction(obj.pos, anchor_obj.pos, anchor_obj.ori)
        print(f"[DEBUG] anchor_obj: {anchor_obj}, obj: {obj}, dir_pair: {dir_pair}")
        if dir_pair.vert == Dir.BACKWARD:
            return False
        else:
            return True
    
    def __repr__(self):
        return f"Room(name={self.name}, objects={self._all_objects_exp})"

    def __init__(
        self,
        objects: List[Object],
        name: str = 'random_room',
        agent: Agent = None,
    ):
        self.name = name
        self.objects = copy.deepcopy(objects)
        self.agent = copy.deepcopy(agent) if agent is not None else None
        self.backup = {
            'agent': copy.deepcopy(self.agent) if self.agent is not None else None,
            'objects': copy.deepcopy(self.objects),
        }
        self.all_objects = [self.agent] + self.objects if self.agent is not None else self.objects

        if agent is not None:
            self.agent_anchor = Object(name="agent_anchor", pos=copy.deepcopy(agent.pos), ori=copy.deepcopy(agent.ori))
            self._all_objects_exp = [self.agent] + self.objects + [self.agent_anchor]
            self.exp_graph = DirectionalGraph(self._all_objects_exp, is_explore=True)
            self.exp_graph.add_edge(self._name_to_id(self.agent.name), self._name_to_id(self.agent_anchor.name), DirPair(Dir.SAME, Dir.SAME))
        else:
            self._all_objects_exp = self.objects
            self.exp_graph = DirectionalGraph(self._all_objects_exp, is_explore=True)
        self._validate()
        

    @classmethod
    def from_dict(cls, room_dict: dict) -> 'Room':
        objects = [Object.from_dict(obj_dict) for obj_dict in room_dict['objects']]
        agent = Object.from_dict(room_dict['agent']) if room_dict['agent'] is not None else None
        agent_anchor = Object.from_dict(room_dict['agent_anchor']) if room_dict['agent_anchor'] is not None else None
        exp_graph = DirectionalGraph.from_dict(room_dict['exp_graph'])
        instance = cls(
            name=room_dict['name'],
            objects=objects,
            agent=agent,
        )
        instance.agent_anchor = agent_anchor
        instance.exp_graph = exp_graph
        instance.backup = {
            'agent': Object.from_dict(room_dict['backup']['agent']) if room_dict['backup']['agent'] is not None else None,
            'objects': [Object.from_dict(obj_dict) for obj_dict in room_dict['backup']['objects']],
        }
        return instance
    
    def to_dict(self):
        return {
            'name': self.name,
            'objects': [obj.to_dict() for obj in self.objects],
            'agent': self.agent.to_dict() if self.agent is not None else None,
            'agent_anchor': self.agent_anchor.to_dict() if self.agent_anchor is not None else None,
            'exp_graph': self.exp_graph.to_dict(),
            'backup': {
                'agent': self.backup['agent'].to_dict() if self.backup['agent'] is not None else None,
                'objects': [obj.to_dict() for obj in self.backup['objects']],
            }
        }
    
    def copy(self):
        room = Room(
            name=self.name,
            objects=copy.deepcopy(self.objects),
            agent=copy.deepcopy(self.agent),
        )
        room.agent_anchor = copy.deepcopy(self.agent_anchor)
        room.exp_graph = self.exp_graph.copy()
        room.backup = {
            'agent': copy.deepcopy(self.backup['agent']),
            'objects': copy.deepcopy(self.backup['objects']),
        }
        return room
    
    def is_object_in_room(self, obj_name: str) -> bool:
        return obj_name in [obj.name for obj in self._all_objects_exp]
    
    def get_boundary(self):
        """
        Get the boundary of the room
        """
        positions = np.array([obj.pos for obj in self._all_objects_exp])
        min_x, min_y = np.min(positions, axis=0)
        max_x, max_y = np.max(positions, axis=0)
        
        # Generate random position within extended boundaries
        # x ranges from (2*min_x - max_x) to (2*max_x - min_x)
        # y ranges from (2*min_y - max_y) to (2*max_y - min_y)
        min_x_bound = min_x - min(max_x - min_x, 1)
        max_x_bound = max_x + min(max_x - min_x, 1)
        min_y_bound = min_y - min(max_y - min_y, 1)
        max_y_bound = max_y + min(max_y - min_y, 1)
        return min_x_bound, max_x_bound, min_y_bound, max_y_bound
    
    def get_room_desc(self):
        """
        Get the description of the room
        """
        desc = f"Imagine yourself named {self.agent.name} in a room.\n" if self.agent is not None else "Imagine looking at a room in a bird's eye view.\n"
        desc += "You are facing north.\n"
        desc += f"Objects in the room are: {', '.join([obj.name for obj in self.objects])}\n"
        return desc
    


    def explore(
            self,
            motion_list: List[Action],
            query_action: Action,
            is_active: bool = False,
    ) -> str:
        """
        Query the spatial relationship between obj1 and obj2
        Parameters:
            action_list: list of actions agent take to move in the room (MOVE, ROTATE, RETURN)
            query_action: action to query the spatial relationship between obj1 and obj2 (QUERY, TERM)
            is_active: if active exploration
                - agent can only see the objects in front of it (NOTE 180 degree)
                - agent can only ask one object relative to itself
                - TODO agent can only move to visible objects
        """
        info = {}
        # perform motion
        assert query_action.action_type in [ActionType.QUERY, ActionType.TERM], "last action must be QUERY or TERM"
        for action in motion_list:
            assert action.action_type in [ActionType.MOVE, ActionType.ROTATE, ActionType.RETURN], "action must be MOVE, ROTATE, or RETURN"
            if action.action_type == ActionType.MOVE:
                # move the agent to the position of the object
                obj_name = action.parameters
                self.move_agent(
                    anchor_obj_name=obj_name,
                    new_pos=self._all_objects_exp[self._name_to_id(obj_name)].pos
                )
                
            elif action.action_type == ActionType.ROTATE:
                degree = action.parameters
                self.rotate_agent(degree)
            elif action.action_type == ActionType.RETURN:
                self.move_agent(
                    anchor_obj_name=self.agent_anchor.name,
                    new_pos=self.agent_anchor.pos
                )
            
            print(f"[DEBUG] after motion {action}, the room is {self}")


        # terminate exploration
        if query_action.action_type == ActionType.TERM:
            return None, info
        

        # perform query and answer
        if is_active:
            assert self.agent is not None, "Agent is required for active exploration"
            obj_name = query_action.parameters
            assert isinstance(obj_name, str), "obj_name must be a string"
            if not self._check_visibility(self.agent, self._get_object_by_name(obj_name)):
                dir_pair = DirPair(Dir.UNKNOWN, Dir.UNKNOWN)
                info['novel_query'] = True
                return DirectionSystem.to_string(dir_pair, perspective='ego'), info
            else:
                obj_dir_pair, obj_dir_pair_str = self.get_direction(obj_name, self.agent.name)
                print(f"[DEBUG] obj_dir_pair: {obj_dir_pair}, obj_dir_pair_str: {obj_dir_pair_str}")
                novel_query = self.exp_graph.add_edge(self._name_to_id(obj_name), self._name_to_id(self.agent.name), obj_dir_pair)
                info['novel_query'] = novel_query
                return obj_dir_pair_str, info

        obj1_name, obj2_name = query_action.parameters
        assert isinstance(obj1_name, str) and isinstance(obj2_name, str), "obj1_name and obj2_name must be strings"
        dir_pair, dir_pair_str = self.get_direction(obj1_name, obj2_name)
        novel_query = self.exp_graph.add_edge(self._name_to_id(obj1_name), self._name_to_id(obj2_name), dir_pair)
        info['novel_query'] = novel_query
        return dir_pair_str, info
    
    def finish_exploration(self, return_original: bool = True):
        """
        Finish the exploration, delete the last agent_anchor
        Parameters:
            return_original: if True, agent returns to the original position and orientation
        TODO add other post-processing
        """
        if len(self._all_objects_exp) == len(self.all_objects): # already finished exploration
            return
        
        if return_original:
            self.move_agent(
                anchor_obj_name=self.agent_anchor.name,
                new_pos=self.agent_anchor.pos
            )
            # Map orientation vectors to degrees
            ori_to_deg = {
                (0, 1): 0,
                (0, -1): 180,
                (1, 0): 90,
                (-1, 0): 270
            }
            deg = ori_to_deg[tuple(self.agent_anchor.ori)]
            self.rotate_agent(deg)

        
        def _delete_elem(matrix: np.ndarray, obj_id: int):
            matrix = np.delete(matrix, obj_id, axis=0)
            matrix = np.delete(matrix, obj_id, axis=1)
            return matrix
        self.exp_graph.size -= 1
        self.exp_graph._v_matrix = _delete_elem(self.exp_graph._v_matrix, self.exp_graph.size)
        self.exp_graph._h_matrix = _delete_elem(self.exp_graph._h_matrix, self.exp_graph.size)
        self.exp_graph._v_matrix_working = _delete_elem(self.exp_graph._v_matrix_working, self.exp_graph.size)
        self.exp_graph._h_matrix_working = _delete_elem(self.exp_graph._h_matrix_working, self.exp_graph.size)
        self.exp_graph._asked_matrix = _delete_elem(self.exp_graph._asked_matrix, self.exp_graph.size)

        self._all_objects_exp.pop()
        
    
    def get_inferable_pairs(self):
        """
        Get all the inferable pairs in the room
        """
        return self.exp_graph.get_inferable_pairs()
    
    def get_unknown_pairs(self):
        """
        Get all the unknown pairs in the room
        """
        return self.exp_graph.get_unknown_pairs()
    







    def get_direction(
            self,
            object1: Union[str, int],
            object2: Union[str, int],
            anchor_obj: Union[str, int] = None,
            from_exp_graph: bool = False,
            perspective: str = None,

    ) -> Tuple[DirPair, str]:
        """
        Get the direction of object2 relative to object1

        Parameters:
            object1: object1
            object2: object2
            anchor_obj: anchor_obj (from which perspective the direction is measured)
            from_exp_graph: if True, get the direction from the exploration graph (maybe unknown)
            perspective: perspective of the direction
                - 'ego': agent's perspective
                - 'allo': allocentric perspective

        Returns:
            direction (DirPair): direction of object2 relative to object1
            direction_str (str): direction of object2 relative to object1 in string format
        """
        if from_exp_graph:
            obj1_id = self._name_to_id(object1) if isinstance(object1, str) else object1
            obj2_id = self._name_to_id(object2) if isinstance(object2, str) else object2
            dir_pair = self.exp_graph.get_direction(obj1_id, obj2_id)
            if anchor_obj is not None:
                anchor_obj_id = self._name_to_id(anchor_obj) if isinstance(anchor_obj, str) else anchor_obj
                dir_pair = DirectionSystem.transform(dir_pair, self._all_objects_exp[anchor_obj_id].ori)

        else:
            if isinstance(object1, str):
                object1 = self._get_object_by_name(object1)
            elif isinstance(object1, int):
                object1 = self._all_objects_exp[object1]

            if isinstance(object2, str):
                object2 = self._get_object_by_name(object2)
            elif isinstance(object2, int):
                object2 = self._all_objects_exp[object2]

            if anchor_obj is not None:
                if isinstance(anchor_obj, str):
                    anchor_obj = self._get_object_by_name(anchor_obj)
                elif isinstance(anchor_obj, int):
                    anchor_obj = self._all_objects_exp[anchor_obj]
                
            dir_pair = DirectionSystem.get_direction(object1.pos, object2.pos, anchor_obj.ori if anchor_obj is not None else None)
        
        perspective = perspective or ('ego' if self.agent is not None else 'allo')
        dir_pair_str = DirectionSystem.to_string(dir_pair, perspective=perspective)
        return dir_pair, dir_pair_str
    
    def move_object(
            self,
            moved_obj_name: str,
            new_pos: np.ndarray,
            anchor_obj_name: str = None,
    ):
        """
        Move <moved_obj_name> to <new_pos>, (optional) relative to <anchor_obj_name>
        1. Move the original object to the new position
        2. Update the graph (if anchor_obj_name is provided)
        """
        moved_obj_id = self._name_to_id(moved_obj_name)
        moved_obj = self._all_objects_exp[moved_obj_id]
        moved_obj.pos = new_pos

        # update the graph
        if anchor_obj_name is not None:
            anchor_obj_id = self._name_to_id(anchor_obj_name)
            anchor_obj = self._all_objects_exp[anchor_obj_id]
            dir_pair = DirectionSystem.get_direction(new_pos, anchor_obj.pos)
            self.exp_graph.move_node(moved_obj_id, anchor_obj_id, dir_pair)


    def move_agent(
            self,
            new_pos: np.ndarray,
            anchor_obj_name: str = None,
    ):
        """
        Move the agent to <new_pos>, (optional) relative to <anchor_obj_name>
        1. Move the agent to the new position
        2. Update the graph
        3. Update so that agent is at (0, 0)
        """
        assert self.agent is not None

        # Step 1: move the agent to the new position
        old_pos = self.agent.pos
        self.move_object(self.agent.name, new_pos, anchor_obj_name)

        # Step 2: move origin to the new position of the agent to keep agent at (0, 0)
        pos_diff = new_pos - old_pos
        for obj in self._all_objects_exp:
            obj.pos = obj.pos - pos_diff


    def rotate_agent(self, degree: int):
        """
        Rotate the agent by <degree> degrees clockwise
        1. Rotate all other objects by <degree> degrees counterclockwise
        2. Update the graph
        """
        assert self.agent is not None, "Agent is required"
        assert np.array_equal(self.agent.pos, np.array([0, 0])), f"Agent position should be [0, 0], but got {self.agent.pos}"
        assert degree in [0, 90, 180, 270]

        if degree == 0:
            return
        elif degree == 90:
            rotation_matrix = np.array([
                [0, 1],
                [-1, 0],
            ])
        elif degree == 180:
            rotation_matrix = np.array([
                [-1, 0],
                [0, -1],
            ])
        elif degree == 270:
            rotation_matrix = np.array([
                [0, -1],
                [1, 0],
            ])


        # update the position and orientation of all objects
        for obj in self._all_objects_exp:
            if obj.name == self.agent.name:
                continue
            pos, ori = obj.pos, obj.ori
            obj.pos = pos @ rotation_matrix
            obj.ori = ori @ rotation_matrix

        self.exp_graph.rotate_axis(degree)
    



if __name__ == "__main__":
    agent = Agent(name="agent", pos=np.array([0, 0]), ori=np.array([0, 1]))
    obj1 = Object(name="obj1", pos=np.array([1, 1]), ori=np.array([0, 1]))
    obj2 = Object(name="obj2", pos=np.array([-1, 1]), ori=np.array([0, 1]))
    obj3 = Object(name="obj3", pos=np.array([0, 2]), ori=np.array([0, 1]))
        
        # Create a room with these objects
    room = Room(
        name="test_room",
        objects=[obj1, obj2, obj3],
        agent=agent
    )
    print(room.get_room_desc())


    dir_pair, dir_pair_str = room.get_direction(obj1.name, obj2.name)
    print(dir_pair, dir_pair_str)

    print(f'[DEBUG] room.exp_graph._v_matrix: \n {room.exp_graph._v_matrix}')

    # res = room.explore([], Action(ActionType.QUERY, obj1.name), is_active=True)
    # print(f'[DEBUG] res: {res}')
    # print(f'[DEBUG] room.exp_graph._v_matrix: \n {room.exp_graph._v_matrix}')
    # print(f'[DEBUG] room.exp_graph._h_matrix: \n {room.exp_graph._h_matrix}')

    # res = room.explore([Action(ActionType.MOVE, obj2.name)], Action(ActionType.QUERY, obj1.name), is_active=True)
    # print(f'[DEBUG] res: {res}')
    # print(f'[DEBUG] room.exp_graph._v_matrix: \n {room.exp_graph._v_matrix}')
    # print(f'[DEBUG] room.exp_graph._h_matrix: \n {room.exp_graph._h_matrix}')

    # res = room.explore([Action(ActionType.ROTATE, 90)], Action(ActionType.QUERY, obj3.name), is_active=True)
    # print(f'[DEBUG] res: {res}')
    # print(f'[DEBUG] room.exp_graph._v_matrix: \n {room.exp_graph._v_matrix}')
    # print(f'[DEBUG] room.exp_graph._h_matrix: \n {room.exp_graph._h_matrix}')
    

    res = room.explore([Action(ActionType.ROTATE, 180)], Action(ActionType.QUERY, obj3.name), is_active=True)
    print(f'[DEBUG] res: {res}')
    print(f'[DEBUG] room.exp_graph._v_matrix: \n {room.exp_graph._v_matrix}')
    print(f'[DEBUG] room.exp_graph._h_matrix: \n {room.exp_graph._h_matrix}')