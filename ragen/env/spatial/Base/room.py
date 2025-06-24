import numpy as np
import json
from typing import List, Union, Dict, Any, Tuple
import copy

from ragen.env.spatial.Base.object import Object, Agent
from ragen.env.spatial.Base.relationship import DirPair, DirectionSystem, Dir
from ragen.env.spatial.Base.graph import DirectionalGraph
    


class Room:
    """
    Simplified Room class focused on state representation.
    Handles basic object management and spatial relationships.
    """

    def __init__(self, objects: List[Object], name: str = 'room', agent: Agent = None):
        self.name = name
        self.objects = copy.deepcopy(objects)
        self.agent = copy.deepcopy(agent) if agent is not None else None
        
        # All objects including agent (for ground truth calculations)
        self.all_objects = ([self.agent] + self.objects) if self.agent else self.objects
        
        # Ground truth graph for evaluation
        self.gt_graph = DirectionalGraph(self.all_objects, is_explore=False)
        
        # Validate unique names
        self._validate_objects()

    def _validate_objects(self):
        """Ensure all objects have unique names"""
        names = [obj.name for obj in self.all_objects]
        assert len(names) == len(set(names)), "All object names must be unique"

    def get_object_by_name(self, name: str) -> Object:
        """Get object by name"""
        for obj in self.all_objects:
            if obj.name == name:
                return obj
        raise ValueError(f"Object '{name}' not found in room")

    def has_object(self, name: str) -> bool:
        """Check if object exists in room"""
        return any(obj.name == name for obj in self.all_objects)

    def get_direction(self, obj1_name: str, obj2_name: str, 
                     anchor_name: str = None, perspective: str = 'allo') -> Tuple[DirPair, str]:
        """Get spatial relationship between two objects"""
        obj1 = self.get_object_by_name(obj1_name)
        obj2 = self.get_object_by_name(obj2_name)
        
        anchor_ori = None
        if anchor_name:
            anchor = self.get_object_by_name(anchor_name)
            anchor_ori = anchor.ori
        
        dir_pair = DirectionSystem.get_direction(obj1.pos, obj2.pos, anchor_ori)
        dir_str = DirectionSystem.to_string(dir_pair, perspective=perspective)
        
        return dir_pair, dir_str

    def get_room_description(self) -> str:
        """Get textual description of the room"""
        if self.agent:
            desc = f"Imagine yourself as {self.agent.name} in a room.\n"
            desc += "You are facing north.\n"
            desc += f"Objects in the room: {', '.join([obj.name for obj in self.objects])}\n"
        else:
            desc = "Imagine looking at a room from above.\n"
            desc += f"Objects in the room: {', '.join([obj.name for obj in self.objects])}\n"
        return desc

    def copy(self) -> 'Room':
        """Create a deep copy of the room"""
        return Room(
            objects=copy.deepcopy(self.objects),
            name=self.name,
            agent=copy.deepcopy(self.agent)
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize room to dictionary"""
        return {
            'name': self.name,
            'objects': [obj.to_dict() for obj in self.objects],
            'agent': self.agent.to_dict() if self.agent else None
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Room':
        """Deserialize room from dictionary"""
        objects = [Object.from_dict(obj_data) for obj_data in data['objects']]
        agent = Agent.from_dict(data['agent']) if data['agent'] else None
        return cls(objects=objects, name=data['name'], agent=agent)

    def __repr__(self):
        return f"Room(name={self.name}, objects={len(self.objects)}, agent={self.agent is not None})"







