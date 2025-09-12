import os
import json
import numpy as np
from typing import Dict, Any, Tuple
from .Base.tos_base.core.room import Room
from .Base.tos_base.core.object import Object, Agent, Gate
from .Base.tos_base.utils.room_utils import RoomGenerator


def load_vision_data(vision_data_path: str, seed: int = None) -> Tuple[str, Dict[str, Any]]:
    """Load vision data from specified path with cycling support."""
    if not os.path.exists(vision_data_path):
        raise ValueError(f"Vision data path does not exist: {vision_data_path}")
    
    subdirs = sorted([d for d in os.listdir(vision_data_path) 
                     if os.path.isdir(os.path.join(vision_data_path, d))])
    
    if not subdirs:
        raise ValueError(f"No subdirectories found in vision data path: {vision_data_path}")
    
    # Use seed to cycle through available data directories
    data_idx = (seed % len(subdirs)) if seed is not None else np.random.randint(0, len(subdirs))
    data_dir = os.path.join(vision_data_path, subdirs[data_idx])
    
    meta_file = os.path.join(data_dir, "meta_data.json")
    if not os.path.exists(meta_file):
        raise ValueError(f"meta_data.json not found in {data_dir}")
    
    with open(meta_file, 'r') as f:
        json_data = json.load(f)
    
    return data_dir, json_data


def initialize_room_from_vision_data(json_data: Dict[str, Any]) -> Tuple[Room, Agent, Dict[str, Any]]:
    """Initialize a Room from vision metadata JSON."""
    rotation_map = {0: np.array([0, 1]), 90: np.array([1, 0]), 
                   180: np.array([0, -1]), 270: np.array([-1, 0])}
    
    offset = np.array(json_data.get('offset', [0, 0]))
    
    # Parse objects (exclude doors)
    objects = []
    for obj in json_data['objects']:
        if 'door' not in obj['name']:
            objects.append(Object(
                name=obj['name'],
                pos=np.array([obj["pos"]["x"], obj["pos"]["z"]]) + offset,
                ori=rotation_map.get(obj["rot"]["y"], np.array([0, 1])) if obj["attributes"].get("has_orientation", False) else np.array([0, 1]),
                has_orientation=obj["attributes"].get("has_orientation", False)
            ))
    
    # Get agent position
    agent_cameras = [cam for cam in json_data['cameras'] if cam['id'] == 'agent']
    if not agent_cameras:
        raise ValueError("Agent camera not found in vision data")
    
    agent_pos = agent_cameras[0]['position']
    agent_pos = np.array([agent_pos["x"], agent_pos["z"]]) + offset
    
    # Create mask and gates
    mask = np.array(json_data['mask'])
    gates = RoomGenerator._gen_gates_from_mask(mask)
    
    # Update gate names from door objects
    door_objects = [obj for obj in json_data['objects'] if 'door' in obj['name']]
    for gate in gates:
        for door in door_objects:
            if set(gate.room_id) == set(door["attributes"]['connected_rooms']):
                gate.name = door['name']
                break
    
    room_name = json_data.get("name", "vision_room")
    agent = Agent(pos=agent_pos, room_id=1, init_room_id=1)
    
    return Room(objects=objects, mask=mask, name=room_name, gates=gates), agent, json_data
