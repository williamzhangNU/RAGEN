"""
Parse the exploration input from agent
"""
from typing import Optional
import re

from ragen.env.spatial.Base.room import Action, ActionType, Room


def parse_action(action_str: str, room: Room, is_active: bool) -> tuple[list[Action], Optional[Action]]:
    """Parse action string into actions and final command"""
    parts = action_str.split(';')
    if len(parts) > 2 or (len(parts) == 2 and not is_active):
        return [], None
    
    # Parse action sequence
    motion_list = []
    if len(parts) == 2 and is_active:
        for item in [i.strip() for i in parts[0].split(',') if i.strip()]:
            if match := re.match(r"Move\(([A-Za-z0-9_-]+)\)", item):
                motion_list.append(Action(ActionType.MOVE, match.group(1)))
            elif match := re.match(r"Rotate\(([0-9-]+)\)", item):
                motion_list.append(Action(ActionType.ROTATE, int(match.group(1))))
            elif item == "Return()":
                motion_list.append(Action(ActionType.RETURN, None))
            else:
                return [], None
            
    # Parse final command
    final_cmd = parts[-1].strip()
    query_action = None
    
    if match := re.match(r"Query\(([A-Za-z0-9_-]+)(?:,\s*([A-Za-z0-9_-]+))?\)", final_cmd):
        obj1, obj2 = match.group(1), match.group(2)
        if obj2 is None and is_active:
            query_action = Action(ActionType.QUERY, obj1)
        elif obj2 is not None and not is_active:
            query_action = Action(ActionType.QUERY, (obj1, obj2))
        else:
            return [], None
    elif final_cmd == "Term()":
        query_action = Action(ActionType.TERM, None)
    else:
        return [], None
    
    # Check if query action exists
    if query_action is None:
        return [], None
    
    # If Term action, motion_list should be empty
    if query_action.action_type == ActionType.TERM and motion_list:
        return [], None
    
    # Validate all actions at once
    def validate_actions():
        # Check Move and Query objects exist in room
        for action in motion_list:
            if action.action_type == ActionType.MOVE and not room.is_object_in_room(action.parameters):
                return False
            if action.action_type == ActionType.ROTATE and action.parameters not in [90, 180, 270]:
                return False
        
        # Check Query objects exist in room
        if query_action.action_type == ActionType.QUERY:
            if isinstance(query_action.parameters, tuple):
                obj1, obj2 = query_action.parameters
                return room.is_object_in_room(obj1) and room.is_object_in_room(obj2)
            else:
                return room.is_object_in_room(query_action.parameters)
        return True
    
    return (motion_list, query_action) if validate_actions() else ([], None)




if __name__ == "__main__":
    from Task.SpatialGym.config import SpatialGymConfig
    from Task.SpatialGym.BaseEnv.utils.room_utils import generate_room
    import numpy as np

    config = SpatialGymConfig()
    room = generate_room(**config.get_room_config(), np_random=np.random.default_rng(0))
    print(room)

    action_str = " Query(scanner)"
    motion_list, query_action = parse_action(action_str, room, False)
    print(motion_list)
    print(query_action)
