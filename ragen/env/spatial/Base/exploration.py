import numpy as np
import copy
from typing import List, Tuple, Dict, Any

from ragen.env.spatial.Base.object import Object, Agent
from ragen.env.spatial.Base.relationship import DirPair, DirectionSystem, Dir
from ragen.env.spatial.Base.graph import DirectionalGraph
from ragen.env.spatial.Base.action import ActionType, Action, ActionSequence
from ragen.env.spatial.Base.room import Room


class ExplorationManager:
    """Manages spatial exploration with agent movement and queries."""
    
    def __init__(self, room: Room):
        self.base_room = room.copy()
        self.current_room = room.copy()
        
        assert room.agent is not None, "Exploration requires an agent in the room"
            
        # Create agent anchor for tracking original position
        self.agent_anchor = Object(
            name="agent_anchor", 
            pos=copy.deepcopy(room.agent.pos), 
            ori=copy.deepcopy(room.agent.ori)
        )
        
        # All objects including agent and anchor
        self._objects = [self.current_room.agent] + self.current_room.objects + [self.agent_anchor]
        self.exp_graph = DirectionalGraph(self._objects, is_explore=True)
        
        # Link agent to anchor
        agent_idx = self._get_index(self.current_room.agent.name)
        anchor_idx = self._get_index(self.agent_anchor.name)
        self.exp_graph.add_edge(agent_idx, anchor_idx, DirPair(Dir.SAME, Dir.SAME))
        
    def _get_index(self, name: str) -> int:
        """Get object index by name."""
        for i, obj in enumerate(self._objects):
            if obj.name == name:
                return i
        raise ValueError(f"Object '{name}' not found")
    
    def _get_object(self, name: str) -> Object:
        """Get object by name."""
        for obj in self._objects:
            if obj.name == name:
                return obj
        raise ValueError(f"Object '{name}' not found")
    
    
    def _move_agent_to(self, new_pos: np.ndarray, anchor_name: str = None):
        """Move agent to position and update coordinate system."""
        old_pos = self.current_room.agent.pos
        
        # Move agent
        agent_idx = self._get_index(self.current_room.agent.name)
        self._objects[agent_idx].pos = new_pos
        
        # Update graph if anchor provided
        if anchor_name:
            anchor_idx = self._get_index(anchor_name)
            anchor = self._objects[anchor_idx]
            dir_pair = DirectionSystem.get_direction(new_pos, anchor.pos, anchor.ori)
            self.exp_graph.move_node(agent_idx, anchor_idx, dir_pair)
        
        # Shift coordinate system to keep agent at origin
        pos_diff = new_pos - old_pos
        for obj in self._objects:
            obj.pos = obj.pos - pos_diff
    
    def _rotate_agent(self, degrees: int):
        """Rotate agent by specified degrees."""
        if degrees == 0:
            return
            
        # Create rotation matrix
        if degrees == 90:
            rotation_matrix = np.array([
                [0, 1],
                [-1, 0],
            ])
        elif degrees == 180:
            rotation_matrix = np.array([
                [-1, 0],
                [0, -1],
            ])
        elif degrees == 270:
            rotation_matrix = np.array([
                [0, -1],
                [1, 0],
            ])
        
        # Rotate all objects except agent
        for obj in self._objects:
            if obj.name != self.current_room.agent.name:
                obj.pos = obj.pos @ rotation_matrix
                obj.ori = obj.ori @ rotation_matrix
                
        # Update exploration graph
        self.exp_graph.rotate_axis(degrees)

    def _is_visible(self, from_obj: Object, to_obj: Object) -> bool:
        """Check if to_obj is visible from from_obj (180-degree visibility)."""
        dir_pair = DirectionSystem.get_direction(to_obj.pos, from_obj.pos, from_obj.ori)
        return dir_pair.vert != Dir.BACKWARD
    
    def _validate_and_execute_action(self, action: Action) -> Tuple[bool, str, Dict[str, Any]]:
        """Validate and execute action in one step.
        
        Returns:
            - success: Whether the action was valid and executed
            - message: Validation/execution message
            - info: Additional execution info

        TODO: add more information to info
        """
        
        if action.action_type == ActionType.MOVE:
            if not self.current_room.has_object(action.parameters):
                return False, action.get_feedback(False, "not_found"), {}
            target = self._get_object(action.parameters)
            if not self._is_visible(self.current_room.agent, target):
                return False, action.get_feedback(False, "not_visible"), {}
            
            # Execute move
            self._move_agent_to(target.pos, action.parameters)
            return True, action.get_feedback(True), {}
            
        elif action.action_type == ActionType.ROTATE:
            if action.parameters not in [0, 90, 180, 270]:
                return False, action.get_feedback(False, "invalid_degree"), {}
            
            # Execute rotation
            self._rotate_agent(action.parameters)
            return True, action.get_feedback(True), {}
            
        elif action.action_type == ActionType.QUERY:
            if not self.current_room.has_object(action.parameters):
                return False, action.get_feedback(False, "not_found"), {}
            target = self._get_object(action.parameters)
            if not self._is_visible(self.current_room.agent, target):
                return False, action.get_feedback(False, "not_visible"), {}
            
            # Execute query
            answer, query_info = self._process_query(action.parameters)
            # Update the success message to include query result
            success_msg = action.get_feedback(True, answer=answer)
            return True, success_msg, query_info
        
        elif action.action_type == ActionType.RETURN:
            # Execute return
            self._move_agent_to(self.agent_anchor.pos, self.agent_anchor.name)
            return True, action.get_feedback(True), {}
            
        elif action.action_type == ActionType.TERM:
            return True, action.get_feedback(True), {}
            
        raise ValueError(f"Unknown action: {action.action_type}")
    
    def _process_query(self, target_name: str) -> Tuple[str, Dict[str, Any]]:
        """Process query and return direction string."""
        info = {}
        dir_pair, dir_str = self.current_room.get_direction(target_name, self.current_room.agent.name, perspective='ego')
        
        # Update graph, if the query is novel, then the edge is added to the graph
        target_idx = self._get_index(target_name)
        agent_idx = self._get_index(self.current_room.agent.name)
        novel_query = self.exp_graph.add_edge(target_idx, agent_idx, dir_pair)
        
        info['novel_query'] = novel_query
        return dir_str, info
    




    
    def execute_action_sequence(self, action_sequence: ActionSequence) -> Tuple[str, Dict[str, Any]]:
        """Execute action sequence with validation using integrated feedback system.
        Args:
            action_sequence: The action sequence to execute
        Returns:
            - message: The feedback of the action sequence
            - info: Additional information of the action sequence
                - novel_query: Whether the query is a novel query
        """
        # Validate sequence structure
        if not action_sequence.final_action:
            return "Action sequence requires a final action.", {}
        
        # Term() should not have motion actions
        if action_sequence.final_action.action_type == ActionType.TERM and action_sequence.motion_actions:
            return "Term() action should not have motion actions.", {}
        
        info = {}
        messages = []
        
        # Execute motion actions
        for i, action in enumerate(action_sequence.motion_actions):
            success, msg, action_info = self._validate_and_execute_action(action)
            info.update(action_info)
            if not success:
                return msg, info
            messages.append(msg)
        
        # Execute final action
        final_action = action_sequence.final_action
        success, msg, action_info = self._validate_and_execute_action(final_action)
        info.update(action_info)
        if not success:
            return msg, info
        messages.append(msg)
        
        # Return combined messages
        return ", ".join(messages), info
    
    def finish_exploration(self, return_to_origin: bool = True) -> Room:
        """Complete exploration and return final room state.
        Args:
            return_to_origin: Whether the agent should return to the original position and orientation
        Returns:
            The final room state
        """
        if return_to_origin:
            # Return to anchor position and orientation
            self._move_agent_to(self.agent_anchor.pos, self.agent_anchor.name)
            
            # Reset orientation, if agent rotate previously, then agent_anchor is rotated reversely (keep agent always face (0, 1))
            ori_to_deg = {(0, 1): 0, (0, -1): 180, (1, 0): 90, (-1, 0): 270}
            target_deg = ori_to_deg[tuple(self.agent_anchor.ori)]
            self._rotate_agent(target_deg)
        
        # Remove anchor from graph and objects
        anchor_idx = len(self._objects) - 1
        self.exp_graph.size -= 1
        for matrix_name in ['_v_matrix', '_h_matrix', '_v_matrix_working', '_h_matrix_working', '_asked_matrix']:
            matrix = getattr(self.exp_graph, matrix_name)
            matrix = np.delete(matrix, anchor_idx, axis=0)
            matrix = np.delete(matrix, anchor_idx, axis=1)
            setattr(self.exp_graph, matrix_name, matrix)
        
        self._objects.pop()
        return self.current_room
    
    def get_unknown_pairs(self) -> List[Tuple[int, int]]:
        """Get pairs of objects with unknown relationships."""
        return self.exp_graph.get_unknown_pairs()
    
    def get_inferable_pairs(self) -> List[Tuple[int, int]]:
        """Get pairs of objects with inferable relationships."""
        return self.exp_graph.get_inferable_pairs() 