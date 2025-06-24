from enum import Enum
from typing import Optional, Union, Tuple, List, Dict, Any
import re


class ActionType(Enum):
    """Types of actions in spatial exploration"""
    MOVE = "Move"
    ROTATE = "Rotate" 
    RETURN = "Return"
    QUERY = "Query"
    TERM = "Term"


class Action:
    """Single action in spatial exploration"""
    
    def __init__(self, action_type: ActionType, parameters: Optional[Union[str, int]] = None):
        self.action_type = action_type
        self.parameters = parameters
    
    def __repr__(self):
        if self.parameters is None:
            return f"{self.action_type.value}()"
        return f"{self.action_type.value}({self.parameters})"
    
    def is_final(self) -> bool:
        """Check if this is a final action (QUERY or TERM)"""
        return self.action_type in [ActionType.QUERY, ActionType.TERM]
    
    def get_feedback(self, success: bool, error_type: str = None, **kwargs) -> str:
        """Generate feedback message based on action execution result.
        
        Args:
            success: Whether the action was executed successfully
            error_type: Type of error ("not_found", "not_visible", "invalid_degree", etc.)
            execution_info: Additional execution information (e.g., query results)
            
        Returns:
            feedback message
        """
        
        if not success:
            return self.get_error(error_type)
        
        return self._get_success_message(**kwargs)
    
    def _get_error(self, error_type: str = None) -> str:
        """Get error message for this action type and error."""
        if self.action_type == ActionType.MOVE:
            if error_type == "not_found":
                return f"Cannot move to '{self.parameters}': object not found."
            elif error_type == "not_visible":
                return f"Cannot move to '{self.parameters}': not visible."
            return f"Cannot move to '{self.parameters}': execution failed."
        elif self.action_type == ActionType.ROTATE:
            if error_type == "invalid_degree":
                return f"Cannot rotate by {self.parameters}°: only 0, 90, 180, 270 allowed."
            return f"Cannot rotate by {self.parameters}°: execution failed."
        elif self.action_type == ActionType.QUERY:
            if error_type == "not_found":
                return f"Cannot query '{self.parameters}': object not found."
            elif error_type == "not_visible":
                return f"Cannot query '{self.parameters}': not visible."
            return f"Cannot query '{self.parameters}': execution failed."
        elif self.action_type == ActionType.RETURN:
            return "Cannot return to anchor: execution failed."
        elif self.action_type == ActionType.TERM:
            return "Cannot terminate exploration: execution failed."
        else:
            return f"Cannot execute {self.action_type.value}: execution failed."
    
    def _get_success_message(self, **kwargs) -> str:
        """Get success message for this action type."""
        if self.action_type == ActionType.MOVE:
            return f"Moved to {self.parameters}."
        elif self.action_type == ActionType.ROTATE:
            return f"Rotated by {self.parameters}°."
        elif self.action_type == ActionType.RETURN:
            return "Returned to anchor."
        elif self.action_type == ActionType.QUERY:
            return f"Queried {self.parameters}. Answer: {kwargs['answer']}"
        elif self.action_type == ActionType.TERM:
            return "Exploration terminated."
        else:
            return f"Executed {self.action_type.value}."


class ActionSequence:
    """Sequence of actions for spatial exploration"""
    
    def __init__(self, motion_actions: List[Action] = None, final_action: Action = None):
        self.motion_actions = motion_actions or []
        self.final_action = final_action
    
    def __repr__(self):
        motions = ", ".join(str(action) for action in self.motion_actions)
        final = str(self.final_action) if self.final_action else "None"
        return f"ActionSequence(motions=[{motions}], final={final})"

    @classmethod
    def parse(cls, action_str: str) -> Optional['ActionSequence']:
        """Parse action string into ActionSequence
        
        Format: "Move(A), Rotate(90); Query(B)" or just "Query(B)" or "Term()"
        """
        parts = action_str.split(';')
        if len(parts) > 2:
            return None
            
        motion_actions = []
        final_action = None
        
        # Parse motion actions (if present)
        if len(parts) == 2:
            motion_str = parts[0].strip()
            for item in [i.strip() for i in motion_str.split(',') if i.strip()]:
                action = cls._parse_single_action(item)
                if not action or action.is_final():
                    return None
                motion_actions.append(action)
        
        # Parse final action
        final_str = parts[-1].strip()
        final_action = cls._parse_single_action(final_str)
        if not final_action or not final_action.is_final():
            return None
            
        # Term() should not have motion actions
        if final_action.action_type == ActionType.TERM and motion_actions:
            return None
            
        return cls(motion_actions, final_action)
    
    @staticmethod
    def _parse_single_action(action_str: str) -> Optional[Action]:
        """Parse a single action string"""
        if match := re.match(r"Move\(([A-Za-z0-9_-]+)\)", action_str):
            return Action(ActionType.MOVE, match.group(1))
        elif match := re.match(r"Rotate\(([0-9-]+)\)", action_str):
            degree = int(match.group(1))
            # Allow parsing any integer, validation is done in ExplorationManager
            return Action(ActionType.ROTATE, degree)
        elif action_str == "Return()":
            return Action(ActionType.RETURN)
        elif match := re.match(r"Query\(([A-Za-z0-9_-]+)\)", action_str):
            return Action(ActionType.QUERY, match.group(1))
        elif action_str == "Term()":
            return Action(ActionType.TERM)
        return None 