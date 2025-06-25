from abc import ABC, abstractmethod
from typing import Optional, List
import re


class BaseAction(ABC):
    """Base class for all actions"""
    
    # Class attributes to be overridden by subclasses
    format_desc = ""
    description = ""
    example = ""
    
    def __init__(self, parameters=None):
        self.parameters = parameters
    
    @abstractmethod
    def format_pattern(self) -> str:
        """Return regex pattern for parsing this action"""
        pass
    
    @abstractmethod
    def success_message(self, **kwargs) -> str:
        """Return success message for this action"""
        pass
    
    @abstractmethod
    def error_message(self, error_type: str) -> str:
        """Return error message for this action"""
        pass
    
    def is_final(self) -> bool:
        """Check if this is a final action (ends the sequence)"""
        return False
    
    def is_term(self) -> bool:
        """Check if this is a termination action"""
        return False
    
    @classmethod
    def parse(cls, action_str: str):
        """Parse action string and return instance if matches"""
        instance = cls()
        if match := re.match(instance.format_pattern(), action_str):
            return cls(*match.groups())
        return None

    def get_feedback(self, success: bool, error_type: str = None, **kwargs) -> str:
        """Generate feedback based on execution result"""
        if success:
            return self.success_message(**kwargs)
        return self.error_message(error_type)


class MoveAction(BaseAction):
    """Move to a target object"""
    
    format_desc = "Move(object_name)"
    description = "Move to a specific object in the room"
    example = "Move(table)"
    
    def __init__(self, target=None):
        super().__init__(target)
        self.target = target
    
    def format_pattern(self) -> str:
        return r"^Move\(([A-Za-z0-9_-]+)\)$"
    
    def success_message(self, **kwargs) -> str:
        return f"Moved to {self.target}."
    
    def error_message(self, error_type: str) -> str:
        if error_type == "not_found":
            return f"Cannot move to '{self.target}': object not found."
        elif error_type == "not_visible":
            return f"Cannot move to '{self.target}': not visible."
        return f"Cannot move to '{self.target}': execution failed."
    
    def __repr__(self):
        return f"Move({self.target})"


class RotateAction(BaseAction):
    """Rotate by specified degrees"""
    
    format_desc = "Rotate(degrees)"
    description = "Rotate by specified degrees (0, 90, 180, 270)"
    example = "Rotate(90)"
    
    def __init__(self, degrees=None):
        super().__init__(degrees)
        self.degrees = int(degrees) if degrees else None
    
    def format_pattern(self) -> str:
        return r"^Rotate\(([0-9-]+)\)$"
    
    def success_message(self, **kwargs) -> str:
        return f"Rotated by {self.degrees}°."
    
    def error_message(self, error_type: str) -> str:
        if error_type == "invalid_degree":
            return f"Cannot rotate by {self.degrees}°: only 0, 90, 180, 270 allowed."
        return f"Cannot rotate by {self.degrees}°: execution failed."
    
    def __repr__(self):
        return f"Rotate({self.degrees})"


class ReturnAction(BaseAction):
    """Return to anchor position"""
    
    format_desc = "Return()"
    description = "Return to the starting anchor position"
    example = "Return()"
    
    def format_pattern(self) -> str:
        return r"^Return\(\)$"
    
    def success_message(self, **kwargs) -> str:
        return "Returned to anchor."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot return to anchor: execution failed."
    
    def __repr__(self):
        return "Return()"


class QueryAction(BaseAction):
    """Query spatial relationship with target object"""
    
    format_desc = "Query(object_name)"
    description = "Query spatial relationship of a target object relative to your current position"
    example = "Query(lamp)"
    
    def __init__(self, target=None):
        super().__init__(target)
        self.target = target
    
    def format_pattern(self) -> str:
        return r"^Query\(([A-Za-z0-9_-]+)\)$"
    
    def success_message(self, **kwargs) -> str:
        return f"Queried {self.target}. Answer: {kwargs.get('answer', 'N/A')}"
    
    def error_message(self, error_type: str) -> str:
        if error_type == "not_found":
            return f"Cannot query '{self.target}': object not found."
        elif error_type == "not_visible":
            return f"Cannot query '{self.target}': not visible."
        return f"Cannot query '{self.target}': execution failed."
    
    def is_final(self) -> bool:
        return True
    
    def __repr__(self):
        return f"Query({self.target})"


class TermAction(BaseAction):
    """Terminate exploration"""
    
    format_desc = "Term()"
    description = "Terminate the exploration phase"
    example = "Term()"
    
    def format_pattern(self) -> str:
        return r"^Term\(\)$"
    
    def success_message(self, **kwargs) -> str:
        return "Exploration terminated."
    
    def error_message(self, error_type: str) -> str:
        return "Cannot terminate exploration: execution failed."
    
    def is_final(self) -> bool:
        return True
    
    def is_term(self) -> bool:
        return True
    
    def __repr__(self):
        return "Term()"


# Action registry for easy lookup
ACTION_CLASSES = [MoveAction, RotateAction, ReturnAction, QueryAction, TermAction]


class ActionSequence:
    """Sequence of actions for spatial exploration"""
    
    def __init__(self, motion_actions: List[BaseAction] = None, final_action: BaseAction = None):
        self.motion_actions = motion_actions or []
        self.final_action = final_action
    
    def __repr__(self):
        motions = ", ".join(str(action) for action in self.motion_actions)
        final = str(self.final_action) if self.final_action else "None"
        return f"ActionSequence(motions=[{motions}], final={final})"

    @classmethod
    def parse(cls, action_str: str) -> Optional['ActionSequence']:
        """Parse action string into ActionSequence"""
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
        if isinstance(final_action, TermAction) and motion_actions:
            return None
            
        return cls(motion_actions, final_action)
    
    @staticmethod
    def _parse_single_action(action_str: str) -> Optional[BaseAction]:
        """Parse a single action string using registered action classes"""
        for action_class in ACTION_CLASSES:
            if action := action_class.parse(action_str):
                return action
        return None
    
    @staticmethod
    def get_usage_instructions() -> str:
        """Get usage instructions for action sequences"""
        motion_actions = [cls for cls in ACTION_CLASSES if not cls().is_final()]
        final_actions = [cls for cls in ACTION_CLASSES if cls().is_final()]
        
        instructions = (
            "## Action Format\n"
            "Use semicolon to separate movement actions from final query/term action.\n"
            "Multiple movements can be chained with commas.\n\n"
            "## Available Actions\n"
        )
        
        if motion_actions:
            instructions += "### Movement Actions\n"
            instructions += "\n".join(f"- {cls.format_desc}: {cls.description}" for cls in motion_actions)
            instructions += "\n\n"
        
        if final_actions:
            instructions += "### Final Actions\n"
            instructions += "\n".join(f"- {cls.format_desc}: {cls.description}" for cls in final_actions)
            instructions += "\n\n"
        
        instructions += (
            "## Examples\n"
            f"Simple query: {QueryAction.example}\n"
            f"Move then query: {MoveAction.example}; {QueryAction.example}\n"
            f"Multiple moves: {MoveAction.example}, {RotateAction.example}; {QueryAction.example}\n"
            f"Return to start: {ReturnAction.example}; {QueryAction.example}\n"
            f"Terminate: {TermAction.example}\n\n"
            "## Rules\n"
            "- Last action must be Query() or Term()\n"
            "- Term() cannot have movement actions before it\n"
            "- You have a field of view of 90 degrees, 45 to the left and 45 to the right\n"
            "- You can only query or move to objects that are within your field of view\n"
        )
        
        return instructions
