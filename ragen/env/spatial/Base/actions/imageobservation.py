import os
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from PIL import Image

from ..utils.room_utils import initialize_room_from_json
from ..managers.exploration_manager import ExplorationManager
from ..actions import ActionSequence, ObserveAction
from ..core.object import Agent

class ImageObservationSystem:
    """System for handling image-based observations during spatial exploration"""
    
    def __init__(self, json_metadata: Dict[str, Any], image_dir: str, image_size: Tuple[int, int] = (512, 512)):
        """
        Initialize the image observation system.
        
        Args:
            json_metadata: Room metadata with objects and image information
            image_dir: Directory containing the image files
            image_size: Target size for resizing images
        """
        self.json_metadata = json_metadata
        self.image_dir = image_dir
        self.image_size = image_size
        self.image_placeholder = "<image>"
        
        # Initialize room and exploration manager
        self.room = initialize_room_from_json(json_metadata)
        self.exploration_manager = ExplorationManager(self.room)
        
        # Create image filename mapping
        self.image_mapping = self._create_image_mapping()
        
    def _create_image_mapping(self) -> Dict[str, Dict[str, str]]:
        """Create mapping from position/direction to image filenames"""
        mapping = {}
        
        for image_info in self.json_metadata.get('images', []):
            position = image_info['position']
            direction = image_info['direction']
            filename = image_info['filename']
            
            if position not in mapping:
                mapping[position] = {}
            mapping[position][direction] = filename
            
        return mapping
    
    def _get_agent_direction_string(self, agent: Agent) -> str:
        """Convert agent orientation to direction string"""
        ori_to_direction = {
            (0, 1): 'north',
            (1, 0): 'east', 
            (0, -1): 'south',
            (-1, 0): 'west'
        }
        return ori_to_direction.get(tuple(agent.ori), 'north')
    
    def _get_current_position_key(self) -> str:
        """Get current position key for image lookup"""
        agent = self.exploration_manager.exploration_room.agent
        
        # Check if agent is at original camera position
        original_pos = self.json_metadata.get('original_cam_position', {})
        if (abs(agent.pos[0] - original_pos.get('x', 0)) < 0.1 and 
            abs(agent.pos[1] - original_pos.get('z', 0)) < 0.1):
            return 'original'
        
        # Check if agent is at any object position
        for obj in self.exploration_manager.exploration_room.objects:
            if np.allclose(agent.pos, obj.pos, atol=0.1):
                # Find matching object in JSON metadata
                for json_obj in self.json_metadata.get('objects', []):
                    json_pos = np.array([json_obj['position']['x'], json_obj['position']['z']])
                    if np.allclose(obj.pos, json_pos, atol=0.1):
                        return json_obj['name']
        
        return 'original'  # fallback
    
    def _load_observation_images(self) -> List[Image.Image]:
        """Load images for current agent position and orientation"""
        position_key = self._get_current_position_key()
        direction = self._get_agent_direction_string(self.exploration_manager.exploration_room.agent)
        
        images = []
        
        # Get filename for current position and direction
        if position_key in self.image_mapping and direction in self.image_mapping[position_key]:
            filename = self.image_mapping[position_key][direction]
            full_path = os.path.join(self.image_dir, f"{filename}.png")  # assuming .png extension
            
            try:
                img = Image.open(full_path)
                img = img.resize(self.image_size, Image.LANCZOS)
                images.append(img)
            except FileNotFoundError:
                print(f"Warning: Image file not found: {full_path}")
        
        return images
    
    def _create_observation_data(self, observe_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create observation data with images and text"""
        # Load images for current view
        images = self._load_observation_images()
        
        # Create observation string
        obs_str = observe_result.get('answer', 'Nothing to observe.')
        
        # Add image placeholder if images are available
        if images:
            obs_str = f"{self.image_placeholder}\n{obs_str}"
        
        return {
            'obs_str': obs_str,
            'multi_modal_data': {
                self.image_placeholder: images
            },
            'text_only': observe_result.get('answer', 'Nothing to observe.'),
            'visible_objects': observe_result.get('visible_objects', []),
            'relationships': observe_result.get('relationships', [])
        }
    
    def execute_action_string(self, action_string: str) -> Dict[str, Any]:
        """
        Execute action string and return observation with images if applicable.
        
        Args:
            action_string: Raw action string to parse and execute
            
        Returns:
            Dictionary containing execution result and observation data
        """
        # Parse action string
        action_sequence = ActionSequence.parse(action_string)
        if not action_sequence:
            return {
                'success': False,
                'message': f"Failed to parse action string: {action_string}",
                'observation': None
            }
        
        try:
            # Execute action sequence
            message, info = self.exploration_manager.execute_action_sequence(action_sequence)
            
            # Check if final action was Observe
            observation_data = None
            if isinstance(action_sequence.final_action, ObserveAction):
                observation_data = self._create_observation_data(info)
            
            return {
                'success': True,
                'message': message,
                'info': info,
                'observation': observation_data,
                'action_sequence': str(action_sequence)
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f"Action execution failed: {str(e)}",
                'observation': None
            }
    
    def get_exploration_status(self) -> Dict[str, Any]:
        """Get current exploration status"""
        agent = self.exploration_manager.exploration_room.agent
        position_key = self._get_current_position_key()
        direction = self._get_agent_direction_string(agent)
        
        efficiency = self.exploration_manager.get_exploration_efficiency()
        
        return {
            'agent_position': agent.pos.tolist(),
            'agent_orientation': agent.ori.tolist(),
            'agent_direction': direction,
            'position_key': position_key,
            'exploration_efficiency': efficiency,
            'available_images': position_key in self.image_mapping,
            'unknown_pairs': len(self.exploration_manager.get_unknown_pairs()),
            'inferable_pairs': len(self.exploration_manager.get_inferable_pairs())
        }
    
    def get_available_actions_help(self) -> str:
        """Get help text for available actions"""
        return ActionSequence.get_usage_instructions()


# Example usage and testing
def example_usage():
    """Example of how to use the ImageObservationSystem"""
    
    # Sample JSON metadata (from your provided example)
    sample_json = {
        "original_cam_position": {"x": 3, "y": 0.8, "z": 3},
        "room_size": [10, 10],
        "objects": [
            {
                "id": 8979800,
                "name": "lapalma_stil_chair_8979800",
                "model": "lapalma_stil_chair",
                "position": {"x": 1, "y": 0, "z": -4},
                "rotation": {"x": 0, "y": 270, "z": 0}
            },
            {
                "id": 434667,
                "name": "brown_leather_side_chair_434667", 
                "model": "brown_leather_side_chair",
                "position": {"x": -1, "y": 0, "z": 3},
                "rotation": {"x": 0, "y": 270, "z": 0}
            }
        ],
        "images": [
            {"filename": "original_pos_facing_north", "position": "original", "direction": "north"},
            {"filename": "original_pos_facing_east", "position": "original", "direction": "east"},
            {"filename": "obj_lapalma_stil_chair_8979800_facing_north", "position": "lapalma_stil_chair_8979800", "direction": "north"},
            {"filename": "obj_brown_leather_side_chair_434667_facing_south", "position": "brown_leather_side_chair_434667", "direction": "south"}
        ]
    }
    
    # Initialize system
    image_dir = "path/to/images"  # Replace with actual image directory
    obs_system = ImageObservationSystem(sample_json, image_dir)
    
    # Example action sequences
    test_actions = [
        "Observe()",  # Simple observation
        "Move(lapalma_stil_chair); Observe()",  # Move then observe
        "Move(brown_leather_side_chair), Rotate(180); Observe()",  # Move, rotate, then observe
        "Return(); Observe()",  # Return to start then observe
        "Term()"  # Terminate exploration
    ]
    
    for action_str in test_actions:
        print(f"\n--- Executing: {action_str} ---")
        result = obs_system.execute_action_string(action_str)
        
        if result['success']:
            print(f"Success: {result['message']}")
            if result['observation']:
                obs = result['observation']
                print(f"Observation: {obs['text_only']}")
                print(f"Images available: {len(obs['multi_modal_data']['<image>'])}")
                print(f"Visible objects: {obs['visible_objects']}")
        else:
            print(f"Failed: {result['message']}")
        
        # Print exploration status
        status = obs_system.get_exploration_status()
        print(f"Agent at: {status['position_key']}, facing: {status['agent_direction']}")
        print(f"Coverage: {status['exploration_efficiency']['coverage']:.2%}")


if __name__ == "__main__":
    example_usage()