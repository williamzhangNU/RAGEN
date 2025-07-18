"""
Generate exploration history using DFS
"""
import numpy as np
from collections import Counter

from typing import List, Tuple, Dict
from ragen.env.spatial.Base.tos_base import (
    Room,
    DirectionalGraph,
    DirPair,
    Dir,
    DirectionSystem,
    BaseAction,
    MoveAction,
    RotateAction,
    ObserveAction,
    TermAction,
    ExplorationManager,
    Object,
    Agent,
)
from ragen.env.spatial.utils.action_utils import action_results_to_text

class AutoExplore:
    """
    Automatically explore the environment
    TODO: use BFS to get shortest path
    """
    
    def __init__(self, room: Room, np_random: np.random.Generator):
        self.room = room.copy()
        self.np_random = np_random
        self.exp_manager = ExplorationManager(self.room)

    def _generate_history_passive(self) -> List[List]:
        """
        Generate exploration history using ExplorationManager.
        Returns:
            action_results_per_turn: ActionResults for each turn
        """
        assert self.room.agent is not None, "Agent is not in the room"

        action_results_per_turn, actions_in_a_turn = [], []
        agent_idx = self.exp_manager._get_index(self.room.agent.name)

        while True:
            unknown_pairs = self.exp_manager.get_unknown_pairs()
            if not unknown_pairs:
                # no unknown pairs --> terminate
                action_results_per_turn.append([])  # Empty turn for termination
                break

            # Get unknowns involving agent
            agent_unknown_pairs = [(pair[1], pair[0]) if pair[0] == agent_idx else pair 
                                 for pair in unknown_pairs if agent_idx in pair]
            
            if not agent_unknown_pairs:
                # Move to best position (next object)
                counts = Counter()
                for i, j in unknown_pairs:
                    counts[i] += 1
                    counts[j] += 1
                next_obj_idx = max(counts, key=counts.get)
                obj_name = self.exp_manager.objects[next_obj_idx].name
                
                # Turn to face target before moving
                rotation = self._find_rotation_to_see_object(next_obj_idx)
                if rotation != 0:
                    result = self.exp_manager.execute_action(RotateAction(rotation))
                    assert result.success, f"Failed to rotate: {result.message}"
                    actions_in_a_turn.append(result)
                
                # Move to target object
                result = self.exp_manager.execute_action(MoveAction(obj_name))
                assert result.success, f"Failed to move: {result.message}"
                actions_in_a_turn.append(result)
                continue
            
            # Check if there are already visible unknowns in current direction
            current_visible_count = sum(1 for target_idx, _ in agent_unknown_pairs 
                                      if self._would_be_visible_after_rotation(target_idx, 0))
            
            # turn to best direction only if current direction has no visible unknowns
            if current_visible_count == 0:
                best_direction = self._find_best_direction(agent_unknown_pairs)
                if best_direction != 0:
                    result = self.exp_manager.execute_action(RotateAction(best_direction))
                    assert result.success, f"Failed to rotate: {result.message}"
                    actions_in_a_turn.append(result)
            
            # Perform observation
            result = self.exp_manager.execute_action(ObserveAction())
            assert result.success, f"Failed to observe: {result.message}"
            actions_in_a_turn.append(result)
            
            # Observation marks end of turn
            action_results_per_turn.append(actions_in_a_turn)
            actions_in_a_turn = []

        return action_results_per_turn
    

    def _would_be_visible_after_rotation(self, target_idx: int, rotation: int) -> bool:
        """Check visibility after rotation"""
        agent = self.exp_manager.exploration_room.agent
        target = self.exp_manager.objects[target_idx]
        if target.name == agent.name:
            return True
        
        rotations = {
            0: np.array([[1, 0], [0, 1]]),
            90: np.array([[0, -1], [1, 0]]),
            270: np.array([[0, 1], [-1, 0]]),
            180: np.array([[-1, 0], [0, -1]]),
        }
        rotated_ori = agent.ori @ rotations[rotation]
        temp_agent = Object(name='temp', pos=agent.pos, ori=rotated_ori)  
        return BaseAction._is_visible(temp_agent, target)
    
    def _find_best_direction(self, agent_unknown_pairs: List[Tuple[int, int]]) -> int:
        """Find direction with most visible unknowns"""
        best_count, best_direction = 0, 0
        
        for rotation in [0, 90, 180, 270]:
            count = sum(1 for target_idx, _ in agent_unknown_pairs 
                       if self._would_be_visible_after_rotation(target_idx, rotation))
            if count > best_count:
                best_count, best_direction = count, rotation
        
        return best_direction

    def _find_rotation_to_see_object(self, target_idx: int) -> int:
        """Find rotation needed to see specific object"""
        for rotation in [0, 90, 180, 270]:
            if self._would_be_visible_after_rotation(target_idx, rotation):
                return rotation
        return 0
    
    def _format_history_to_obs(self, action_results_per_turn: List[List]) -> Dict:
        """Convert action results to obs format with multi_modal_data."""
        turn_strings = []
        
        for turn_num, turn_results in enumerate(action_results_per_turn, 1):
            if not turn_results:  # Empty turn (termination)
                continue
                
            turn_text = action_results_to_text(turn_results)
            turn_strings.append(f"{turn_num}. {turn_text}")
        
        obs = "\n".join(turn_strings)
        return obs
            
    def gen_exp_history(self) -> str:
        """Generate exploration history in obs format."""
        return self._format_history_to_obs(self._generate_history_passive())

if __name__ == "__main__":
    import re
    from ragen.env.spatial.Base.tos_base import Object, Agent, generate_room, CANDIDATE_OBJECTS
    from gymnasium.utils import seeding

    BaseAction.set_field_of_view(180)
    rng1 = seeding.np_random(2)[0]
    room = generate_room(
        room_range=(-5, 5),
        n_objects=3,
        candidate_objects=CANDIDATE_OBJECTS,
        generation_type='rand',
        np_random=rng1,
    )
    print(room)
    room.plot(render_mode='text')

    explorer = AutoExplore(room, rng1)
    exploration_str = explorer.gen_exp_history()
    print(exploration_str)
