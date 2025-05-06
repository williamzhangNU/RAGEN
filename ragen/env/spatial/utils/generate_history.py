"""
Generate exploration history using DFS
"""
import numpy as np
from typing import List, Optional, Tuple
from ragen.env.spatial.Base.room import Room
from ragen.env.spatial.Base.graph import DirectionalGraph          
from ragen.env.spatial.Base.relationship import DirPair, Dir, DirectionSystem
class AutoExplore:
    """
    Automatically explore the environment
    TODO: use BFS to get shortest path
    """
    
    def __init__(self, room: Room, np_random: np.random.Generator):
        self.room = room.copy()
        self.np_random = np_random

    def _get_exp_history_str(self, history: List[Tuple], perspective: str = 'ego') -> str:
        """
        Get the exploration history in a readable format
        Format: 1. A to <dir> to B
        """
        if perspective == 'ego':
            return "\n".join([f"{i + 1}. {obj1} {DirectionSystem.to_string(dir_pair, perspective=perspective)} to {obj2}" for i, ((obj1, obj2), dir_pair) in enumerate(history)])
        else:
            return "\n".join([f"{i + 1}. {obj1} {DirectionSystem.to_string(dir_pair, perspective=perspective)} to {obj2}" for i, ((obj1, obj2), dir_pair) in enumerate(history)])


    def generate_history(
            self,
            no_inferable: bool = True,
            perspective: str = None,
        ) -> List[Tuple]:
        """
        Generate exploration history using DFS

        Parameters:
            no_inferable : boolean indicating if we are excluding inferable relationships
            perspective: 'allo' or 'ego' 

        Returns:
            List[Tuple]: List of tuples
                - Tuple: ((obj1_id, obj2_id), dir_pair)
        """
        graph = DirectionalGraph(self.room.all_objects, is_explore=True)
        history = []
        if not no_inferable:
            for i in range(len(self.room.all_objects)):
                for j in range(i + 1, len(self.room.all_objects)):
                    (i, j) = (j, i) if self.np_random.random() < 0.5 else (i, j)
                    dir_pair: DirPair = self.room.get_direction(i, j)[0]
                    obj1, obj2 = self.room.all_objects[i].name, self.room.all_objects[j].name
                    history.append(((obj1, obj2), dir_pair))
            return history

        perspective = perspective or ('ego' if self.room.agent is not None else 'allo')
        
        unknown_pairs = graph.get_unknown_pairs()
        while unknown_pairs:
                
            # Choose a random unknown pair
            pair_idx = self.np_random.integers(0, len(unknown_pairs))
            obj1_id, obj2_id = unknown_pairs[pair_idx]
            
            # Random orientation for variety
            if self.np_random.random() < 0.5:
                obj1_id, obj2_id = obj2_id, obj1_id
                
            # Get direction between objects
            dir_pair, _ = self.room.get_direction(obj1_id, obj2_id)
            
            # Add edge to graph
            graph.add_edge(obj1_id, obj2_id, dir_pair)
            
            # Record step in readable form
            obj1_name = self.room.all_objects[obj1_id].name
            obj2_name = self.room.all_objects[obj2_id].name
            history.append(((obj1_name, obj2_name), dir_pair))

            unknown_pairs = graph.get_unknown_pairs()
        

        return self._get_exp_history_str(history, perspective=perspective)
    



    
# ────────────────────────────────────────────────────────────────────
# 📜  SELF‑TESTS (run `python auto_explore.py` to execute)
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import re
    from ragen.env.spatial.Base.object import Object, Agent
    from ragen.env.spatial.Base.utils.room_utils import generate_room
    from ragen.env.spatial.Base.constant import CANDIDATE_OBJECTS
    from gymnasium.utils import seeding


    # ---------------------------------------------------------------
    # Helper: build a very small deterministic Room
    # ---------------------------------------------------------------
    seed = 10
    rng1 = seeding.np_random(seed)[0]
    room = generate_room(
        room_range=(-10, 10),
        n_objects=3,
        candidate_objects=CANDIDATE_OBJECTS,
        generation_type='rand',
        perspective='ego',
        np_random=rng1,
    )
    print(room)

    hist1 = AutoExplore(room).generate_history(np_random=rng1)
    print(hist1)


    print("All AutoExplore self‑tests passed")
