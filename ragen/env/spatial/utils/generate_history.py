"""
Generate exploration history using DFS
"""
import numpy as np
from collections import Counter

from typing import List, Tuple, Dict
from ragen.env.spatial.Base.room import Room
from ragen.env.spatial.Base.graph import DirectionalGraph          
from ragen.env.spatial.Base.relationship import DirPair, Dir, DirectionSystem
from ragen.env.spatial.Base.room import Action, ActionType

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

    def _generate_history_ego(self) -> List[Tuple]:
        """
        Generate exploration history of egocentric exploration
        Generate in an iterative manner:
        1. Move to one object, take action to explore its relationship with other objects
        2. Move to another object, repeat the process

        Heuristics:
            - Always move to the object that has the most unknown relationships
            - Before asking relationship, first determine whether to turn around
        Oracle:
            - Know all relationships
            - Always first turn to the direction where most objects are visible

        1. we use Oracle
        2. Facing north at beginning
        3. No redundancy in history

        Returns:
            history: list of ((obj1, obj2), dir_pair)
            actions: list of Action instances in chronological order

        Exploration heuristic:
         1. At each location, bucket unknown pairs into front/back/left/right
            (based on v>=0, v<0, h>0, h<=0).
         2. Pick the bucket with the most unknowns, QUERY all in it.
         3. If there remain unknowns in the opposite-facing orientation,
            record a ROTATE(180) and QUERY that bucket (with flipped v,h).
         4. When current object has no unknowns, MOVE to the object with most remaining unknowns.
         5. Repeat until no unknown pairs remain, then TERM.
        """
        assert self.room.agent is not None, "Agent is not in the room"

        history, actions = [], []

        while True:
            unknown_pairs = self.room.exp_graph.get_unknown_pairs()
            if not unknown_pairs:
                actions.append(Action(ActionType.TERM))
                break

            # unknowns at current location
            local = [(j, i) for (i, j) in unknown_pairs if i == 0]
            if not local:
                # MOVE to next object with max unknowns
                counts = Counter()
                for i, j in unknown_pairs:
                    counts[i] += 1
                    counts[j] += 1
                next_idx = max(counts, key=counts.get)
                obj_name = self.room.all_objects[next_idx].name
                actions.append(Action(ActionType.MOVE, obj_name))
                # update graph and agent position
                self.room.move_agent(self.room.all_objects[next_idx].pos, obj_name)
                continue
            
            buckets = {'front':[], 'back':[], 'left':[], 'right':[]}
            for i,j in local:
                v, h = self.room.gt_graph._v_matrix[i,j], self.room.gt_graph._h_matrix[i,j]
                if v >= 0: buckets['front'].append((i,j))
                else: buckets['back'].append((i,j))
                if h >= 0: buckets['right'].append((i,j))
                else: buckets['left'].append((i,j))
                
            best_dir, best_pairs = max(buckets.items(), key=lambda kv: len(kv[1]))
            if best_dir == 'back':
                best_dir, best_pairs = 'front', buckets['front']

            # 3. Phase 1: first turn to the best direction and query best_dir
            rotation_map = {'front': 0, 'right': 90, 'back': 180, 'left': 270}
            degree = rotation_map[best_dir]
            if degree != 0:
                actions.append(Action(ActionType.ROTATE, degree))
                self.room.rotate_agent(degree)
            for i,j in best_pairs:
                dir_pair = self.room.get_direction(i,j)[0]
                name1 = self.room.all_objects[i].name
                name2 = self.room.all_objects[j].name
                history.append(((name1, name2), dir_pair))
                actions.append(Action(ActionType.QUERY, name1))
                self.room.exp_graph.add_edge(i, j, dir_pair)

            # 4. Check opposite direction
            opposite = {'front':'back', 'back':'front', 'left':'right', 'right':'left'}[best_dir]
            opp_pairs = buckets[opposite]
            if not opp_pairs:
                continue

            actions.append(Action(ActionType.ROTATE, 180))
            self.room.rotate_agent(180)
            print(self.room.all_objects)
            # phase 2: query opposite bucket
            for i,j in opp_pairs:
                dir_pair = self.room.get_direction(i,j)[0]
                name1 = self.room.all_objects[i].name
                name2 = self.room.all_objects[j].name
                history.append(((name1, name2), dir_pair))
                actions.append(Action(ActionType.QUERY, name1))
                self.room.exp_graph.add_edge(i, j, dir_pair)

            # loop to MOVE or finish
        return history, actions







        
        
        


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
    


if __name__ == "__main__":
    import re
    from ragen.env.spatial.Base.object import Object, Agent
    from ragen.env.spatial.Base.utils.room_utils import generate_room
    from ragen.env.spatial.Base.constant import CANDIDATE_OBJECTS
    from gymnasium.utils import seeding

    rng1 = seeding.np_random(11)[0]
    room = generate_room(
        room_range=(-10, 10),
        n_objects=3,
        candidate_objects=CANDIDATE_OBJECTS,
        generation_type='rand',
        perspective='ego',
        np_random=rng1,
    )
    print(room)

    history, actions = AutoExplore(room, rng1)._generate_history_ego()
    print(history)
    print(actions)
