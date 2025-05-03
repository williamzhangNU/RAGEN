"""
Generate exploration history using DFS
"""
import numpy as np
from typing import List, Optional
from ragen.env.spatial.Base.room import Room
from ragen.env.spatial.Base.graph import DirectionalGraph          
from ragen.env.spatial.Base.relationship import DirPair, Dir, DirectionSystem
class AutoExplore:
    """
    Automatically explore the environment using DFS
    """
    def __init__(self, room: Room):
        self.room = room

    def generate_history(
            self,
            np_random,
            no_inferable: bool = True,
            perspective: str = None,
        ) -> str:
        """
        Generate exploration history using DFS

        Parameters:
            no_inferable : boolean indicating if we are excluding inferable relationships
            perspective: 'allo' or 'ego' 
        """
        
        if np_random is None:
            np_random = np.random.default_rng(42)
        graph = DirectionalGraph(self.room.objects, is_explore=True)
        history: List[str] = []
        if not no_inferable:
            lines: List[str] = []
            for i in range(len(self.room.objects)):
                for j in range(i + 1, len(self.room.objects)):
                    dir_pair: DirPair = self.room.get_direction(i, j)[0]
                    obj1, obj2 = self.room.objects[i].name, self.room.objects[j].name
                    lines.append(f"{obj1}, {obj2} {dir_pair}")
            return "\n".join(lines)
        perspective = perspective or ('ego' if self.room.agent is not None else 'allo')
        #perspective = None
        def dfs(cur_graph: DirectionalGraph,
        cur_hist: List[str]) -> Optional[List[str]]:
            unknown_pairs = cur_graph.get_unknown_pairs()     # still‑unknown
            if not unknown_pairs:                            # ✓ finished
                return cur_hist

            # reproducibly shuffle the exploration order
            for idx in np_random.permutation(len(unknown_pairs)):
                obj1_id, obj2_id = unknown_pairs[idx]

                # random orientation – just for variety
                if np_random.random() < 0.5:
                    obj1_id, obj2_id = obj2_id, obj1_id

                # Direction Room (obj1  →  obj2)  in absolute coordinates
                dir_pair, dir_pair_str = self.room.get_direction(obj1_id, obj2_id, perspective=perspective)
                # next‑state graph (deep copy) so we don't need to back‑track
                nxt_graph = cur_graph.copy()
                nxt_graph.add_edge(obj1_id, obj2_id, dir_pair)

                # record the step in a *readable* form
                obj1_name = self.room.objects[obj1_id].name
                obj2_name = self.room.objects[obj2_id].name
                step_str = (
                    f"{obj1_name}, {obj2_name} "
                    f"{dir_pair_str}"
                )

                # recurse
                done_path = dfs(nxt_graph, cur_hist + [step_str])
                if done_path is not None:                     # first full path wins
                    return done_path

            # no branch from this node leads to full exploration
            return None

        full_path = dfs(graph, history) or []
        
        return "\n".join(full_path)
    



    
# ────────────────────────────────────────────────────────────────────
# 📜  SELF‑TESTS (run `python auto_explore.py` to execute)
# ────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import re
    from ragen.env.spatial.Base.object import Object
    from gymnasium.utils import seeding


    # ---------------------------------------------------------------
    # Helper: build a very small deterministic Room
    # ---------------------------------------------------------------
    def _make_tiny_room():
        objs = [
            Object(name="A", pos=(0, 0)),
            Object(name="B", pos=(1, 0)),
            Object(name="C", pos=(0, 1)),
            Object(name="D", pos=(2, 0))
        ]
        return Room(objects=objs)

    room = _make_tiny_room()
    # ---------------------------------------------------------------
    # Deterministic with same seed
    # ---------------------------------------------------------------
    seed = 99
    rng1 = seeding.np_random(seed)[0]
    rng2 = seeding.np_random(seed)[0]
    hist1 = AutoExplore(room).generate_history(np_random=rng1)
    hist2 = AutoExplore(room).generate_history(np_random=rng2)
    print(hist1)
    print(hist2)
    assert hist1 == hist2, "Same seed should yield identical history"


    print("All AutoExplore self‑tests passed")
