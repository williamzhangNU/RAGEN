"""
Generate exploration history using DFS
"""

from ragen.env.spatial.Base.room import Room

class AutoExplore:
    """
    Automatically explore the environment using DFS
    """
    def __init__(self, room: Room):
        self.room = room

    def generate_history(self) -> str:
        """
        Generate exploration history using DFS
        """
        return "EXPLORATION HISTORY PLACEHOLDER"
        # raise NotImplementedError("Not implemented")
