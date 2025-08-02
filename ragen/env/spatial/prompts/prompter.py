import numpy as np
from ragen.env.spatial.Base.tos_base import ActionSequence
from ragen.env.spatial.utils.generate_history import AutoExplore
from ragen.env.spatial.Base.tos_base import Room
from .prompts import *

class Prompter:
    """A class to generate prompts for the SpatialGym environment."""
    ACTIVE_INSTRUCTION = ACTIVE_INSTRUCTION_SHORTER
    PASSIVE_INSTRUCTION = PASSIVE_INSTRUCTION
    COGNITION_MAP_INSTRUCTION = COGNITION_MAP_INSTRUCTION
    EVALUATION_INSTRUCTION = EVALUATION_INSTRUCTION
    SHORT_EXPLORATION_PROMPT = SHORT_EXPLORATION_PROMPT
    SHORT_EVALUATION_PROMPT = SHORT_EVALUATION_PROMPT

    def __init__(self, config):
        self.config = config

    def get_initial_observation_prompt(self, room: Room, np_random: np.random.RandomState, **kwargs) -> str:
        """
        Generates the initial observation prompt based on the exploration type.
        """
        room_desc = room.get_room_description(with_topdown=self.config.prompt_with_topdown)
        if self.config.exp_type == 'active':
            exp_instructions = f"## Action Instructions\n{ActionSequence.get_usage_instructions()}\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."
            
            return self.ACTIVE_INSTRUCTION.format(
              room_info=room_desc,
              exp_instructions=exp_instructions
            )
        else:
            exp_history = f"## Exploration History\n{AutoExplore(room, np_random).gen_exp_history()}" if not self.config.prompt_with_topdown else ""
            obs = self.PASSIVE_INSTRUCTION.format(
              room_info=room_desc,
              exp_history=exp_history
            )
            return obs

    def get_evaluation_prompt(self, eval_question: str) -> str:
        """
        Generates the evaluation prompt, optionally including the cognitive map instructions.
        """
        if self.config.prompt_with_cogmap:
            return f"{self.COGNITION_MAP_INSTRUCTION}\n{self.EVALUATION_INSTRUCTION.format(eval_question=eval_question)}"
        return self.EVALUATION_INSTRUCTION.format(eval_question=eval_question)
