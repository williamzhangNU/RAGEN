import numpy as np
from typing import Optional
from ragen.env.spatial.Base.tos_base import ActionSequence, EvaluationManager, CognitiveMapManager
from ragen.env.spatial.utils.generate_history import AutoExplore
from ragen.env.spatial.Base.tos_base import Room
from ragen.env.spatial.Base.tos_base.managers.cognitive_map_manager import COGMAP_REQUIRED_INSTRUCTION
from .prompts import *

class Prompter:
    """A class to generate prompts for the SpatialGym environment."""
    ACTIVE_INSTRUCTION = ACTIVE_INSTRUCTION_SHORTER
    PASSIVE_INSTRUCTION = PASSIVE_INSTRUCTION
    EVALUATION_INSTRUCTION = EVALUATION_INSTRUCTION
    SHORT_EXPLORATION_PROMPT = SHORT_EXPLORATION_PROMPT
    SHORT_EVALUATION_PROMPT = SHORT_EVALUATION_PROMPT
    COGMAP_REQUIRED_INSTRUCTION = COGMAP_REQUIRED_INSTRUCTION

    def __init__(self, config, np_random: np.random.RandomState):
        self.config = config
        self.np_random = np_random

    def get_initial_observation_prompt(
            self, 
            room: Room, 
            eval_manager: Optional[EvaluationManager] = None,
            cogmap_manager: Optional[CognitiveMapManager] = None,
            **kwargs
        ) -> str:
        """
        Generates the complete observation prompt including exploration, evaluation, and cognitive map instructions.
        """
        room_desc = room.get_room_description(with_topdown=self.config.prompt_with_topdown)
        
        # Build main prompt based on exploration type
        if self.config.exp_type == 'active':
            exp_instructions = ""
            if cogmap_manager:
                cogmap_instruction = cogmap_manager.get_cognitive_map_instruction()
                exp_instructions += f"\n{cogmap_instruction}"
            exp_instructions += f"## Action Instructions\n{ActionSequence.get_usage_instructions()}\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."
            prompt = self.ACTIVE_INSTRUCTION.format(
                room_info=room_desc,
                exp_instructions=exp_instructions
            )
        else:
            exp_history = f"## Exploration History\n{AutoExplore(room, self.np_random).gen_exp_history()}" if not self.config.prompt_with_topdown else ""
            prompt = self.PASSIVE_INSTRUCTION.format(
                room_info=room_desc,
                exp_history=exp_history
            )

            if eval_manager:
                prompt += f"\n{self.get_evaluation_prompt(eval_manager)}"
        

        prompt += f"\n{self.COGMAP_REQUIRED_INSTRUCTION}" if cogmap_manager else ""
        return prompt

    def get_evaluation_prompt(self, eval_manager: EvaluationManager) -> str:
        """Generate the evaluation prompt."""
        eval_question = eval_manager.get_current_question()
        assert eval_question, "No question found after exploration phase"
        return self.EVALUATION_INSTRUCTION.format(eval_question=f"## Evaluation Question\n{eval_question}")