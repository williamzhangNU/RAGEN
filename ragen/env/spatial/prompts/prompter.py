import numpy as np
from typing import Optional
from ragen.env.spatial.Base.tos_base import ActionSequence, EvaluationManager, CognitiveMapManager
from ragen.env.spatial.Base.tos_base import Room, Agent
from ragen.env.spatial.Base.tos_base.utils.room_utils import get_room_description
from ragen.env.spatial.Base.tos_base.managers.cognitive_map_manager import COGMAP_EXP_REQUIRED_INSTRUCTION, COGMAP_EVAL_REQUIRED_INSTRUCTION
from ragen.env.spatial.Base.tos_base.core.relationship import PairwiseRelationship, PairwiseRelationshipDiscrete
from .prompts import *

class Prompter:
    """A class to generate prompts for the SpatialGym environment."""
    ACTIVE_INSTRUCTION = ACTIVE_INSTRUCTION
    PASSIVE_INSTRUCTION = PASSIVE_INSTRUCTION
    EVALUATION_INSTRUCTION = EVALUATION_INSTRUCTION
    SHORT_EXPLORATION_PROMPT = SHORT_EXPLORATION_PROMPT
    SHORT_EVALUATION_PROMPT = SHORT_EVALUATION_PROMPT
    COGMAP_EXP_REQUIRED_INSTRUCTION = COGMAP_EXP_REQUIRED_INSTRUCTION
    COGMAP_EVAL_REQUIRED_INSTRUCTION = COGMAP_EVAL_REQUIRED_INSTRUCTION

    def __init__(self, config, np_random: np.random.RandomState):
        self.config = config
        self.np_random = np_random

    def get_initial_observation_prompt(
            self, 
            room: Room, 
            agent: Agent,
            eval_manager: Optional[EvaluationManager] = None,
            cogmap_manager: Optional[CognitiveMapManager] = None,
            **kwargs
        ) -> str:
        """
        Generates the complete observation prompt including exploration, evaluation, and cognitive map instructions.
        """
        room_desc = get_room_description(room, agent, with_topdown=self.config.prompt_config["topdown"])
        cogmap_instruction = cogmap_manager.get_cognitive_map_instruction() if cogmap_manager else ""
        # Build main prompt based on exploration type
        if self.config.exp_type == 'active':
            exp_instructions = (
                ActionSequence.get_usage_instructions()
                + f"\n\n{PairwiseRelationshipDiscrete.prompt()}"
                + f"\n\nYou have a maximum of {self.config.max_exp_steps} exploration steps."
            )
            active_instruction = self.ACTIVE_INSTRUCTION
            prompt = active_instruction.format(
                room_info=room_desc,
                cogmap_instruction=cogmap_instruction,
                exp_instructions=exp_instructions
            )
            prompt += f"\n{self.COGMAP_EXP_REQUIRED_INSTRUCTION}" if cogmap_manager else ""
        else:
            exp_history = f"## Exploration History\n{kwargs.get('exp_history', '')}" if not self.config.prompt_config["topdown"] else ""
            prompt = self.PASSIVE_INSTRUCTION.format(
                room_info=room_desc,
                cogmap_instruction=cogmap_instruction,
                # action_instructions=ActionSequence.get_usage_instructions() + f"\n\n{PairwiseRelationship.prompt()}",
                action_instructions=f"{PairwiseRelationshipDiscrete.prompt()}",
                exp_history=exp_history
            )
            prompt += f"\n{self.get_evaluation_prompt(eval_manager)}"
            prompt += f"\n{self.COGMAP_EVAL_REQUIRED_INSTRUCTION}" if cogmap_manager else ""
        
        return prompt

    def get_evaluation_prompt(self, eval_manager: EvaluationManager) -> str:
        """Generate the evaluation prompt."""
        eval_question = eval_manager.get_current_question()
        assert eval_question, "No question found after exploration phase"
        return self.EVALUATION_INSTRUCTION.format(eval_question=f"## Evaluation Question\n{eval_question}")