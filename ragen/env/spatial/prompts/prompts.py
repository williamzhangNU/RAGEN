

# ACTIVE_INSTRUCTION = """\
# # Spatial Exploration Task

# Goal: Build a global understanding of the whole scene: resolve spatial relationships for EVERY object pair across ALL rooms. Stop immediately once complete.

ACTIVE_INSTRUCTION = """\
# Spatial Exploration Task

You are a spatial reasoner operating in a 2D, text-only world. 
Imagine yourself at N by M grid and each object is a point on the grid with integer coordinates (including yourself). 

Goal:
Your objective is to **minimize total COST** while gaining knowledge of spatial relationships between each pair of objects.
For each relation, you should determine which allocentric bin it corresponds to. The required spatial relationship only needs to fall into the same bin category, not be precisely accurate.

{allo_bins}

Observation:
{observation_instructions}

Multi-room: 
- You can not look through the gate when you are not at the gate.
- Rooms are connected by gates/doors on vertical (N–S) or horizontal (E–W) walls. When you stand at a door, you can see objects from both connected rooms (within FOV).

Rules:
- Achieve complete coverage with the fewest steps; continue only while any pair is unknown
- Prefer actions that reveal many unknowns; avoid redundancy
- FOV is 90°
- Track your current and initial pose

## Room Layout
{room_info}

{cogmap_instruction}

## Action Instructions
{exp_instructions}

After exploration, you will return to your starting position facing north.
"""


PASSIVE_INSTRUCTION = """\
# Spatial Understanding Task

You will be given a multi-room layout and a tour (you return to start). Then answer the question.
For spatial relationships, you need to determine which allocentric bin each corresponds to. The required spatial relationship only needs to fall into the same bin category, not be precisely accurate.

{allo_bins}

Observation:
{observation_instructions}

Multi-room: 
- You can not look through the gate when you are not at the gate.
- Rooms are connected by gates/doors on vertical (N–S) or horizontal (E–W) walls. When you stand at a door, you can see objects from both connected rooms (within FOV).

Rules: FOV is 90°

## Room Layout
{room_info}

{cogmap_instruction}

{exp_history}
"""

# NOTE: COGNITION_MAP_INSTRUCTION has been moved to CognitiveMap class for flexible formatting
# The dynamic instruction is now provided by CognitiveMap.get_json_format_instruction()

EVALUATION_INSTRUCTION = "NOTE: Now you return to your starting position and face north.\n{eval_question}"
SHORT_EXPLORATION_PROMPT = "Please respond with valid actions to explore the rooms."
SHORT_EVALUATION_PROMPT = "Please respond with a valid answer to the question."
