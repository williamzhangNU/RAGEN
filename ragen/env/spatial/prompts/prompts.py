

# ACTIVE_INSTRUCTION = """\
# # Spatial Exploration Task

# Goal: Build a global understanding of the whole scene: resolve spatial relationships for EVERY object pair across ALL rooms. Stop immediately once complete.

ACTIVE_INSTRUCTION = """\
# Spatial Exploration Task

You are a spatial reasoner operating in a 2D, text-only world. 
Imagine yourself at N by M grid and each object is a point on the grid with integer coordinates (including yourself). 

Goal:
Your objective is to **minimize total COST** while **maximizing global scene understanding**.
Ensure directional relationships between objects are accurate and consistent. 
For distance, angle, and rough 2D coordinates, only maintain approximate understanding.

Observation:
- Observation will include approximate direction and distance.
- Oriented objects also include facing: "faces forward/backward/left/right". Gates report wall side: "gate at front/back/left/right wall". When facing north: forward=north, back=south, right=east, left=west.
- Local relations may appear: e.g., "A is right of B and closer from agent's view".

Multi-room: The scene may have multiple rectangular rooms connected by gates/doors on vertical (N–S) or horizontal (E–W) walls. Stand at a door and use GoThroughDoor(name) to traverse.

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

Observation:
- Observation will include approximate direction and distance.
- Oriented objects also include facing: "faces forward/backward/left/right". Gates report wall side: "gate at front/back/left/right wall". When facing north: forward=north, back=south, right=east, left=west.
- Local relations may appear: e.g., "A is right of B and closer from agent's view".

Multi-room: The scene may have multiple rectangular rooms connected by gates/doors on vertical (N–S) or horizontal (E–W) walls. Stand at a door and use GoThroughDoor(name) to traverse.

Rules: FOV is 90°

## Room Layout
{room_info}

{cogmap_instruction}

## Action Instructions
{action_instructions}

{exp_history}
"""

# NOTE: COGNITION_MAP_INSTRUCTION has been moved to CognitiveMap class for flexible formatting
# The dynamic instruction is now provided by CognitiveMap.get_json_format_instruction()

EVALUATION_INSTRUCTION = "NOTE: Now you return to your starting position and facing north.\n{eval_question}"
SHORT_EXPLORATION_PROMPT = "Please respond with valid actions to explore the rooms."
SHORT_EVALUATION_PROMPT = "Please respond with a valid answer to the question."
