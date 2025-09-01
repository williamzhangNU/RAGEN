

# ACTIVE_INSTRUCTION = """\
# # Spatial Exploration Task

# Goal: Build a global understanding of the whole scene: resolve spatial relationships for EVERY object pair across ALL rooms. Stop immediately once complete.

ACTIVE_INSTRUCTION = """\
# Spatial Exploration

You are a spatial reasoner in a 2D, text-only N×M grid. Every object including you is a point at integer (x, y) coordinates.

## Multi-room rules
- You cannot see objects in other rooms.
- You cannot see through a door unless you are standing on it. When at a door, it's open and invisible.
- Rooms connect via doors on vertical (front/back) or horizontal (left/right) walls.
- When standing on a door, you can see objects from both connected rooms (within FOV).

## Objective
- Minimize total COST.
- Achieve complete spatial understanding: every pair of objects must be assigned exactly one relation from:
{allo_bins}

## Observation
You egocentric observation is provided in following format:
{observation_instructions}

## Rules
- Achieve complete coverage with the fewest steps;
- Prefer actions that reveal more unknowns; avoid redundancy
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

You are a spatial reasoner in a 2D, text-only N×M grid. Every object including you is a point at integer (x, y) coordinates.

## Multi-room rules
- You cannot see objects in other rooms.
- You cannot see through a door unless you are standing on it. When at a door, it's open and invisible.
- Rooms connect via doors on vertical (front/back) or horizontal (left/right) walls.
- When standing on a door, you can see objects from both connected rooms (within FOV).

## Observation
You egocentric observation is provided in following format:
{observation_instructions}

## Rules
- FOV is 90°
- Track your current and initial pose

## Room Layout
{room_info}

{cogmap_instruction}

{exp_history}
"""

# NOTE: COGNITION_MAP_INSTRUCTION has been moved to CognitiveMap class for flexible formatting
# The dynamic instruction is now provided by CognitiveMap.get_json_format_instruction()

EVALUATION_INSTRUCTION = "\n{eval_question}"
SHORT_EXPLORATION_PROMPT = "Please respond with valid actions to explore the rooms."
SHORT_EVALUATION_PROMPT = "Please respond with a valid answer to the question."
