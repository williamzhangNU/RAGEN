

# ACTIVE_INSTRUCTION = """\
# # Spatial Exploration Task

# Goal: Build a global understanding of the whole scene: resolve spatial relationships for EVERY object pair across ALL rooms. Stop immediately once complete.

ACTIVE_INSTRUCTION = """\
# Spatial Exploration

You are a spatial reasoner in a 2D, text-only N×M grid. Every object including you is a point at integer (x, y) coordinates.

## Multi-room rules
- You cannot see objects in other rooms.
- You cannot see through a door unless you are standing on it. When at a door, door will be open and invisible, and you can see objects from both connected rooms (within your FOV).
- Rooms connect via doors on vertical or horizontal walls, e.g., door on front/left wall. 

## Objective
- Minimize total COST.
- Achieve complete spatial understanding: every pair of objects (including your initial position) must be assigned exactly one relation from:
{allo_bins}

## Observation
You egocentric observation rule is provided in following format:
{observation_instructions}

## Rules
- Achieve complete coverage with the fewest steps;
- Prefer actions that reveal more unknowns; avoid redundancy
- FOV is 90°, you can NOT see objects outside your FOV.
- Track your current and initial pose

## Room Layout
{room_info}

## Action Instructions
{exp_instructions}
"""


PASSIVE_INSTRUCTION = """\
# Spatial Understanding Task

You are a spatial reasoner in a 2D, text-only N×M grid. Every object including you is a point at integer (x, y) coordinates.

## Multi-room rules
- You cannot see objects in other rooms.
- You cannot see through a door unless you are standing on it. When at a door, door will be open and invisible, and you can see objects from both connected rooms (within your FOV).
- Rooms connect via doors on vertical or horizontal walls, e.g., door on front wall. 

## Observation
You egocentric observation rule is provided in following format:
{observation_instructions}

## Rules
- FOV is 90°, you can NOT see objects outside your FOV.
- Track your current and initial pose

## Room Layout
{room_info}

{exp_history}
"""

# NOTE: COGNITION_MAP_INSTRUCTION has been moved to CognitiveMap class for flexible formatting
# The dynamic instruction is now provided by CognitiveMap.get_json_format_instruction()

EVALUATION_INSTRUCTION = "\n{eval_question}"
SHORT_EXPLORATION_PROMPT = "Please respond with valid actions to explore the rooms."
SHORT_EVALUATION_PROMPT = "Please respond with a valid answer to the question."
