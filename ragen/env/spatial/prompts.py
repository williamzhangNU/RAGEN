ACTIVE_INSTRUCTION = """\
# Spatial Exploration Task

Explore the room to know the spatial relationships between objects.
You should explore the room efficiently:
- Avoid redundant actions
- Terminate IMMEDIATELY when you know spatial relationships between all object pairs.

After exploration, you will return to your starting position and orientation.
Then you need to answer question(s) based on your exploration.

## Room Layout
{room_info}

{exp_instructions}
"""

PASSIVE_INSTRUCTION = """\
# Spatial Understanding Task

You will be given a room layout and a tour around the room. 
NOTE: After the tour, you will return to your starting position and orientation.
Then you need to answer the question based on the tour.

## Room Layout
{room_info}

{exp_history}
"""

EVALUATION_INSTRUCTION = """\
You return to your starting position and orientation.
{eval_question}
"""

SHORT_EXPLORATION_PROMPT = """\
Please respond with valid actions to explore the room.
"""

SHORT_EVALUATION_PROMPT = """\
Please respond with a valid answer to the question.
"""