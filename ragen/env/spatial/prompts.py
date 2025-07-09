ACTIVE_INSTRUCTION = """\
# Spatial Exploration Task

Explore the room to know the spatial relationships between objects.
You should explore the room efficiently:
- Avoid redundant actions
- Terminate IMMEDIATELY when you know spatial relationships between all object pairs.

## Room Layout
{room_info}

{exp_instructions}
"""

PASSIVE_INSTRUCTION = """\
# Spatial Understanding Task

You will be given a room layout and a tour around the room. 
After the tour, you will return to your starting position and orientation.
Then you need to answer the question based on the tour.

## Room Layout
{room_info}

{exp_history}

{eval_question}
"""