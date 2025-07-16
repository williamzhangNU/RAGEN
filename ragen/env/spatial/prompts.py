ACTIVE_INSTRUCTION = """\
# Spatial Exploration Task

Your goal: Learn ALL spatial relationships between EVERY pair of objects in the room.

## Direction Format:
Spatial relationships are described using (<horizontal>, <vertical>) format:
- **horizontal**: left, right, same, unknown
- **vertical**: front, back, same, unknown
- "same" means objects are aligned in that dimension (e.g., same horizontal line)

## Critical Requirements:
1. **Complete Coverage**: Explore until you know where every object is relative to every other object
2. **Be Efficient** (Avoid redundant observations):
   - Focus on areas where you expect to eliminate some unknown relationships
   - If you know all objects are in one general direction but lack specific details, focus your exploration there 
      - (e.g., if all objects are to your left but you don't know which are in front vs. back, explore the left side systematically)
3. **Stop When Done**: End exploration as soon as you have all spatial relationships


## Important Notes:
- Focus on **directional relationships** between objects, not exact distances

After exploration, you return to starting position to answer questions.

## Room Layout
{room_info}

{exp_instructions}
"""

PASSIVE_INSTRUCTION = """\
# Spatial Understanding Task

You will be given a room layout and a tour around the room. 
NOTE: After the tour, you will return to your starting position and orientation.
Then you need to answer the question based on the tour.

## Direction Format:
Spatial relationships are described using (<horizontal>, <vertical>) format:
- **horizontal**: left, right, same, unknown
- **vertical**: front, back, same, unknown
- "same" means objects are aligned in that dimension (e.g., same horizontal line)

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