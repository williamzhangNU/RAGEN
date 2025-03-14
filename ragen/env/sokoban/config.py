class SokobanEnvConfig:
    def __init__(self, 
                 dim_room=(6, 6), 
                 max_steps=100, 
                 num_boxes=3, 
                 search_depth=300,
                 grid_lookup=None, 
                 action_lookup=None, 
                 invalid_act=0, 
                 invalid_act_score=-1):
        self.grid_lookup = grid_lookup or {0:"#", 1:"_", 2:"O", 3:"√", 4:"X", 5:"P", 6:"S"}
        
        self.action_lookup = action_lookup or {0:"None", 1:"Up", 2:"Down", 3:"Left", 4:"Right"}
        self.invalid_act = invalid_act
        self.invalid_act_score = invalid_act_score
        
        self.dim_room = dim_room
        self.max_steps = max_steps
        self.num_boxes = num_boxes
        self.search_depth = search_depth