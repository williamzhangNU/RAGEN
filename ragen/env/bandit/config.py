class BiArmBanditEnvConfig:
    def __init__(
            self,
            lo_arm_name: str = "phoenix",
            hi_arm_name: str = "dragon",
            action_space_start: int = 1,
            lo_arm_score: float = 0.2,
            hi_arm_loscore: float = 0.1,
            hi_arm_hiscore: float = 1.0,
            hi_arm_hiscore_prob: float = 0.25,
            invalid_act: int = 0,
            invalid_act_score: float = 0,
    ):
        # Action space configuration
        self.action_space_start = action_space_start

        self.lo_arm_name = lo_arm_name
        self.lo_arm_score = lo_arm_score

        self.hi_arm_name = hi_arm_name
        self.hi_arm_loscore = hi_arm_loscore
        self.hi_arm_hiscore = hi_arm_hiscore
        self.hi_arm_hiscore_prob = hi_arm_hiscore_prob
        
        self.invalid_act = invalid_act
        self.invalid_act_score = invalid_act_score