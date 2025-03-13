class BiArmBanditEnvConfig:
    def __init__(
            self,
            low_risk_name: str = "phoenix",
            high_risk_name: str = "dragon",
            action_space_start: int = 1,
            low_risk_reward: float = 0.2,
            high_risk_low_reward: float = 0.1,
            high_risk_high_reward: float = 1.0,
            high_risk_high_reward_prob: float = 0.25,
    ):
        # Arm names
        self.low_risk_name = low_risk_name
        self.high_risk_name = high_risk_name
        
        # Action space configuration
        self.action_space_start = action_space_start
        self.low_risk_reward = low_risk_reward
        self.high_risk_low_reward = high_risk_low_reward
        self.high_risk_high_reward = high_risk_high_reward
        self.high_risk_high_reward_prob = high_risk_high_reward_prob

