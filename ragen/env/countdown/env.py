import gymnasium as gym
from ragen.env.base import BaseLanguageBasedEnv
import datasets
import re
import itertools
from ragen.env.countdown.config import CountdownEnvConfig


def check_format(equation, nums):
    try:
        nums_in_eq = [int(n) for n in re.findall(r'\d+', equation)]
        return sorted(nums_in_eq) == sorted(nums)
    except:
        return False

def check_correctness(equation_str, target):
    try:
        result = eval(equation_str, {"__builtins__": None}, {})
        return abs(result - target) < 1e-5
    except:
        return False

def has_solution(nums, target):
    """Check if there is a valid equation using each number exactly once."""
    # pad nums all to 4 numbers
    length = 4
    nums = nums + [0] * (length - len(nums))
    # +- num1 +- num2 +- num3 +- num4 = target, try all
    combinations = list(itertools.product([1, -1], repeat=length))
    for combination in combinations:
        if sum(combination[i] * nums[i] for i in range(length)) == target:
            return True
    return False

class CountdownEnv(BaseLanguageBasedEnv, gym.Env):
    def __init__(self, config=None):
        BaseLanguageBasedEnv.__init__(self)
        self.config = config if config is not None else CountdownEnvConfig()
        self.INVALID_ACTION = self.config.invalid_act
        self.PENALTY_FOR_INVALID = self.config.invalid_act_score
        self.data = self._get_data_from_parquet(self.config.train_path)
        self.index = None

    def _get_data_from_parquet(self, path):
        df = datasets.load_dataset("parquet", data_files=path)['train'].select(range(self.config.max_instances))
        df = df.filter(lambda x: has_solution(x['nums'], x['target']))
        return df

    def reset(self, seed=None, mode='text'):
        gym.Env.reset(self, seed=seed)
        self.index = seed % len(self.data)
        data = self.data[self.index]
        return f"Target: {data['target']}, nums: {data['nums']}"

    def step(self, action):
        if not isinstance(action, str) or action == self.INVALID_ACTION:
            return "You have made an invalid move.", self.PENALTY_FOR_INVALID, True, {"action_is_effective": False, "action_is_valid": False, "success": False}
        
        reward = self.compute_reward(action, self.data[self.index])
        next_obs, done, info = f"Your answer get {reward} points.", True, {"action_is_effective": reward > 0, "action_is_valid": True, "success": reward == self.config.score}
        return next_obs, reward, done, info

    def compute_reward(self, action, ground_truth):
        """Score the countdown task solution."""
        target = ground_truth['target']
        nums = ground_truth['nums']
        if not check_format(action, nums):
            return 0
        if not check_correctness(action, target):
            return self.config.format_score
        else:
            return self.config.score

if __name__ == "__main__":
    def test(path, seed=43):
        config = CountdownEnvConfig(train_path=path)
        env = CountdownEnv(config)
        obs = env.reset(seed=seed)
        problem = env.data[env.index]
        solution = f"- {problem['nums'][0]} + {problem['nums'][1]} + {problem['nums'][2]}"
        _, reward, _, _ = env.step(solution)
        print(f"{obs}\nSolution: {solution}, Reward: {reward}")
    
    test("data/countdown/countdown_train.parquet")