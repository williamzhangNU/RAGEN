import numpy as np
from ragen.env.spatial.Evaluation import (
    BaseEvaluationTask,
    DirEvaluationTask,
    RotEvaluationTask,
    AllPairsEvaluationTask,
)
from ragen.env.spatial.config import (
    DirEvaluationConfig,
    RotEvaluationConfig,
    AllPairsEvaluationConfig,
)

def get_eval_task(eval_task: str, np_random: np.random.Generator, eval_kwargs: dict = None) -> BaseEvaluationTask:
    """
    Get the evaluation task from the config
    """
    task_map = {
        "dir": (DirEvaluationTask, DirEvaluationConfig),
        "rot": (RotEvaluationTask, RotEvaluationConfig),
        "all_pairs": (AllPairsEvaluationTask, AllPairsEvaluationConfig)
    }
    
    if eval_task in task_map:
        task_class, config_class = task_map[eval_task]
        return task_class(np_random, config_class(**eval_kwargs))
    else:
        raise ValueError(f"Unknown evaluation task: {eval_task}")
