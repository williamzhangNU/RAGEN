import numpy as np
from ragen.env.spatial.Evaluation import (
    BaseEvaluationTask,
    DirEvaluationTask,
    RotEvaluationTask,
    AllPairsEvaluationTask,
    ReverseDirEvaluationTask,
    PovEvaluationTask,
    A2EEvaluationTask,
    E2AEvaluationTask,
)
from ragen.env.spatial.config import (
    DirEvaluationConfig,
    RotEvaluationConfig,
    AllPairsEvaluationConfig,
    ReverseDirEvaluationConfig,
    PovEvaluationConfig,
    A2EEvaluationConfig,
    E2AEvaluationConfig,
)

def get_eval_task(eval_task: str, np_random: np.random.Generator, eval_kwargs: dict = None) -> BaseEvaluationTask:
    """
    Get the evaluation task from the config
    """
    task_map = {
        "dir": (DirEvaluationTask, DirEvaluationConfig),
        "rot": (RotEvaluationTask, RotEvaluationConfig),
        "all_pairs": (AllPairsEvaluationTask, AllPairsEvaluationConfig),
        "rev": (ReverseDirEvaluationTask, ReverseDirEvaluationConfig),
        "pov": (PovEvaluationTask, PovEvaluationConfig),
        "a2e": (A2EEvaluationTask, A2EEvaluationConfig),
        "e2a": (E2AEvaluationTask, E2AEvaluationConfig),
    }
    
    if eval_task in task_map:
        task_class, config_class = task_map[eval_task]
        return task_class(np_random, config_class(**eval_kwargs))
    else:
        raise ValueError(f"Unknown evaluation task: {eval_task}")
