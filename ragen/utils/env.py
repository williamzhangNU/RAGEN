# env_utils.py
import random
import logging
import numpy as np
import torch
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from contextlib import contextmanager
import os

@contextmanager
def all_seed(seed):
    random_state = random.getstate()
    np_random_state = np.random.get_state()

    try:
        random.seed(seed)
        np.random.seed(seed)
        yield
    finally:
        random.setstate(random_state)
        np.random.set_state(np_random_state)

def get_train_val_env(env_class, config: dict):

    val_env = None
    if config.env.name == 'frozenlake':
        env = env_class(size=config.env.size, p=config.env.p)
    elif config.env.name == 'bandit':
        env = env_class(n_arms=config.env.n_arms)
    elif config.env.name == 'bi_arm_bandit':
        lo_name, hi_name = config.env.low_risk_name, config.env.high_risk_name
        lo_val_name = config.env.low_risk_name if config.env.low_risk_val_name is None else config.env.low_risk_val_name
        hi_val_name = config.env.high_risk_name if config.env.high_risk_val_name is None else config.env.high_risk_val_name
        env = env_class(low_risk_name=lo_name, high_risk_name=hi_name)
        val_env = env_class(low_risk_name=lo_val_name, high_risk_name=hi_val_name)
        print(f"[INFO] val_env low_risk_name: {val_env.low_risk_name}, high_risk_name: {val_env.high_risk_name}")
        if val_env.low_risk_name is None or val_env.high_risk_name is None:
            print("[WARNING] val_env arm are None, falling back to not create val_env")
            val_env = None
    elif config.env.name == 'sokoban':
        env = env_class(dim_room=(config.env.dim_x, config.env.dim_y), num_boxes=config.env.num_boxes, max_steps=config.env.max_steps, search_depth=config.env.search_depth)
    elif config.env.name == 'countdown':
        env = env_class(parquet_path=config.env.train_path)
        val_env = env_class(parquet_path=config.env.val_path)
    else:
        raise ValueError(f"Environment {config.env.name} not supported")

    return env, val_env