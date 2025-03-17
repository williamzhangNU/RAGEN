# Experiment Commands Reference

## Basic Experiment
bash train.sh sokoban \
    model.experiment_name=new_test \
    training.train_batch_size=4 \
    training.n_rollout=8 \
    training.ppo_batch_size=8 \
    training.micro_batch_size=1 \
    training.max_turns=5 \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=gae

bash train.sh bi_arm_bandit \
    model.experiment_name=new_test \
    training.train_batch_size=8 \
    training.n_rollout=16 \
    training.ppo_batch_size=32 \
    training.micro_batch_size=16 \
    optimization.kl_coef=0.000

## Common Parameters
```bash
# Base parameters used across experiments.
BASE_PARAMS="
    training.micro_batch_size=4 \
    training.ppo_batch_size=32 \
    optimization.kl_coef=0.001 \
    optimization.adv_estimator=brpo"

# Create log directory
mkdir -p ./log/terminal
```

## Hyperparameter Search
Running with 3B models using FSDP strategy. Total 27 runs on Qwen2.5-3B-Instruct for Sokoban.

```bash
# Search different parameter groups
bash scripts/hyperparam_search.sh \
    --env_name=bi_arm_bandit \
    --exp_base_name="hyperparam_searching" \
    --search_group [1|2|3|4] \
    --micro_batch_size 4 \
    --parallel \
    --n_gpus 1
```

Search groups:
1. train_batch_size & n_rollout: [4,8,16]
2. actor_lr: [5e-7, 1e-6, 5e-6, 1e-5]
3. kl_coef: [0.001, 0.005, 0.01, 0.04, 0.1, 0.5]
4. max_turns: [2,5,8] & temperature: [0.1, 0.5, 1]

**Note:** Remove `--parallel` if GPU issues occur.

## Environment-Specific Commands

### Two-Armed Bandit
```bash
# RAGEN - Base command
bash train.sh bi_arm_bandit \
    model.base_model=Qwen/Qwen2.5-[0.5B|3B]-Instruct \
    model.experiment_name=bi_arm_bandit_[0_5B|3B]_instruct_ragen_main \
    training.train_batch_size=32 \
    training.max_turns=1 \
    training.n_rollout=1 \
    ${BASE_PARAMS}

# Add for RAGEN w/o thinking
    training.no_think_rl=True
```

### Sokoban
```bash
# RAGEN - Base command
bash train.sh sokoban \
    model.base_model=Qwen/Qwen2.5-[0.5B|1.5B|3B|7B]-Instruct \
    model.experiment_name=sokoban_[0_5B|1_5B|3B|7B]_instruct_ragen_main \
    training.train_batch_size=4 \
    training.max_turns=5 \
    training.n_rollout=8 \
    ${BASE_PARAMS}

# Add for 7B model
    system.n_gpus=2

# Add for RAGEN w/o thinking
    training.no_think_rl=True
```

### FrozenLake
```bash
# RAGEN - Base command
bash train.sh frozenlake \
    model.base_model=Qwen/Qwen2.5-[0.5B|3B]-Instruct \
    model.experiment_name=frozenlake_[0_5B|3B]_instruct_ragen_main \
    training.train_batch_size=4 \
    training.max_turns=5 \
    training.n_rollout=8 \
    ${BASE_PARAMS}

# Add for RAGEN w/o thinking
    training.no_think_rl=True
```

Usage notes:
1. Replace [0.5B|3B] with desired model size
2. Adjust experiment names accordingly
3. All commands output logs to ./log/terminal/
4. For model scaling experiments, use appropriate GPU counts for larger models