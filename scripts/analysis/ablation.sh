#!/bin/bash

mkdir -p ./log/terminal

##############################
# Section 5.3.1 - Batch Size vs Rollout
# Format: GPU_ID, batch_size, n_rollout
CONFIGS=(
    "0  2   64"  # p2n64
    "1  4   32"  # p4n32
    "2  16  8"   # p16n8
    "3  32  4"   # p32n4
    "0  64  2"   # p64n2
)

for config in "${CONFIGS[@]}"; do
    read -r gpu batch rollout <<< "$config"
    bash train.sh sokoban \
        model.experiment_name=sokoban_ablation_p${batch}n${rollout} \
        system.cuda_visible_devices=$gpu \
        training.micro_batch_size=2 \
        training.total_training_steps=100 \
        training.train_batch_size=$batch \
        training.n_rollout=$rollout \
        trainer.test_freq=10 \
        >> ./log/terminal/sokoban_ablation_p${batch}n${rollout}.log &
done


##############################
# Section 5.3.3 - Offline Training
# Format: GPU_ID, batch_size, total_steps, test_freq
OFFLINE_CONFIGS=(
    "0  16  50  5"   # offline2
    "1  40  20  2"   # offline5
    "2  80  10  1"   # offline10
    "3  160 5   1"   # offline20
)

for config in "${OFFLINE_CONFIGS[@]}"; do
    read -r gpu batch steps freq <<< "$config"
    bash train.sh sokoban \
        model.experiment_name=sokoban_ablation_offline${batch} \
        system.cuda_visible_devices=$gpu \
        training.micro_batch_size=2 \
        training.ppo_batch_size=128 \
        training.n_rollout=16 \
        training.train_batch_size=$batch \
        training.total_training_steps=$steps \
        trainer.test_freq=$freq \
        >> ./log/terminal/sokoban_ablation_offline${batch}.log &
done

##############################
# Section 5.3.4 - Base Model Study
# Format: env_name, gpu_id
BASE_CONFIGS=(
    "sokoban    1"
    "frozenlake 3"
)

for config in "${BASE_CONFIGS[@]}"; do
    read -r env gpu <<< "$config"
    bash train.sh $env \
        model.base_model=Qwen/Qwen2.5-0.5B \
        model.experiment_name=${env}_abl_base \
        system.cuda_visible_devices=$gpu \
        training.micro_batch_size=2 \
        training.total_training_steps=100 \
        trainer.test_freq=10 \
        >> ./log/terminal/${env}_abl_base.log &
done