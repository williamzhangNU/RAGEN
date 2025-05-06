# Tested with 2 & 4 GPUs

set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_spatial_sft_lora.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

if [ ! -d "$save_path" ]; then
    echo "Creating directory: $save_path"
    mkdir -p "$save_path"
fi


# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m ragen.trainer.fsdp_sft_trainer \
    data.train_files=data/spatial_qa/sft_format/train_no_think.json \
    data.val_files=data/spatial_qa/sft_format/val_no_think.json \
    data.prompt_key=question \
    data.response_key=answer \
    optim.lr=1e-4 \
    data.micro_batch_size_per_gpu=16 \
    data.max_length=512 \
    model.partial_pretrain=Qwen/Qwen2.5-3B-Instruct \
    trainer.default_local_dir=$save_path \
    trainer.project_name=spatial-qa-sft \
    trainer.experiment_name=spatial-qa-sft-qwen-2.5-3b-instruct \
    trainer.logger=['console'] \
    trainer.total_epochs=50 \
    trainer.default_hdfs_dir=null $@ \
    model.lora_rank=32 \
    model.lora_alpha=16 \
    model.target_modules=all-linear 2>&1 | tee $save_path/train.log

    # Or you can do this:
    # model.target_modules=[q_proj,v_proj] \