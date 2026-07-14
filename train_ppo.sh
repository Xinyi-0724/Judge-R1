#!/usr/bin/env bash
set -euo pipefail

export DATA_DIR="${DATA_DIR:-data/judge_r1}"
export BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-8B}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-judge-r1-ppo-qwen3-8b}"
export WANDB_PROJECT="${WANDB_PROJECT:-Judge-R1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_NAME="${WANDB_NAME:-$EXPERIMENT_NAME}"
export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-judge-r1}"
export RETRIEVER_URL="${RETRIEVER_URL:-http://127.0.0.1:8000/retrieve}"
export RETRIEVER_TOPK="${RETRIEVER_TOPK:-3}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-800}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export MAX_TURNS="${MAX_TURNS:-3}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-XFORMERS}"

if [ -z "${N_GPUS:-}" ]; then
    if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
        N_GPUS=1
    else
        N_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)
    fi
fi

export RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray-$(whoami)}"
mkdir -p "$RAY_TMPDIR" logs
export RAY_ADDRESS="${RAY_ADDRESS:-local}"

PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$DATA_DIR/test.parquet" \
    data.train_batch_size=64 \
    data.val_batch_size=16 \
    data.max_prompt_length=12500 \
    data.max_response_length=1000 \
    data.max_start_length=9000 \
    data.max_obs_length=500 \
    data.shuffle=True \
    algorithm.adv_estimator=gae \
    actor_rollout_ref.model.path="$BASE_MODEL" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.285 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.max_model_len=13500 \
    actor_rollout_ref.rollout.max_num_batched_tokens=15000 \
    actor_rollout_ref.rollout.temperature=1 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.actor.state_masking=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.optim.lr=1e-5 \
    critic.model.use_remove_padding=True \
    critic.optim.lr_warmup_steps_ratio=0.015 \
    critic.model.path="$BASE_MODEL" \
    critic.model.enable_gradient_checkpointing=true \
    critic.ppo_micro_batch_size_per_gpu=8 \
    critic.model.fsdp_config.param_offload=true \
    critic.model.fsdp_config.optimizer_offload=true \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.no_think_rl=false \
    trainer.critic_warmup=0 \
    trainer.logger=['wandb'] \
    trainer.val_before_train=true \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
    trainer.default_local_dir="verl_checkpoints/$EXPERIMENT_NAME" \
    max_turns="$MAX_TURNS" \
    retriever.url="$RETRIEVER_URL" \
    retriever.topk="$RETRIEVER_TOPK" \
    2>&1 | tee "logs/$EXPERIMENT_NAME.log"
