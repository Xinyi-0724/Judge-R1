#!/usr/bin/env bash
set -euo pipefail

METHOD="${METHOD:-grpo}"

if [ "$METHOD" = "grpo" ]; then
    DEFAULT_RUN_NAME="judge-r1-grpo-qwen3-8b"
elif [ "$METHOD" = "ppo" ]; then
    DEFAULT_RUN_NAME="judge-r1-ppo-qwen3-8b"
else
    echo "Unsupported METHOD='$METHOD'. Use METHOD=grpo or METHOD=ppo." >&2
    exit 1
fi

LOCAL_DIR="${LOCAL_DIR:-verl_checkpoints/$DEFAULT_RUN_NAME/global_step_300/actor}"
TARGET_DIR="${TARGET_DIR:-models/$DEFAULT_RUN_NAME}"

mkdir -p "$TARGET_DIR"

echo "Merging Judge-R1 FSDP checkpoint"
echo "  method: $METHOD"
echo "  source: $LOCAL_DIR"
echo "  target: $TARGET_DIR"

if [ ! -d "$LOCAL_DIR" ]; then
    echo "Missing checkpoint directory: $LOCAL_DIR" >&2
    echo "Set LOCAL_DIR to the actor checkpoint you want to publish." >&2
    exit 1
fi

python3 -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$LOCAL_DIR" \
    --target_dir "$TARGET_DIR"

echo "Merged model saved to $TARGET_DIR"
