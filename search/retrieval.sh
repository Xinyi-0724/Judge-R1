#!/usr/bin/env bash
set -euo pipefail

DATA_NAME="${DATA_NAME:-judge_r1}"

DATASET_PATH="${DATASET_PATH:-data/$DATA_NAME}"

SPLIT="${SPLIT:-test}"
TOPK="${TOPK:-3}"

INDEX_PATH="${INDEX_PATH:-data/retrieval/judge_r1_index}"
CORPUS_PATH="${CORPUS_PATH:-data/retrieval/judge_r1_corpus.jsonl}"

# INDEX_PATH=data/retrieval/other_index
# CORPUS_PATH=data/retrieval/other_corpus.jsonl

python retrieval.py --retrieval_method e5 \
                    --retrieval_topk "$TOPK" \
                    --index_path "$INDEX_PATH" \
                    --corpus_path "$CORPUS_PATH" \
                    --dataset_path "$DATASET_PATH" \
                    --data_split "$SPLIT" \
                    --retrieval_model_path "intfloat/e5-base-v2" \
                    --retrieval_pooling_method "mean" \
                    --retrieval_batch_size 512
