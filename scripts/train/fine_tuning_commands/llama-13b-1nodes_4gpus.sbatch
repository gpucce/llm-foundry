#!/usr/bin/bash

printf -v exp_date '%(%Y-%m-%d_%H-%M-%S)T' -1
export RUN_NAME="llama-13b_camoscio_1node_4gpus_${exp_date}"
echo "${SLURM_NODEID}: LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"

export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_MODE=offline
echo "${SLURM_NODEID}: WANDB_MODE: ${WANDB_MODE}"

composer -v train.py yamls/finetune/llama-13b_camoscio_1nodes.yaml
