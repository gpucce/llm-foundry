#!/usr/bin/bash

printf -v exp_date '%(%Y-%m-%d_%H-%M-%S)T' -1
export RUN_NAME="llama-70b_camoscio_${exp_date}"
# export RUN_NAME=llama-13b_camoscio_2023-09-22_19

export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_MODE=offline

export LOCAL_WORLD_SIZE=4
echo "${SLURM_NODEID}: LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
export NPROC=$LOCAL_WORLD_SIZE
echo "${SLURM_NODEID}: NPROC=${NPROC}"
export WORLD_SIZE=32
echo "${SLURM_NODEID}: WORLD SIZE=${WORLD_SIZE}"

master_addr="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
export MASTER_ADDR=$master_addr
echo "${SLURM_NODEID}: MASTER_ADDR=${MASTER_ADDR}"
export MASTER_PORT=54321
echo "${SLURM_NODEID}: MASTER PORT=${MASTER_PORT}"
export PYTHONUNBUFFERED=1
echo "${SLURM_NODEID}: PYTHONUNBUFFERED=${PYTHONUNBUFFERED}"

export NODENAME=${SLURM_NODEID}

export BASE_RANK=$(($SLURM_PROCID * $NPROC))
echo "${SLURM_NODEID}: BASE_RANK=${BASE_RANK}, NODEID=${SLURM_NODEID}"
export LOCAL_RANK=$SLURM_LOCALID
echo "${SLURM_NODEID}: LOCAL_RANK=${LOCAL_RANK}"

# export CUDA_VISIBLE_DEVICES="0,1"
# echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"



if [ $SLURM_NODEID != 0 ] 
then
    sleep 1
fi

composer train.py yamls/finetune/mistral-7b_camoscio_8nodes.yaml
