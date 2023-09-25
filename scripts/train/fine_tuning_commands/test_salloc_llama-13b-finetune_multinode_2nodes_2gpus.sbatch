#!/usr/bin/bash

module purge
module load gcc
module load cuda
module load profile/deeplrn
module load cineca-ai

source /leonardo_scratch/large/userexternal/gpuccett/llm-foundry/venv/bin/activate

printf -v exp_date '%(%Y-%m-%d_%H-%M-%S)T' -1
export RUN_NAME="llama-13b_camoscio_${exp_date}"
echo "${SLURM_NODEID}: LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"

export TF_ENABLE_ONEDNN_OPTS=0
export WANDB_MODE=offline
echo "${SLURM_NODEID}: WANDB_MODE: ${WANDB_MODE}"

export LOCAL_WORLD_SIZE=2
echo "${SLURM_NODEID}: LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
export NPROC=$LOCAL_WORLD_SIZE
echo "${SLURM_NODEID}: NPROC=${NPROC}"
export WORLD_SIZE=4
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

composer train.py yamls/finetune/llama-13b_camoscio.yaml

# echo $(echo $SLURM_NODEID)
# echo $SLURM_NODEID