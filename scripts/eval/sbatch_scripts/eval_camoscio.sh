#!/bin/bash
#SBATCH --account IscrC_GELATINO \
#SBATCH --partition boost_usr_prod \
#SBATCH --time 01:00:00 \
#SBATCH --gres gpu:4 \
#SBATCH --nodes 1 \
#SBATCH --ntasks-per-node 1 \
#SBATCH --cpus-per-task 32 \

module purge
module load gcc
module load cuda
module load profile/deeplrn
module load cineca-ai

source /leonardo_work/IscrC_GELATINO/gpuccett/Repos/llm-foundry_gpucce/venv/bin/activate

export HF_HOME=/leonardo_work/IscrC_GELATINO/gpuccett/Repos/llm-foundry_gpucce/scripts/hf_cache
export HF_DATASETS_CACHE=/leonardo_work/IscrC_GELATINO/gpuccett/Repos/llm-foundry_gpucce/scripts/hf_cache
export HF_TRANSFORMERS_CACHE=/leonardo_work/IscrC_GELATINO/gpuccett/Repos/llm-foundry_gpucce/scripts/hf_cache
export TF_ENABLE_ONEDNN_OPTS=0

srun --cpu_bind=none,v composer eval.py yamls/ita_hf_eval.yaml