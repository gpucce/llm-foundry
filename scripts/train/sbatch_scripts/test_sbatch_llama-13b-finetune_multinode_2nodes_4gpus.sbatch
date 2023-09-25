#!/bin/bash
#SBATCH --account IscrC_GELATINO
#SBATCH --partition boost_usr_prod
#SBATCH --exclude lrdn[0809]
#SBATCH --time 00:30:00
#SBATCH --gres gpu:4
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --wait-all-nodes 1
#SBATCH --output /leonardo_scratch/large/userexternal/gpuccett/llm-foundry/scripts/train/slurm_logs/llama-13b_2nodes_4gpus_%j.out

module purge
module load gcc
module load cuda
module load profile/deeplrn
module load cineca-ai

source /leonardo_scratch/large/userexternal/gpuccett/llm-foundry/venv/bin/activate

srun --cpu_bind=none,v /leonardo_scratch/large/userexternal/gpuccett/llm-foundry/scripts/train/fine_tuning_commands/test_salloc_llama-13b-finetune_multinode_2nodes_4gpus.sbatch
