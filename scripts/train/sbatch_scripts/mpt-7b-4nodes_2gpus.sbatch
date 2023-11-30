#!/bin/bash -x
#SBATCH --account IscrC_GELATINO
#SBATCH --partition boost_usr_prod
#SBATCH --exclude lrdn[0809]
#SBATCH --time 00:30:00
#SBATCH --gres gpu:2
#SBATCH --nodes 4
#SBATCH --mem 250G
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 8
#SBATCH --wait-all-nodes 1
#SBATCH --output /leonardo_scratch/large/userexternal/gpuccett/llm-foundry/scripts/train/slurm_logs/mpt-7b_2nodes_2gpus_%j.out

module purge
module load gcc
module load cuda
module load profile/deeplrn
module load cineca-ai

source /leonardo_scratch/large/userexternal/gpuccett/llm-foundry/venv/bin/activate

srun /leonardo_scratch/large/userexternal/gpuccett/llm-foundry/scripts/train/fine_tuning_commands/test_salloc_mpt-7b-finetune_multinode_4nodes.sbatch
