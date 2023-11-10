#!/bin/bash -x
#SBATCH --account IscrC_GELATINO
#SBATCH --partition boost_usr_prod
#SBATCH --qos boost_qos_dbg
#SBATCH --exclude lrdn[0809]
#SBATCH --time 00:30:00
#SBATCH --gres gpu:4
#SBATCH --nodes 2
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --wait-all-nodes 1
#SBATCH --output /leonardo_work/IscrC_GELATINO/gpuccett/Repos/llm-foundry_gpucce/scripts/train/slurm_logs/llama-13b_2nodes_2gpus_%j.out

# module purge
# module load gcc
# module load cuda
# module load profile/deeplrn
# module load cineca-ai

source /leonardo_scratch/large/userexternal/gpuccett/llm-foundry/venv/bin/activate

srun /leonardo_work/IscrC_GELATINO/gpuccett/Repos/llm-foundry_gpucce/scripts/train/fine_tuning_commands/test_salloc_llama-13b-finetune_multinode_2nodes_2gpus.sbatch
