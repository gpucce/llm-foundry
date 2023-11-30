#!/bin/bash -x

#SBATCH --account IscrC_GELATINO
#SBATCH --partition boost_usr_prod
#SBATCH --time 04:00:00
#SBATCH --gres gpu:4
#SBATCH --nodes 8
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --wait-all-nodes 1
#SBATCH --output ./slurm_logs/llama-13b_8nodes_4gpus_%j.out

module purge
module load gcc
module load cuda

source /leonardo_work/IscrC_GELATINO/gpuccett/Repos/foundry_venv/bin/activate

srun /leonardo_work/IscrC_GELATINO/gpuccett/Repos/llm-foundry_gpucce/scripts/train/fine_tuning_commands/test_salloc_llama-13b-finetune_multinode_8nodes_4gpus.sbatch
