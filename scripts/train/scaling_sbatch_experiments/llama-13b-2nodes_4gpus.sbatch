#!/bin/bash
#SBATCH --account IscrC_GELATINO
#SBATCH --partition boost_usr_prod
#SBATCH --exclude lrdn[0809]
#SBATCH --time 00:30:00
#SBATCH --gres gpu:4
#SBATCH --nodes 2
#SBATCH --qos=boost_qos_dbg
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 32
#SBATCH --wait-all-nodes 1
#SBATCH --output /leonardo_work/IscrC_GELATINO/gpuccett/Repos/llm-foundry_gpucce/scripts/train/slurm_logs/llama-13b_2nodes_4gpus_%j.out

module purge
module load gcc
module load cuda

source /leonardo_work/IscrC_GELATINO/gpuccett/Repos/foundry_venv/bin/activate

srun --cpu_bind=none,v /leonardo_work/IscrC_GELATINO/gpuccett/Repos/llm-foundry_gpucce/scripts/train/fine_tuning_commands/llama-13b-2nodes_4gpus.sbatch