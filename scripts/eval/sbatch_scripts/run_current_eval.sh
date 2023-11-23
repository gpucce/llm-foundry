#!/usr/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=32
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=./slurm_logs/slurm-%j.out

module load gcc
module load cuda
module load profile/deeplrn
module load cineca-ai
source /leonardo_work/IscrC_GELATINO/gpuccett/Repos/foundry_venv/bin/activate

export PYTHONPATH=""

srun python /leonardo_work/IscrC_GELATINO/gpuccett/Repos/llm-foundry_gpucce/scripts/eval/eval.py /leonardo_work/IscrC_GELATINO/gpuccett/Repos/llm-foundry_gpucce/scripts/eval/yamls/ita_hf_eval.yaml
