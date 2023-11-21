#!/bin/bash

#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --account=IscrC_GELATINO
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --time=00:30:00
#SBATCH --output slurm_logs/generate-%j.out

module purge 
module load gcc
module load cuda

source /leonardo_work/IscrC_GELATINO/gpuccett/Repos/vllm_venv/bin/activate

# ray start

python /leonardo_work/IscrC_GELATINO/gpuccett/Repos/llm-foundry_gpucce/scripts/inference/vllm_generate.py \
    --name_or_path /leonardo_scratch/large/userexternal/gpuccett/models/hf_camoscio/camoscio2-70b-lora-hf/ \
    --temperature 0.8 \
    --top_p 0.95 \
    --top_k 50 \
    --seed 1 \
    --max_new_tokens 384 \
    --model_dtype bf16 \
    --max_batch_size 16 \
    --system_prompt "### Istruzione: Dato il testo '{}' scrivete un articolo di almeno 400 parole in Italiano di cui quello è il titolo.\n\n### Risposta:" \
    --dataset_name CHANGE-it \
    --prompts "file::/leonardo_scratch/large/userexternal/gpuccett/data/CHANGE-it/train.jsonl" \
    --max_prompts 5000 \
    --output_path /leonardo_scratch/large/userexternal/gpuccett/data/camoscio2_70b_m4_no_beam_search \
    --human_key full_text \
    --tensor_parallel_size 4 \
    --use_beam_search false


    # --max_seq_len 100 \
    # --prompts "/p/home/jusers/puccetti1/juwels/puccetti1/llm/data/CHANGE-it/train/change-it.repubblica.train.csv" \
