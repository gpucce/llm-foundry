#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --account=transfernetx
#SBATCH --partition=develbooster
#SBATCH --time=02:00:00
#SBATCH --output /p/home/jusers/puccetti1/juwels/puccetti1/llm/slurm_logs/generate-%j.out

eval "$(/p/home/jusers/puccetti1/juwels/puccetti1/miniconda3/bin/conda shell.bash hook)" # init conda
# conda activate llm
export CUDA_VISIBLE_DEVICES=0,1,2,3
export MASTER_PORT=12804

master_addr="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)i"
export MASTER_ADDR=$master_addr

cd /p/home/jusers/puccetti1/juwels/puccetti1/llm/llm-foundry/scripts

export tasks=(
    wikihow_chatGPT \
    wikipedia_chatGPT \
    arxiv_chatGPT \
    germanwikipedia_chatgpt \
    id-newspaper_chatGPT \
    pearread_chatgpt \
    qazh_chatgpt \
    reddit_chatGPT \
    russian_chatGPT \
    urdu_chatGPT \
)

for i in "${tasks[@]}"
do
srun --cpu_bind=none,v --accel-bind=gn apptainer exec \
    --nv "/p/home/jusers/puccetti1/juwels/puccetti1/llm/llm-foundry/llm-foundry_2.0.1_cu118-latest.sif" \
    /composer-python/python /p/home/jusers/puccetti1/juwels/puccetti1/llm/llm-foundry/scripts/inference/vllm_generate.py \
    --name_or_path /p/fastdata/mmlaion/puccetti1/llama-2-models-hf/llama-2-7b-chat/ \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --max_new_tokens 256 \
    --prompts "file::/p/home/jusers/puccetti1/juwels/puccetti1/llm/M4/data/$i.jsonl" \
    --max_prompts 3000 \
    --output_path "/p/home/jusers/puccetti1/juwels/puccetti1/llm/llm-foundry/outputs/CHANGE-it/llama-2-7b-chat/repubblica/$i/"
done