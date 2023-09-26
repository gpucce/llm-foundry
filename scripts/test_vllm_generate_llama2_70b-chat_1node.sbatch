#!/bin/bash -x
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --wait-all-nodes=1
#SBATCH --job-name=generate
#SBATCH --account=transfernetx
#SBATCH --partition=develbooster
#SBATCH --time=02:00:00
#SBATCH --output /p/home/jusers/puccetti1/juwels/puccetti1/llm/slurm_logs/generate-%j.out

export tasks=(
    wikihow_chatGPT.jsonl \
    wikipedia_chatgpt.jsonl \
    arxiv_chatGPT.jsonl \
    germanwikipedia_chatgpt.jsonl \
    id-newspaper_chatGPT.jsonl \
    peerread_chatgpt.jsonl \
    qazh_chatgpt.jsonl \
    reddit_chatGPT.jsonl \
    russian_chatGPT.jsonl \
    urdu_chatGPT.jsonl \
)

for i in "${tasks[@]}"
do
python /p/home/jusers/puccetti1/juwels/puccetti1/llm/llm-foundry/scripts/inference/vllm_generate.py \
    --name_or_path /p/fastdata/mmlaion/puccetti1/llama-2-models-hf/llama-2-70b-chat/ \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --seed 1 \
    --model_dtype bf16 \
    --max_batch_size 16 \
    --prompts "file::/p/home/jusers/puccetti1/juwels/puccetti1/llm/M4/data/$i" \
    --max_prompts 100 \
    --output_path "/p/home/jusers/puccetti1/juwels/puccetti1/llm/llm-foundry/outputs/llama-2-70b-chat_prompted/${i:0:-6}/"
done

    # --max_new_tokens 256 \
