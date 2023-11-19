
tasks=(wikihow_chatGPT.jsonl wikipedia_chatGPT.jsonl arxiv_chatGPT.jsonl germanwikipedia_chatgpt.jsonl id-newspaper_chatGPT.jsonl pearread_chatgpt.jsonl qazh_chatgpt.jsonl reddit_chatGPT.jsonl russian_chatGPT.jsonl urdu_chatGPT.jsonl)

for i in "${tasks[@]}"
do
srun python /p/home/jusers/puccetti1/juwels/puccetti1/llm/llm-foundry/scripts/inference/vllm_generate.py \
    --name_or_path /p/fastdata/mmlaion/puccetti1/llama-2-models-hf/llama-2-7b-chat/ \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --seed 1 \
    --max_new_tokens 256 \
    --model_dtype bf16 \
    --max_batch_size 16 \
    --prompts "file::/p/home/jusers/puccetti1/juwels/puccetti1/llm/M4/data/$i" \
    --max_prompts 3000 \
    --output_path "/p/home/jusers/puccetti1/juwels/puccetti1/llm/llm-foundry/outputs/CHANGE-it/llama-2-7b-chat/repubblica/${i:0:-5]}/"
done

    # --max_seq_len 100 \
    # --prompts "/p/home/jusers/puccetti1/juwels/puccetti1/llm/data/CHANGE-it/train/change-it.repubblica.train.csv" \