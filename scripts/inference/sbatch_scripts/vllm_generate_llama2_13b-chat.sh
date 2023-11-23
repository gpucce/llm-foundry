python /p/home/jusers/puccetti1/juwels/puccetti1/llm/llm-foundry/scripts/inference/vllm_generate.py \
    --name_or_path /leonardo_scratch/large/userexternal/gpuccett/models/hf_camoscio/camoscio2_13b \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --seed 1 \
    --max_new_tokens 256 \
    --model_dtype bf16 \
    --max_batch_size 16 \
    --prompts "file::/p/home/jusers/puccetti1/juwels/puccetti1/llm/M4/data/wikihow_chatGPT.jsonl" \
    --max_prompts 3000 \
    --output_path "/p/home/jusers/puccetti1/juwels/puccetti1/llm/llm-foundry/outputs/CHANGE-it/llama-2-13b-chat/repubblica/"


    # --max_seq_len 100 \
    # --prompts "/p/home/jusers/puccetti1/juwels/puccetti1/llm/data/CHANGE-it/train/change-it.repubblica.train.csv" \