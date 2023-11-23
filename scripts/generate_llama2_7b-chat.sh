python /p/home/jusers/puccetti1/juwels/puccetti1/llm/llm-foundry/scripts/inference/hf_generate.py \
    --name_or_path /p/fastdata/mmlaion/puccetti1/llama-2-models-hf/llama-2-7b-chat/ \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --seed 1 \
    --max_new_tokens 256 \
    --model_dtype bf16 \
    --max_batch_size 16 \
    --prompts "file::/p/home/jusers/puccetti1/juwels/puccetti1/llm/M4/data/wikihow_chatGPT.jsonl" \
    --output_path "/p/home/jusers/puccetti1/juwels/puccetti1/llm/llm-foundry/outputs/CHANGE-it/llama-2-7b-chat/repubblica/"

    # --max_prompts 100 \
    # --max_seq_len 100 \
    # --prompts "/p/home/jusers/puccetti1/juwels/puccetti1/llm/data/CHANGE-it/train/change-it.repubblica.train.csv" \