python /leonardo_work/IscrC_GELATINO/gpuccett/Repos/llm-foundry_gpucce/scripts/inference/vllm_generate.py \
    --name_or_path /leonardo_scratch/large/userexternal/gpuccett/models/hf_camoscio/camoscio2_13b \
    --temperature 1.0 \
    --top_p 0.95 \
    --top_k 50 \
    --seed 1 \
    --max_new_tokens 512 \
    --model_dtype bf16 \
    --max_batch_size 16 \
    --system_prompt "### Istruzione: Dato il testo '{}' scrivete un articolo di almeno 400 parole di cui quello è il titolo.\n\n### Risposta:" \
    --prompts "file::/leonardo_scratch/large/userexternal/gpuccett/data/CHANGE-it/train.jsonl" \
    --max_prompts 5000 \
    --output_path /leonardo_scratch/large/userexternal/gpuccett/data/camoscio_m4 \
    --human_key full_text


    # --max_seq_len 100 \
    # --prompts "/p/home/jusers/puccetti1/juwels/puccetti1/llm/data/CHANGE-it/train/change-it.repubblica.train.csv" \