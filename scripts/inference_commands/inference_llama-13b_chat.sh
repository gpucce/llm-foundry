


python inference/hf_interactive_generate.py -n /leonardo_work/IscrC_GELATINO/gpuccett/models/camoscio2_13b \
  --max_new_tokens=512 \
  --temperature 0.8 \
  --top_k 0 \
  --model_dtype bf16 \
  --trust_remote_code \
  --stop_tokens "</s>" \
  --user_msg_fmt "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n" \
  --assistant_msg_fmt "### Response:\n"

 