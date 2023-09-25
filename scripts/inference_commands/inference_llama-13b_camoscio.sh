


python inference/hf_interactive_generate.py -n /leonardo_scratch/large/userexternal/gpuccett/models/llama-13b_camoscio_hf \
  --max_new_tokens=512 \
  --temperature 0.8 \
  --top_k 0 \
  --model_dtype bf16 \
  --trust_remote_code \
  --stop_tokens "</s>" \
  --user_msg_fmt "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\n{}\n" \
  --assistant_msg_fmt "### Response:\n" \

 Scrivi un programma che stampa i numeri da 1 a 100. Ma per i multipli di tre stampa 'Fizz' al posto del numero e per i multipli di cinque stampa 'Buzz'. Per i numeri che sono multipli sia di tre che di cinque stampa 'FizzBuzz'.



RIORDINA LE PAROLE E COMPONI LA FRASE
1 Barbara / bel / compleanno / un / blu / il / indossato / per / vestito/ ha

2 indossato/ bella/ Giulia/ ha / sciarpa / una

3 dove / comprato / cappello / hai / ? / quel

4 indosso / per / l’infradito / mare / il / sempre

5 comprato/ ho / maglietta / una/ ieri
