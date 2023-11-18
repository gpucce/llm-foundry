import itertools
import os
import json
from pathlib import Path
import random
import time
import warnings
from argparse import ArgumentParser, ArgumentTypeError, Namespace
from contextlib import nullcontext
from tqdm import tqdm
import pandas as pd


import numpy as np
import torch
from vllm import LLM, SamplingParams

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ArgumentTypeError('Boolean value expected.')


def str_or_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        return v

def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description='Load a HF CausalLM Model and use it to generate text.')
    parser.add_argument('-n', '--name_or_path', type=str, required=True)
    parser.add_argument(
        '-p',
        '--prompts',
        help='Generation prompts. Use syntax "file::/path/to/prompt.txt" to load a ' +\
             'prompt contained in a txt file.'
        )
    parser.add_argument('--max_seq_len', type=int, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=1250)
    parser.add_argument('--max_batch_size', type=int, default=None)
    #####
    # Note: Generation config defaults are set to match Hugging Face defaults
    parser.add_argument('--temperature', type=float, nargs='+', default=[1.0])
    parser.add_argument('--top_k', type=int, nargs='+', default=[50])
    parser.add_argument('--top_p', type=float, nargs='+', default=[1.0])
    parser.add_argument('--repetition_penalty',
                        type=float,
                        nargs='+',
                        default=[1.0])
    parser.add_argument('--no_repeat_ngram_size',
                        type=int,
                        nargs='+',
                        default=[0])
    #####
    parser.add_argument('--seed', type=int, nargs='+', default=[42])
    parser.add_argument('--do_sample',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--use_cache',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--eos_token_id', type=int, default=None)
    parser.add_argument('--pad_token_id', type=int, default=None)
    parser.add_argument('--model_dtype',
                        type=str,
                        choices=['fp32', 'fp16', 'bf16'],
                        default=None)
    parser.add_argument('--autocast_dtype',
                        type=str,
                        choices=['fp32', 'fp16', 'bf16'],
                        default=None)
    parser.add_argument('--warmup',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--trust_remote_code',
                        type=str2bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument('--use_auth_token',
                        type=str_or_bool,
                        nargs='?',
                        const=True,
                        default=None)
    parser.add_argument('--revision', type=str, default=None)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--device_map', type=str, default=None)
    parser.add_argument('--attn_impl', type=str, default=None)
    parser.add_argument("--headline", action="store_true", default=False)
    parser.add_argument("--max_prompts", type=int, default=20)
    parser.add_argument("--output_path", type=str, default="generation_output")
    parser.add_argument("--num_nodes", type=int, default=1)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--human_key", type=str, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    return parser.parse_args()


def load_prompt_string_from_file(prompt_path_str: str):
    if not prompt_path_str.startswith('file::'):
        raise ValueError('prompt_path_str must start with "file::".')
    _, prompt_file_path = prompt_path_str.split('file::', maxsplit=1)
    prompt_file_path = os.path.expanduser(prompt_file_path)
    if not os.path.isfile(prompt_file_path):
        raise FileNotFoundError(
            f'{prompt_file_path=} does not match any existing files.')
    with open(prompt_file_path, 'r') as f:
        # prompt_string = ''.join(f.readlines())
        prompt_string = f.readlines()
        prompt_string = [json.loads(i) if i.startswith("{") else i.replace("\n", "") for i in prompt_string]
    return prompt_string

def main(args):
    tensor_parallel_size = args.tensor_parallel_size
    print(f"########################### Tensor Parallel {tensor_parallel_size}")
    prompt_dicts = load_prompt_string_from_file(
        args.prompts
    )

    prompt_dicts = prompt_dicts[:args.max_prompts]
    prompts = [i["prompt"] for i in prompt_dicts]
    if args.system_prompt is None:
        args.system_prompt = "[INST]<<SYS>>\n \n<</SYS>>\n\n {} [/INST]"
    prompts = [args.system_prompt.format(i) for i in prompts]

    llm = LLM(model=args.name_or_path, tensor_parallel_size=tensor_parallel_size)

    for temp, topp, topk, in itertools.product(
            args.temperature, args.top_p, args.top_k):

        future_df_path = Path(args.output_path)
        future_df_path.mkdir(parents=True, exist_ok=True)
        _future_df_path = "vllm_"
        _future_df_path += f"temp_{temp}_"
        _future_df_path += f"topp_{topp}_"
        _future_df_path += f"topk_{topk}_"
        # _future_df_path += f"repp_{repp}_"
        # _future_df_path += f"nrnz_{nrnz}_"
        # _future_df_path += f"seed_{seed}"
        _future_df_path += ".csv"
        future_df_path = future_df_path / _future_df_path

        params = SamplingParams(
            temperature=temp,
            top_p=topp,
            top_k=topk,
            max_tokens=args.max_new_tokens,
            # ignore_eos=True
        )

        generations = llm.generate(prompts, params)
        future_df = []
        if args.human_key is None:
            args.human_key = "human_text"
        with open(str(future_df_path).replace(".csv", ".json"), "w") as jf:
            for prompt_dict, generated in tqdm(zip(prompt_dicts, generations)):
                future_df.append({
                    "prompt": prompt_dict["prompt"],
                    "original_continuation": prompt_dict[args.human_key],
                    "continuation": generated.outputs[0].text
                })
                prompt_dict["machine_text"] = generated.outputs[0].text
                prompt_dict["model"] = (
                    args.name_or_path.split("/")[-1]
                    if len(args.name_or_path.split("/")[-1]) > 0
                    else args.name_or_path.split("/")[-2]
                )
                jf.write(json.dumps(prompt_dict) + "\n")
            pd.DataFrame(future_df).to_csv(future_df_path)

if __name__ == "__main__":
    main(parse_args())
