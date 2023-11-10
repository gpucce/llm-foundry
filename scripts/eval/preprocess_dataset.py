from datasets import load_dataset
import huggingface_hub
from argparse import ArgumentParser
from typing import Dict
from pathlib import Path

def squad_it_preprocessing_function(inp: Dict) -> Dict[str, str]:
    """Format the text string."""
    ITA_PROMPT_FORMAT = 'Dopo aver letto il paragrafo che segue, rispondi correttamente alla successiva domanda. \n\n Paragrafo: {instruction} \n\n Risposta:'
    try:
        prompt = ITA_PROMPT_FORMAT.format(instruction=inp["source"])
        response = inp['target']
    except Exception as e:
        raise ValueError(
            f'Unable to extract prompt/response from {inp=}') from e
    return {'context': prompt, 'continuation': response}

def ns_it_preprocessing_function(inp: Dict) -> Dict[str, str]:
    """Format the text string."""
    ITA_PROMPT_FORMAT = 'Dopo aver letto il testo qui sotto, riassumilo adeguatamente. {target}'
    try:
        prompt = ITA_PROMPT_FORMAT.format(target=inp["target"])
        response = inp['target']
    except Exception as e:
        raise ValueError(
            f'Unable to extract prompt/response from {inp=}') from e
    return {'context': prompt, 'continuation': response}


def fts_f2i_it_preprocessing_function(inp: Dict) -> Dict[str, str]:
    """Format the text string."""
    ITA_PROMPT_FORMAT = 'Dato il seguente testo scritto in modo formale, riscrivilo in modo informale. {formal}'
    try:
        prompt = ITA_PROMPT_FORMAT.format(formal=inp["formal"])
        response = inp['informal']
    except Exception as e:
        raise ValueError(
            f'Unable to extract prompt/response from {inp=}') from e
    return {'context': prompt, 'continuation': response}

def fts_i2f_it_preprocessing_function(inp: Dict) -> Dict[str, str]:
    """Format the text string."""
    ITA_PROMPT_FORMAT = 'Dato il seguente testo scritto in modo informale, riscrivilo in modo formale. {informal}'
    try:
        prompt = ITA_PROMPT_FORMAT.format(informal=inp["informal"])
        response = inp['formal']
    except Exception as e:
        raise ValueError(
            f'Unable to extract prompt/response from {inp=}') from e
    return {'context': prompt, 'continuation': response}

PREPROCESSING_MAP = {
    "qa": squad_it_preprocessing_function,
    "ns": ns_it_preprocessing_function,
    "fst_formal": fts_f2i_it_preprocessing_function,
    "fst_informal": fts_i2f_it_preprocessing_function,
}

def main(args):
    huggingface_hub.login()
    ds_name, ds_type = args.dataset.split(':')
    _ds_type = ds_type.split("_")[0]
    ds = load_dataset(ds_name, _ds_type)
    preprocessing_fn = PREPROCESSING_MAP[ds_type]
    for i in ds.keys():
        ds[i] = ds[i].map(preprocessing_fn)
        ds[i].to_json(Path(args.output) / f"{ds_name}_{ds_type}_{i}.jsonl")
    


if __name__ == "__main__":
    

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='it5/datasets:qa')
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    main(args)

