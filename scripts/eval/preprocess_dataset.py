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

def main(args):
    huggingface_hub.login()
    ds_name, ds_type = args.dataset.split(':')
    ds = load_dataset(ds_name, ds_type)
    for i in ds.keys():
        ds[i] = ds[i].map(squad_it_preprocessing_function)
        ds[i].to_json(Path(args.output) / f"{ds_name}_{ds_type}_{i}.jsonl")
    


if __name__ == "__main__":
    

    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='it5/datasets:qa')
    parser.add_argument('--output', type=str)
    args = parser.parse_args()
    main(args)

