# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

r"""Example custom preprocessing function.

This is here to help illustrate the way to set up finetuning
on a local dataset. One step of that process is to create
a preprocessing function for your dataset, and that is what
is done below. Check out the LLM Finetuning section of
`../README.md` for more context.

For this example, we're going to pretend that our local dataset
is `./train.jsonl`.

Note: this dataset is actually a copy of one of our ARC-Easy
multiple-choice ICL eval datasets. And you would never actually
train on eval data! ... But this is just a demonstration.

Every example within the dataset has the format:
{
    'query': <query text>,
    'choices': [<choice 0 text>, <choice 1 text>, ...],
    'gold': <int> # index of correct choice
}

To enable finetuning, we want to turn this into a prompt/response
format. We'll structure prompts and responses like this:
{
    'prompt': <query text>\nOptions:\n - <choice 0 text>\n - <choice 1 text>\nAnswer: ,
    'response': <correct choice text>
}
"""

from typing import Dict, List, Union


def multiple_choice(
        inp: Dict[str, Union[str, List[str], int]]) -> Dict[str, str]:
    PROMPT_FORMAT = '{query}\nOptions:{options}\nAnswer: '
    options = ''
    assert isinstance(inp['choices'], List)
    for option in inp['choices']:
        options += f'\n - {option}'
    query = inp['query']

    assert isinstance(inp['gold'], int)
    return {
        'prompt': PROMPT_FORMAT.format(query=query, options=options),
        'response': inp['choices'][inp['gold']],
    }


def camoscio_preprocessing_function(inp: Dict) -> Dict[str, str]:
    """Format the text string."""
    ITA_PROMPT_FORMAT_NO_INPUT = 'Di seguito è riportata una istruzione che descrive un task. Scrivete una risposta che completi in modo appropriato la richiesta.\n\n### Istruzione:\n{instruction}\n\n### Risposta:\n'
    ITA_PROMPT_FORMAT = 'Di seguito è riportata una istruzione che descrive un task. Scrivete una risposta che completi in modo appropriato la richiesta.\n\n### Istruzione:\n{instruction}\n\n### Input:\n{input}\n\n### Risposta:\n'
    try:
        if inp['input'] != '':
            prompt = ITA_PROMPT_FORMAT.format(instruction=inp["instruction"], input=inp['input'])
        else:
            prompt = ITA_PROMPT_FORMAT_NO_INPUT.format(instruction=inp["instruction"])
        response = inp['output']
    except Exception as e:
        raise ValueError(
            f'Unable to extract prompt/response from {inp=}') from e
    return {'prompt': prompt, 'response': response}
