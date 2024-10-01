# -*- coding:utf-8 _*-

import re
import json
import argparse
import torch
from vllm import LLM, SamplingParams


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True)
    parser.add_argument('--inp_path', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--prompt_path', type=str, required=True)
    parser.add_argument('--max_tokens', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top_p', type=float, default=1.0)
    parser.add_argument('--presence_penalty', type=float, default=0.0)
    parser.add_argument('--frequency_penalty', type=float, default=0.0)
    parser.add_argument('--stop', type=str, nargs='+', default=['Question:'])
    parser.add_argument('--max_num_batched_tokens', type=int, default=4096)
    args = parser.parse_args()

    # reading data with target prompt
    data, prompts = [], []
    prompt = open(args.prompt_path, 'r', encoding="utf-8").read().strip()
    with open(args.inp_path, 'r', encoding="utf-8") as f:
        for line in f.readlines():
            line = json.loads(line)
            span = ''
            for idx in line['hard_labels']:
                span += line['model_output_text'][idx[0]:idx[1]] + ' || '
            line['gold'] = span.strip(' || ')
            prompts.append(prompt.format(
                line['model_input'],
                line['model_output_text']))
            data.append(line)

    num_gpus = torch.cuda.device_count()
    another_args = {
        'max_num_batched_tokens': args.max_num_batched_tokens,
        'max_model_len': args.max_num_batched_tokens,
        'trust_remote_code': True,
        'tokenizer_mode': 'auto',
        'enforce_eager': True,
    }
    llm = LLM(
        model=args.model_dir,
        tensor_parallel_size=num_gpus,
        **another_args)

    # sampling params
    sampling_params = SamplingParams(
        top_p=args.top_p,
        stop=args.stop,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        presence_penalty=args.presence_penalty,
        frequency_penalty=args.frequency_penalty
    )

    # vllm is async so extra sorting is needed
    outputs = llm.generate(prompts, sampling_params)
    sorted_outputs = sorted(outputs, key=lambda output: int(output.request_id))

    with open(args.out_path, "w", encoding="utf-8") as f:
        for i, output in enumerate(sorted_outputs):
            data[i]['prediction'] = output.outputs[0].text
            
        json.dump(data, f, indent=1, ensure_ascii=False)

