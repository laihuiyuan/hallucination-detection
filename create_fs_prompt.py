# -*- coding:utf-8 _*-

import json

# language
lang = 'EN'

# task instruction
instruction = "Given the following question and answer, " \
              "identify any portions of the answer that contain " \
              "hallucinated information or unsupported claims.\n\n"

with open('prompt_zs.txt', 'r') as fp, \
    open('sample/sample_set.v1.json', 'r') as fi, \
    open('prompt_fs.txt', 'w') as fo:
    fo.write(instruction)
    prompt = fp.read()
    for line in fi.readlines():
        line = json.loads(line)

        # check language, you may want to use multilingual data
        if line['lang'] != lang:
            continue

        #collect hallucinated span
        span = ''
        for idx in line['hard_labels']:
            span += line['model_output_text'][idx[0]:idx[1]] + ' || '
        span = span.strip(' || ')
        sample = prompt.format(
            line['model_input'].strip(),
            line['model_output_text'].strip(),
            span)
        fo.write(sample + '\n\n')
    fo.write(prompt.strip('{}'))
