## Quick Start

```python
# create few-shot prompt using sample data
python create_fs_prompt.py

# run few-shot method to detect hallucination spans
python infer_fs.py \
  --model_dir meta-llama/Meta-Llama-3.1-8B-Instruct \
  --prompt_path prompt_fs.txt \
  --inp_path val/mushroom.en-val.v1.jsonl \
  --out_path outputs/llama-3.1-8b-en.json
```
