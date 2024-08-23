
import json
from transformers import AutoTokenizer


filename = 'test.json'

model_dir = 'Meta-Llama-3-8B'

tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

print("当前模型的词表大小为：{}".format(len(tokenizer)))

raw_num = 0
token_num = 0
res = []
with open(filename, 'r') as f:
    for line in f.readlines():
        data = json.loads(line.strip())
        text = data['text']
        text_token = tokenizer.tokenize(text)

        raw_num += len(text)
        token_num += len(text_token)
        

print("当前模型压缩率为：{}".format(token_num/raw_num))