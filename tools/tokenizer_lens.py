


import json
from transformers import AutoTokenizer


model_dir = 'Qwen1.5-7B'
tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=False, trust_remote_code=True)

filename = 'text.json'
out_filename = 'text_lens.json'


res = {}

judge = 1000

with open(filename, 'r') as f:
    dataes = json.load(f)
    
    for data in dataes:
        content = data['text']
        text_token = tokenizer.tokenize(content)
        length = (len(text_token) // judge + 1) * judge
        if length in res:
            res[length]  += 1
        else:
            res[length]  = 1
            

with open(out_filename, 'w') as f:
    json.dump(res, f, ensure_ascii=False)
    
        
        
        
        
        
    