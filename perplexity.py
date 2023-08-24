from datasets import load_dataset
from transformers import GPT2TokenizerFast
from utils import get_model
import torch, random

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

model_name = "PACINGV4"
model = get_model(model_name, load_latest=True).cuda()

import torch
from tqdm import tqdm

def pad(tokens, max_length=1024, pad_token=50256):
    return torch.nn.functional.pad(tokens, (max_length-tokens.shape[1], 0), value=pad_token)

max_length = 1024
stride = 512
seq_len = encodings.input_ids.size(1)
nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc
    input_ids = encodings.input_ids[:, begin_loc:end_loc].cuda()
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        neg_log_likelihood = model(pad(input_ids), pad(target_ids))

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl)