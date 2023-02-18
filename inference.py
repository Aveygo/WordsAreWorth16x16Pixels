import torch
from transformers import GPT2TokenizerFast
from utils import get_model, MODEL_NAMES, latest_state_dict, set_seed

set_seed(43)

tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
model_id = 1
model = get_model(MODEL_NAMES[model_id]).eval().cuda()
model.load_state_dict(latest_state_dict(MODEL_NAMES[model_id]))

start_idx = 1000
prompt = open("shakespeare.txt").read()[start_idx:start_idx+2048]

print("Prompt:")
print("-"*80)
print(prompt)
print("-"*80)

tokens = tokenizer.encode(prompt)

def sample(tokens, logits):

    logits[tokens[-1]] = -100
    logits[tokens[-2]] = -100
    logits[220] = -100

    n = 50
    probs, indices = torch.topk(logits, n)
    probs = torch.ones_like(probs)
    token = indices[torch.multinomial(probs, 1)]

    return token.item()

for i in range(100):

    with torch.no_grad():
        logits = model(torch.tensor(tokens).unsqueeze(0).cuda())

    logits = logits.detach().cpu().squeeze(0)
    token = sample(tokens, logits)

    tokens.append(token)

print("Output:")
print("-"*80)
print(tokenizer.decode(tokens[-100:]))
print("-"*80)