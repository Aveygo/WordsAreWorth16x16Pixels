from transformers import GPT2TokenizerFast
import numpy as np, os

if not os.path.exists("tokens.npy"):    
    tokenizer = GPT2TokenizerFast.from_pretrained('gpt2')
    
    text = open("shakespeare.txt", "r").read()
    tokens = tokenizer.encode(text)
    tokens = np.array(tokens)
    
    np.save("tokens.npy", tokens)

class TokenDataset:
    def __init__(self, max_tokens=1024):
        self.max_tokens = max_tokens
        self.tokens = np.load("tokens.npy")
    
    def __len__(self):
        return len(self.tokens) - self.max_tokens

    def __getitem__(self, index):
        x = self.tokens[index:index+self.max_tokens]
        y = self.tokens[self.max_tokens+index]
        return x, y