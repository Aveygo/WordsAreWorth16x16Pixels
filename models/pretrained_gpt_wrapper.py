from transformers import GPT2LMHeadModel
from torch import nn

class Generator(nn.Module):
    def __init__(self):
        """
        GPT2LMHeadModel wrapper, mainly to match gpt2 and pacing structure for logit outputs
        """
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained("gpt2")

    def forward(self, x):
        return self.model(x).logits[0, -1]
    
    def cuda(self):
        self.model.cuda()
        return self

    def cpu(self):
        self.model.cpu()
        return self
    
    def eval(self):
        self.model.eval()
        return self

    def train(self):
        self.model.train()
        return self