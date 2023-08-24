"""

Trying to transfer GPT attention "blocks" and convert them into ConvNext ones.

"""

from models.PACINGV4 import Block
from transformers import GPT2Model

import torch
import torch.nn as nn

class CustomBlock(nn.Module):
    def __init__(self, max_length=1024, emb_size=(32, 24)):
        super().__init__()
        self.emb_size = emb_size
        self.block = Block(max_length)
        self.loss_fn = nn.MSELoss()

    def forward(self, inputs, outputs=None, **args):
        inputs_emb = torch.reshape(inputs, (inputs.shape[0], inputs.shape[1], self.emb_size[0], self.emb_size[1]))
        pred = self.block(inputs_emb)
        pred = torch.reshape(pred, inputs.shape)

        if self.training and outputs is not None:
            return self.loss_fn(pred, outputs)  
        else:
            return pred, None


class PACINGHOOKED(nn.Module):
    def __init__(self, max_len=1024, pad_token=50256, gpt_model:GPT2Model|None = None, tie_weights=True):
        super().__init__()
        self.gpt_model = gpt_model
        self.max_len = max_len
        self.pad_token = pad_token

        if gpt_model is None:
            self.gpt_model = GPT2Model.from_pretrained("gpt2").eval().cuda()
        
        self.losses = None
        children = list(self.gpt_model.h.named_children())
        self.last_idx = len(children)
        self.h = nn.ModuleList([CustomBlock().train().cuda() for i in range(self.last_idx)])

        for i, (name, layer) in enumerate(children):
            layer.register_forward_hook(
                lambda layer, input, output, i=i: self.process_data(layer, input, output, i)
            )
        
    def train(self, mode):
        self.gpt_model.train(False)
        self.h.train(mode)
                
    def process_data(self, layer, input, output, idx):
        bl = self.h[idx](input[0], output[0])
        self.losses = self.losses + bl if idx else bl  

    def forward(self, input_ids, labels=None):

        if self.training:
            self.gpt_model(input_ids)
            return self.losses
        else:
            self.gpt_model.h = self.h.eval()
            return self.gpt_model(input_ids)[0][:, -1]
    
if __name__ == "__main__":
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    x = torch.zeros(2, 1024).long()

    model = PACINGHOOKED().eval()

    outputs = model(x.cuda())
    print(outputs)
        