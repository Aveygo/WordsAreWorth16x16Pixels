"""
Replacing the Attention layers in GPT with ConvNext
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from typing import Optional, Tuple, Union


class GRN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        
class Block(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, past=None):
        input = x if past is None else past
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        return input + self.drop_path(x)

from models.GPT import GPT

class NewGPTBlock(nn.Module):
    def __init__(self, seq_len=1024, emb_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.emb_dim = emb_dim
        self.convnext = Block(emb_dim)
        self.register_buffer("bias", torch.tril(torch.ones(self.seq_len, self.seq_len)).view(1, 1, self.seq_len, self.seq_len))

    def forward(self, x, past):
        
        # x: [b, seq_len, emb_dim] -> [b, emb_dim, seq_len]
        x = torch.transpose(x, 1, 2)

        # [b, emb_dim, seq_len] -> [b, emb_dim, sqrt(seq_len), sqrt(seq_len)]
        x = x.view(-1, self.emb_dim, int(self.seq_len**(1/2)), int(self.seq_len**(1/2)))

        # Main block
        x = self.convnext(x, past)

        # Revert back
        x = x.view(-1, self.emb_dim, self.seq_len)
        x = torch.transpose(x, 1, 2)

        nd, ns = x.size(-2), x.size(-1)
        b = self.bias[:, :, ns-nd:ns, :ns]
        print(b.shape, nd, ns)
        x = x * b - 1e10 * (1 - b)

        return x, None
        

class GPTAUG(nn.Module):
    def __init__(self):
        super().__init__()

        self.gpt = GPT()
        self.gpt.transformer.h = nn.ModuleList([NewGPTBlock() for i in range(self.gpt.transformer.n_layer)])
    
    def forward(self, input_ids, lm_labels=None):
        return self.gpt(input_ids, lm_labels)






