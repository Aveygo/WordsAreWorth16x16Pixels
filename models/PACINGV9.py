"""

Tried to use a more GPT oriented model, so instead of making a 3x512x512 image from the input,
I instead just reshape it into a 1024x16x16 "hidden state".

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

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
    def __init__(self, dim, drop_path=0.0, drop_out=0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_out = nn.Dropout2d(drop_out)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.drop_out(x)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        x = input + self.drop_path(x)
        return x

class PACINGV9(nn.Module):
    def __init__(self, seq_len=1024, emb_dim=768, num_blocks=10, vocab_size=50257):
        super().__init__()
        assert (seq_len**(1/2)).is_integer(), "Embedding must be square!"

        self.width = int(seq_len**(1/2))
        self.seq_len = seq_len
        self.emb_dim = emb_dim

        self.blocks = nn.Sequential(*[Block(emb_dim) for i in range(num_blocks)])

        self.ln_f = nn.LayerNorm(emb_dim)
        self.wte = nn.Embedding(vocab_size, emb_dim)
        self.head = nn.Linear(self.wte.weight.shape[1], self.wte.weight.shape[0], bias=False)
        self.head.weight = self.wte.weight

        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=1/vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if m.bias is not None:
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.Embedding)):
            nn.init.uniform_(m.weight, a=-1e-4, b=1e-4)


    def forward(self, input_ids, labels=None):
        
        # Convert to embeddings
        # [b, l] -> [b, l, emb_dim]
        x = self.wte(input_ids)
        x = self.ln_f(x)

        x = torch.transpose(x, 2, 1)

        # Reshape to input images
        # [b, emb_dim, seq_len] - > [b, emb_dim, width, width]
        x = torch.reshape(x, (x.shape[0], x.shape[1], self.width, self.width))
        
        # Main blocks
        x = self.blocks(x)

        # Flatten
        # [b, emb_dim, width, width] -> [b, emb_dim, seq_len]
        x = torch.reshape(x, (x.shape[0], x.shape[1], self.seq_len))
        x = torch.transpose(x, 2, 1)

        # Norm
        x = self.ln_f(x)

        # Token probability
        lm_logits = self.head(x)

        if labels is not None:

            return self.loss_fn(
                lm_logits.view(-1, lm_logits.size(-1)),
                labels.view(-1)
            )
            
        return lm_logits[:, -1]
               
if __name__ == "__main__":
    model = PACINGV9().cuda()
    x = torch.zeros(2, 1024).long().cuda()
    y = model(x, torch.zeros((2,1024)).long().cuda())
    print(y)
