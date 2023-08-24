"""

PACINGV5, translation tasks

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
    def __init__(self, dim, drop_path=0.0, drop_out=0.1):
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

class PACING(nn.Module):
    def __init__(self, seq_len=1024, emb_dim=768, features=128, num_blocks=10, vocab_size=50257):
        super().__init__()
        self.emb_dim = emb_dim
        self.seq_len = seq_len

        self.pool = nn.AdaptiveAvgPool2d((seq_len, emb_dim))
        self.blocks = nn.Sequential(*[Block(features) for i in range(num_blocks)])

        self.ln_f = nn.LayerNorm(emb_dim)
        self.wte = nn.Embedding(vocab_size, emb_dim)
        self.apply(self._init_weights)

        self.down1 = nn.Conv2d(emb_dim//256, features//2, kernel_size=3, stride=2, padding=1)
        self.down2 = nn.Conv2d(features//2, features, kernel_size=3, stride=2, padding=1)
        self.up1 = nn.ConvTranspose2d(features, features//2, kernel_size=2, stride=2, padding=0)
        self.up2 = nn.ConvTranspose2d(features//2, emb_dim//256, kernel_size=2, stride=2, padding=0)

        self.relu = nn.ReLU()
        self.shfl = nn.PixelShuffle(16)
        self.unshfl = nn.PixelUnshuffle(16)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if m.bias is not None:
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)
        if isinstance(m, (nn.Embedding)):
            nn.init.uniform_(m.weight, a=-1e-4, b=1e-4)

    def imageify(self, x:torch.Tensor):
        # [b, emb_dim, seq_len] -> [b, seq_len, emb_dim] -> [b, 3, width, height]
        #x = torch.transpose(x, 1, 2)
        #x = x.view(-1, self.emb_dim, int(self.seq_len**(1/2)), int(self.seq_len**(1/2)))
        #return self.shfl(x)
        # [b, emb_dim, seq_len] -> [b, 1024, 2, 16, 16]
        x = x.reshape(x.shape[0], 32, 32, 3, 16, 16)
        x = x.permute(0, 3, 1, 4, 2, 5)
        return x.reshape(x.shape[0], 3, 512, 512)

    def deimageify(self, x:torch.Tensor):
        # [b, 3, width, height] -> [b, seq_len, emb_dim] -> [b, emb_dim, seq_len]
        #x = self.unshfl(x)
        #x = x.view(-1, self.emb_dim, self.seq_len)
        #return torch.transpose(x, 1, 2)
        x = x.reshape(x.shape[0], 32, 32, 3, 16, 16)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return torch.reshape(x, (x.shape[0], self.seq_len, self.emb_dim))

    def main_blocks(self, x, memory=None):
        for block in self.blocks:
            x = block(x)
            if not memory is None:
                x = x + memory

        return x
        
    def forward(self, input_ids):
        # Convert to embeddings
        # [b, seq_len] -> [b, seq_len, emb_dim]
        x = self.wte(input_ids)
        #x = self.ln_f(x)

        x1 = self.imageify(x)

        # Downscale
        x2 = self.relu(self.down1(x1))
        x3 = self.relu(self.down2(x2))

        # Main blocks
        hidden_state = self.main_blocks(x3) + x3

        # Upscale
        x5 = self.relu(self.up1(hidden_state)) + x2
        x6 = self.relu(self.up2(x5)) + x1

        return self.deimageify(x6)

class Head(nn.Module):
    def __init__(self, wte:nn.Embedding, emb_dim):
        super().__init__()
        self.ln_f = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(wte.weight.shape[1], wte.weight.shape[0], bias=False)
        self.head.weight = wte.weight
    
    def forward(self, x):
        # Norm, Token probability
        # [b, 1, 768] -> [b, vocab_size]
        #x = self.ln_f(x)
        return self.head(x)

class PACINGV6TINY(nn.Module):
    def __init__(self, seq_len=1024, emb_dim=768, features=128, num_blocks=10, vocab_size=50257):
        super().__init__()
        assert (emb_dim/256).is_integer(), "Embedding must be divisible my 256!"

        self.encoder = PACING(seq_len, emb_dim, features, num_blocks, vocab_size)
        self.head = Head(self.encoder.wte, emb_dim)
        self.loss_fn = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
    
    def forward(self, input_ids, outputs=None):
        lm_logits = self.encoder(input_ids)

        pred = self.head(lm_logits)
        
        if not outputs is None:
            
            return self.loss_fn(
                pred.view(-1, pred.size(-1)),
                outputs.view(-1)
            )
            #+ self.mse(
            #    self.encoder.ln_f(self.encoder.wte(outputs)),
            #    lm_logits
            #)
        
        return pred[:, -1]

               
if __name__ == "__main__":
    model = PACINGV6TINY().cuda()
    x1 = torch.zeros(2, 1024).long().cuda()
    x2 = torch.zeros(2, 1024).long().cuda()

    #y1 = torch.zeros(2, 1).long().cuda()

    loss = model(x1, x2)
    
    print(loss)
