"""

Refactoring and combining hidden layers for a single token prediction.
This is the implementation used in the paper that I submitted for A.T.3 for my university.

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
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)

        return input + self.drop_path(x)

class ConvNeXtV2(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., head_init_scale=1.):
        super().__init__()
        self.depths = depths
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

from transformers import GPT2LMHeadModel

class PACINGV3(nn.Module):
    def __init__(self, seq_len=1024, n_embd=768, num_layers=10, vocab_size=50257, in_channels=3, distil:GPT2LMHeadModel|None|bool = True):
        super().__init__()
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.num_blocks = num_layers
        self.vocab_size = vocab_size
        self.in_channels = in_channels

        self.patch_num = int((self.seq_len)**(1/2))
        self.patch_size = int((n_embd//in_channels)**(1/2))

        if type(distil) == bool and distil:
            distil = GPT2LMHeadModel.from_pretrained("gpt2").eval().cuda()

        self.distil = distil

        self.ln_f = nn.LayerNorm(n_embd)
        self.wte = nn.Embedding(vocab_size, n_embd)
        self.convnext = ConvNeXtV2(in_channels, vocab_size, depths=[3, 3, 9, 3], dims=[96, 192, 384, n_embd])
        self.convnext.head.weight = self.wte.weight

        self.loss_fct = nn.CrossEntropyLoss()
        self.loss_distil_fn = nn.CosineSimilarity()

    def build_image(self, embeddings):
        patches = torch.reshape(embeddings, (embeddings.shape[0], embeddings.shape[1], self.in_channels, self.patch_size, self.patch_size))
        patches = patches.reshape(patches.shape[0], self.patch_num, self.patch_num, self.in_channels, self.patch_size, self.patch_size)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        return patches.reshape(patches.shape[0], self.in_channels, self.patch_size*self.patch_num, self.patch_size*self.patch_num)
    
    def forward(self, input_ids, lm_labels=None, gpt_logits=None):
        x = self.wte(input_ids)
        x = self.ln_f(x)
        x = self.build_image(x)

        lm_logits = self.convnext(x)

        if lm_labels is not None:
            if not type(self.distil) == bool:
                if gpt_logits is None:
                    gpt_logits = self.distil(input_ids)[0][:, -1]
                
                return (1 - self.loss_distil_fn(
                    lm_logits.view(-1, lm_logits.size(-1)),
                    gpt_logits.view(-1, gpt_logits.size(-1))
                ).mean()) #+ self.loss_fn(lm_logits.view(-1, lm_logits.size(-1)), lm_labels[:, -1].view(-1))
            else:
                return self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels[:, -1].view(-1))
        
        return lm_logits
    
if __name__ == "__main__":
    model = PACINGV3()
    x = torch.zeros(2, 1024).long()
    y = model(x, torch.zeros((2,1024)).long())
    print(y)