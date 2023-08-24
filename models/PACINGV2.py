"""

This version was an attempt to address the bottlenecks in PACINGV1.
While it had improvements, I was still a little out of my league and as the model reflected that.

"""

import torch, numpy as np, os
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

PAD_TOKEN = 50256
VOCAB_SIZE, N_EMDB, N_POSITIONS = 50257, 768, 1024

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

        x = input + self.drop_path(x)
        return x

class PACINGV2(nn.Module):
    def __init__(self, in_chans=3,depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., head_init_scale=1.):
        super().__init__()
        
        self.downsample_layers = nn.ModuleList() 
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=16, stride=16),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=3, stride=1, padding=1),
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

        self.wte = nn.Embedding(VOCAB_SIZE, N_EMDB)
        self.wpe = nn.Embedding(N_POSITIONS, N_EMDB)
        
        #self.pre_head = nn.ConvTranspose1d(dims[-1], 1024, kernel_size=12, stride=12)
        self.pre_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1024),
            nn.InstanceNorm2d(768),
            #nn.GELU(),
            nn.ConvTranspose1d(dims[-1], 768, kernel_size=3, padding=1),
        )
        
        self.head = nn.Linear(self.wte.weight.shape[1], self.wte.weight.shape[0], bias=False)
        self.head.weight = self.wte.weight

        self.apply(self._init_weights)

    def pad_indices(self, indices):

        if indices.shape[1] < N_POSITIONS:
            indices = torch.nn.functional.pad(indices, (0, N_POSITIONS - indices.shape[1]), value=50256)
        else:
            indices = indices[-N_POSITIONS:]
        return indices
    
    def build_image(self, patches):
        # patches: (B, 1024, 3, 16, 16)
        
        # Fast
        patches = patches.reshape(patches.shape[0], 32, 32, 3, 16, 16)
        patches = patches.permute(0, 3, 1, 4, 2, 5)
        image = patches.reshape(patches.shape[0], 3, 512, 512)

        return image

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # Check if bias is present (ignore final logit output layer)
            if m.bias is not None:
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        # [b, 768, 32, 32] -> [b, 768, 1024]
        x = x.view(x.shape[0], x.shape[1], -1)

        # [b, 768, 1024]
        x = self.pre_head(x)

        # [b, 768, 1024] -> [b, 1024, 768]
        x = x.permute(0, 2, 1)
        return x

    def forward(self, raw_input_ids, target_ids=None):
        # Reverse the order of the tokens
        input_ids = torch.flip(raw_input_ids, [1])

        # Padd with 50256
        input_ids = self.pad_indices(input_ids)

        # Prepare the position ids / embeddings
        position_ids = torch.arange(0, input_ids.size(-1) + 0, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        # Embeddings and position encoding
        x = self.wte(input_ids)# + self.wpe(position_ids)

        # Reshape to (B, 1024, 3, 16, 16)
        x = torch.reshape(x, (x.shape[0], x.shape[1], 3, 16, 16))
        
        # Build an image from the patches
        x = self.build_image(x)

        # Run though the convnet
        x = self.forward_features(x)

        # Output logits
        lm_logits = self.head(x)

        #if target_ids is not None:
            #loss = F.cross_entropy(x.view(-1, x.size(-1)), target_ids.view(-1))
            #return x[:, -1], loss

        if target_ids is not None:
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = target_ids[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            return loss


        return lm_logits[:, -1]