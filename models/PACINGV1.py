"""

This is the first version of PACING.
Major bottlenecks with the final layer disconnected it from the inputs.
For future references, this implementation should be used as example of how not to do large language models.

"""

import torch
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

class PACINGV1(nn.Module):
    def __init__(self, in_chans=3, depths=[3, 3, 20, 3], dims=[128, 256, 512, 1024], drop_path_rate=0., head_init_scale=1.):
        super().__init__()
        
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

        self.wte = nn.Embedding(VOCAB_SIZE, N_EMDB)
        self.wpe = nn.Embedding(N_POSITIONS, N_EMDB)
        self.pre_head = nn.ConvTranspose1d(dims[-1], 1024, kernel_size=12, stride=12)
        
        self.head = nn.Linear(self.wte.weight.shape[1], self.wte.weight.shape[0], bias=False)
        self.head.weight = self.wte.weight

        self.softmax = nn.Softmax(dim=1)

        self.apply(self._init_weights)

    def pad_indices(self, indices):

        if indices.shape[1] < N_POSITIONS:
            indices = torch.nn.functional.pad(indices, (0, N_POSITIONS - indices.shape[1]), value=50256)
        else:
            indices = indices[-N_POSITIONS:]
        return indices
    
    def build_image(self, patches):
        image = torch.zeros((patches.shape[0], 3, 256, 256)).cuda()
        for i in range(16):
            for j in range(16):
                image[:, :, i*16:(i+1)*16, j*16:(j+1)*16] = patches[:, i*16+j, :, :, :]

        return image

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            if m.bias is not None:
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        
        x = x.view(x.shape[0], x.shape[1], -1)
        x = self.pre_head(x)
        return x

    def forward(self, input_ids):
        # Reverse the order of the tokens
        input_ids = torch.flip(input_ids, [1])

        # Padd with 50256
        input_ids = self.pad_indices(input_ids)

        # Prepare the position ids / embeddings
        position_ids = torch.arange(0, input_ids.size(-1) + 0, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        # Embeddings and position encoding
        x = self.wte(input_ids) + self.wpe(position_ids)

        # Reshape to (B, 1024, 3, 16, 16)
        x = torch.reshape(x, (x.shape[0], x.shape[1], 3, 16, 16))
        
        # Build an image from the patches
        x = self.build_image(x)

        # Run though the convnet
        x = self.forward_features(x)

        # Output logits
        x = self.head(x)

        return x[:, -1]